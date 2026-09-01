"""ConvCNP station downscaler with optional TESSERA conditioning.

A convolutional conditional neural process in the style of Vaughan et al.
(2022), written as a plain ``torch.nn.Module``. Given a gridded ERA5
context and a set of station locations it predicts, for each station and
each target variable, the parameters of a predictive distribution.

Forward pass::

    context grid  (batch, C, n_lat, n_lon)
        → GridCNN               residual CNN, spatial size preserved
        → grid-to-point interp  BilinearInterp (default) or RBFSetConv
        → DecoderMLP            per-station MLP over
                                  [interpolated grid features,
                                   elevation / Δelevation / mTPI (÷1000),
                                   precomputed TESSERA vector (optional)]
        → LikelihoodHeadDict    one Linear per target variable
        → {var: {param: (batch, n_targets)}}

Grid-to-point interpolation
---------------------------
``interpolation="bilinear"`` (the default) samples the CNN feature grid at
the station coordinates with parameter-free bilinear interpolation.
``interpolation="setconv"`` is the vanilla ConvCNP SetConv: a separable RBF
kernel with a single learned log length-scale (``setconv_length_scale``
initialises it, in degrees); it adds one parameter, ``interp.log_scale``.

TESSERA conditioning
--------------------
The model does not encode TESSERA patches itself. When
``tessera_features_precomputed=True`` the batch carries a precomputed
per-station vector of width ``precomputed_tessera_dim`` (the paper uses the
16-d latent of a VAE trained on TESSERA patches), and with
``tessera_injection="concat"`` that vector is concatenated onto the decoder
MLP input alongside the topographic features. ``tessera_injection="none"``
disables the injection (used by the cross-lead baselines, which share a
data pipeline with the TESSERA runs).

Checkpoint layout
-----------------
State-dict keys are ``cnn.net.*``, ``interp.log_scale`` (SetConv only),
``mlp.net.*`` and ``heads.heads.<var>.linear.*``; they must not change, so
that checkpoints on disk keep loading with ``strict=True``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads_dispatch import LikelihoodHeadDict

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Single residual block: two convolutions with a skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.relu(self.conv1(x))) + x)


class GridCNN(nn.Module):
    """Residual CNN that processes the ERA5 grid.

    Spatial dimensions are preserved (no striding or pooling).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output feature channels.
        hidden_channels: Width of the residual blocks.
        num_layers: Total number of conv layers.
        kernel_size: Convolution kernel size.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 7,
        kernel_size: int = 3,
    ):
        super().__init__()
        pad = kernel_size // 2

        # Input projection → residual blocks → output projection.
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad),
            nn.ReLU(),
        ]
        # Each residual block contains 2 conv layers.
        n_blocks = (num_layers - 1) // 2
        layers.extend(
            ResidualBlock(hidden_channels, kernel_size) for _ in range(n_blocks)
        )
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBFSetConv(nn.Module):
    """Separable RBF kernel interpolation from a regular grid to points.

    The vanilla ConvCNP SetConv: a Gaussian kernel with a single learned
    length-scale (stored as ``log_scale``), applied separably in latitude
    and longitude and normalised by the kernel density.

    Args:
        init_length_scale: Initial RBF length scale in degrees. Default is
            0.5° (~55 km at mid-latitudes), chosen to start with a bias
            toward locality rather than region-wide smoothing.

    """

    def __init__(self, init_length_scale: float = 0.5):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_length_scale)))

    def forward(
        self,
        grid_features: torch.Tensor,
        grid_lats: torch.Tensor,
        grid_lons: torch.Tensor,
        target_lats: torch.Tensor,
        target_lons: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate grid features to target locations.

        Args:
            grid_features: ``(batch, C, n_lat, n_lon)``
            grid_lats: ``(n_lat,)``
            grid_lons: ``(n_lon,)``
            target_lats: ``(batch, n_targets)``
            target_lons: ``(batch, n_targets)``

        Returns:
            ``(batch, C, n_targets)``

        """
        scale = torch.exp(self.log_scale)

        # 1-D RBF weights along each axis.
        weights_lat = torch.exp(
            -0.5 * ((grid_lats[None, :, None] - target_lats[:, None, :]) / scale) ** 2
        )
        weights_lon = torch.exp(
            -0.5 * ((grid_lons[None, :, None] - target_lons[:, None, :]) / scale) ** 2
        )

        # Separable interpolation: contract lat then lon.
        out = torch.einsum("bcij,bit->bcjt", grid_features, weights_lat)
        out = torch.einsum("bcjt,bjt->bct", out, weights_lon)

        # Density normalisation.
        density = torch.einsum("bit,bjt->bt", weights_lat, weights_lon)
        return out / (density[:, None, :] + 1e-8)


class BilinearInterp(nn.Module):
    """Bilinear interpolation from a regular grid to arbitrary points.

    Unlike :class:`RBFSetConv` this has no learnable parameters and applies
    standard bilinear interpolation at the native grid resolution, so the
    weather pathway carries only smooth, grid-scale features; any local
    station correction has to come from the per-station decoder inputs
    (topography and, when present, the TESSERA vector).

    Uses ``torch.nn.functional.grid_sample`` for efficient, differentiable
    bilinear interpolation.
    """

    def forward(
        self,
        grid_features: torch.Tensor,
        grid_lats: torch.Tensor,
        grid_lons: torch.Tensor,
        target_lats: torch.Tensor,
        target_lons: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate grid features to target locations.

        Args:
            grid_features: ``(batch, C, n_lat, n_lon)``
            grid_lats: ``(n_lat,)`` — must be regularly spaced.
            grid_lons: ``(n_lon,)`` — must be regularly spaced.
            target_lats: ``(batch, n_targets)``
            target_lons: ``(batch, n_targets)``

        Returns:
            ``(batch, C, n_targets)``

        """
        # Convert target coordinates to normalised [-1, 1] grid for grid_sample.
        # grid_sample expects (x, y) in [-1, 1] where -1 is left/top, +1 is
        # right/bottom.
        lat_min, lat_max = grid_lats[0], grid_lats[-1]
        lon_min, lon_max = grid_lons[0], grid_lons[-1]

        # Normalise to [-1, 1].
        norm_lats = 2.0 * (target_lats - lat_min) / (lat_max - lat_min) - 1.0
        norm_lons = 2.0 * (target_lons - lon_min) / (lon_max - lon_min) - 1.0

        # grid_sample expects (batch, n_targets, 1, 2) with (x=lon, y=lat).
        grid = torch.stack([norm_lons, norm_lats], dim=-1)  # (batch, n_targets, 2)
        grid = grid.unsqueeze(2)  # (batch, n_targets, 1, 2)

        # grid_sample takes input (batch, C, H, W) and grid (batch, H_out, W_out, 2)
        # and returns (batch, C, n_targets, 1).
        sampled = F.grid_sample(
            grid_features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return sampled.squeeze(-1)  # (batch, C, n_targets)


class DecoderMLP(nn.Module):
    """Pointwise MLP body. Maps ``(in_features) → (hidden_dim)`` per station.

    ``n_hidden_layers`` blocks of ``Linear → ReLU``; the output is the
    post-ReLU hidden state, which the per-variable likelihood heads (one
    ``nn.Linear`` each, see :mod:`.heads`) project to distribution
    parameters. There is deliberately no output projection here.

    State-dict layout (even indices are ``Linear``, odd are ``ReLU``)::

        mlp.net.0.{weight,bias}          Linear(in_features, hidden_dim)
        mlp.net.2.{weight,bias}          Linear(hidden_dim, hidden_dim)
        ...
        mlp.net.{2*(n-1)}.{weight,bias}  last hidden Linear

    Args:
        in_features: Total input dimension.
        hidden_dim: Width of hidden layers and of the body output.
        n_hidden_layers: Number of Linear+ReLU blocks in the body.

    """

    def __init__(
        self,
        in_features: int = 130,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = in_features
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(current, hidden_dim), nn.ReLU()])
            current = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, n_targets, in_features) → (batch, n_targets, hidden_dim)``."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class ConvCNPDownscaler(nn.Module):
    """ConvCNP for station downscaling, with optional TESSERA conditioning.

    Args:
        n_context_channels: Number of channels in the ERA5 context grid
            (dynamic + static + coordinate + temporal channels).
        cnn_hidden: CNN hidden channel width (also the number of grid
            feature channels reaching the decoder).
        cnn_layers: Number of CNN conv layers.
        cnn_kernel: CNN kernel size.
        setconv_length_scale: Initial RBF length scale in degrees. Only
            used when ``interpolation="setconv"``.
        interpolation: Grid-to-station interpolator: ``"bilinear"``
            (default; parameter-free) or ``"setconv"`` (vanilla SetConv
            with a learned length-scale).
        mlp_hidden: Decoder MLP hidden width (also the input width of each
            likelihood head).
        mlp_n_hidden: Number of ``Linear → ReLU`` blocks in the decoder MLP.
        n_elev_features: Number of per-station topographic scalars appended
            to the decoder input when ``include_elevation`` is True: 2 =
            (elevation, Δelevation to the ERA5 orography); 3 additionally
            appends mTPI, matching the auxiliary vector of Vaughan et al.
            (2022). All are divided by 1000 (m → km) in :meth:`forward`.
        include_elevation: Whether to append the topographic scalars at all.
        target_variables: Ordered list of target variable names. Fixes the
            iteration order of the heads and of the returned dict.
        likelihood_per_variable: Mapping ``{var_name: dist_name}`` where
            ``dist_name`` is a key of
            :data:`~tessera_downscaling.model.heads.HEAD_REGISTRY`
            (``"gaussian"`` or ``"truncated_normal"``). Must be 1:1 with
            ``target_variables``. Defaults to Gaussian for every variable.
        tessera_injection: ``"concat"`` (default) appends the precomputed
            TESSERA vector to the decoder input; ``"none"`` ignores it.
        tessera_features_precomputed: When True the forward pass expects
            ``target_tessera`` of shape ``(batch, n_targets,
            precomputed_tessera_dim)`` — a per-station vector computed
            offline (the paper's 16-d VAE latent).
        precomputed_tessera_dim: Width of that vector. Required (> 0) when
            ``tessera_features_precomputed=True``.

    """

    def __init__(
        self,
        n_context_channels: int = 38,
        cnn_hidden: int = 128,
        cnn_layers: int = 7,
        cnn_kernel: int = 3,
        setconv_length_scale: float = 0.5,
        interpolation: str = "bilinear",
        mlp_hidden: int = 128,
        mlp_n_hidden: int = 3,
        n_elev_features: int = 2,
        include_elevation: bool = True,
        target_variables: list[str] | None = None,
        likelihood_per_variable: dict[str, str] | None = None,
        tessera_injection: str = "concat",
        tessera_features_precomputed: bool = False,
        precomputed_tessera_dim: int = 0,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if target_variables is None or len(target_variables) == 0:
            raise ValueError(
                "target_variables must be a non-empty list of variable names"
            )
        if interpolation not in ("bilinear", "setconv"):
            raise ValueError(
                f"Unknown interpolation {interpolation!r}. "
                "Supported: 'bilinear', 'setconv'."
            )
        if tessera_injection not in ("concat", "none"):
            raise ValueError(
                f"tessera_injection must be 'concat' or 'none'; "
                f"got {tessera_injection!r}."
            )
        if tessera_features_precomputed and precomputed_tessera_dim <= 0:
            raise ValueError(
                "precomputed_tessera_dim must be > 0 when "
                "tessera_features_precomputed=True"
            )
        if likelihood_per_variable is None:
            likelihood_per_variable = dict.fromkeys(target_variables, "gaussian")

        # ------------------------------------------------------------------
        # Bookkeeping
        # ------------------------------------------------------------------
        self.tessera_features_precomputed = tessera_features_precomputed
        self.precomputed_tessera_dim = (
            precomputed_tessera_dim if tessera_features_precomputed else 0
        )
        self.include_elevation = include_elevation
        self.n_elev_features = n_elev_features
        self.target_variables = list(target_variables)
        self.n_target_variables = len(target_variables)
        self.likelihood_per_variable = dict(likelihood_per_variable)
        self.interpolation_method = interpolation
        self.tessera_injection = tessera_injection
        self.mlp_hidden = mlp_hidden

        # ------------------------------------------------------------------
        # Decoder input width
        # ------------------------------------------------------------------
        mlp_in = cnn_hidden
        if include_elevation:
            mlp_in += n_elev_features
        if tessera_injection == "concat":
            mlp_in += self.precomputed_tessera_dim

        # ------------------------------------------------------------------
        # Modules. The attribute names fix the state-dict prefixes
        # (cnn. / interp. / mlp. / heads.) — do not rename.
        # ------------------------------------------------------------------
        self.cnn = GridCNN(
            in_channels=n_context_channels,
            out_channels=cnn_hidden,
            hidden_channels=cnn_hidden,
            num_layers=cnn_layers,
            kernel_size=cnn_kernel,
        )
        if interpolation == "setconv":
            self.interp = RBFSetConv(init_length_scale=setconv_length_scale)
        else:
            self.interp = BilinearInterp()
        self.mlp = DecoderMLP(
            in_features=mlp_in,
            hidden_dim=mlp_hidden,
            n_hidden_layers=mlp_n_hidden,
        )
        self.heads = LikelihoodHeadDict(
            likelihood_per_variable=likelihood_per_variable,
            target_variables=target_variables,
            hidden_dim=mlp_hidden,
        )

    def forward(
        self,
        context_grid: torch.Tensor,
        grid_lats: torch.Tensor,
        grid_lons: torch.Tensor,
        target_coords: torch.Tensor,
        target_elev: torch.Tensor,
        target_delta_elev: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        target_tessera: torch.Tensor | None = None,
        target_mtpi: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            context_grid: ``(batch, C, n_lat, n_lon)``
            grid_lats: ``(n_lat,)``
            grid_lons: ``(n_lon,)``
            target_coords: ``(batch, n_targets, 2)`` — (lat, lon).
            target_elev: ``(batch, n_targets)`` station elevation (m).
            target_delta_elev: ``(batch, n_targets)`` station minus ERA5
                orography (m).
            target_mask: ``(batch, n_targets)`` — accepted for call-site
                compatibility but unused; masking happens in the loss.
            target_tessera: ``(batch, n_targets, precomputed_tessera_dim)``
                precomputed TESSERA vector, or None when the model was
                built without one.
            target_mtpi: ``(batch, n_targets)`` multi-scale topographic
                position index, required iff ``n_elev_features >= 3``.

        Returns:
            Nested dict ``{var_name: {param_name: tensor}}``; every
            parameter tensor has shape ``(batch, n_targets)``. Variables
            iterate in ``self.target_variables`` order.

        """
        del target_mask  # unused at forward time; kept for positional callers.
        target_lats = target_coords[:, :, 0]
        target_lons = target_coords[:, :, 1]

        # 1. Grid encoder → per-station interpolation.
        grid_features = self.cnn(context_grid)  # (batch, cnn_hidden, n_lat, n_lon)
        interp_features = self.interp(
            grid_features, grid_lats, grid_lons, target_lats, target_lons
        ).permute(0, 2, 1)  # (batch, n_targets, cnn_hidden)

        # 2. Assemble the per-station decoder input.
        mlp_parts = [interp_features]
        if self.include_elevation:
            mlp_parts.append((target_elev / 1000.0).unsqueeze(-1))
            mlp_parts.append((target_delta_elev / 1000.0).unsqueeze(-1))
            if self.n_elev_features >= 3:
                if target_mtpi is None:
                    raise ValueError(
                        "Model was built with n_elev_features >= 3 (mTPI "
                        "enabled) but target_mtpi was not passed to forward()."
                    )
                # Same 1/1000 (m → km) scaling as the other topographic inputs.
                mlp_parts.append((target_mtpi / 1000.0).unsqueeze(-1))

        if self.tessera_features_precomputed:
            if target_tessera is None:
                raise ValueError(
                    "Model uses precomputed TESSERA features but "
                    "target_tessera was not provided."
                )
            if target_tessera.shape[-1] != self.precomputed_tessera_dim:
                raise ValueError(
                    f"Precomputed TESSERA feature dim {target_tessera.shape[-1]} "
                    f"does not match model's precomputed_tessera_dim "
                    f"{self.precomputed_tessera_dim}."
                )
            if self.tessera_injection == "concat":
                mlp_parts.append(target_tessera)

        mlp_input = torch.cat(mlp_parts, dim=-1)

        # 3. Decoder body → per-variable likelihood heads.
        hidden = self.mlp(mlp_input)  # (batch, n_targets, mlp_hidden)
        return self.heads(hidden)
