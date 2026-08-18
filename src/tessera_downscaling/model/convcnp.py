"""ConvCNP downscaling model — unified baseline and TESSERA variant.

Implements the MLP variant from Vaughan et al. (2022) as a plain
``torch.nn.Module``. When a TESSERA encoder is provided, the model
additionally processes station-centred TESSERA patches and feeds
the compressed representation into the decoder MLP.

Architecture::

    ERA5 grid (n_channels, H, W)
        → ResNet CNN (7 layers, kernel 3, 128 channels)
        → RBF SetConv interpolation to station locations
        → Decoder MLP body (+ optional FiLM modulation from TESSERA)
        → Per-variable likelihood heads (one nn.Linear each)
        → Per-variable predictive-distribution parameters dict

Per-variable likelihood heads
-----------------------------

The model owns a ``LikelihoodHeadDict`` (one head per target variable),
constructed from ``likelihood_per_variable: dict[str, str]`` mapping
each variable name to its distribution key (``gaussian``, ``weibull``,
or ``bernoulli_gamma``). The forward pass returns a nested dict
``{var_name: {param_name: tensor}}`` rather than a flat tuple, and the
loss path dispatches per variable through ``model.heads.heads[var].nll``.

The decoder MLP body emits the post-activation hidden state of shape
``(batch, n_targets, mlp_hidden)`` directly; the heads consume that
hidden state. There is no longer a single shared output projection that
packs all variables' parameters into one ``[2 · V]`` tensor — that
legacy structure required every variable to share both its parameter
count and its parameterisation, which doesn't hold once Weibull and
Bernoulli-Gamma enter the mix.

The TESSERA path is controlled by ``tessera_encoder`` and
``tessera_injection``:

  - ``tessera_encoder=None``  → baseline (no TESSERA features).
  - ``tessera_injection="concat"``  → TESSERA features appended to MLP input.
  - ``tessera_injection="film"``    → TESSERA generates per-layer (γ, β)
    that modulate hidden activations.

The legacy ``hypernet`` injection mode is no longer supported. It was
an underperforming experiment, was incompatible with the per-variable
heads abstraction (the hypernet generated the entire translator including
its output layer, leaving no static weight tensor to split into per-head
projections), and removing it cleans up the construction logic
significantly.

For the migration of legacy checkpoints (saved before the heads
abstraction landed) into this model, see
``scripts/migrate_legacy_checkpoints.py``.
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
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden_channels, kernel_size))
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RBFSetConv(nn.Module):
    """Separable RBF kernel interpolation from a regular grid to points.

    Args:
        init_length_scale: Initial RBF length scale in degrees. Default is
            0.5° (~55 km at mid-latitudes), chosen to start with a bias
            toward locality rather than region-wide smoothing.
    """

    def __init__(self, init_length_scale: float = 0.5):
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_length_scale))
        )

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

    Unlike RBFSetConv, this has NO learnable parameters and applies standard
    bilinear interpolation at the native grid resolution. This deliberately
    limits the weather pathway to smooth, grid-scale features, forcing any
    local station corrections onto other model components (e.g. TESSERA).

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
        # grid_sample expects (x, y) in [-1, 1] where -1 is left/top, +1 is right/bottom.
        lat_min, lat_max = grid_lats[0], grid_lats[-1]
        lon_min, lon_max = grid_lons[0], grid_lons[-1]

        # Normalise to [-1, 1].
        norm_lats = 2.0 * (target_lats - lat_min) / (lat_max - lat_min) - 1.0
        norm_lons = 2.0 * (target_lons - lon_min) / (lon_max - lon_min) - 1.0

        # grid_sample expects (batch, n_targets, 1, 2) with (x=lon, y=lat).
        grid = torch.stack([norm_lons, norm_lats], dim=-1)  # (batch, n_targets, 2)
        grid = grid.unsqueeze(2)  # (batch, n_targets, 1, 2)

        # grid_sample: input (batch, C, H, W), grid (batch, H_out, W_out, 2)
        # Output: (batch, C, n_targets, 1)
        sampled = F.grid_sample(
            grid_features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return sampled.squeeze(-1)  # (batch, C, n_targets)


# ---------------------------------------------------------------------------
# Embedding-aware decoder mechanisms (methodological extensions)
#
# Three new modules layered on top of the existing decoder pipeline:
#
#   - ``EmbeddingConditionedSetConv``  (§5.1 in the scoping doc):
#       Decoder SetConv whose per-target lengthscale is a function of the
#       target's embedding (via a small shape MLP). Drop-in replacement
#       for :class:`RBFSetConv`. Zero-init on the shape MLP's final layer
#       means the model starts behaving exactly like ``RBFSetConv`` with
#       the same ``init_length_scale``; deviations are learned.
#
#   - ``EmbeddingStreamSetConv``  (§5.2):
#       Encoder-direction SetConv that aggregates sparse target-station
#       embeddings onto the CNN's internal grid. The output is concatenated
#       channel-wise with the CNN's F before the decoder SetConv reads it.
#       This lets cross-target embedding context flow into each target's
#       prediction without ever materialising per-grid-cell embeddings.
#
#   - ``TargetEmbeddingAttention``:
#       Self-attention over the target set, aggregating embeddings by
#       similarity. Two modes:
#         ``mode='embedding'``  — pure cosine-similarity weights.
#         ``mode='hybrid'``     — learnable mix of cosine similarity and
#         a Gaussian spatial term over (lat, lon) distance. The mix
#         coefficient ``α`` (sigmoid of a learnable logit) starts at 0.5
#         so gradient descent picks the balance.
#
# All three consume the SAME canonical embedding tensor that the existing
# concat / FiLM path consumes (i.e., after the optional ``precomputed_projection``
# in :class:`ConvCNPDownscaler`). The projection is applied once at the top
# of the forward pass; every downstream consumer sees the same projected
# representation.
# ---------------------------------------------------------------------------


class EmbeddingConditionedSetConv(nn.Module):
    """Decoder SetConv with per-target kernel lengthscale (§5.1).

    The lengthscale is parameterised as

        log λ(e_*) = log_scale_base + shape_mlp(e_*)

    where ``shape_mlp`` is a small 2-layer feed-forward net mapping an
    embedding to two delta-log-lengthscale values (separable: one for
    lat, one for lon). The final layer of ``shape_mlp`` is zero-init so
    that at initialisation ``shape_mlp(e_*) = 0`` and the module behaves
    exactly like :class:`RBFSetConv` with ``init_length_scale``. The
    model has to actively learn to deviate.

    Args:
        embed_dim: Input embedding dim (the same canonical dim consumed
            by the concat / FiLM path).
        init_length_scale: Initial base lengthscale (degrees).
        shape_mlp_hidden: Hidden width of the 2-layer shape MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        init_length_scale: float = 0.5,
        shape_mlp_hidden: int = 32,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(
                f"EmbeddingConditionedSetConv requires embed_dim > 0; "
                f"got {embed_dim}."
            )
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_length_scale))
        )
        self.shape_mlp = nn.Sequential(
            nn.Linear(embed_dim, shape_mlp_hidden),
            nn.ReLU(),
            nn.Linear(shape_mlp_hidden, 2),
        )
        # Zero-init the final layer so the initial delta is exactly zero
        # and behaviour matches RBFSetConv with the same init_length_scale.
        nn.init.zeros_(self.shape_mlp[-1].weight)
        nn.init.zeros_(self.shape_mlp[-1].bias)

    def forward(
        self,
        grid_features: torch.Tensor,
        grid_lats: torch.Tensor,
        grid_lons: torch.Tensor,
        target_lats: torch.Tensor,
        target_lons: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate grid features to target locations.

        Args:
            grid_features: ``(batch, C, n_lat, n_lon)``
            grid_lats: ``(n_lat,)``
            grid_lons: ``(n_lon,)``
            target_lats: ``(batch, n_targets)``
            target_lons: ``(batch, n_targets)``
            target_embeddings: ``(batch, n_targets, embed_dim)``

        Returns:
            ``(batch, C, n_targets)``
        """
        # Per-target log-lengthscale deltas.
        delta = self.shape_mlp(target_embeddings)  # (batch, n_targets, 2)
        scale_lat = torch.exp(self.log_scale + delta[..., 0])  # (batch, n_targets)
        scale_lon = torch.exp(self.log_scale + delta[..., 1])

        # 1-D RBF weights along each axis, now with per-target scales.
        # Broadcasting:
        #   grid_lats[None, :, None]     -> (1, n_lat, 1)
        #   target_lats[:, None, :]      -> (batch, 1, n_targets)
        #   scale_lat[:, None, :]        -> (batch, 1, n_targets)
        weights_lat = torch.exp(
            -0.5 * (
                (grid_lats[None, :, None] - target_lats[:, None, :])
                / scale_lat[:, None, :]
            ) ** 2
        )  # (batch, n_lat, n_targets)
        weights_lon = torch.exp(
            -0.5 * (
                (grid_lons[None, :, None] - target_lons[:, None, :])
                / scale_lon[:, None, :]
            ) ** 2
        )  # (batch, n_lon, n_targets)

        # Same separable contractions as RBFSetConv.
        out = torch.einsum("bcij,bit->bcjt", grid_features, weights_lat)
        out = torch.einsum("bcjt,bjt->bct", out, weights_lon)
        density = torch.einsum("bit,bjt->bt", weights_lat, weights_lon)
        return out / (density[:, None, :] + 1e-8)


class EmbeddingStreamSetConv(nn.Module):
    """Encoder-side SetConv: stations → internal CNN grid (§5.2).

    Maps the sparse set ``{(x_i, e_i)}`` of target stations to a smoothed
    embedding field ``E[x_g]`` on the CNN grid. The output is intended to
    be concatenated channel-wise with the CNN's F before the decoder
    SetConv reads it.

    Args:
        embed_dim: Embedding dimensionality (D).
        init_length_scale: Initial lengthscale in degrees. Default 0.5°
            (matches the locality-biased decoder SetConv default).
    """

    def __init__(
        self,
        embed_dim: int,
        init_length_scale: float = 0.5,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(
                f"EmbeddingStreamSetConv requires embed_dim > 0; "
                f"got {embed_dim}."
            )
        self.embed_dim = embed_dim
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_length_scale))
        )

    def forward(
        self,
        target_lats: torch.Tensor,
        target_lons: torch.Tensor,
        target_embeddings: torch.Tensor,
        grid_lats: torch.Tensor,
        grid_lons: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate target embeddings onto the grid.

        Args:
            target_lats: ``(batch, n_targets)``
            target_lons: ``(batch, n_targets)``
            target_embeddings: ``(batch, n_targets, embed_dim)``
            grid_lats: ``(n_lat,)``
            grid_lons: ``(n_lon,)``

        Returns:
            ``(batch, embed_dim, n_lat, n_lon)`` — E[grid].
        """
        scale = torch.exp(self.log_scale)

        # Separable RBF weights from each target station to each grid axis.
        # weights_lat: (batch, n_targets, n_lat)
        weights_lat = torch.exp(
            -0.5 * (
                (target_lats[:, :, None] - grid_lats[None, None, :])
                / scale
            ) ** 2
        )
        # weights_lon: (batch, n_targets, n_lon)
        weights_lon = torch.exp(
            -0.5 * (
                (target_lons[:, :, None] - grid_lons[None, None, :])
                / scale
            ) ** 2
        )

        # Numerator: contract n_targets axis through both spatial axes.
        # Output: (batch, embed_dim, n_lat, n_lon).
        numerator = torch.einsum(
            "bnl,bnm,bnd->bdlm",
            weights_lat, weights_lon, target_embeddings,
        )
        # Density normalisation: (batch, n_lat, n_lon).
        density = torch.einsum("bnl,bnm->blm", weights_lat, weights_lon)
        return numerator / (density[:, None, :, :] + 1e-8)


class TargetEmbeddingAttention(nn.Module):
    """Self-attention over the target set aggregating embeddings.

    Modes:
      ``embedding``: weights = softmax( cosine_sim(e_i, e_j) / τ )
      ``hybrid``:    weights = softmax( α · sim_logits + (1 − α) · spatial_logits )
                     where ``spatial_logits = -0.5 · ((Δlat)² + (Δlon)²) / λ²``.

    α is sigmoid(``alpha_logit``) and initialised at 0.5 so gradient
    descent picks the mix. λ (spatial scale) is log-parameterised and
    initialised to 0.5° in hybrid mode.

    Args:
        embed_dim: Embedding dimensionality.
        mode: ``"embedding"`` or ``"hybrid"``.
        init_temp: Initial softmax temperature τ.
        init_alpha: Initial balance coefficient α ∈ (0, 1) for hybrid mode.
        init_spatial_scale: Initial spatial lengthscale (degrees) for
            hybrid mode.
    """

    def __init__(
        self,
        embed_dim: int,
        mode: str = "embedding",
        init_temp: float = 1.0,
        init_alpha: float = 0.5,
        init_spatial_scale: float = 0.5,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(
                f"TargetEmbeddingAttention requires embed_dim > 0; "
                f"got {embed_dim}."
            )
        if mode not in ("embedding", "hybrid"):
            raise ValueError(
                f"mode must be 'embedding' or 'hybrid'; got {mode!r}."
            )
        self.embed_dim = embed_dim
        self.mode = mode
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        if mode == "hybrid":
            if not 0.0 < init_alpha < 1.0:
                raise ValueError(
                    f"init_alpha must be in (0, 1); got {init_alpha}."
                )
            self.alpha_logit = nn.Parameter(
                torch.logit(torch.tensor(float(init_alpha)))
            )
            self.log_spatial_scale = nn.Parameter(
                torch.tensor(math.log(init_spatial_scale))
            )

    def forward(
        self,
        target_lats: torch.Tensor,
        target_lons: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate target embeddings via attention.

        Args:
            target_lats: ``(batch, n_targets)``
            target_lons: ``(batch, n_targets)``
            target_embeddings: ``(batch, n_targets, embed_dim)``

        Returns:
            ``(batch, n_targets, embed_dim)`` — per-target aggregated
            embeddings.
        """
        # Cosine-similarity logits.
        e_norm = F.normalize(target_embeddings, dim=-1, eps=1e-8)
        sim_logits = torch.einsum("bnd,bmd->bnm", e_norm, e_norm) / torch.exp(self.log_temp)

        if self.mode == "hybrid":
            d_lat = target_lats[:, :, None] - target_lats[:, None, :]
            d_lon = target_lons[:, :, None] - target_lons[:, None, :]
            spatial_scale = torch.exp(self.log_spatial_scale)
            spatial_logits = -0.5 * (
                (d_lat / spatial_scale) ** 2 + (d_lon / spatial_scale) ** 2
            )
            alpha = torch.sigmoid(self.alpha_logit)
            logits = alpha * sim_logits + (1.0 - alpha) * spatial_logits
        else:
            logits = sim_logits

        attn = torch.softmax(logits, dim=-1)  # (batch, n_targets, n_targets)
        return torch.einsum("bnm,bmd->bnd", attn, target_embeddings)


# ---------------------------------------------------------------------------
# Decoder MLP body — now emits the post-activation hidden state directly,
# leaving the per-variable projection to the heads dispatcher.
# ---------------------------------------------------------------------------

class DecoderMLP(nn.Module):
    """Pointwise MLP body. Maps ``(in_features) → (mlp_hidden)`` per token.

    The body has ``n_hidden_layers`` ``Linear → ReLU`` blocks. The final
    output is the post-ReLU hidden state of width ``hidden_dim``; the
    likelihood heads (one ``nn.Linear`` each) consume this hidden state
    and produce per-variable distribution parameters.

    State-dict layout (preserved from the legacy ``DecoderMLP``, minus the
    final projection layer that's now lifted into the heads):

      ``mlp.net.0.weight``     — Linear(in_features, hidden_dim)
      ``mlp.net.0.bias``
      ``mlp.net.2.weight``     — Linear(hidden_dim, hidden_dim)
      ``mlp.net.2.bias``
      ...
      ``mlp.net.{2*(n-1)}.weight`` — last hidden Linear
      ``mlp.net.{2*(n-1)}.bias``

    (Even-indexed positions are Linear; odd are ReLU.)

    The legacy code had an additional ``mlp.net.{2*n}`` Linear that
    projected to ``2 * n_target_variables`` and packed all variables'
    (μ, log_var) pairs row-wise into one tensor. The migration script
    splits that legacy tensor row-wise into per-variable head weights.

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
        # NOTE: no output projection here. The heads dispatcher owns the
        # per-variable Linear(hidden_dim, n_params) projections.
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, n_targets, in_features) → (batch, n_targets, hidden_dim)``."""
        return self.net(x)


class FiLMDecoderMLP(nn.Module):
    """Decoder MLP body with FiLM conditioning from TESSERA features.

    Per-layer FiLM modulation: each hidden Linear's post-activation output
    is rescaled by ``γ * h + β`` where ``γ, β`` are produced from TESSERA
    features by a per-layer generator. When ``tessera_features`` is None
    (baseline mode), the modulation is identity (γ=1, β=0) — this allows
    the same module to be used in baseline runs without behavioural change.

    State-dict layout (preserved from the legacy ``FiLMDecoderMLP``, minus
    the final ``output_layer``):

      ``mlp.input_layer.{weight,bias}``       — Linear(in_features, hidden_dim)
      ``mlp.hidden_layers.{i}.{weight,bias}`` — Linear(hidden_dim, hidden_dim)
      ``mlp.film_generators.{i}.{weight,bias}`` — Linear(tessera_dim, 2*hidden_dim)

    The FiLM generators' biases were carefully initialised so that
    ``γ ≈ 1`` and ``β ≈ 0`` at start (identity modulation); that
    initialisation passes through unchanged because the migration only
    touches the now-removed ``output_layer``.

    Args:
        in_features: Input dimension (weather features + optional elevation,
            NOT including TESSERA dim).
        hidden_dim: Width of hidden layers and of the body output.
        n_hidden_layers: Number of Linear layers in the body. Equals the
            number of FiLM generators (one per Linear).
        tessera_dim: Dimension of the TESSERA encoder output. Required
            when FiLM conditioning is used.
    """

    def __init__(
        self,
        in_features: int = 130,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        tessera_dim: int = 16,
    ):
        super().__init__()
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim

        # Body layers: input projection + (n_hidden_layers - 1) hidden Linears.
        # Total = n_hidden_layers Linear modules in the body. No output
        # projection here — the heads dispatcher owns the per-variable
        # projections.
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_hidden_layers - 1)
        ])
        self.activation = nn.ReLU()

        # FiLM generators: one (γ, β) pair per hidden Linear (n_hidden_layers
        # total — one for input_layer, one per element of hidden_layers).
        # Each generator maps tessera_dim → 2 * hidden_dim (γ then β).
        self.film_generators = nn.ModuleList([
            nn.Linear(tessera_dim, 2 * hidden_dim)
            for _ in range(n_hidden_layers)
        ])

        # Initialise FiLM generators so γ≈1, β≈0 at start (identity
        # modulation). Output bias is split: first hidden_dim entries
        # are γ-bias (set to 1), second hidden_dim are β-bias (set to 0).
        # Output weight is small so γ and β stay near these values
        # initially even with non-trivial TESSERA inputs.
        for fg in self.film_generators:
            nn.init.normal_(fg.weight, std=0.01)
            nn.init.ones_(fg.bias[:hidden_dim])      # γ ≈ 1
            nn.init.zeros_(fg.bias[hidden_dim:])     # β ≈ 0

    def forward(
        self,
        x: torch.Tensor,
        tessera_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional FiLM conditioning.

        Args:
            x: ``(batch, n_targets, in_features)`` — weather + elevation.
            tessera_features: ``(batch, n_targets, tessera_dim)`` — encoded
                TESSERA patches. When None, modulation is identity.

        Returns:
            ``(batch, n_targets, hidden_dim)`` — post-FiLM hidden state.
        """
        # Input layer.
        h = self.activation(self.input_layer(x))

        # Apply FiLM modulation from first generator.
        if tessera_features is not None:
            film_params = self.film_generators[0](tessera_features)
            gamma = film_params[..., :self.hidden_dim]
            beta = film_params[..., self.hidden_dim:]
            h = gamma * h + beta

        # Remaining hidden layers with FiLM.
        for i, layer in enumerate(self.hidden_layers):
            h = self.activation(layer(h))
            if tessera_features is not None:
                film_params = self.film_generators[i + 1](tessera_features)
                gamma = film_params[..., :self.hidden_dim]
                beta = film_params[..., self.hidden_dim:]
                h = gamma * h + beta

        return h


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ConvCNPDownscaler(nn.Module):
    """ConvCNP for climate downscaling (baseline + optional TESSERA).

    Args:
        n_context_channels: Total ERA5 grid input channels.
        cnn_hidden: CNN hidden channel width.
        cnn_layers: Number of CNN conv layers.
        cnn_kernel: CNN kernel size.
        setconv_length_scale: Initial SetConv RBF length scale (degrees).
        interpolation: Grid-to-station interpolator; ``"setconv"`` (default)
            or ``"bilinear"``.
        mlp_hidden: Decoder MLP hidden width (also the input width to each
            likelihood head).
        mlp_n_hidden: Number of MLP hidden layers in the body.
        n_elev_features: Number of elevation features per target.
        include_elevation: Whether to include elevation features in the MLP.
        target_variables: Ordered list of target variable names. Order
            determines per-variable iteration in the heads dispatcher and
            in the legacy state-dict migration logic.
        likelihood_per_variable: Mapping ``{var_name: dist_name}`` where
            ``dist_name`` is a key in
            :data:`~tessera_downscaling.model.heads.HEAD_REGISTRY`
            (``gaussian``, ``weibull``, ``bernoulli_gamma``). Strict 1:1
            with ``target_variables``: every variable must have an entry,
            and every entry must correspond to a variable. When omitted,
            defaults to ``{var: "gaussian" for var in target_variables}``
            — matching the implicit Gaussian-everywhere behaviour of the
            legacy code.
        tessera_encoder: Optional ``TesseraPatchEncoder``. When provided,
            the model expects ``target_tessera`` in the forward call and
            either concatenates encoded patches with the MLP input
            (``concat``) or feeds them to the FiLM generators (``film``).
        tessera_injection: ``"concat"`` (default), ``"film"``, or
            ``"none"``. The ``"none"`` mode disables the per-target
            concat / FiLM injection entirely (useful for ablations that
            test the new mechanisms below standalone). The legacy
            ``"hypernet"`` mode is no longer supported.
        tessera_features_precomputed: When True, the model expects
            already-encoded TESSERA features in the batch (e.g. frozen
            VAE latents) instead of raw 64×64 patches.
        precomputed_tessera_dim: Dimension of the precomputed feature.
            Required when ``tessera_features_precomputed=True``.
        precomputed_drop_prob: Full-embedding dropout probability applied
            to precomputed TESSERA features during training.
        precomputed_proj_dim: When > 0, learnable projection of the
            precomputed feature down to this dimension before injection
            into the decoder. The projection is applied ONCE and the
            result is the canonical embedding consumed by every
            downstream consumer (concat / FiLM, the new kernel
            modulator, the embedding stream, and the attention path).
        precomputed_proj_mlp: When True (and ``precomputed_proj_dim > 0``),
            use a 2-layer MLP for the projection instead of a single
            ``nn.Linear``.
        decoder_kernel: ``"isotropic"`` (default — uses ``RBFSetConv`` or
            ``BilinearInterp`` depending on ``interpolation``) or
            ``"embedding_conditioned"`` (uses
            :class:`EmbeddingConditionedSetConv` — §5.1). The latter
            requires an embedding source (tessera_encoder or precomputed
            latents) and is only valid when ``interpolation="setconv"``.
        use_target_embed_stream: When True, adds the §5.2 station→grid
            embedding SetConv branch; its output is concatenated
            channel-wise to F before the decoder SetConv reads it.
            Requires an embedding source.
        target_embed_attention: ``"none"`` (default), ``"embedding"``, or
            ``"hybrid"``. Adds a :class:`TargetEmbeddingAttention`
            aggregator over the target set; its per-target output is
            concatenated to the MLP input. Requires an embedding source.
        detach_attn_embed: Ablation flag (default ``False``). When True and
            ``target_embed_attention != "none"``, the per-target embedding is
            detached at the attention module's input. The forward pass is
            numerically identical to the live version with the same weights,
            but gradients no longer flow from the attention output back to
            the projection (or end-to-end TESSERA encoder) via the attention
            path. Used to isolate whether the mechanism's gain comes from its
            forward-pass content (gain preserved when detached) or from extra
            gradient signal on the projection (gain disappears when detached).
    """

    def __init__(
        self,
        n_context_channels: int = 38,
        cnn_hidden: int = 128,
        cnn_layers: int = 7,
        cnn_kernel: int = 3,
        setconv_length_scale: float = 0.5,
        interpolation: str = "setconv",
        mlp_hidden: int = 128,
        mlp_n_hidden: int = 3,
        n_elev_features: int = 2,
        include_elevation: bool = True,
        target_variables: list[str] | None = None,
        likelihood_per_variable: dict[str, str] | None = None,
        tessera_encoder: nn.Module | None = None,
        tessera_injection: str = "concat",
        tessera_features_precomputed: bool = False,
        precomputed_tessera_dim: int = 0,
        precomputed_drop_prob: float = 0.0,
        precomputed_proj_dim: int = 0,
        precomputed_proj_mlp: bool = False,
        decoder_kernel: str = "isotropic",
        use_target_embed_stream: bool = False,
        target_embed_attention: str = "none",
        detach_attn_embed: bool = False,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if target_variables is None or len(target_variables) == 0:
            raise ValueError(
                "target_variables must be a non-empty list of variable names"
            )

        if tessera_injection not in ("concat", "film", "none"):
            raise ValueError(
                f"tessera_injection must be 'concat', 'film', or 'none'; "
                f"got {tessera_injection!r}. The legacy 'hypernet' mode "
                "is no longer supported (it was an underperforming experiment "
                "and is incompatible with the per-variable heads abstraction)."
            )

        if decoder_kernel not in ("isotropic", "embedding_conditioned"):
            raise ValueError(
                f"decoder_kernel must be 'isotropic' or "
                f"'embedding_conditioned'; got {decoder_kernel!r}."
            )
        if decoder_kernel == "embedding_conditioned" and interpolation != "setconv":
            raise ValueError(
                f"decoder_kernel='embedding_conditioned' is only valid "
                f"with interpolation='setconv'; got interpolation="
                f"{interpolation!r}."
            )

        if target_embed_attention not in ("none", "embedding", "hybrid"):
            raise ValueError(
                f"target_embed_attention must be 'none', 'embedding', or "
                f"'hybrid'; got {target_embed_attention!r}."
            )
        
        if detach_attn_embed and target_embed_attention == "none":
            raise ValueError(
                "detach_attn_embed=True requires target_embed_attention to "
                "be enabled ('embedding' or 'hybrid'); got 'none'."
            )

        # Precomputed TESSERA features (e.g. frozen VAE latents) are mutually
        # exclusive with an end-to-end tessera_encoder: only one source of
        # surface features can feed the decoder.
        if tessera_features_precomputed and tessera_encoder is not None:
            raise ValueError(
                "tessera_features_precomputed=True is incompatible with "
                "passing a tessera_encoder. Use one or the other."
            )
        if tessera_features_precomputed and precomputed_tessera_dim <= 0:
            raise ValueError(
                "precomputed_tessera_dim must be > 0 when "
                "tessera_features_precomputed=True"
            )
        if precomputed_proj_dim > 0 and not tessera_features_precomputed:
            raise ValueError(
                "precomputed_proj_dim > 0 requires "
                "tessera_features_precomputed=True"
            )
        if precomputed_proj_mlp and precomputed_proj_dim <= 0:
            raise ValueError(
                "precomputed_proj_mlp=True requires precomputed_proj_dim > 0 "
                "(the MLP projects to precomputed_proj_dim, which must be set)."
            )

        # Default likelihood: all-Gaussian, matching the legacy implicit
        # behaviour. The dispatcher will validate strict 1:1 between the
        # spec and ``target_variables``.
        if likelihood_per_variable is None:
            likelihood_per_variable = {var: "gaussian" for var in target_variables}

        # ------------------------------------------------------------------
        # Bookkeeping
        # ------------------------------------------------------------------
        self.tessera_encoder = tessera_encoder
        self.tessera_features_precomputed = tessera_features_precomputed
        self.precomputed_drop_prob = precomputed_drop_prob
        self.precomputed_proj_dim = precomputed_proj_dim
        self.precomputed_proj_mlp = precomputed_proj_mlp
        self.include_elevation = include_elevation
        # Number of per-station scalar auxiliary features concatenated onto
        # the decoder MLP input when ``include_elevation`` is True. 2 =
        # (elevation, delta_elevation); 3 additionally appends mTPI, matching
        # the (elevation, elevation-difference, mTPI) auxiliary vector of
        # Vaughan et al. (2022). Stored so ``forward`` knows how many of the
        # optional per-station tensors to consume, keeping pre-mTPI (2-feature)
        # checkpoints loadable unchanged.
        self.n_elev_features = n_elev_features
        self.target_variables = list(target_variables)
        self.n_target_variables = len(target_variables)
        self.likelihood_per_variable = dict(likelihood_per_variable)
        self.interpolation_method = interpolation
        self.tessera_injection = tessera_injection
        self.mlp_hidden = mlp_hidden
        self.decoder_kernel = decoder_kernel
        self.use_target_embed_stream = use_target_embed_stream
        self.target_embed_attention = target_embed_attention
        self.detach_attn_embed = detach_attn_embed

        # ------------------------------------------------------------------
        # Compute TESSERA feature dim
        # ------------------------------------------------------------------
        # TESSERA dim comes from either the encoder (end-to-end path), the
        # precomputed latent dim, or the task-adapted projection (linear or
        # 2-layer MLP) of the precomputed latent.
        if tessera_encoder is not None:
            tessera_dim = tessera_encoder.output_dim
        elif tessera_features_precomputed:
            # If a learnable projection head is requested, it takes the raw
            # latent from ``precomputed_tessera_dim`` and compresses it to
            # ``precomputed_proj_dim``. The downstream MLP / FiLM then sees
            # ``precomputed_proj_dim``-sized features.
            #
            # Two shapes supported:
            #   precomputed_proj_mlp=False (default): single nn.Linear
            #     — what all the Linear+proj experiments used.
            #   precomputed_proj_mlp=True: Linear(d_in, 2*proj_dim) → ReLU
            #     → Linear(2*proj_dim, proj_dim).
            if precomputed_proj_dim > 0:
                if precomputed_proj_mlp:
                    hidden = 2 * precomputed_proj_dim
                    self.precomputed_projection = nn.Sequential(
                        nn.Linear(precomputed_tessera_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, precomputed_proj_dim),
                    )
                else:
                    self.precomputed_projection = nn.Linear(
                        precomputed_tessera_dim, precomputed_proj_dim,
                    )
                tessera_dim = precomputed_proj_dim
            else:
                self.precomputed_projection = None
                tessera_dim = precomputed_tessera_dim
        else:
            self.precomputed_projection = None
            tessera_dim = 0
        self._tessera_dim = tessera_dim
        self._raw_precomputed_dim = precomputed_tessera_dim

        # ------------------------------------------------------------------
        # Validate that any new mechanism that consumes embeddings has an
        # embedding source. (Existing concat / FiLM checks already cover
        # those paths.)
        # ------------------------------------------------------------------
        new_mechanisms_enabled = (
            decoder_kernel == "embedding_conditioned"
            or use_target_embed_stream
            or target_embed_attention != "none"
        )
        if new_mechanisms_enabled and tessera_dim == 0:
            raise ValueError(
                f"decoder_kernel={decoder_kernel!r}, "
                f"use_target_embed_stream={use_target_embed_stream}, "
                f"target_embed_attention={target_embed_attention!r} all require "
                "an embedding source (either tessera_encoder or "
                "tessera_features_precomputed). None was configured."
            )

        # ------------------------------------------------------------------
        # MLP input dim
        # ------------------------------------------------------------------
        actual_elev_dim = n_elev_features if include_elevation else 0

        # Base: CNN-output channels reaching the per-target MLP input.
        # When the embedding stream is on, it concatenates ``tessera_dim``
        # additional channels onto the grid before the decoder SetConv, so
        # the per-target features coming out of the SetConv carry both.
        if use_target_embed_stream:
            decoder_in_channels = cnn_hidden + tessera_dim
        else:
            decoder_in_channels = cnn_hidden

        mlp_in = decoder_in_channels + actual_elev_dim
        # Target-side attention output (per-target embed_dim-sized vector).
        if target_embed_attention != "none":
            mlp_in += tessera_dim
        # Existing concat injection (FiLM doesn't add to mlp_in; 'none' is no-op).
        if tessera_injection == "concat":
            mlp_in += tessera_dim

        # ------------------------------------------------------------------
        # CNN + interpolation
        # ------------------------------------------------------------------
        self.cnn = GridCNN(
            in_channels=n_context_channels,
            out_channels=cnn_hidden,
            hidden_channels=cnn_hidden,
            num_layers=cnn_layers,
            kernel_size=cnn_kernel,
        )

        if interpolation == "setconv":
            if decoder_kernel == "embedding_conditioned":
                self.interp = EmbeddingConditionedSetConv(
                    embed_dim=tessera_dim,
                    init_length_scale=setconv_length_scale,
                )
            else:
                self.interp = RBFSetConv(init_length_scale=setconv_length_scale)
        elif interpolation == "bilinear":
            self.interp = BilinearInterp()
        else:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Supported: 'setconv', 'bilinear'."
            )

        # ------------------------------------------------------------------
        # §5.2 embedding stream (station → grid)
        # ------------------------------------------------------------------
        if use_target_embed_stream:
            self.embed_stream = EmbeddingStreamSetConv(
                embed_dim=tessera_dim,
                init_length_scale=setconv_length_scale,
            )
        else:
            self.embed_stream = None

        # ------------------------------------------------------------------
        # Target-side attention aggregator
        # ------------------------------------------------------------------
        if target_embed_attention != "none":
            self.embed_attention = TargetEmbeddingAttention(
                embed_dim=tessera_dim,
                mode=target_embed_attention,
                init_spatial_scale=setconv_length_scale,
            )
        else:
            self.embed_attention = None

        # ------------------------------------------------------------------
        # Decoder MLP body (no output projection — heads own that)
        # ------------------------------------------------------------------
        has_tessera_features = (
            tessera_encoder is not None or tessera_features_precomputed
        )

        if tessera_injection == "film" and has_tessera_features:
            self.mlp = FiLMDecoderMLP(
                in_features=mlp_in,
                hidden_dim=mlp_hidden,
                n_hidden_layers=mlp_n_hidden,
                tessera_dim=tessera_dim,
            )
        else:
            self.mlp = DecoderMLP(
                in_features=mlp_in,
                hidden_dim=mlp_hidden,
                n_hidden_layers=mlp_n_hidden,
            )

        # ------------------------------------------------------------------
        # Per-variable likelihood heads
        # ------------------------------------------------------------------
        self.heads = LikelihoodHeadDict(
            likelihood_per_variable=likelihood_per_variable,
            target_variables=target_variables,
            hidden_dim=mlp_hidden,
        )

    @property
    def uses_tessera(self) -> bool:
        return (
            self.tessera_encoder is not None
            or self.tessera_features_precomputed
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
            target_elev: ``(batch, n_targets)``
            target_delta_elev: ``(batch, n_targets)``
            target_mask: ``(batch, n_targets)`` — ``True`` for valid targets.
                Currently unused in the forward pass (masking happens in
                the loss / metric layer); kept in the signature for API
                continuity.
            target_tessera: ``(batch, n_targets, 128, H, W)`` raw patches,
                or ``(batch, n_targets, d)`` precomputed feature, or None.
            target_mtpi: ``(batch, n_targets)`` multi-scale topographic
                position index per station, or None. Required only when the
                model was built with ``n_elev_features >= 3``; ignored
                otherwise. Last/optional so positional call sites that pre-date
                mTPI are unaffected.

        Returns:
            Nested dict ``{var_name: {param_name: tensor}}``. Each
            parameter tensor has shape ``(batch, n_targets)``. Iteration
            order over variables follows
            ``self.target_variables``.
        """
        del target_mask  # unused at forward time; carried for API continuity.
        batch_size = context_grid.shape[0]
        n_targets = target_coords.shape[1]
        target_lats = target_coords[:, :, 0]
        target_lons = target_coords[:, :, 1]

        # ------------------------------------------------------------------
        # 1. Compute the canonical per-target embedding tensor up front.
        #
        #    All downstream consumers (§5.1 kernel modulation, §5.2 grid
        #    stream, target-side attention, and the existing concat / FiLM
        #    injection) operate on the SAME ``t_encoded`` tensor — encoded
        #    (if end-to-end), projected (if precomputed + proj_dim > 0),
        #    and dropout-masked (during training). Sharing one tensor
        #    means autograd routes gradients to all consumers correctly
        #    and the projection is trained jointly by all of them.
        # ------------------------------------------------------------------
        t_encoded = None
        if self.tessera_encoder is not None:
            if target_tessera is None:
                raise ValueError(
                    "Model has a TESSERA encoder but target_tessera was not provided."
                )
            t_flat = target_tessera.reshape(
                batch_size * n_targets, *target_tessera.shape[2:]
            )
            t_encoded = self.tessera_encoder(t_flat).reshape(
                batch_size, n_targets, -1
            )
        elif self.tessera_features_precomputed:
            if target_tessera is None:
                raise ValueError(
                    "Model uses precomputed TESSERA features but "
                    "target_tessera was not provided."
                )
            if target_tessera.shape[-1] != self._raw_precomputed_dim:
                raise ValueError(
                    f"Precomputed TESSERA feature dim {target_tessera.shape[-1]} "
                    f"does not match model's precomputed_tessera_dim "
                    f"{self._raw_precomputed_dim}."
                )
            t_encoded = target_tessera
            # Optional learnable projection of the (frozen) precomputed
            # latent down to a task-specific dimension. Applied ONCE here;
            # every downstream consumer sees the projected result.
            if self.precomputed_projection is not None:
                t_encoded = self.precomputed_projection(t_encoded)
            # Full-embedding dropout: zero entire vectors for a fraction
            # of stations during training, scaled by 1/(1-p) for eval
            # consistency. Mirrors TesseraPatchEncoder's internal dropout.
            # Applied AFTER the projection so the mask shape matches the
            # final feature dim, and BEFORE any downstream consumer so
            # all of them see the same masking pattern.
            if self.training and self.precomputed_drop_prob > 0:
                keep_mask = (
                    torch.rand(
                        t_encoded.shape[0], t_encoded.shape[1], 1,
                        device=t_encoded.device,
                    ) > self.precomputed_drop_prob
                ).float()
                t_encoded = t_encoded * keep_mask / (1.0 - self.precomputed_drop_prob)

        # ------------------------------------------------------------------
        # 2. CNN encodes the ERA5 grid. Spatial dims are preserved by
        #    GridCNN; output shape: (batch, cnn_hidden, n_lat, n_lon).
        # ------------------------------------------------------------------
        grid_features = self.cnn(context_grid)

        # ------------------------------------------------------------------
        # 3. §5.2 embedding stream: if enabled, aggregate station
        #    embeddings onto the internal grid and concatenate the result
        #    channel-wise to grid_features BEFORE the decoder SetConv.
        # ------------------------------------------------------------------
        if self.embed_stream is not None:
            if t_encoded is None:
                raise RuntimeError(
                    "use_target_embed_stream is enabled but no embedding "
                    "source produced t_encoded. This indicates a "
                    "configuration error."
                )
            E_grid = self.embed_stream(
                target_lats, target_lons, t_encoded,
                grid_lats, grid_lons,
            )  # (batch, tessera_dim, n_lat, n_lon)
            grid_features = torch.cat([grid_features, E_grid], dim=1)

        # ------------------------------------------------------------------
        # 4. Decoder SetConv: grid → per-target. §5.1's
        #    EmbeddingConditionedSetConv takes the embedding as an extra
        #    argument to shape its kernel per-target; the standard
        #    RBFSetConv / BilinearInterp don't.
        # ------------------------------------------------------------------
        if self.decoder_kernel == "embedding_conditioned":
            if t_encoded is None:
                raise RuntimeError(
                    "decoder_kernel='embedding_conditioned' is enabled but "
                    "no embedding source produced t_encoded."
                )
            interp_features = self.interp(
                grid_features, grid_lats, grid_lons,
                target_lats, target_lons, t_encoded,
            )
        else:
            interp_features = self.interp(
                grid_features, grid_lats, grid_lons, target_lats, target_lons,
            )
        interp_features = interp_features.permute(0, 2, 1)
        # interp_features: (batch, n_targets, decoder_in_channels)

        # ------------------------------------------------------------------
        # 5. Assemble per-target MLP input.
        # ------------------------------------------------------------------
        mlp_parts = [interp_features]

        if self.include_elevation:
            mlp_parts.append((target_elev / 1000.0).unsqueeze(-1))
            mlp_parts.append((target_delta_elev / 1000.0).unsqueeze(-1))
            # Third per-station feature (mTPI), present iff the model was
            # built with n_elev_features >= 3. Scaled by the same 1/1000 (m→km)
            # factor as the other elevation features for a comparable range.
            if self.n_elev_features >= 3:
                if target_mtpi is None:
                    raise ValueError(
                        "Model was built with n_elev_features >= 3 (mTPI "
                        "enabled) but target_mtpi was not passed to forward()."
                    )
                mlp_parts.append((target_mtpi / 1000.0).unsqueeze(-1))

        # Target-side attention output.
        if self.embed_attention is not None:
            if t_encoded is None:
                raise RuntimeError(
                    "target_embed_attention is enabled but no embedding "
                    "source produced t_encoded."
                )
            attn_input = t_encoded.detach() if self.detach_attn_embed else t_encoded
            attn_out = self.embed_attention(target_lats, target_lons, attn_input)
            mlp_parts.append(attn_out)

        # Existing concat injection (still optional; injection='none'
        # / 'film' skips this).
        if self.tessera_injection == "concat" and t_encoded is not None:
            mlp_parts.append(t_encoded)

        mlp_input = torch.cat(mlp_parts, dim=-1)

        # ------------------------------------------------------------------
        # 6. Run decoder body. FiLM mode passes t_encoded into the body
        #    for per-layer modulation; concat / none modes just run the
        #    plain MLP body.
        # ------------------------------------------------------------------
        if self.tessera_injection == "film" and isinstance(self.mlp, FiLMDecoderMLP):
            hidden = self.mlp(mlp_input, tessera_features=t_encoded)
        else:
            hidden = self.mlp(mlp_input)
        # hidden: (batch, n_targets, mlp_hidden)

        # ------------------------------------------------------------------
        # 7. Per-variable likelihood heads.
        # ------------------------------------------------------------------
        return self.heads(hidden)