"""The patch-encoder VAE: encoder, decoder, auxiliary heads.

The encoder takes a ``(B, C, S, S)`` patch of surface embeddings (``C = 128``
channels; ``S = 64`` pixels, i.e. a 640 m window at 10 m resolution, for the
paper's TESSERA run), pushes it through four stride-2 convolution blocks
(128 -> 256 -> 256 -> 512), flattens the resulting ``512 x (S/16) x (S/16)``
feature map and projects it to the mean and log-variance of a diagonal Gaussian
over a ``latent_dim``-dimensional latent. The decoder mirrors that path back to
the input shape with transposed convolutions and no output activation (the
target is a z-scored embedding, not an image). During training small MLP heads
regress auxiliary targets -- elevation, latitude, longitude -- off the latent.

Flattening rather than pooling the final feature map is deliberate: the latent
keeps *where* in the patch a feature sits, which a global average pool would
discard.

Neither the channel count nor the number of stages is baked in, which is what
lets the foundation-model benchmark reuse this model unchanged: ``in_channels``
follows the patch file (128 TESSERA, 64 AlphaEarth, 768 OlmoEarth) and the
number of stride-2 stages is the length of ``encoder_channels``, so the 16x16
OlmoEarth token grid is encoded by a three-stage configuration (16 -> 8 -> 4 ->
2) while the 10 m rasters use four.

The posterior mean of the trained encoder (:meth:`TesseraVAE.encode`) is the
per-station surface descriptor the downscaler consumes.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock, DeconvBlock

# The TESSERA geometry: four stride-2 stages, hence patches must be divisible
# by 2**4. A config's own encoder_channels / decoder_channels override these,
# and their length sets the number of stages (three for OlmoEarth's 16x16 grid).
DEFAULT_ENCODER_CHANNELS = (128, 256, 256, 512)
DEFAULT_DECODER_CHANNELS = (512, 256, 256, 128)


class Encoder(nn.Module):
    """``(B, in_channels, S, S)`` patch -> posterior ``(mu, logvar)``.

    Args:
        in_channels: Channels of the input patch (128 TESSERA, 64 AlphaEarth,
            768 OlmoEarth).
        layer_channels: Output channels of the successive stride-2 blocks.
        latent_dim: Dimension of the latent.
        dropout: Dropout2d probability inside each conv block.
        input_size: Patch side length ``S``; must be divisible by
            ``2 ** len(layer_channels)``.
    """

    def __init__(
        self,
        in_channels: int = 128,
        layer_channels: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        layer_channels = list(layer_channels or DEFAULT_ENCODER_CHANNELS)

        blocks = []
        ch_in = in_channels
        for ch_out in layer_channels:
            blocks.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out
        self.blocks = nn.Sequential(*blocks)

        n_down = len(layer_channels)
        if input_size % (2**n_down) != 0:
            raise ValueError(
                f"input_size={input_size} is not divisible by 2^{n_down}="
                f"{2**n_down}; pick a patch size that is a multiple of {2**n_down}."
            )
        self.final_spatial = input_size // (2**n_down)

        flat_dim = layer_channels[-1] * self.final_spatial**2
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """``(B, latent_dim)`` -> reconstructed ``(B, out_channels, S, S)`` patch.

    A linear layer expands the latent to the encoder's final feature-map shape;
    transposed convolutions then double the resolution ``len(layer_channels)``
    times. The last one is bare (no BatchNorm, no ReLU) because the target is a
    z-scored patch that takes both signs.
    """

    def __init__(
        self,
        out_channels: int = 128,
        layer_channels: tuple[int, ...] | list[int] | None = None,
        latent_dim: int = 16,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        layer_channels = list(layer_channels or DEFAULT_DECODER_CHANNELS)

        self.init_ch = layer_channels[0]
        self.final_spatial = input_size // (2 ** len(layer_channels))
        self.fc = nn.Linear(latent_dim, self.init_ch * self.final_spatial**2)

        blocks: list[nn.Module] = [
            DeconvBlock(layer_channels[i], layer_channels[i + 1])
            for i in range(len(layer_channels) - 1)
        ]
        blocks.append(
            nn.ConvTranspose2d(
                layer_channels[-1], out_channels, kernel_size=4, stride=2, padding=1
            )
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.final_spatial
        h = self.fc(z).view(-1, self.init_ch, s, s)
        return self.blocks(h)


class AuxHead(nn.Module):
    """Two-layer MLP predicting one scalar target from the latent."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class TesseraVAE(nn.Module):
    """Encoder + decoder + auxiliary heads.

    Args:
        in_channels: Channels of the input patch (128 TESSERA, 64 AlphaEarth,
            768 OlmoEarth).
        latent_dim: Dimension of the latent (16 in the paper).
        encoder_channels, decoder_channels: Per-stage channel counts.
        dropout: Dropout2d probability inside the encoder blocks.
        input_size: Patch side length the model is built for.
        aux_targets: Names of the auxiliary regression heads; empty disables
            them. Head names become ``aux_heads.<name>`` in the state dict and
            ``aux_<name>`` in the forward output.
        aux_hidden_dim: Hidden width of each auxiliary head.
    """

    def __init__(
        self,
        in_channels: int = 128,
        latent_dim: int = 16,
        encoder_channels: tuple[int, ...] | list[int] | None = None,
        decoder_channels: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.1,
        input_size: int = 64,
        aux_targets: list[str] | None = None,
        aux_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder = Encoder(
            in_channels=in_channels,
            layer_channels=encoder_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            input_size=input_size,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            layer_channels=decoder_channels,
            latent_dim=latent_dim,
            input_size=input_size,
        )

        self.aux_heads = nn.ModuleDict(
            {name: AuxHead(latent_dim, aux_hidden_dim) for name in aux_targets or []}
        )

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample the posterior while training; return its mean when evaluating."""
        logvar = torch.clamp(logvar, min=-10, max=10)
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``x_recon``, ``mu``, ``logvar``, ``z`` and one ``aux_*`` per head."""
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)

        out = {"x_recon": self.decoder(z), "mu": mu, "logvar": logvar, "z": z}
        for name, head in self.aux_heads.items():
            out[f"aux_{name}"] = head(z)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean -- the descriptor used downstream."""
        mu, _ = self.encoder(x)
        return mu


def build_model(cfg: dict) -> TesseraVAE:
    """Construct a :class:`TesseraVAE` from a run config.

    ``cfg`` is the dict parsed from one of the ``scripts/patch_encoder/vae*.yaml``
    configs (``vae.yaml`` for TESSERA, ``vae_alphaearth.yaml`` and
    ``vae_olmoearth.yaml`` for the benchmark arms) or the copy stored inside a
    checkpoint (``ckpt["config"]``); rebuilding from the latter reproduces the
    trained model exactly, so ``load_state_dict(..., strict=True)`` succeeds.

    ``model.input_size`` and ``model.in_channels`` are written by
    ``train_vae.py`` from the data the run actually saw (the crop size and the
    patch file's channel count) rather than being set by hand.
    """
    model_cfg = cfg["model"]
    aux_cfg = cfg.get("auxiliary", {})

    # Two ablation switches of the original research code -- Squeeze-and-
    # Excitation blocks and a global-average-pool bottleneck -- were dropped in
    # the migration: no run behind the paper or its checkpoints used them.
    # Refuse such a config rather than silently building a different model.
    if model_cfg.get("use_se", False):
        raise ValueError(
            "model.use_se is not supported: Squeeze-and-Excitation blocks were "
            "dropped when the patch encoder moved into this repository."
        )
    bottleneck = model_cfg.get("bottleneck", "flatten")
    if bottleneck != "flatten":
        raise ValueError(
            f"model.bottleneck={bottleneck!r} is not supported: only the "
            "'flatten' bottleneck was kept when the patch encoder moved here."
        )

    aux_targets = aux_cfg.get("targets", []) if aux_cfg.get("enable", False) else []

    return TesseraVAE(
        in_channels=model_cfg.get("in_channels", 128),
        latent_dim=model_cfg["latent_dim"],
        encoder_channels=model_cfg.get("encoder_channels"),
        decoder_channels=model_cfg.get("decoder_channels"),
        dropout=model_cfg.get("dropout", 0.1),
        input_size=model_cfg.get("input_size", 64),
        aux_targets=aux_targets,
        aux_hidden_dim=aux_cfg.get("hidden_dim", 64),
    )
