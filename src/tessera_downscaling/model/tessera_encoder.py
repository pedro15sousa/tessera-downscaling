"""TESSERA patch encoder for compressing station-centred land surface patches.

Provides multiple compression strategies for reducing a 64×64×128 TESSERA
patch to a fixed-length vector, with varying parameter counts and
expressivity:

  - ``meanpool``: Spatial mean pooling (zero learnable parameters).
  - ``linear``: Mean pool + linear projection (minimal parameters).
  - ``cnn``: Small strided CNN with BatchNorm + global average pool.

The encoder is shared across all target stations — same weights process
every patch regardless of location. During training, all target stations'
patches are batched together for efficiency.

Full-embedding dropout (``drop_prob``): During training, the entire encoded
TESSERA vector is zeroed out for a random fraction of samples. This forces
the downstream MLP to not rely on TESSERA for station identification,
preventing memorisation and encouraging the encoder to learn generalisable
land surface features. Rescaled by 1/(1-p) so expected values match at eval.
"""

import torch
import torch.nn as nn


class TesseraPatchEncoder(nn.Module):
    """Compress a TESSERA patch into a fixed-length vector.

    Args:
        embed_dim: Input embedding dimension per pixel (128 for TESSERA).
        output_dim: Desired output vector dimension.
        method: Compression strategy — ``"meanpool"``, ``"linear"``, or
            ``"cnn"``.
        drop_prob: Probability of zeroing out the entire encoded vector
            during training (Rich's full-embedding dropout). Default 0.0.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        output_dim: int = 64,
        method: str = "meanpool",
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.method = method
        self.drop_prob = drop_prob

        if method == "cnn":
            self.output_dim = output_dim
            # Three strided conv layers halve spatial dims each time:
            # 64→32→16→8 (or 16→8→4→2 for patch16), then global avg pool.
            # BatchNorm after each conv stabilises training and acts as regularisation
            self.cnn = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
        elif method == "meanpool":
            # No learnable parameters. Output dim = embed_dim.
            self.output_dim = embed_dim
        elif method == "linear":
            self.output_dim = output_dim
            self.proj = nn.Linear(embed_dim, output_dim)
        else:
            raise ValueError(f"Unknown TESSERA encoder method: {method}")

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Compress patches to vectors.

        Args:
            patches: ``(N, embed_dim, H, W)`` in channels-first format.

        Returns:
            ``(N, output_dim)`` compressed representations.
        """
        if self.method == "cnn":
            out = self.cnn(patches).squeeze(-1).squeeze(-1)
        elif self.method == "meanpool":
            out = patches.mean(dim=(-2, -1))
        elif self.method == "linear":
            if patches.ndim == 4:
                pooled = patches.mean(dim=(-2, -1))
            else:
                pooled = patches  # Point embeddings: already (N, 128)
            out = self.proj(pooled)

        # Full-embedding dropout: zero out entire vectors for a fraction of
        # samples during training. Rescale by 1/(1-p) so expected values
        # match between train and eval.
        if self.training and self.drop_prob > 0:
            mask = (torch.rand(out.shape[0], 1, device=out.device)
                    > self.drop_prob).float()
            out = out * mask / (1.0 - self.drop_prob)

        return out