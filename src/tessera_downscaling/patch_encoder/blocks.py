"""Convolutional building blocks of the patch-encoder VAE.

Each block changes the spatial resolution by a factor of two, so a stack of
``n`` blocks maps an ``S``-pixel patch to ``S / 2**n`` pixels (encoder) or back
up again (decoder). The submodule names fix the keys of a saved
``state_dict`` -- ``conv.0`` / ``block.0`` is the convolution, ``conv.1`` /
``block.1`` its BatchNorm -- so they must not be renamed while checkpoints of
the paper's runs are still in use.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Stride-2 ``Conv2d -> BatchNorm2d -> ReLU``, optionally with Dropout2d.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        dropout: Channel-dropout probability applied after the block
            (``0`` disables it and creates no submodule).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class DeconvBlock(nn.Module):
    """Stride-2 ``ConvTranspose2d -> BatchNorm2d -> ReLU`` (decoder mirror)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
