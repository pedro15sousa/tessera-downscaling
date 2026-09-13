"""Aurora latent crops: token-grid geometry, region cropping, storage encoding,
self-describing metadata and forward-hook capture.

Aurora 0.25 deg works on a token grid: after ``batch.crop(patch_size=4)``
drops the 721st latitude row, the 720 x 1440 field is patch-embedded into
180 x 360 tokens, one per 4 x 4 block of cells, at ``latent_levels`` (4)
latent levels. Token (level, h, w) covers global rows ``4h..4h+3``
(lat ``90 - 0.25*row``) and cols ``4w..4w+3`` (lon ``0.25*col``); level 0 is
the surface level (the decoder's surface heads read it directly), levels 1..3
are de-aggregated into the pressure levels. The encoder output has
``embed_dim`` (512) channels per token, the backbone output ``2*embed_dim``
(the last Swin stage concatenates the first skip connection).

A region crop on this grid is exact and lossless for the region: every
dataset bbox has integer-degree edges, so the region's 0.25 deg grid starts
and ends on token boundaries, and Aurora's decoder mixes nothing across
tokens, so the fields inside the region depend only on the tokens inside the
region. The crop is derived from the SAME ``compute_grid_crop_indices`` the
dataset preprocessor uses (including the longitude roll that makes Europe's
0-deg-crossing box contiguous) and then widened by a margin of tokens so a
future consumer can look beyond the region without another rollout. The
interior offsets are recorded, so ``crop[..., cell rows, cell cols]`` at 0.25
deg is the paper's region grid.

Stored array layout: ``(latent_levels * D, nh, nw)`` with channel index
``level * D + d`` (level-major), so a level subset is a contiguous slice.

This module is numpy-only at import time; :class:`LatentCapture` and
:func:`tokens_to_grid` operate on torch tensors through duck typing.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tessera_downscaling.preprocessing.helpers import compute_grid_crop_indices

# --------------------------------------------------------------------------- #
# Aurora 0.25 deg token-grid constants
# --------------------------------------------------------------------------- #

PATCH_SIZE = 4
RES_DEG = 0.25
N_ROWS = 720  # after Aurora's batch.crop(4): the 721st (south pole) row is dropped
N_COLS = 1440
LAT_TOP = 90.0
LON_LEFT = 0.0
N_LAT_PATCHES = N_ROWS // PATCH_SIZE  # 180
N_LON_PATCHES = N_COLS // PATCH_SIZE  # 360
STEP_HOURS = 6  # one rollout step = 6 h
LEVEL_SEMANTICS = {
    "0": "surface (read directly by the decoder's surface heads)",
    "1..": "aggregated atmospheric levels (de-aggregated by the decoder)",
}
CHANNEL_LAYOUT = "level-major: channel = level * embed_dim + d"
FORMAT_VERSION = 1

# fp16 holds |x| < 65504; keep a 2x safety factor for the storage decision.
FP16_MAX = 65504.0
FP16_SAFE_MAX = 3.0e4
# A channel may lose at most this fraction of its spread to fp16 rounding.
FP16_MAX_REL_ERR = 1e-2
STORAGE_DTYPES = ("float16", "float32", "centred_float16")


# --------------------------------------------------------------------------- #
# Crop geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PatchCrop:
    """A region crop on Aurora's token grid, margin included.

    Rows are global token rows ``h0 .. h0 + nh - 1``. Columns live in a
    *rolled* frame: crop column ``j`` is global token column
    ``(j - patch_roll) % n_lon_patches`` (``np.roll`` semantics), which keeps a
    0-deg-crossing box contiguous exactly as the cell-level crop does. The
    region interior occupies token rows ``interior_h .. +interior_nh`` and
    columns ``interior_w .. +interior_nw`` of the crop (each including the
    one token that the inclusive region edge overshoots into), and its 0.25
    deg grid occupies cell rows ``cell_row0 .. +n_cell_rows`` and columns
    ``cell_col0 .. +n_cell_cols`` of the unpatchified crop.
    """

    name: str
    bbox: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    patch_size: int
    margin: int
    n_lat_patches: int
    n_lon_patches: int
    h0: int
    nh: int
    patch_roll: int
    nw: int
    interior_h: int
    interior_w: int
    interior_nh: int
    interior_nw: int
    cell_row0: int
    cell_col0: int
    n_cell_rows: int
    n_cell_cols: int

    @property
    def shape(self) -> tuple[int, int]:
        """``(nh, nw)`` -- token rows and columns of the stored crop."""
        return (self.nh, self.nw)

    def global_patch_col(self, j: int) -> int:
        """Global token column of crop column ``j``."""
        return (j - self.patch_roll) % self.n_lon_patches

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PatchCrop:
        d = dict(d)
        d["bbox"] = tuple(float(v) for v in d["bbox"])
        return cls(**d)


def _to_lon180(lon: np.ndarray | float):
    """0..360 -> -180..180, with the same ``> 180`` rule as the cell-level crop."""
    return np.where(np.asarray(lon) > 180, np.asarray(lon) - 360, lon)


def patch_crop_from_bbox(
    name: str,
    bbox: tuple[float, float, float, float],
    margin_patches: int = 4,
    patch_size: int = PATCH_SIZE,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
    res_deg: float = RES_DEG,
) -> PatchCrop:
    """Derive the token-grid crop of ``bbox`` from the cell-level crop.

    Runs :func:`compute_grid_crop_indices` on Aurora's (cropped) global grid,
    checks that the cell crop is token-aligned -- the top edge and the roll
    must fall on token boundaries, which holds when ``lat_max`` and
    ``lon_min`` are whole degrees -- and widens it by ``margin_patches`` tokens
    on every side (clipped at the poles, wrapped in longitude).
    """
    if margin_patches < 0:
        raise ValueError("margin_patches must be >= 0")
    lat_min, lat_max, lon_min, lon_max = (float(v) for v in bbox)
    lats = LAT_TOP - res_deg * np.arange(n_rows, dtype=np.float64)
    lons = LON_LEFT + res_deg * np.arange(n_cols, dtype=np.float64)
    lat_idx, lon_idx, roll, _lats_c, _lons_c = compute_grid_crop_indices(
        lats, lons, (lat_min, lat_max), (lon_min, lon_max)
    )
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(f"Region {name!r}: bbox {bbox} selects no cells.")
    if not (np.all(np.diff(lat_idx) == 1) and np.all(np.diff(lon_idx) == 1)):
        raise ValueError(f"Region {name!r}: cell crop is not contiguous.")
    if lon_idx[0] != 0:
        raise ValueError(
            f"Region {name!r}: rolled crop does not start at column 0 ({lon_idx[0]})."
        )
    if lat_idx[0] % patch_size != 0:
        raise ValueError(
            f"Region {name!r}: top edge lat={lat_max} is not token-aligned "
            f"(row {lat_idx[0]} is not a multiple of {patch_size}); the bbox's "
            f"lat_max must be a multiple of {patch_size * res_deg} deg."
        )
    if roll % patch_size != 0:
        raise ValueError(
            f"Region {name!r}: longitude roll {roll} cells is not a multiple of "
            f"{patch_size}; the bbox's lon_min must be a multiple of "
            f"{patch_size * res_deg} deg."
        )
    n_lat_patches = n_rows // patch_size
    n_lon_patches = n_cols // patch_size

    h_first = int(lat_idx[0]) // patch_size
    h_last = int(lat_idx[-1]) // patch_size
    w_last = int(lon_idx[-1]) // patch_size
    interior_nh = h_last - h_first + 1
    interior_nw = w_last + 1

    h0 = max(h_first - margin_patches, 0)
    h1 = min(h_last + margin_patches, n_lat_patches - 1)
    nh = h1 - h0 + 1
    nw = interior_nw + 2 * margin_patches
    if nw > n_lon_patches:
        raise ValueError(
            f"Region {name!r}: crop width {nw} tokens exceeds the grid "
            f"({n_lon_patches}); reduce margin_patches."
        )
    patch_roll = int(roll) // patch_size + margin_patches
    interior_h = h_first - h0
    return PatchCrop(
        name=name,
        bbox=(lat_min, lat_max, lon_min, lon_max),
        patch_size=patch_size,
        margin=margin_patches,
        n_lat_patches=n_lat_patches,
        n_lon_patches=n_lon_patches,
        h0=h0,
        nh=nh,
        patch_roll=patch_roll,
        nw=nw,
        interior_h=interior_h,
        interior_w=margin_patches,
        interior_nh=interior_nh,
        interior_nw=interior_nw,
        cell_row0=interior_h * patch_size,
        cell_col0=margin_patches * patch_size,
        n_cell_rows=int(len(lat_idx)),
        n_cell_cols=int(len(lon_idx)),
    )


def crop_columns(pc: PatchCrop) -> np.ndarray:
    """Global token columns of the crop's columns, in crop order."""
    return (np.arange(pc.nw) - pc.patch_roll) % pc.n_lon_patches


def crop_latent_tokens(tokens: np.ndarray, pc: PatchCrop) -> np.ndarray:
    """``(C, n_lat_patches, n_lon_patches, D)`` -> stored ``(C*D, nh, nw)``.

    Rows are sliced, columns gathered in the rolled order, then the array is
    re-laid out level-major (``channel = level*D + d``).
    """
    if tokens.ndim != 4:
        raise ValueError(f"tokens must be (C, H, W, D); got shape {tokens.shape}")
    c, h, w, d = tokens.shape
    if (h, w) != (pc.n_lat_patches, pc.n_lon_patches):
        raise ValueError(
            f"tokens grid {h}x{w} does not match the crop's "
            f"{pc.n_lat_patches}x{pc.n_lon_patches} token grid"
        )
    sub = tokens[:, pc.h0 : pc.h0 + pc.nh]  # (C, nh, W, D) view
    sub = np.take(sub, crop_columns(pc), axis=2)  # (C, nh, nw, D) copy
    out = np.ascontiguousarray(sub.transpose(0, 3, 1, 2))  # (C, D, nh, nw)
    return out.reshape(c * d, pc.nh, pc.nw)


def patch_centre_coords(pc: PatchCrop, res_deg: float = RES_DEG):
    """``(lats (nh,), lons (nw,))`` of the crop's token centres, -180/180 lons."""
    half = (pc.patch_size - 1) / 2.0
    rows = pc.patch_size * (pc.h0 + np.arange(pc.nh)) + half
    lats = LAT_TOP - res_deg * rows
    cols = pc.patch_size * crop_columns(pc) + half
    lons = _to_lon180(LON_LEFT + res_deg * cols)
    return lats.astype(np.float64), np.asarray(lons, dtype=np.float64)


def crop_cell_coords(pc: PatchCrop, res_deg: float = RES_DEG):
    """0.25 deg coordinates of the unpatchified crop: ``(lats (nh*p,), lons (nw*p,))``."""
    p = pc.patch_size
    rows = p * pc.h0 + np.arange(pc.nh * p)
    lats = LAT_TOP - res_deg * rows
    cols = (p * crop_columns(pc))[:, None] + np.arange(p)[None, :]
    lons = _to_lon180(LON_LEFT + res_deg * cols.reshape(-1))
    return lats.astype(np.float64), np.asarray(lons, dtype=np.float64)


def interior_patch_slices(pc: PatchCrop) -> tuple[slice, slice]:
    """Token rows/cols of the crop that cover the region (incl. the overshoot token)."""
    return (
        slice(pc.interior_h, pc.interior_h + pc.interior_nh),
        slice(pc.interior_w, pc.interior_w + pc.interior_nw),
    )


def interior_cell_slices(pc: PatchCrop) -> tuple[slice, slice]:
    """Cell rows/cols of the unpatchified crop that ARE the region's 0.25 deg grid."""
    return (
        slice(pc.cell_row0, pc.cell_row0 + pc.n_cell_rows),
        slice(pc.cell_col0, pc.cell_col0 + pc.n_cell_cols),
    )


def level_channel_slices(embed_dim: int, levels: list[int]) -> list[slice]:
    """Channel slices of the requested latent levels in the level-major layout."""
    return [slice(lv * embed_dim, (lv + 1) * embed_dim) for lv in levels]


# --------------------------------------------------------------------------- #
# Token tensor <-> grid
# --------------------------------------------------------------------------- #


def tokens_to_grid(
    x,
    latent_levels: int,
    n_lat_patches: int = N_LAT_PATCHES,
    n_lon_patches: int = N_LON_PATCHES,
) -> np.ndarray:
    """``(1, L, D)`` (torch or numpy, ``L = C*H*W`` in ``(level, row, col)``
    order -- Aurora's ``"B (C H W) D"``) -> ``(C, H, W, D)`` float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1; got shape {x.shape}")
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected (1, L, D) or (L, D); got shape {x.shape}")
    n_tokens, d = x.shape
    expected = latent_levels * n_lat_patches * n_lon_patches
    if n_tokens != expected:
        raise ValueError(
            f"Token count {n_tokens} != latent_levels*H*W = "
            f"{latent_levels}*{n_lat_patches}*{n_lon_patches} = {expected}"
        )
    return x.reshape(latent_levels, n_lat_patches, n_lon_patches, d).astype(
        np.float32, copy=False
    )


class LatentCapture:
    """Collect a module's outputs across a rollout, keyed by 1-based call index.

    Aurora's ``rollout`` calls ``model.forward`` once per step, and the
    forward pass calls ``model.encoder`` and ``model.backbone`` exactly once
    each, so the k-th call of either module is rollout step k. Only the steps
    in ``keep_steps`` are retained (``None`` keeps every call); tensors are
    detached and, by default, moved to the CPU as float32 so GPU memory is
    not held across steps. ``pre=True`` captures the first positional input
    instead of the output (used to check that the decoder consumes exactly
    the backbone output).
    """

    def __init__(self, module, keep_steps=None, to_cpu: bool = True, pre: bool = False):
        self.module = module
        self.keep_steps = None if keep_steps is None else set(keep_steps)
        self.to_cpu = to_cpu
        self.pre = pre
        self.calls = 0
        self.outputs: dict[int, object] = {}
        if pre:
            self._handle = module.register_forward_pre_hook(self._pre_hook)
        else:
            self._handle = module.register_forward_hook(self._hook)

    def _keep(self, tensor) -> None:
        if self.keep_steps is None or self.calls in self.keep_steps:
            t = tensor.detach()
            if self.to_cpu:
                t = t.float().cpu()
            self.outputs[self.calls] = t

    def _hook(self, _module, _inputs, output) -> None:
        self.calls += 1
        self._keep(output[0] if isinstance(output, tuple) else output)

    def _pre_hook(self, _module, inputs) -> None:
        self.calls += 1
        self._keep(inputs[0])

    def reset(self, keep_steps=None) -> None:
        """Start a new rollout: zero the call counter, drop retained tensors."""
        self.calls = 0
        self.outputs = {}
        if keep_steps is not None:
            self.keep_steps = set(keep_steps)

    def pop(self, step: int):
        return self.outputs.pop(step)

    def remove(self) -> None:
        self._handle.remove()


# --------------------------------------------------------------------------- #
# Storage encoding and fp16 calibration
# --------------------------------------------------------------------------- #


def encode_for_storage(arr: np.ndarray, dtype: str, offset=None) -> np.ndarray:
    """float32 ``(C, h, w)`` -> array to save, per the storage decision."""
    if dtype == "float32":
        return np.asarray(arr, dtype=np.float32)
    if dtype == "float16":
        return np.asarray(arr, dtype=np.float32).astype(np.float16)
    if dtype == "centred_float16":
        off = np.asarray(offset, dtype=np.float32)[:, None, None]
        return (np.asarray(arr, dtype=np.float32) - off).astype(np.float16)
    raise ValueError(f"Unknown storage dtype {dtype!r}; expected {STORAGE_DTYPES}")


def decode_from_storage(stored: np.ndarray, dtype: str, offset=None) -> np.ndarray:
    """Inverse of :func:`encode_for_storage`; always returns float32."""
    if dtype == "float32":
        return np.asarray(stored, dtype=np.float32)
    if dtype == "float16":
        return np.asarray(stored, dtype=np.float32)
    if dtype == "centred_float16":
        off = np.asarray(offset, dtype=np.float32)[:, None, None]
        return np.asarray(stored, dtype=np.float32) + off
    raise ValueError(f"Unknown storage dtype {dtype!r}; expected {STORAGE_DTYPES}")


def storage_numpy_dtype(dtype: str) -> np.dtype:
    return np.dtype(np.float32 if dtype == "float32" else np.float16)


def check_storable(encoded: np.ndarray) -> None:
    """Refuse to write fp16 overflow (``inf`` after the cast, or the largest
    representable value) or any non-finite value."""
    if encoded.dtype == np.float16 and (
        np.isinf(encoded).any() or float(np.abs(encoded).max()) >= FP16_MAX
    ):
        raise ValueError("latent overflows float16")
    if not np.isfinite(encoded).all():
        raise ValueError("latent contains non-finite values")


class CalibrationAccumulator:
    """Streaming per-channel statistics for the fp16 storage decision.

    Feed float32 ``(C, h, w)`` crops; accumulates per-channel mean/std/abs-max
    and the RMS error of an fp16 round trip, both plain and centred on a
    provisional per-channel offset (the first sample's channel means, which
    is what makes the centred error computable in one pass). The final
    ``offset`` reported is the full-sample channel mean.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.n_values = 0
        self.n_samples = 0
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sumsq = np.zeros(n_channels, dtype=np.float64)
        self.absmax = np.zeros(n_channels, dtype=np.float64)
        self.centred_absmax = np.zeros(n_channels, dtype=np.float64)
        self.err_plain_sq = np.zeros(n_channels, dtype=np.float64)
        self.err_centred_sq = np.zeros(n_channels, dtype=np.float64)
        self.provisional_offset: np.ndarray | None = None

    def update(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] != self.n_channels:
            raise ValueError(
                f"expected (C={self.n_channels}, h, w); got shape {arr.shape}"
            )
        flat = arr.reshape(self.n_channels, -1).astype(np.float64)
        if self.provisional_offset is None:
            self.provisional_offset = flat.mean(axis=1)
        self.n_samples += 1
        self.n_values += flat.shape[1]
        self.sum += flat.sum(axis=1)
        self.sumsq += (flat**2).sum(axis=1)
        self.absmax = np.maximum(self.absmax, np.abs(flat).max(axis=1))
        centred = flat - self.provisional_offset[:, None]
        self.centred_absmax = np.maximum(
            self.centred_absmax, np.abs(centred).max(axis=1)
        )
        # Values beyond the fp16 range become inf here; the resulting infinite
        # error is exactly what should disqualify fp16, so silence the warning.
        with np.errstate(over="ignore"):
            rt = flat.astype(np.float32).astype(np.float16).astype(np.float64)
            rt_c = centred.astype(np.float32).astype(np.float16).astype(np.float64)
        self.err_plain_sq += ((flat - rt) ** 2).sum(axis=1)
        self.err_centred_sq += ((centred - rt_c) ** 2).sum(axis=1)

    def summary(self) -> dict:
        if self.n_values == 0:
            raise ValueError("no samples accumulated")
        mean = self.sum / self.n_values
        var = np.maximum(self.sumsq / self.n_values - mean**2, 0.0)
        std = np.sqrt(var)
        safe_std = np.maximum(std, 1e-12)
        rel_plain = np.sqrt(self.err_plain_sq / self.n_values) / safe_std
        rel_centred = np.sqrt(self.err_centred_sq / self.n_values) / safe_std
        # The stored offset is the full-sample mean, the accumulated centred
        # magnitude used the provisional one; the difference bounds the gap.
        centred_absmax = self.centred_absmax + np.abs(mean - self.provisional_offset)
        return {
            "n_samples": int(self.n_samples),
            "n_values_per_channel": int(self.n_values),
            "mean": mean,
            "std": std,
            "absmax": self.absmax.copy(),
            "rel_err_float16": rel_plain,
            "rel_err_centred_float16": rel_centred,
            "centred_absmax_bound": centred_absmax,
        }


def decide_storage(
    summary: dict,
    max_abs: float = FP16_SAFE_MAX,
    max_rel_err: float = FP16_MAX_REL_ERR,
) -> dict:
    """Pick the storage encoding from a :meth:`CalibrationAccumulator.summary`.

    ``float16`` if every channel stays under ``max_abs`` and loses at most
    ``max_rel_err`` of its spread to rounding; else ``centred_float16`` (a
    per-channel offset is subtracted first) if that suffices; else
    ``float32``. Returns ``{"dtype", "offset", "calibration"}`` ready to be
    embedded in the metadata (``calibration`` holds only summary scalars and
    the worst channels; per-channel arrays are kept out of the JSON except
    the offset).
    """
    absmax = np.asarray(summary["absmax"])
    rel_plain = np.asarray(summary["rel_err_float16"])
    rel_centred = np.asarray(summary["rel_err_centred_float16"])
    centred_absmax = np.asarray(summary["centred_absmax_bound"])
    mean = np.asarray(summary["mean"])

    plain_ok = bool(absmax.max() < max_abs and rel_plain.max() <= max_rel_err)
    centred_ok = bool(
        centred_absmax.max() < max_abs and rel_centred.max() <= max_rel_err
    )
    if plain_ok:
        dtype, offset = "float16", None
    elif centred_ok:
        dtype, offset = "centred_float16", [float(v) for v in mean]
    else:
        dtype, offset = "float32", None

    def worst(v: np.ndarray, k: int = 5) -> list[dict]:
        idx = np.argsort(v)[::-1][:k]
        return [{"channel": int(i), "value": float(v[i])} for i in idx]

    return {
        "dtype": dtype,
        "offset": offset,
        "calibration": {
            "n_samples": summary["n_samples"],
            "max_abs_threshold": float(max_abs),
            "max_rel_err_threshold": float(max_rel_err),
            "absmax_max": float(absmax.max()),
            "rel_err_float16_max": float(rel_plain.max()),
            "rel_err_centred_float16_max": float(rel_centred.max()),
            "worst_channels_absmax": worst(absmax),
            "worst_channels_rel_err_float16": worst(rel_plain),
            "worst_channels_rel_err_centred_float16": worst(rel_centred),
            "float16_ok": plain_ok,
            "centred_float16_ok": centred_ok,
        },
    }


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

# Keys of the metadata that must agree between shards writing the same
# directory (geometry, layout, storage); provenance keys may differ.
GEOMETRY_KEYS = (
    "format_version",
    "kind",
    "lead_hours",
    "rollout_step",
    "embed_dim",
    "latent_levels",
    "n_channels",
    "channel_layout",
    "crop",
    "shape",
    "storage_dtype",
    "storage_offset",
)


def latent_meta(
    pc: PatchCrop,
    *,
    kind: str,
    lead_hours: int,
    rollout_step: int,
    embed_dim: int,
    latent_levels: int,
    storage: dict,
    model: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """The self-describing sidecar written next to a directory of latent files."""
    if kind not in ("backbone", "encoder"):
        raise ValueError("kind must be 'backbone' or 'encoder'")
    if storage["dtype"] not in STORAGE_DTYPES:
        raise ValueError(f"storage dtype must be one of {STORAGE_DTYPES}")
    if storage["dtype"] == "centred_float16":
        if storage.get("offset") is None or len(storage["offset"]) != (
            latent_levels * embed_dim
        ):
            raise ValueError("centred_float16 needs one offset per channel")
    lats, lons = patch_centre_coords(pc)
    meta = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "description": (
            "Aurora backbone output (the decoder's input)"
            if kind == "backbone"
            else "Aurora encoder output (embedding of the two input frames)"
        ),
        "lead_hours": int(lead_hours),
        "rollout_step": int(rollout_step),
        "step_hours": STEP_HOURS,
        "embed_dim": int(embed_dim),
        "latent_levels": int(latent_levels),
        "n_channels": int(latent_levels * embed_dim),
        "channel_layout": CHANNEL_LAYOUT,
        "level_semantics": LEVEL_SEMANTICS,
        "token_grid": {
            "patch_size": pc.patch_size,
            "n_lat_patches": pc.n_lat_patches,
            "n_lon_patches": pc.n_lon_patches,
            "lat_of_row": f"{LAT_TOP} - {RES_DEG} * row (row 0..{N_ROWS - 1})",
            "lon_of_col": f"{LON_LEFT} + {RES_DEG} * col (col 0..{N_COLS - 1})",
            "token_order": "(level, row, col), Aurora's 'B (C H W) D'",
        },
        "crop": pc.to_dict(),
        "shape": [int(latent_levels * embed_dim), pc.nh, pc.nw],
        "interior": {
            "note": (
                "crop[:, cell_row0:cell_row0+n_cell_rows, cell_col0:cell_col0+"
                "n_cell_cols] of the unpatchified (0.25 deg) crop is the region "
                "grid; token rows/cols interior_h/interior_w cover it"
            ),
        },
        "patch_centre_lats": [float(v) for v in lats],
        "patch_centre_lons": [float(v) for v in lons],
        "storage_dtype": storage["dtype"],
        "storage_offset": storage.get("offset"),
        "storage_calibration": storage.get("calibration"),
        "model": model or {},
    }
    if extra:
        meta.update(extra)
    return meta


def patch_crop_from_meta(meta: dict) -> PatchCrop:
    return PatchCrop.from_dict(meta["crop"])


def meta_geometry_equal(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in GEOMETRY_KEYS)


def validate_latent_meta(
    meta: dict, grid_lats: np.ndarray, grid_lons: np.ndarray, atol: float = 1e-4
) -> None:
    """Check a directory's meta: stored shape and channel count agree with the
    crop, and the crop's interior IS the region grid ``(grid_lats, grid_lons)``.

    Raises ``ValueError`` with the first discrepancy.
    """
    pc = patch_crop_from_meta(meta)
    if list(meta["shape"]) != [meta["n_channels"], pc.nh, pc.nw]:
        raise ValueError(f"shape {meta['shape']} != [n_channels, nh, nw] of the crop")
    if meta["n_channels"] != meta["latent_levels"] * meta["embed_dim"]:
        raise ValueError("n_channels != latent_levels * embed_dim")
    validate_patch_crop(pc, grid_lats, grid_lons, atol=atol)


def validate_patch_crop(
    pc: PatchCrop, grid_lats: np.ndarray, grid_lons: np.ndarray, atol: float = 1e-4
) -> None:
    """Check that the interior cells of ``pc`` are exactly the region grid
    ``(grid_lats, grid_lons)`` (the dataset's ``regions/<r>/lats.npy`` /
    ``lons.npy``). Raises ``ValueError`` with the first discrepancy."""
    grid_lats = np.asarray(grid_lats, dtype=np.float64)
    grid_lons = np.asarray(grid_lons, dtype=np.float64)
    if len(grid_lats) != pc.n_cell_rows or len(grid_lons) != pc.n_cell_cols:
        raise ValueError(
            f"region grid {len(grid_lats)}x{len(grid_lons)} != crop interior "
            f"{pc.n_cell_rows}x{pc.n_cell_cols}"
        )
    cell_lats, cell_lons = crop_cell_coords(pc)
    rs, cs = interior_cell_slices(pc)
    if not np.allclose(cell_lats[rs], grid_lats, atol=atol):
        raise ValueError(
            f"interior latitudes start at {cell_lats[rs][0]} but the region grid "
            f"starts at {grid_lats[0]}"
        )
    if not np.allclose(cell_lons[cs], grid_lons, atol=atol):
        raise ValueError(
            f"interior longitudes start at {cell_lons[cs][0]} but the region grid "
            f"starts at {grid_lons[0]}"
        )
    ps = pc.patch_size
    if ps * pc.interior_nh < pc.n_cell_rows or ps * pc.interior_nw < pc.n_cell_cols:
        raise ValueError("interior tokens do not cover the region grid")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #

META_NAME = "latent_meta.json"
LATS_NAME = "latent_lats.npy"
LONS_NAME = "latent_lons.npy"


def write_latent(path: Path, arr: np.ndarray, storage: dict) -> None:
    """Encode ``arr`` (float32 ``(C, h, w)``) per ``storage`` and save atomically."""
    encoded = encode_for_storage(arr, storage["dtype"], storage.get("offset"))
    check_storable(encoded)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, encoded)
    os.replace(tmp, path)


def load_latent(path: Path, meta: dict) -> np.ndarray:
    """Read one latent file back to float32 ``(C, h, w)`` using its directory's meta."""
    stored = np.load(path)
    return decode_from_storage(
        stored, meta["storage_dtype"], meta.get("storage_offset")
    )


def write_sidecars(directory: Path, meta: dict, pc: PatchCrop) -> None:
    """Write ``latent_meta.json`` + patch-centre coordinate files once per directory.

    If a meta already exists it must agree on geometry/layout/storage
    (another shard wrote it); provenance keys may differ.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / META_NAME
    if meta_path.exists():
        existing = json.loads(meta_path.read_text())
        if not meta_geometry_equal(existing, meta):
            diff = [k for k in GEOMETRY_KEYS if existing.get(k) != meta.get(k)]
            raise RuntimeError(
                f"{meta_path} exists with different geometry/storage (keys {diff}); "
                "refusing to mix shards written with different settings."
            )
        return
    lats, lons = patch_centre_coords(pc)
    np.save(directory / LATS_NAME, lats.astype(np.float32))
    np.save(directory / LONS_NAME, lons.astype(np.float32))
    tmp = meta_path.with_name(META_NAME + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, meta_path)


def read_meta(directory: Path) -> dict:
    return json.loads((Path(directory) / META_NAME).read_text())


__all__ = [
    "CHANNEL_LAYOUT",
    "FORMAT_VERSION",
    "FP16_MAX_REL_ERR",
    "FP16_SAFE_MAX",
    "GEOMETRY_KEYS",
    "LATS_NAME",
    "LONS_NAME",
    "META_NAME",
    "N_COLS",
    "N_LAT_PATCHES",
    "N_LON_PATCHES",
    "N_ROWS",
    "PATCH_SIZE",
    "RES_DEG",
    "STEP_HOURS",
    "STORAGE_DTYPES",
    "CalibrationAccumulator",
    "LatentCapture",
    "PatchCrop",
    "check_storable",
    "crop_cell_coords",
    "crop_columns",
    "crop_latent_tokens",
    "decide_storage",
    "decode_from_storage",
    "encode_for_storage",
    "interior_cell_slices",
    "interior_patch_slices",
    "latent_meta",
    "level_channel_slices",
    "load_latent",
    "meta_geometry_equal",
    "patch_centre_coords",
    "patch_crop_from_bbox",
    "patch_crop_from_meta",
    "read_meta",
    "storage_numpy_dtype",
    "tokens_to_grid",
    "validate_latent_meta",
    "write_latent",
    "write_sidecars",
]
