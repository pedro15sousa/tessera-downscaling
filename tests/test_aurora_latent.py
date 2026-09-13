"""Aurora latent crops: token-grid geometry against the cell-level crop, the
level-major channel layout, interior recoverability, storage encoding and
calibration, metadata round trips, and the hook capture logic (on a stand-in
module, so no Aurora install is needed). Pure numpy/torch, no data root.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from tessera_downscaling.preprocessing import aurora_latent as al
from tessera_downscaling.preprocessing.helpers import compute_grid_crop_indices

# The five dataset regions (dataset_timestamp_global/metadata.json bboxes).
BBOXES = {
    "europe": (35.0, 75.0, -24.0, 40.0),
    "us": (24.0, 50.0, -125.0, -66.0),
    "east_asia": (20.0, 46.0, 100.0, 146.0),
    "australia": (-44.0, -10.0, 112.0, 154.0),
    "southern_africa": (-35.0, -15.0, 15.0, 35.0),
}


def global_grid():
    lats = al.LAT_TOP - al.RES_DEG * np.arange(al.N_ROWS)
    lons = al.LON_LEFT + al.RES_DEG * np.arange(al.N_COLS)
    return lats, lons


def region_grid(bbox):
    """The dataset's regions/<r>/lats.npy, lons.npy (computed the same way)."""
    lats, lons = global_grid()
    lat_min, lat_max, lon_min, lon_max = bbox
    _, _, _, lats_c, lons_c = compute_grid_crop_indices(
        lats, lons, (lat_min, lat_max), (lon_min, lon_max)
    )
    return lats_c, lons_c


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("margin", [0, 4])
@pytest.mark.parametrize("name", sorted(BBOXES))
def test_interior_cells_equal_the_cell_level_crop(name, margin):
    pc = al.patch_crop_from_bbox(name, BBOXES[name], margin_patches=margin)
    lats_c, lons_c = region_grid(BBOXES[name])
    cell_lats, cell_lons = al.crop_cell_coords(pc)
    rs, cs = al.interior_cell_slices(pc)
    assert np.allclose(cell_lats[rs], lats_c)
    assert np.allclose(cell_lons[cs], lons_c)
    al.validate_patch_crop(pc, lats_c, lons_c)  # the same check, as the CLI runs it
    # Interior tokens cover the region with less than one token of overshoot.
    assert 0 <= pc.patch_size * pc.interior_nh - pc.n_cell_rows < pc.patch_size
    assert 0 <= pc.patch_size * pc.interior_nw - pc.n_cell_cols < pc.patch_size
    # Margin bookkeeping (none of the five bboxes touches a pole).
    assert pc.interior_w == margin and pc.interior_h == margin
    assert pc.nw == pc.interior_nw + 2 * margin
    assert pc.nh == pc.interior_nh + 2 * margin
    assert pc.cell_row0 == margin * pc.patch_size
    assert pc.cell_col0 == margin * pc.patch_size


def test_europe_geometry_and_zero_meridian_wrap():
    pc = al.patch_crop_from_bbox("europe", BBOXES["europe"], margin_patches=4)
    assert (pc.h0, pc.nh, pc.nw) == (11, 49, 73)
    assert (pc.interior_nh, pc.interior_nw) == (41, 65)
    assert (pc.n_cell_rows, pc.n_cell_cols) == (161, 257)
    lats, lons = al.patch_centre_coords(pc)
    assert lats[pc.interior_h] == pytest.approx(74.625)
    assert lons[pc.interior_w] == pytest.approx(-23.625)
    assert lons[0] == pytest.approx(-27.625)  # 4 tokens of margin west of -24
    cols = al.crop_columns(pc)
    assert cols[pc.interior_w] == 336  # lon -24 deg = global token column 336
    assert 359 in cols and 0 in cols  # the crop crosses the 0 deg meridian
    assert lons[-1] == pytest.approx(44.375)


def test_east_asia_and_southern_hemisphere_rows():
    ea = al.patch_crop_from_bbox("east_asia", BBOXES["east_asia"], margin_patches=4)
    assert (ea.h0, ea.nh, ea.nw) == (40, 35, 55)
    lats, lons = al.patch_centre_coords(ea)
    assert lats[ea.interior_h] == pytest.approx(45.625)
    assert lons[ea.interior_w] == pytest.approx(100.375)
    au = al.patch_crop_from_bbox("australia", BBOXES["australia"], margin_patches=0)
    assert (au.h0, au.interior_nh) == (100, 35)  # lat -10 -> row 400 -> token 100
    assert al.patch_centre_coords(au)[0][0] == pytest.approx(-10.375)


def test_misaligned_bbox_raises():
    with pytest.raises(ValueError, match="token-aligned"):
        al.patch_crop_from_bbox("x", (35.0, 74.5, -24.0, 40.0))
    with pytest.raises(ValueError, match="roll"):
        al.patch_crop_from_bbox("x", (35.0, 75.0, -24.5, 40.0))
    with pytest.raises(ValueError, match="margin"):
        al.patch_crop_from_bbox("x", BBOXES["europe"], margin_patches=200)


# --------------------------------------------------------------------------- #
# Cropping and layout
# --------------------------------------------------------------------------- #


def test_crop_latent_tokens_is_level_major_and_rolled():
    c, d = 2, 3
    h, w = al.N_LAT_PATCHES, al.N_LON_PATCHES
    # Unique integer per (level, row, col, d); exact in float32.
    idx = np.arange(c * h * w * d, dtype=np.float32).reshape(c, h, w, d)
    pc = al.patch_crop_from_bbox("europe", BBOXES["europe"], margin_patches=4)
    out = al.crop_latent_tokens(idx, pc)
    assert out.shape == (c * d, pc.nh, pc.nw)
    cols = al.crop_columns(pc)
    for ch in (0, 1, d, c * d - 1):
        level, dd = divmod(ch, d)
        expected = idx[level, pc.h0 : pc.h0 + pc.nh][:, cols, dd]
        assert np.array_equal(out[ch], expected)
    assert al.level_channel_slices(d, [1]) == [slice(d, 2 * d)]
    with pytest.raises(ValueError):
        al.crop_latent_tokens(idx[:, :10], pc)


@pytest.mark.parametrize("name", ["europe", "australia"])
def test_interior_unpatchify_recovers_the_region_field(name):
    """Patchify a synthetic global field into 16-d tokens, crop, unpatchify the
    interior: it must equal the cell-level crop of the same field."""
    lats, lons = global_grid()
    field = (
        np.arange(al.N_ROWS)[:, None] * 10_000 + np.arange(al.N_COLS)[None, :]
    ).astype(np.float64)
    p = al.PATCH_SIZE
    tokens = (
        field.reshape(al.N_LAT_PATCHES, p, al.N_LON_PATCHES, p)
        .transpose(0, 2, 1, 3)
        .reshape(1, al.N_LAT_PATCHES, al.N_LON_PATCHES, p * p)
    )
    pc = al.patch_crop_from_bbox(name, BBOXES[name], margin_patches=4)
    crop = al.crop_latent_tokens(tokens, pc)  # (16, nh, nw)
    cells = (
        crop.reshape(p, p, pc.nh, pc.nw)
        .transpose(2, 0, 3, 1)
        .reshape(pc.nh * p, pc.nw * p)
    )
    rs, cs = al.interior_cell_slices(pc)
    lat_min, lat_max, lon_min, lon_max = BBOXES[name]
    lat_idx, lon_idx, roll, _, _ = compute_grid_crop_indices(
        lats, lons, (lat_min, lat_max), (lon_min, lon_max)
    )
    expected = np.roll(field, roll, axis=1)[lat_idx][:, lon_idx]
    assert np.array_equal(cells[rs, cs], expected)


def test_tokens_to_grid_orders_level_row_col():
    lv, h, w, d = 4, al.N_LAT_PATCHES, al.N_LON_PATCHES, 2
    x = torch.arange(lv * h * w * d, dtype=torch.float32).reshape(1, lv * h * w, d)
    g = al.tokens_to_grid(x, lv)
    assert g.shape == (lv, h, w, d)
    assert g[1, 0, 0, 0] == h * w * d  # second level starts after one full grid
    assert g[0, 1, 0, 0] == w * d  # next row after one row of tokens
    with pytest.raises(ValueError, match="Token count"):
        al.tokens_to_grid(x[:, :-1], lv)


# --------------------------------------------------------------------------- #
# Hook capture
# --------------------------------------------------------------------------- #


class _Tuple(nn.Module):
    def forward(self, x):
        return x * 2, "aux"


def test_latent_capture_keeps_only_requested_steps():
    lin = nn.Linear(3, 3)
    cap = al.LatentCapture(lin, keep_steps={1, 4, 12})
    pre = al.LatentCapture(lin, keep_steps={4}, pre=True)
    xs = [torch.randn(1, 5, 3) for _ in range(12)]
    with torch.no_grad():
        outs = [lin(x) for x in xs]
    assert cap.calls == 12 and pre.calls == 12
    assert sorted(cap.outputs) == [1, 4, 12]
    assert torch.equal(cap.pop(4), outs[3]) and 4 not in cap.outputs
    assert torch.equal(pre.pop(4), xs[3])  # pre-hook captures the input
    cap.reset(keep_steps={2})
    assert cap.calls == 0 and cap.outputs == {}
    with torch.no_grad():
        lin(xs[0])
        lin(xs[1])
    assert sorted(cap.outputs) == [2]
    cap.remove()
    pre.remove()
    with torch.no_grad():
        lin(xs[0])
    assert cap.calls == 2  # detached hook no longer counts

    tup = al.LatentCapture(_Tuple(), keep_steps=None)
    tup.module(torch.ones(2))
    assert torch.equal(tup.pop(1), torch.full((2,), 2.0))


# --------------------------------------------------------------------------- #
# Storage encoding and calibration
# --------------------------------------------------------------------------- #


def _accumulate(samples):
    acc = al.CalibrationAccumulator(samples[0].shape[0])
    for s in samples:
        acc.update(s)
    return acc.summary()


def test_calibration_picks_float16_for_well_scaled_channels():
    rng = np.random.default_rng(0)
    samples = [rng.normal(0, 3, size=(8, 6, 7)).astype(np.float32) for _ in range(4)]
    decision = al.decide_storage(_accumulate(samples))
    assert decision["dtype"] == "float16" and decision["offset"] is None
    assert decision["calibration"]["rel_err_float16_max"] < 1e-2


def test_calibration_picks_centred_float16_for_an_offset_channel():
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(4):
        s = rng.normal(0, 1, size=(4, 6, 7)).astype(np.float32)
        s[2] += 2.0e4  # large mean, unit spread: fp16 spacing there is 16
        samples.append(s)
    summary = _accumulate(samples)
    assert summary["rel_err_float16"][2] > 1e-2
    decision = al.decide_storage(summary)
    assert decision["dtype"] == "centred_float16"
    assert decision["offset"][2] == pytest.approx(2.0e4, abs=1.0)
    enc = al.encode_for_storage(samples[0], "centred_float16", decision["offset"])
    assert enc.dtype == np.float16
    dec = al.decode_from_storage(enc, "centred_float16", decision["offset"])
    assert np.abs(dec - samples[0]).max() < 1e-2


def test_calibration_falls_back_to_float32_on_overflow_risk():
    rng = np.random.default_rng(2)
    samples = [
        rng.normal(0, 4.0e4, size=(3, 6, 7)).astype(np.float32) for _ in range(3)
    ]
    decision = al.decide_storage(_accumulate(samples))
    assert decision["dtype"] == "float32"
    assert al.encode_for_storage(samples[0], "float32").dtype == np.float32
    with np.errstate(over="ignore"):
        overflowed = np.array([7.0e4], dtype=np.float32).astype(np.float16)
    with pytest.raises(ValueError, match="overflow"):
        al.check_storable(overflowed)
    with pytest.raises(ValueError, match="overflow"):
        al.check_storable(np.array([al.FP16_MAX], dtype=np.float16))
    with pytest.raises(ValueError, match="non-finite"):
        al.check_storable(np.array([np.nan], dtype=np.float16))
    with pytest.raises(ValueError, match="non-finite"):
        al.check_storable(np.array([np.inf], dtype=np.float32))


# --------------------------------------------------------------------------- #
# Metadata and files
# --------------------------------------------------------------------------- #


def _meta(pc, storage=None, **kw):
    storage = storage or {"dtype": "float16", "offset": None, "calibration": None}
    return al.latent_meta(
        pc,
        kind=kw.get("kind", "backbone"),
        lead_hours=kw.get("lead_hours", 6),
        rollout_step=kw.get("rollout_step", 1),
        embed_dim=kw.get("embed_dim", 8),
        latent_levels=4,
        storage=storage,
        model={"name": "small"},
        extra={"created": "now"},
    )


def test_meta_round_trip_and_validation_against_the_region_grid():
    pc = al.patch_crop_from_bbox("europe", BBOXES["europe"], margin_patches=4)
    meta = json.loads(json.dumps(_meta(pc)))  # must be JSON-serialisable
    assert al.patch_crop_from_meta(meta) == pc
    assert meta["shape"] == [32, pc.nh, pc.nw]
    lats_c, lons_c = region_grid(BBOXES["europe"])
    al.validate_latent_meta(meta, lats_c, lons_c)
    with pytest.raises(ValueError, match="latitudes"):
        al.validate_latent_meta(meta, lats_c - 0.25, lons_c)
    with pytest.raises(ValueError, match="region grid"):
        al.validate_latent_meta(meta, lats_c[:-1], lons_c)
    bad = dict(meta, shape=[32, pc.nh, pc.nw + 1])
    with pytest.raises(ValueError, match="shape"):
        al.validate_latent_meta(bad, lats_c, lons_c)
    with pytest.raises(ValueError, match="offset"):
        _meta(pc, storage={"dtype": "centred_float16", "offset": [0.0]})


def test_sidecars_and_latent_files_round_trip(tmp_path):
    pc = al.patch_crop_from_bbox("east_asia", BBOXES["east_asia"], margin_patches=4)
    meta = _meta(pc)
    al.write_sidecars(tmp_path, meta, pc)
    assert al.read_meta(tmp_path)["crop"] == pc.to_dict()
    lats = np.load(tmp_path / al.LATS_NAME)
    assert lats.shape == (pc.nh,) and lats[pc.interior_h] == pytest.approx(45.625)
    al.write_sidecars(
        tmp_path, dict(meta, created="later"), pc
    )  # provenance may differ
    other = al.patch_crop_from_bbox("east_asia", BBOXES["east_asia"], margin_patches=2)
    with pytest.raises(RuntimeError, match="different geometry"):
        al.write_sidecars(tmp_path, _meta(other), other)

    rng = np.random.default_rng(3)
    arr = rng.normal(0, 2, size=(32, pc.nh, pc.nw)).astype(np.float32)
    al.write_latent(tmp_path / "2020-01-01-00.npy", arr, {"dtype": "float16"})
    back = al.load_latent(tmp_path / "2020-01-01-00.npy", meta)
    assert back.dtype == np.float32 and back.shape == arr.shape
    assert np.abs(back - arr).max() < 5e-3
    assert not (tmp_path / "2020-01-01-00.npy.tmp").exists()
