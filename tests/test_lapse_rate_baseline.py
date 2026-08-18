"""Tests for the lapse-rate elevation correction in the simple baselines.

The sign convention here is the one thing in this baseline that can be wrong
*silently*: flipping it still produces finite, plausible-looking temperatures,
just systematically worse ones. These tests pin it down against the definition
of ``delta_elevation`` used upstream in
``scripts/preprocessing/helpers.py:compute_delta_elevation``, namely

    delta_elevation = station_elevation - ERA5_orography

so a station *above* its ERA5 cell must be *cooled*.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# The baselines live under scripts/, which is not an importable package, so
# load the module by path rather than restructuring the tree for one test.
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "baselines" / "evaluate_simple_baselines.py"
)
_spec = importlib.util.spec_from_file_location("_simple_baselines", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_simple_baselines"] = _mod
_spec.loader.exec_module(_mod)

apply_lapse_rate_correction = _mod.apply_lapse_rate_correction
DEFAULT_LAPSE_RATE = _mod.DEFAULT_LAPSE_RATE_K_PER_M


def test_station_above_era5_cell_is_cooled():
    """Δelev > 0 (station higher than the grid cell) must lower temperature."""
    interp = np.array([10.0], dtype=np.float32)
    delta_elev = np.array([1000.0], dtype=np.float32)  # 1 km above the cell

    out = apply_lapse_rate_correction(interp, delta_elev, DEFAULT_LAPSE_RATE)

    # 6.5 K/km over 1 km => 10 - 6.5 = 3.5 °C
    assert out[0] == pytest.approx(3.5, abs=1e-5)
    assert out[0] < interp[0], "a station above its ERA5 cell must be cooled"


def test_station_below_era5_cell_is_warmed():
    """Δelev < 0 (station lower than the grid cell) must raise temperature."""
    interp = np.array([10.0], dtype=np.float32)
    delta_elev = np.array([-500.0], dtype=np.float32)

    out = apply_lapse_rate_correction(interp, delta_elev, DEFAULT_LAPSE_RATE)

    assert out[0] == pytest.approx(13.25, abs=1e-5)
    assert out[0] > interp[0], "a station below its ERA5 cell must be warmed"


def test_zero_delta_elev_is_identity():
    """A station at exactly the grid-cell height is left untouched."""
    interp = np.array([-3.0, 0.0, 21.5], dtype=np.float32)
    delta_elev = np.zeros(3, dtype=np.float32)

    out = apply_lapse_rate_correction(interp, delta_elev, DEFAULT_LAPSE_RATE)

    np.testing.assert_allclose(out, interp, atol=1e-6)


def test_correction_is_linear_in_delta_elev():
    """Doubling Δelev doubles the correction (constant-lapse-rate assumption)."""
    interp = np.zeros(2, dtype=np.float32)
    delta_elev = np.array([250.0, 500.0], dtype=np.float32)

    out = apply_lapse_rate_correction(interp, delta_elev, DEFAULT_LAPSE_RATE)

    assert out[1] == pytest.approx(2.0 * out[0], rel=1e-5)


def test_non_finite_delta_elev_leaves_prediction_uncorrected():
    """NaN Δelev must pass the interpolated value through, not emit NaN.

    Emitting NaN would drop the station from the metrics and silently change
    the station set relative to the other rows of the same table.
    """
    interp = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    delta_elev = np.array([np.nan, 1000.0, np.inf], dtype=np.float32)

    out = apply_lapse_rate_correction(interp, delta_elev, DEFAULT_LAPSE_RATE)

    assert np.isfinite(out).all(), "no prediction may become non-finite"
    assert out[0] == pytest.approx(5.0)   # NaN Δelev -> untouched
    assert out[1] == pytest.approx(0.5)   # 7.0 - 6.5
    assert out[2] == pytest.approx(9.0)   # inf Δelev -> untouched


def test_dtype_is_float32():
    """Predictions feed a float32 npz alongside the other baselines."""
    out = apply_lapse_rate_correction(
        np.array([1.0]), np.array([100.0]), DEFAULT_LAPSE_RATE,
    )
    assert out.dtype == np.float32


def test_default_lapse_rate_is_per_metre_not_per_km():
    """Guards against a 1000x unit slip in the default constant."""
    assert DEFAULT_LAPSE_RATE == pytest.approx(0.0065)


def test_recovers_a_known_lapse_rate_from_synthetic_data():
    """End-to-end sanity: if obs are generated with Γ, the correction undoes it.

    Mirrors what the fitted-Γ estimator assumes — that the signed
    interpolation error (interp − obs) is Γ·Δelev — and confirms the
    correction inverts exactly that relationship.
    """
    rng = np.random.default_rng(0)
    gamma = 0.0045
    delta_elev = rng.uniform(-800, 1600, size=5000).astype(np.float32)
    truth = rng.uniform(-20, 35, size=5000).astype(np.float32)
    # Construct the interpolated field that the lapse relationship implies.
    interp = (truth + gamma * delta_elev).astype(np.float32)

    out = apply_lapse_rate_correction(interp, delta_elev, gamma)

    np.testing.assert_allclose(out, truth, atol=1e-3)

    # And the slope the fitter would estimate is Γ itself.
    slope = np.linalg.lstsq(
        delta_elev[:, None].astype(np.float64),
        (interp - truth).astype(np.float64),
        rcond=None,
    )[0][0]
    assert slope == pytest.approx(gamma, rel=1e-6)
