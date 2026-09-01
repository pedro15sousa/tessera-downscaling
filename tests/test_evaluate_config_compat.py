"""``tessera-evaluate`` must interpret every checkpoint family on the data root.

:func:`tessera_downscaling.evaluate.build_model_from_config` is the one place
a stored training config is turned back into a model. This test feeds it the
``config`` dict of one checkpoint per family that exists on the data root --
the three paper runs of ``test_checkpoint_compat.py`` (TESSERA t2m, TESSERA
wind / truncated normal, ERA5-only baseline), a Norway station-rollout run
(``--region-specs-train-file`` + ``--probe-active-from-file`` +
``--train-end-override``) and a lead-conditioned cross-lead run
(``--lead-datasets`` + ``--drop-context-channels``) -- and checks that the
rebuilt model loads the stored weights with ``strict=True``.

The input width is derived from the stored first-conv weight, so the test
does not need the datasets; the dataset-side channel arithmetic is covered by
``tests/test_data_helpers.py``. Skipped when the data root is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tessera_downscaling.evaluate import (
    build_model_from_config,
    load_state_dict_compat,
    precomputed_vector_files,
    resolve_sidecar_path,
)
from tessera_downscaling.paths import data_root

pytestmark = pytest.mark.skipif(
    not data_root().exists(), reason=f"data root {data_root()} not present"
)

_RUNS = {
    "tessera_t2m": (
        "training_runs/snapshot_14y_eu_tessera_1B-M_2017/"
        "t2m_snap_vae_crop64_lat16_auxon_concat_mtpi_seed42"
    ),
    "tessera_wind_truncnormal": (
        "training_runs/snapshot_14y_eu_tessera_1B-M_2017/"
        "wind_truncnormal_snap_vae_crop64_lat16_auxon_concat_mtpi_seed42"
    ),
    "era5_baseline_t2m": (
        "training_runs/snapshot_14y_eu/t2m_snap_bilinear_baseline_mtpi_wd_seed42"
    ),
    "norway_rollout": (
        "training_runs/snapshot_14y_eu_temporal_rollout_norway_tessera_1B-M_2017/"
        "t2m_snap_vae_lat16_concat_with_elev_mtpi_no_static_wd_r1y_seed42"
    ),
    "cross_lead": (
        "training_runs/snapshot_14y_cross_lead_tessera_1B-M_2017/europe/"
        "t2m_xlead_snap_vae_lat16_concat_with_elev_no_static_wd_seed42"
    ),
}


def _load(run: str) -> dict:
    ckpt_path = data_root() / run / "best_model.pt"
    if not ckpt_path.exists():
        pytest.skip(f"checkpoint not present: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize("run", _RUNS.values(), ids=list(_RUNS))
def test_checkpoint_family_rebuilds_and_loads_strict(run: str) -> None:
    ckpt = _load(run)
    config = ckpt["config"]
    state_dict = ckpt["model_state_dict"]

    vectors_path, station_csv = precomputed_vector_files(config)
    if vectors_path is not None and not vectors_path.exists():
        pytest.skip(f"per-station vector file not present: {vectors_path}")

    n_context_channels = state_dict["cnn.net.0.weight"].shape[1]
    model = build_model_from_config(config, n_context_channels)
    assert set(model.state_dict()) == set(state_dict)
    load_state_dict_compat(model, state_dict)  # strict=True inside

    # The stored config describes the model completely: every paper run is
    # bilinear, and the per-station vector is present exactly when the config
    # names one.
    assert model.interpolation_method == config["interpolation"] == "bilinear"
    assert model.tessera_features_precomputed == (vectors_path is not None)
    if vectors_path is not None:
        assert station_csv is not None
        assert station_csv.exists()
    assert model.target_variables == list(config["target_variables"])


def test_cross_lead_config_carries_the_lead_channel_and_precip_drop() -> None:
    """The cross-lead run's config embeds the precip drop (no post-hoc patching)."""
    ckpt = _load(_RUNS["cross_lead"])
    config = ckpt["config"]
    assert config["drop_context_channels"] == ["total_precipitation_sum"]
    assert config["lead_datasets"]
    # 19 ERA5 channels (20 minus precip) + lat/lon + 4 time channels + lead.
    assert ckpt["model_state_dict"]["cnn.net.0.weight"].shape[1] == 19 + 2 + 4 + 1


def test_rollout_config_records_the_experiment_sidecars() -> None:
    """Probe activation map and train-end override survive in the stored config."""
    config = _load(_RUNS["norway_rollout"])["config"]
    assert config["train_end_override"]
    probe_file = Path(str(config["probe_active_from_file"]))
    assert probe_file.name.startswith("probe_active_from_")
    assert config["region_specs_train_file"]


def test_rollout_sidecars_fall_back_to_the_repo_copies() -> None:
    """A stored sidecar path from another machine resolves to this repo's copy.

    The rollout configs record ``probe_active_from_file`` under the checkout
    the runs were launched from. When that path does not exist here,
    :func:`resolve_sidecar_path` must find the committed
    ``scripts/experiments/<folder>/<file>`` instead — otherwise re-evaluating
    a stored rollout checkpoint silently collapses the probe / always_on
    station split into ``train_stations``.
    """
    config = _load(_RUNS["norway_rollout"])["config"]
    resolved = resolve_sidecar_path(config, "probe_active_from_file")
    assert resolved is not None and resolved.exists(), (
        f"probe_active_from_file did not resolve: {config['probe_active_from_file']}"
    )
    assert resolved.name.startswith("probe_active_from_")

    # A path that exists is returned untouched (a fresh run's own sidecar).
    own = {"probe_active_from_file": str(resolved)}
    assert resolve_sidecar_path(own, "probe_active_from_file") == resolved
