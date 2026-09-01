"""``paths.resolve``: legacy-prefix rewrite and layout relocations.

Stored configs and checkpoints record absolute paths from the machine and
data-root layout a run was trained under. ``resolve`` must map (a) the legacy
HPC prefixes onto the current data root, (b) pre-reorganisation layouts onto
the current one, and (c) leave current-layout and unrelated paths alone.
No data root is needed -- these are pure path computations.
"""

from __future__ import annotations

from pathlib import Path

from tessera_downscaling.paths import LEGACY_ROOT_PREFIXES, data_root, resolve

LEGACY = LEGACY_ROOT_PREFIXES[0]


def test_legacy_prefix_with_old_training_runs_layout():
    old = f"{LEGACY}/training_runs_snapshot_14y_eu/run/best_model.pt"
    assert resolve(old) == data_root() / "training_runs/snapshot_14y_eu/run/best_model.pt"


def test_legacy_prefix_with_old_dataset_and_vector_layout():
    assert resolve(f"{LEGACY}/dataset_timestamp_global") == (
        data_root() / "datasets/dataset_timestamp_global"
    )
    assert resolve(f"{LEGACY}/processed/extra_descriptors.npy") == (
        data_root() / "processed/station_vectors/extra_descriptors.npy"
    )
    # Unmoved processed/ subdirectories match no relocation rule.
    assert resolve(f"{LEGACY}/processed/vae_tessera_1B-M/x.npy") == (
        data_root() / "processed/vae_tessera_1B-M/x.npy"
    )


def test_relative_old_layout_relocates():
    assert resolve("dataset_timestamp_aurora_lead6h") == (
        data_root() / "datasets/dataset_timestamp_aurora_lead6h"
    )
    assert resolve("processed/station_latents_lat16_grad0.5.npy") == (
        data_root() / "processed/station_vectors/station_latents_lat16_grad0.5.npy"
    )
    assert resolve("_staging/processed") == data_root() / "ingest/processed"


def test_current_layout_passes_through():
    assert resolve("training_runs/snapshot_14y_eu") == (
        data_root() / "training_runs/snapshot_14y_eu"
    )
    assert resolve("datasets/dataset_timestamp_global") == (
        data_root() / "datasets/dataset_timestamp_global"
    )
    assert resolve("ingest/raw/ghcnh") == data_root() / "ingest/raw/ghcnh"


def test_absolute_under_current_root_relocates_old_layout():
    old_abs = data_root() / "training_runs_snapshot_14y_eu/run"
    assert resolve(old_abs) == data_root() / "training_runs/snapshot_14y_eu/run"


def test_unrelated_absolute_path_unchanged():
    assert resolve("/somewhere/else/file.npy") == Path("/somewhere/else/file.npy")
