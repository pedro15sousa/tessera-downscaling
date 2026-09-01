"""Location of the data root and helpers to resolve paths under it.

Everything the project reads or writes outside the repository -- raw staging
data, processed descriptors and latents, the training/evaluation datasets, the
per-experiment training runs and the figure inputs -- lives under a single
directory, the *data root*. It is taken from the ``TESSERA_DATA_ROOT``
environment variable and defaults to ``/data/weather-downscaling``. See
``DATA.md`` for the layout beneath it.

The paper's runs were produced on an HPC where the same tree lived under a
project-local ``.tmp_output`` directory, in an older layout. Those absolute
prefixes are recorded inside every ``config.json`` and checkpoint;
:func:`resolve` rewrites them onto the current data root and maps the old
layout onto the current one (:data:`RELOCATIONS`), so stored runs stay usable.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

ENV_VAR = "TESSERA_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("/data/weather-downscaling")

# Prefixes under which the data tree lived on the training HPC (both spellings
# of the same filesystem were used). Order matters only for readability.
LEGACY_ROOT_PREFIXES: tuple[str, ...] = (
    "/projects/u6do/pmms2/end-to-end-forecasting/projects/tessera_downscaling/.tmp_output",
    "/lus/lfs1aip2/projects/u6do/pmms2/end-to-end-forecasting/projects/tessera_downscaling/.tmp_output",
)

# Root-relative paths that moved when the data root was reorganised.
# Applied by :func:`resolve` after the legacy-prefix
# rewrite (and to absolute paths under the current root), so stored configs
# keep resolving. First match wins; current-layout paths match no rule.
RELOCATIONS: tuple[tuple[str, str], ...] = (
    ("training_runs_", "training_runs/"),
    ("dataset_timestamp", "datasets/dataset_timestamp"),
    ("_staging/", "ingest/"),
    ("processed/station_latents_", "processed/station_vectors/station_latents_"),
    (
        "processed/station_summary_stats_",
        "processed/station_vectors/station_summary_stats_",
    ),
    ("processed/extra_descriptors", "processed/station_vectors/extra_descriptors"),
    (
        "processed/station_extra_descriptors.csv",
        "processed/station_vectors/station_extra_descriptors.csv",
    ),
    ("processed/station_mtpi.csv", "processed/station_vectors/station_mtpi.csv"),
)

# Canonical dataset used by every experiment in the paper.
DEFAULT_DATASET_NAME = "dataset_timestamp_global"


def data_root() -> Path:
    """Return the data root (``$TESSERA_DATA_ROOT`` or the default)."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_ROOT)).expanduser()


def _relocate(rel: str) -> str:
    """Map a root-relative path of the pre-reorganisation layout onto the
    current one. Paths already in the current layout match no rule and pass
    through unchanged."""
    posix = PurePosixPath(rel).as_posix()
    for old, new in RELOCATIONS:
        if posix.startswith(old):
            return new + posix[len(old) :]
    return posix


def resolve(path: str | os.PathLike[str]) -> Path:
    """Map ``path`` onto the current data root and layout.

    Absolute paths that start with one of the legacy HPC prefixes are
    rewritten onto the data root; absolute paths under the current data root
    and relative paths are kept root-relative. In all three cases the
    root-relative part is passed through :data:`RELOCATIONS` so paths stored
    before the data-root reorganisation still land on the right files. Any
    other absolute path is returned unchanged.
    """
    text = os.fspath(path)
    for prefix in LEGACY_ROOT_PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            return data_root() / _relocate(text[len(prefix) :].lstrip("/"))
    p = Path(text).expanduser()
    if not p.is_absolute():
        return data_root() / _relocate(text)
    root = data_root()
    if p == root or root in p.parents:
        return root / _relocate(p.relative_to(root).as_posix())
    return p


def dataset_dir(name: str = DEFAULT_DATASET_NAME) -> Path:
    """``<root>/datasets/<name>`` -- a dataset built by ``scripts/preprocessing``."""
    return data_root() / "datasets" / name


def training_runs_dir(experiment_folder: str) -> Path:
    """``<root>/training_runs/<experiment_folder>`` -- runs of one experiment folder."""
    return data_root() / "training_runs" / experiment_folder


def processed_dir(*parts: str) -> Path:
    """``<root>/processed/...`` -- station descriptors, latents, dense grids, caches."""
    return data_root().joinpath("processed", *parts)


def station_vectors_dir(*parts: str) -> Path:
    """``<root>/processed/station_vectors/...`` -- loose per-station vector files."""
    return data_root().joinpath("processed", "station_vectors", *parts)


def ingest_dir(*parts: str) -> Path:
    """``<root>/ingest/...`` -- downloaded raw and intermediate ERA5 / GHCNh / Aurora files."""
    return data_root().joinpath("ingest", *parts)


def patch_encoder_dir(*parts: str) -> Path:
    """``<root>/tessera_patch_encoder/...`` -- patch-encoder runs and caches."""
    return data_root().joinpath("tessera_patch_encoder", *parts)


def paper_figure_inputs_dir() -> Path:
    """``<root>/paper_figure_outputs/maps_outputs`` -- cached inputs of the map figures."""
    return data_root() / "paper_figure_outputs" / "maps_outputs"
