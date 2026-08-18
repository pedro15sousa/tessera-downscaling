"""Location of the data root and helpers to resolve paths under it.

Everything the project reads or writes outside the repository -- raw staging
data, processed descriptors and latents, the training/evaluation datasets, the
per-experiment training runs and the figure inputs -- lives under a single
directory, the *data root*. It is taken from the ``TESSERA_DATA_ROOT``
environment variable and defaults to ``/data/weather-downscaling``. See
``DATA.md`` for the layout beneath it.

The paper's runs were produced on the Isambard HPC, where the same tree lived
under a project-local ``.tmp_output`` directory. Those absolute prefixes are
recorded inside every ``config.json`` and checkpoint; :func:`resolve` rewrites
them onto the current data root so stored runs stay usable after the move.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "TESSERA_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("/data/weather-downscaling")

# Prefixes under which the data tree lived on Isambard (both spellings of the
# same filesystem were used). Order matters only for readability.
LEGACY_ROOT_PREFIXES: tuple[str, ...] = (
    "/projects/u6do/pmms2/end-to-end-forecasting/projects/tessera_downscaling/.tmp_output",
    "/lus/lfs1aip2/projects/u6do/pmms2/end-to-end-forecasting/projects/tessera_downscaling/.tmp_output",
)

# Canonical dataset used by every experiment in the paper.
DEFAULT_DATASET_NAME = "dataset_timestamp_global"


def data_root() -> Path:
    """Return the data root (``$TESSERA_DATA_ROOT`` or the default)."""
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_ROOT)).expanduser()


def resolve(path: str | os.PathLike[str]) -> Path:
    """Map ``path`` onto the current data root.

    Absolute paths that start with one of the legacy Isambard prefixes are
    rewritten; every other absolute path is returned unchanged; relative paths
    are interpreted relative to the data root.
    """
    text = os.fspath(path)
    for prefix in LEGACY_ROOT_PREFIXES:
        if text == prefix or text.startswith(prefix + "/"):
            return data_root() / text[len(prefix) :].lstrip("/")
    p = Path(text).expanduser()
    return p if p.is_absolute() else data_root() / p


def dataset_dir(name: str = DEFAULT_DATASET_NAME) -> Path:
    """``<root>/<name>`` -- a dataset built by ``scripts/preprocessing``."""
    return data_root() / name


def training_runs_dir(experiment_folder: str) -> Path:
    """``<root>/training_runs_<experiment_folder>`` -- runs of one experiment folder."""
    return data_root() / f"training_runs_{experiment_folder}"


def processed_dir(*parts: str) -> Path:
    """``<root>/processed/...`` -- station descriptors, latents, dense grids, caches."""
    return data_root().joinpath("processed", *parts)


def staging_dir(*parts: str) -> Path:
    """``<root>/_staging/...`` -- raw and intermediate ERA5 / GHCNh / Aurora files."""
    return data_root().joinpath("_staging", *parts)


def paper_figure_inputs_dir() -> Path:
    """``<root>/paper_figure_outputs/maps_outputs`` -- cached inputs of the map figures."""
    return data_root() / "paper_figure_outputs" / "maps_outputs"
