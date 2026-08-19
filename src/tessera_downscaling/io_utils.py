"""Small I/O helpers shared by the ERA5 / GHCNh ingestion scripts.

Vendored from the ``dataprocessing`` package of the ``end-to-end-forecasting``
monorepo (``src/dataprocessing/utils.py``); the logic is unchanged.

Three things live here:

* :func:`atomic` / :func:`atomic_completed` -- write a file in a way that makes
  a partial or failed write detectable: a ``<path>-problem.txt`` sidecar exists
  while ``f`` runs and is removed only on success. Every downloader checks
  :func:`atomic_completed` before redoing work, which is what makes the
  ingestion scripts resume-safe.
* :func:`compute_file_name` -- the ``YYYY-MM-DD-HH.nc`` naming convention of the
  raw staging tree (``_staging/processed/era5_wb2_quarter_<var>/data/`` and
  ``_staging/processed/ghcnh/data/``). Every preprocessor assumes it.
* :func:`parallel_foreach` -- a queue-based multiprocessing map with a per-worker
  ``init`` (used to open one zarr store / GCS client per process).

It also fixes the ERA5 variable set that the downscaling datasets are built
from: the 5 surface + 5 atmospheric variables below (the atmospheric ones at
the 3 pressure levels 500/700/850 hPa in the paper's staging; the 13
WeatherBench2 levels only for the Aurora initial-condition staging).
"""

import datetime as dt
import faulthandler
import io
import os
import time
import traceback
from collections.abc import Callable, Iterable
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import xarray as xr

# ERA5 pressure-level variables (WeatherBench2 names).
ATMOS_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
]

# ERA5 single-level variables (WeatherBench2 names).
SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
]

# The 3 pressure levels the downscaling datasets use (Vaughan et al. 2022).
PAPER_LEVELS = [500, 700, 850]

# All 13 WeatherBench2 pressure levels (needed only as Aurora initial conditions).
WB_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def parallel_foreach(
    f: Callable[..., None], init: Callable, items: Iterable, num_processes: int
) -> None:
    """Apply `f` to each element in `item`s using `num_processes` processes.

    Creates `num_processes` new processes, and calls `init` on each of them. `items`
    are placed in a queue, and each process pulls elements from it. For each `item`,
    `f(item, init_output)` is called, where `init_output = init(rank)`.
    """
    # Push all of the items onto a queue.
    queue: Queue = Queue()
    for item in items:
        queue.put(item)

    # Push `None`s onto the end of the queue to signify when we're done.
    for _ in range(num_processes):
        queue.put(None)

    # Set off all of the workers.
    procs = []
    for n in range(num_processes):
        p = Process(target=_worker, args=(queue, f, init, n))
        p.start()
        procs.append(p)

    # Block until all workers have completed.
    while True:
        time.sleep(0.1)
        codes = [p.exitcode for p in procs]
        if all(c == 0 for c in codes):
            return
        for j, c in enumerate(codes):
            if c not in (0, None):
                message = f"Process {j} terminated with exit code {c}."
                raise RuntimeError(message)


# Used by `parallel_foreach` -- don't call directly.
def _worker(queue: Queue, f: Callable[..., None], init: Callable, rank: int) -> None:
    # Ensure that we see tracebacks.
    faulthandler.enable()

    # Initialise this worker.
    init_output = init(rank)

    while True:
        # Pop an item off the queue. If there's nothing left to process, exit.
        x = queue.get(block=True, timeout=10)
        if x is None:
            return

        # Operate on the item.
        f(x, init_output)


def compute_problem_path(output_path: Path) -> Path:
    """Return the location of the problem file associated to `output_path`."""
    return Path(f"{output_path!s}-problem.txt")


def atomic_completed(output_path: Path) -> bool:
    """Return `True` if atomic to create `output_path` succeeded, else `False`."""
    return output_path.exists() and (not compute_problem_path(output_path).exists())


def atomic(
    f: Callable[..., None], output_path: Path, *args: Any, **kwargs: Any
) -> None:
    """Execute `f` in a manner that makes it obvious if it fails.

    `f` must be a function `f(output_path, *args) -> None`. It is expected that it
    will have the side-effect of either writing a file `output_path`, or doing nothing.

    This function assumes that `f` may not complete successfully (this could happen
    for one of many reasons, including that I got bored and interupted the process!).
    When this happens, a file may not have been created at `output_path`, or a file may
    have been created but only partially written to.

    To handle this possibility, this function ensures that a file called
    `{output_path}-problem.txt` exists if this function did not complete successfully.
    We call this the "problem file". This problem file is a text file, and contains as
    much debugging information as is available (typically including a stack trace).

    In summary:
    - if the problem blob exists, something went wrong
    - if there is no problem blob, but the output blob exists, assume success
    - if there is nothing, something went wrong

    Args:
        f: callable to execute
        output_path: path to which `f` will write its output.
        args: additional positional arguments to pass to `f`, after `output_path`.
        kwargs: keyword arguments to pass to `f`.

    """
    # Construct blob clients for output file, processing file, and problem file.
    problem = compute_problem_path(output_path)

    try:
        # Ensure that we have a problem file while processing, so that we know if
        # something goes wrong half way through.
        with problem.open("w") as h:
            h.write("A placeholder.")

        # Do the work.
        f(output_path, *args, **kwargs)

        # Completed successfully, remove the problem file.
        problem.unlink()

    except Exception as e:  # noqa: BLE001
        print(f"Encountered exception {e} with traceback:")
        print(traceback.format_exc())
        print(f"Writing traceback to {problem}.")
        with problem.open("w") as h:
            h.write(traceback.format_exc())


def compute_file_name(t: dt.datetime, ext: str = "nc") -> str:
    """``YYYY-MM-DD-HH.<ext>`` -- the file-name convention of the staging tree."""
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}.{ext}"


def flush_and_sync(f: io.IOBase) -> None:
    """Ensure that `f` is fully written to disk.

    See https://docs.python.org/3/library/os.html#os.fsync .
    """
    f.flush()
    os.fsync(f.fileno())


def write_and_flush_dataset(path: Path, ds: xr.Dataset, **kwargs: Any) -> None:
    """Serialise ``ds`` to NetCDF and write it to ``path`` with an fsync."""
    data = ds.to_netcdf(path=None, **kwargs)
    with path.open("wb") as f:
        f.write(data)
        flush_and_sync(f)
