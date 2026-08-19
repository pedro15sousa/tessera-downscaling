"""Build the ``rollout_schedule.json`` for a temporal-rollout experiment.

Takes an existing ``probe_station_ids.json`` (e.g. one produced by
``pick_probe_set.py``) and emits the full per-sweep schedule:

* A linear staggered ``probe_active_from`` map per sweep point
  (per-station activation timestamps spread evenly over ``T_rollout``).
* A uniform ``train_end_override`` per sweep point.
* Schedule metadata (``T_0``, rollout completion, dataset train_start/end,
  activation order seed, cadence shape).

Output shape matches
``snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/rollout_schedule.json``
byte-for-byte, modulo the input probe-set contents.

Anchoring rule
--------------

``T_0`` is chosen such that the *largest* sweep point lands exactly at
``dataset.train_end``:

    T_0 = train_end − max(sweep_months)

So at the smallest sweep point ``r1mo``, always-on stations still have
~(train_end − train_start − max_sweep_months) years of pre-rollout history,
and at the largest sweep point the rollout has been fully online for
``(max_sweep_months − T_rollout)`` extra months past completion.

The deployment order is a fixed numpy permutation of the probe set seeded
with ``--activation-seed`` — the same order is used across all model-init
seeds (the order is part of the *experiment design*, not noise).

Refuses to overwrite an existing file (use ``--force`` to override).

Example: build the Norway rollout schedule of the paper
------------------------------------------------------

    uv run python scripts/experiments/build_rollout_schedule.py \\
        --dataset-dir <data root>/dataset_timestamp_global \\
        --probe-ids-json scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/probe_station_ids.json \\
        --out scripts/experiments/snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi/rollout_schedule.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from dateutil.relativedelta import relativedelta
except ImportError as e:
    sys.exit(
        "ERROR: python-dateutil is required (relativedelta). Install it "
        f"or run via uv with the core group. Original error: {e}"
    )


# Default sweep grid — the paper's Norway rollout (10 points; r0 = the
# pre-rollout anchor with no probes deployed, r3y = rollout complete).
DEFAULT_SWEEP_POINTS = (
    "r0:0,r1mo:1,r3mo:3,r6mo:6,r1y:12,r2y:24,r3y:36,r4y:48,r5y:60,r6y:72"
)

# Future-sentinel string used to mark a probe as "not yet deployed at this
# sweep point". Lexicographically greater than every real timestamp in
# the dataset, so the dataset's probe-row filter excludes the station
# entirely for that sweep. Must match the value the dataset class expects.
FUTURE_SENTINEL = "9999-12-31-23"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_sweep_points(spec: str) -> dict[str, int]:
    """Parse "label:months,label:months,..." into a dict.

    Order is preserved (Python 3.7+ dicts are insertion-ordered).
    Months must be positive integers; labels must be unique.
    """
    out: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            sys.exit(
                f"ERROR: bad --sweep-points entry {chunk!r}; expected "
                f"'label:months' (e.g. 'r1y:12')."
            )
        label, months_str = chunk.split(":", 1)
        label = label.strip()
        if not label:
            sys.exit(f"ERROR: empty label in --sweep-points entry {chunk!r}.")
        if label in out:
            sys.exit(f"ERROR: duplicate sweep label {label!r}.")
        try:
            months = int(months_str)
        except ValueError:
            sys.exit(f"ERROR: months value in {chunk!r} is not an integer.")
        if months < 0:
            sys.exit(
                f"ERROR: sweep months in {chunk!r} must be >= 0 "
                f"(0 = pre-rollout anchor: no probes deployed yet)."
            )
        out[label] = months
    if not out:
        sys.exit("ERROR: --sweep-points yielded an empty dict.")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the dataset directory (must contain metadata.json with "
        "train_start in valid_timestamps and temporal_split.train_end).",
    )
    p.add_argument(
        "--probe-ids-json",
        type=Path,
        required=True,
        help="Path to the probe_station_ids.json produced by pick_probe_set.py.",
    )
    p.add_argument(
        "--t-rollout-months",
        type=int,
        default=36,
        help="Rollout duration in months; default 36 (the paper's value). "
        "Must be ≤ max(sweep months).",
    )
    p.add_argument(
        "--activation-seed",
        type=int,
        default=0,
        help="numpy.random.default_rng seed for the per-probe-station "
        "deployment order. Same value across model-init seeds; the "
        "order is part of the experiment design, not noise. Ignored when "
        "--order-ids-json is given.",
    )
    p.add_argument(
        "--order-ids-json",
        type=Path,
        default=None,
        help="Optional JSON with an 'ordered_station_ids' list giving the probe "
        "DEPLOYMENT ORDER, earliest-deployed first. When set, it REPLACES "
        "the random --activation-seed permutation; the ids must be a "
        "permutation of the probe set. (The paper uses the random order.)",
    )
    p.add_argument(
        "--sweep-points",
        type=str,
        default=DEFAULT_SWEEP_POINTS,
        help=f"Comma-separated 'label:months' pairs. Default (the paper's "
        f"Norway grid): {DEFAULT_SWEEP_POINTS}",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for rollout_schedule.json.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it exists. Default: refuse.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    """Parse 'YYYY-MM-DD' (day-end ⇒ hour 23) or 'YYYY-MM-DD-HH'.

    Day-only strings interpreted as "end of that day", matching the
    temporal-split semantics where the whole train_end day is in train.
    """
    s = s.strip()
    if len(s) <= 10:
        s = s + "-23"
    return datetime.strptime(s, "%Y-%m-%d-%H")


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d-%H")


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------


def _load_dataset_window(dataset_dir: Path) -> tuple[str, str, datetime, datetime]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        sys.exit(f"ERROR: dataset metadata.json not found at {metadata_path}.")
    meta = json.loads(metadata_path.read_text())
    try:
        train_start_str = meta["valid_timestamps"][0]
        train_end_str = meta["temporal_split"]["train_end"]
    except (KeyError, IndexError) as e:
        sys.exit(
            f"ERROR: {metadata_path} is missing expected fields "
            f"(valid_timestamps[0] / temporal_split.train_end): {e}"
        )
    return (
        train_start_str,
        train_end_str,
        _parse_ts(train_start_str),
        _parse_ts(train_end_str),
    )


def _build_schedule(
    probe_ids: list[str],
    sweep_points: dict[str, int],
    t_rollout_months: int,
    activation_seed: int,
    train_start_str: str,
    train_end_str: str,
    train_start: datetime,
    train_end: datetime,
    deployment_order: list[str] | None = None,
    order_source: str | None = None,
) -> dict:
    if not probe_ids:
        sys.exit("ERROR: probe_station_ids list is empty.")
    n = len(probe_ids)
    max_sweep_months = max(sweep_points.values())
    if t_rollout_months > max_sweep_months:
        sys.exit(
            f"ERROR: --t-rollout-months={t_rollout_months} exceeds max "
            f"sweep months={max_sweep_months}. Either shorten T_rollout or "
            f"add a larger sweep point."
        )

    # Anchor T_0 so the largest sweep ends at train_end.
    t_0 = train_end - relativedelta(months=max_sweep_months)
    rollout_completion = t_0 + relativedelta(months=t_rollout_months)
    if t_0 < train_start:
        sys.exit(
            f"ERROR: anchored T_0 = {_fmt_ts(t_0)} precedes "
            f"train_start = {_fmt_ts(train_start)}. The largest sweep "
            f"({max_sweep_months} months) is wider than the training "
            f"window; widen the dataset or shrink the sweep grid."
        )

    # Linear cadence: probe i opens at T_0 + (i / N) * T_rollout, on the
    # ORDERED list (deployment order). Use timedelta for the float math —
    # relativedelta doesn't support multiplication by a float.
    #
    # Deployment order is EITHER a caller-supplied permutation (coverage-driven
    # placement) OR a seeded random permutation (the baseline rollout). Earlier
    # in the list == deployed earlier == more accumulated history at each sweep.
    if deployment_order is not None:
        if sorted(deployment_order) != sorted(probe_ids):
            sys.exit(
                "ERROR: --order-ids-json is not a permutation of the probe "
                "set — id sets differ (same length required, same ids)."
            )
        deployment = list(deployment_order)
    else:
        rng = np.random.default_rng(activation_seed)
        deployment = [probe_ids[i] for i in rng.permutation(n)]
    t_rollout = rollout_completion - t_0
    t_open = {deployment[i]: t_0 + (i / n) * t_rollout for i in range(n)}

    # Per-sweep maps. Iterate sweep_points in the user's input order.
    sweep_out: dict[str, dict] = {}
    print()
    print("Per-sweep schedule:")
    for label, months in sweep_points.items():
        snapshot_time = t_0 + relativedelta(months=months)
        # months == 0 is the pre-rollout ANCHOR: snapshot_time == T_0 and, by
        # convention, NO probe is deployed yet (all future-sentinel), giving a
        # European-only baseline at elapsed year 0. (The i=0 probe opens exactly
        # at T_0, so a naive `<=` would count it online at the anchor — the
        # `months > 0` guard keeps the anchor at a clean zero deployed.)
        paf = {
            sid: (
                _fmt_ts(t_open[sid])
                if months > 0 and t_open[sid] <= snapshot_time
                else FUTURE_SENTINEL
            )
            for sid in probe_ids
        }
        n_online = sum(1 for v in paf.values() if v != FUTURE_SENTINEL)
        sweep_out[label] = {
            "elapsed_months": months,
            "train_end_override": _fmt_ts(snapshot_time),
            "probe_active_from": paf,
        }
        print(
            f"  {label:6s} elapsed={months:>3} mo  "
            f"train_end_override={_fmt_ts(snapshot_time)}  "
            f"online={n_online}/{n} ({100 * n_online / n:.1f}%)"
        )

    if deployment_order is not None:
        order_desc = (
            f"deployment order supplied by {order_source or 'order-ids-json'} "
            f"(coverage-driven placement); same order across all model-init seeds."
        )
    else:
        order_desc = (
            f"fixed permutation seeded with {activation_seed}; same order "
            f"across all model-init seeds."
        )
    return {
        "schedule_metadata": {
            "t_rollout_months": t_rollout_months,
            "rollout_anchor_t_0": _fmt_ts(t_0),
            "rollout_completion_t": _fmt_ts(rollout_completion),
            "cadence_shape": "linear",
            "activation_order_seed": None
            if deployment_order is not None
            else activation_seed,
            "activation_order_source": order_source
            if deployment_order is not None
            else f"random_seed_{activation_seed}",
            "n_stations": n,
            "dataset_train_start": train_start_str,
            "dataset_train_end": train_end_str,
            "description": (
                f"Linear rollout of {n} probe stations over "
                f"{t_rollout_months} months. Station {order_desc}"
            ),
        },
        "sweep_points": sweep_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    out_path: Path = args.out
    if out_path.exists() and not args.force:
        sys.exit(f"ERROR: {out_path} already exists. Pass --force to overwrite.")

    if not args.probe_ids_json.exists():
        sys.exit(f"ERROR: --probe-ids-json not found at {args.probe_ids_json}.")
    probe_payload = json.loads(args.probe_ids_json.read_text())
    probe_ids = probe_payload.get("probe_station_ids")
    if not isinstance(probe_ids, list) or not probe_ids:
        sys.exit(
            f"ERROR: {args.probe_ids_json} has no 'probe_station_ids' list "
            f"or it's empty."
        )
    probe_ids = [str(s) for s in probe_ids]

    deployment_order = None
    order_source = None
    if args.order_ids_json is not None:
        if not args.order_ids_json.exists():
            sys.exit(f"ERROR: --order-ids-json not found at {args.order_ids_json}.")
        order_payload = json.loads(args.order_ids_json.read_text())
        deployment_order = order_payload.get("ordered_station_ids")
        if not isinstance(deployment_order, list) or not deployment_order:
            sys.exit(
                f"ERROR: {args.order_ids_json} has no 'ordered_station_ids' "
                f"list or it's empty."
            )
        deployment_order = [str(s) for s in deployment_order]
        order_source = order_payload.get("strategy", args.order_ids_json.stem)

    sweep_points = _parse_sweep_points(args.sweep_points)

    train_start_str, train_end_str, train_start, train_end = _load_dataset_window(
        args.dataset_dir
    )
    print(f"Dataset train_start = {train_start_str}")
    print(f"Dataset train_end   = {train_end_str}")
    print(f"Probe set size      = {len(probe_ids)} (from {args.probe_ids_json.name})")
    print(f"T_rollout (months)  = {args.t_rollout_months}")
    if deployment_order is not None:
        print(
            f"Activation order    = {order_source} "
            f"({args.order_ids_json.name}, {len(deployment_order)} ids)"
        )
    else:
        print(f"Activation seed     = {args.activation_seed}")
    print(f"Sweep grid          = {sweep_points}")

    schedule = _build_schedule(
        probe_ids=probe_ids,
        sweep_points=sweep_points,
        t_rollout_months=args.t_rollout_months,
        activation_seed=args.activation_seed,
        train_start_str=train_start_str,
        train_end_str=train_end_str,
        train_start=train_start,
        train_end=train_end,
        deployment_order=deployment_order,
        order_source=order_source,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schedule, indent=2))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
