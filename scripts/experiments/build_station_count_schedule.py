"""Build a STATION-COUNT (step) schedule for the placement experiment.

Companion to ``build_rollout_schedule.py``. Where that script simulates a
realistic *temporal* rollout (probes come online linearly over T_rollout, and
``train_end_override`` is backed off so early sweeps see less history for
everyone), this one answers the decoupled **placement** question:

    "Given a budget of k FULLY-observed stations, where should they go?"

So there is NO temporal axis: at budget k the first k stations of the
deployment order are active from ``train_start`` (their full record, the same
window every always-on European station gets — deployed "now"), the rest are
the future-sentinel, and ``train_end_override`` is the dataset's real
``train_end`` at *every* budget (no back-off, no maturation). One model per
(k, strategy); the only variable is which k stations are in.

This isolates WHERE-you-place from HOW-LONG-a-station-has-been-live (which the
temporal rollout deliberately confounds), and samples the low-budget regime
(k << pool) where the deployed sets are near-disjoint across strategies — the
regime the temporal grid can't resolve because large-k subsets are forced to
overlap. The temporal rollout remains the realistic/deployable companion; this
is the controlled experiment (and the steady-state / network-rationalisation
question in its own right).

Output shape is byte-compatible with ``rollout_schedule.json`` (a
``schedule_metadata`` block + ``sweep_points`` mapping label -> {
train_end_override, probe_active_from}), so the placement ``submit.sh`` and its
per-sweep materialisation consume it unchanged. Sweep labels are ``k{NNN}``.

Refuses to overwrite an existing file (use ``--force``).

Example:
    uv run --project projects/tessera_downscaling python \\
        projects/tessera_downscaling/scripts/experiments/build_station_count_schedule.py \\
        --dataset-dir .tmp_output/dataset_timestamp_global \\
        --probe-ids-json .../snapshot_14y_eu_placement_norway/probe_station_ids.json \\
        --order-ids-json .../snapshot_14y_eu_placement_norway/orders/probe_order_kcenter_tessera.json \\
        --budgets 0,10,25,50,75,100,150,250 \\
        --out .../snapshot_14y_eu_placement_norway_stationcount/schedules/station_count_schedule_kcenter_tessera.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the temporal builder's date parsing + future-sentinel so the two
# schedule formats stay in exact sync (imported read-only; its __main__ guard
# means nothing runs on import). Same-dir import: this file's dir is sys.path[0]
# when run as a script.
from build_rollout_schedule import (   # noqa: E402
    FUTURE_SENTINEL,
    _fmt_ts,
    _load_dataset_window,
)


def _parse_budgets(spec: str, n: int) -> list[int]:
    """Parse "0,10,25,..." into a sorted, de-duplicated list of budgets in [0, n]."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            k = int(chunk)
        except ValueError:
            sys.exit(f"ERROR: budget {chunk!r} is not an integer.")
        if k < 0 or k > n:
            sys.exit(f"ERROR: budget {k} outside [0, {n}] (probe set size).")
        out.add(k)
    if not out:
        sys.exit("ERROR: --budgets yielded an empty list.")
    return sorted(out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Dataset dir with metadata.json (valid_timestamps, "
                        "temporal_split.train_end).")
    p.add_argument("--probe-ids-json", type=Path, required=True,
                   help="probe_station_ids.json (the candidate pool).")
    p.add_argument("--order-ids-json", type=Path, required=True,
                   help="A probe_order_*.json with 'ordered_station_ids' — the "
                        "deployment order; budget k deploys its first k.")
    p.add_argument("--budgets", type=str, default="0,10,25,50,75,100,150,250",
                   help="Comma-separated station-count budgets (k). Default: "
                        "0,10,25,50,75,100,150,250.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output schedule JSON path.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the output file if it exists.")
    return p.parse_args()


def _build(probe_ids: list[str], order: list[str], budgets: list[int],
           train_start_str: str, train_end_str: str, order_source: str) -> dict:
    if sorted(order) != sorted(probe_ids):
        sys.exit("ERROR: --order-ids-json is not a permutation of the probe set.")
    n = len(probe_ids)
    # Normalise both anchors through the shared formatter so the strings match
    # the temporal schedules exactly. active_from = train_start ⇒ full history.
    from build_rollout_schedule import _parse_ts
    start = _fmt_ts(_parse_ts(train_start_str))
    end = _fmt_ts(_parse_ts(train_end_str))

    sweep_out: dict[str, dict] = {}
    print("\nPer-budget schedule (full history, real train_end at every k):")
    for k in budgets:
        label = f"k{k:03d}"
        deployed = set(order[:k])
        paf = {sid: (start if sid in deployed else FUTURE_SENTINEL)
               for sid in probe_ids}
        sweep_out[label] = {
            "budget_k":           k,
            "n_online":           len(deployed),
            "train_end_override": end,
            "probe_active_from":  paf,
        }
        print(f"  {label}  online={k:>4}/{n}  active_from={start}  train_end={end}")

    return {
        "schedule_metadata": {
            "mode":                  "station_count",
            "cadence_shape":         "step",
            "activation_order_source": order_source,
            "n_stations":            n,
            "budgets":               budgets,
            "dataset_train_start":   train_start_str,
            "dataset_train_end":     train_end_str,
            "description": (
                f"Station-count (step) schedule over {n} probe stations. At "
                f"budget k the first k of the '{order_source}' order are active "
                f"from train_start (full history); train_end is the dataset's "
                f"real train_end at every budget (no temporal back-off)."
            ),
        },
        "sweep_points": sweep_out,
    }


def main() -> None:
    args = _parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"ERROR: {args.out} already exists. Pass --force to overwrite.")
    for pth in (args.probe_ids_json, args.order_ids_json):
        if not pth.exists():
            sys.exit(f"ERROR: input not found: {pth}")

    probe_ids = [str(s) for s in json.loads(args.probe_ids_json.read_text())["probe_station_ids"]]
    order_payload = json.loads(args.order_ids_json.read_text())
    order = [str(s) for s in order_payload["ordered_station_ids"]]
    order_source = order_payload.get("strategy", args.order_ids_json.stem)

    train_start_str, train_end_str, _, _ = _load_dataset_window(args.dataset_dir)
    budgets = _parse_budgets(args.budgets, len(probe_ids))

    print(f"Dataset train_start = {train_start_str}")
    print(f"Dataset train_end   = {train_end_str}  (used at EVERY budget — no back-off)")
    print(f"Probe set size      = {len(probe_ids)} (from {args.probe_ids_json.name})")
    print(f"Deployment order    = {order_source} ({args.order_ids_json.name})")
    print(f"Budgets             = {budgets}")

    schedule = _build(probe_ids, order, budgets, train_start_str, train_end_str, order_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(schedule, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
