#!/bin/bash
set -euo pipefail

# =====================================================================
# Norway PLACEMENT experiment submitter.
#
# Loops the SAME architecture x sweep x seed grid as the temporal-rollout
# folder over NINE probe DEPLOYMENT ORDERINGS (schedules/rollout_schedule_
# <strategy>.json), one per active-learning-style selection strategy:
#     random
#     {kcenter,maxcov}_{geographic,elevation_mtpi,era5_static,tessera}
# Each strategy's schedule is materialised into _materialised/<strategy>/ and
# its runs land in <OUTPUT_ROOT_BASE>/<strategy>/. The deterministic ERA5-interp
# simple baselines are strategy-independent and run ONCE.
#
# SAFETY: defaults to DRY_RUN=1 (prints the sbatch commands, submits nothing).
#         Set DRY_RUN=0 to actually submit.
#
# Useful knobs (all env-overridable):
#   STRATEGIES="random kcenter_tessera"   # subset of strategies (default: all 9)
#   SEEDS="42"                            # subset of seeds       (default: 42 123 456)
#   DRY_RUN=0                             # actually submit
#   SKIP_UV_SYNC=1                        # skip the pre-submit uv sync
# =====================================================================

# ---- Paths ----
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset_timestamp_global}"

TESSERA_PATH="${TESSERA_PATH:-${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy}"
TESSERA_CSV="${TESSERA_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH:-${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy}"
export VAE_LATENTS_CSV="${VAE_LATENTS_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"

OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-${BASE_DIR}/training_runs_snapshot_14y_eu_placement_norway}"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"
SIMPLE_BASELINE_SCRIPT="projects/tessera_downscaling/scripts/baselines/evaluate_simple_baselines.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"
PROBE_IDS_JSON="${SCRIPT_DIR}/probe_station_ids.json"
SCHEDULES_DIR="${SCRIPT_DIR}/schedules"
MATERIALISED_DIR="${SCRIPT_DIR}/_materialised"

NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-}"

# ---- Hyperparameters (mirror the temporal-rollout folder) ----
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-42 123 456}"
BATCH_SIZE=1
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
LR_WARMUP_PCT="0.05"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

# ---- Run mode ----
DRY_RUN="${DRY_RUN:-1}"        # default: print only, submit nothing
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"

# ---- Strategy selection ----
# Default: every schedule on disk EXCEPT 'random'. The random baseline is
# byte-identical (same seed-0 permutation, same probe set, same architectures)
# to the already-trained temporal-rollout experiment
# (training_runs_snapshot_14y_eu_temporal_rollout_norway_lat16_mtpi), so the
# analysis reuses THAT tree instead of re-running 72 jobs. Force it back with
# STRATEGIES="random" if you ever want a self-contained copy.
if [ -n "${STRATEGIES:-}" ]; then
    IFS=' ' read -r -a STRATEGY_LIST <<< "${STRATEGIES}"
else
    STRATEGY_LIST=()
    for f in "${SCHEDULES_DIR}"/rollout_schedule_*.json; do
        b=$(basename "$f" .json)
        s="${b#rollout_schedule_}"
        [ "${s}" = "random" ] && continue   # reuse the temporal-rollout tree
        STRATEGY_LIST+=("${s}")
    done
fi
if [ "${#STRATEGY_LIST[@]}" -eq 0 ]; then
    echo "ERROR: no strategies found (schedules dir empty and STRATEGIES unset)." >&2
    exit 1
fi

# ---- Preflight ----
if [ ! -f "${DATASET_DIR}/metadata.json" ]; then
    echo "ERROR: ${DATASET_DIR}/metadata.json does not exist." >&2
    exit 1
fi
LAYOUT_VERSION=$(
    python3 -c "import json; print(json.load(open('${DATASET_DIR}/metadata.json')).get('layout_version', ''))"
)
if [ "${LAYOUT_VERSION}" != "multi_region_snapshot_v1" ]; then
    echo "ERROR: expected multi_region_snapshot_v1 dataset; got '${LAYOUT_VERSION}'." >&2
    exit 1
fi
if [ ! -f "${PROBE_IDS_JSON}" ]; then
    echo "ERROR: probe_station_ids.json does not exist at ${PROBE_IDS_JSON}." >&2
    exit 1
fi

if [ "${SKIP_UV_SYNC}" = "1" ]; then
    echo "SKIP_UV_SYNC=1, skipping uv sync."
else
    echo "Pre-syncing environment..."
    ( cd "${REPO_ROOT}" && uv sync --group core )
fi

# ---- Region-specs JSONs (shared across all strategies) ----
REGION_SPECS_TRAIN_JSON="${SCRIPT_DIR}/region_specs_train.json"
REGION_SPECS_TEST_JSON="${SCRIPT_DIR}/region_specs_test.json"
echo '{"europe":"train"}' > "${REGION_SPECS_TRAIN_JSON}"
echo '{"europe":"all"}'   > "${REGION_SPECS_TEST_JSON}"

# ---- Load architecture + sweep_point + simple-baseline lists from YAML ----
mapfile -t ARCH_ENTRIES < <(python3 <<PYEOF
import os, yaml
from pathlib import Path
spec = yaml.safe_load(Path("${EXPERIMENTS_YAML}").read_text())
for a in spec["architectures"]:
    tv_arg = "--target-variables " + " ".join(a["target_variables"])
    extra = os.path.expandvars(a["extra_args"])
    print(f"{a['name']}|{tv_arg}|{extra}")
PYEOF
)
mapfile -t SWEEP_ENTRIES < <(python3 <<PYEOF
import yaml
from pathlib import Path
spec = yaml.safe_load(Path("${EXPERIMENTS_YAML}").read_text())
for sp in spec["sweep_points"]:
    print(f"{sp['label']}")
PYEOF
)
mapfile -t SIMPLE_BASELINE_ENTRIES < <(python3 <<PYEOF
import yaml
from pathlib import Path
spec = yaml.safe_load(Path("${EXPERIMENTS_YAML}").read_text())
for sb in spec.get("simple_baselines", []):
    tv = " ".join(sb["target_variables"])
    print(f"{sb['name']}|{sb['baseline']}|{tv}")
PYEOF
)
if [ "${#ARCH_ENTRIES[@]}" -eq 0 ] || [ "${#SWEEP_ENTRIES[@]}" -eq 0 ]; then
    echo "ERROR: empty architectures or sweep_points list." >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT_BASE}"

# Shared training args (region-spec is per-run-dir below).
COMMON_TRAIN_ARGS="--dataset-dir ${DATASET_DIR} \
    --tessera-path ${TESSERA_PATH} \
    --tessera-station-csv ${TESSERA_CSV} \
    --normalisation-policy ${NORMALISATION_POLICY} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --lr ${LR} \
    --lr-warmup-pct ${LR_WARMUP_PCT} \
    --cnn-hidden ${CNN_HIDDEN} \
    --cnn-layers ${CNN_LAYERS} \
    --mlp-hidden ${MLP_HIDDEN} \
    --mlp-n-hidden ${MLP_N_HIDDEN} \
    --num-workers ${NUM_WORKERS} \
    --region-specs-train-file ${REGION_SPECS_TRAIN_JSON}"

echo "============================================"
echo "Placement experiment: snapshot_14y_eu_placement_norway"
echo "============================================"
echo "Strategies:    ${#STRATEGY_LIST[@]}  (${STRATEGY_LIST[*]})"
echo "Architectures: ${#ARCH_ENTRIES[@]}"
echo "Sweep points:  ${#SWEEP_ENTRIES[@]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Trained jobs:  $(( ${#STRATEGY_LIST[@]} * ${#ARCH_ENTRIES[@]} * ${#SWEEP_ENTRIES[@]} * ${#SEEDS[@]} ))  (before skip-existing)"
echo "Simple jobs:   ${#SIMPLE_BASELINE_ENTRIES[@]}  (once, strategy-independent)"
echo "DRY_RUN=${DRY_RUN}  (set DRY_RUN=0 to actually submit)"
echo ""

JOB_COUNT=0

# ---- Trained jobs: strategy x architecture x sweep x seed ----
for strategy in "${STRATEGY_LIST[@]}"; do
    SCHED="${SCHEDULES_DIR}/rollout_schedule_${strategy}.json"
    if [ ! -f "${SCHED}" ]; then
        echo "ERROR: schedule for strategy '${strategy}' not found: ${SCHED}" >&2
        exit 1
    fi
    MAT="${MATERIALISED_DIR}/${strategy}"
    mkdir -p "${MAT}"

    # Materialise per-sweep probe_active_from_<label>.json + train_end_overrides.json
    # for THIS strategy's schedule (idempotent: leave existing files).
    python3 <<PYEOF
import json
from pathlib import Path

schedule = json.loads(Path("${SCHED}").read_text())
mat = Path("${MAT}")
required_top = {"schedule_metadata", "sweep_points"}
missing = required_top - set(schedule)
if missing:
    raise SystemExit(f"${SCHED}: missing top-level keys {sorted(missing)}.")
md = schedule["schedule_metadata"]
print(f"  [{'${strategy}'}] order_source={md.get('activation_order_source')} "
      f"t_rollout={md.get('t_rollout_months')}mo n={md.get('n_stations')}")
override_map = {}
required_sweep = {"train_end_override", "probe_active_from"}
for label, sweep in schedule["sweep_points"].items():
    sm = required_sweep - set(sweep)
    if sm:
        raise SystemExit(f"${SCHED} sweep[{label!r}] missing {sorted(sm)}.")
    override_map[label] = sweep["train_end_override"]
    out_path = mat / f"probe_active_from_{label}.json"
    if not out_path.exists():
        out_path.write_text(json.dumps(sweep["probe_active_from"], indent=2))
(mat / "train_end_overrides.json").write_text(json.dumps(override_map, indent=2))
PYEOF

    OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${strategy}"
    mkdir -p "${OUTPUT_ROOT}"

    for arch_entry in "${ARCH_ENTRIES[@]}"; do
        IFS='|' read -r arch_name target_args arch_extra_args <<< "${arch_entry}"

        for sweep_label in "${SWEEP_ENTRIES[@]}"; do
            PROBE_ACTIVE_FROM_JSON="${MAT}/probe_active_from_${sweep_label}.json"
            if [ ! -f "${PROBE_ACTIVE_FROM_JSON}" ]; then
                echo "ERROR: ${PROBE_ACTIVE_FROM_JSON} missing (materialisation failed)." >&2
                exit 1
            fi
            TRAIN_END_OVERRIDE=$(python3 -c "
import json, sys
d = json.load(open('${MAT}/train_end_overrides.json'))
if '${sweep_label}' not in d:
    sys.exit(f\"sweep '${sweep_label}' not in overrides. Available: {sorted(d)}\")
print(d['${sweep_label}'])
")

            for seed in "${SEEDS[@]}"; do
                run_name="${arch_name}_${sweep_label}_seed${seed}"
                run_dir="${OUTPUT_ROOT}/${run_name}"
                job_name="place_${strategy}_${arch_name}_${sweep_label}_s${seed}"

                if [ -f "${run_dir}/test_summary.json" ]; then
                    echo "SKIP: ${strategy}/${run_name} (already complete)"
                    continue
                fi

                TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} \
                    ${COMMON_TRAIN_ARGS} \
                    ${target_args} \
                    ${arch_extra_args} \
                    --probe-active-from-file ${PROBE_ACTIVE_FROM_JSON} \
                    --train-end-override ${TRAIN_END_OVERRIDE} \
                    --seed ${seed} \
                    --output-dir ${run_dir}"
                EVAL_CMD="${REPO_ROOT}/.venv/bin/python ${EVAL_SCRIPT} \
                    --checkpoint ${run_dir}/best_model.pt \
                    --batch-size ${BATCH_SIZE} \
                    --num-workers ${NUM_WORKERS} \
                    --region-specs-test-file ${REGION_SPECS_TEST_JSON}"

                SBATCH_CMD="sbatch --job-name=${job_name} \
                    --output=${REPO_ROOT}/logs/${job_name}_%j.out \
                    --error=${REPO_ROOT}/logs/${job_name}_%j.err \
                    --gpus=1 --time=${TIME} \
                    ${PARTITION:+--partition=${PARTITION}} \
                    --wrap=\"cd ${REPO_ROOT} && ${TRAIN_CMD} && ${EVAL_CMD}\""

                if [ "${DRY_RUN}" = "1" ]; then
                    echo "DRY RUN: ${job_name}"
                else
                    mkdir -p "${run_dir}"
                    JOB_ID=$(eval "${SBATCH_CMD}")
                    echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
                fi
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
done

# ---- Simple non-trained baselines (once — strategy/sweep independent) ----
SB_OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/_simple_baselines"
mkdir -p "${SB_OUTPUT_ROOT}"
for sb_entry in "${SIMPLE_BASELINE_ENTRIES[@]}"; do
    IFS='|' read -r sb_name sb_kind sb_target_variables <<< "${sb_entry}"
    sb_run_dir="${SB_OUTPUT_ROOT}/${sb_name}_seed42"
    sb_job_name="place_${sb_name}"

    if [ -f "${sb_run_dir}/test_summary.json" ]; then
        echo "SKIP: ${sb_job_name} (already complete)"
        continue
    fi
    SB_CMD="${REPO_ROOT}/.venv/bin/python ${SIMPLE_BASELINE_SCRIPT} \
        --baseline ${sb_kind} \
        --dataset-dir ${DATASET_DIR} \
        --target-variables ${sb_target_variables} \
        --train-regions europe \
        --normalisation-policy ${NORMALISATION_POLICY} \
        --station-split all \
        --output-dir ${sb_run_dir} \
        --seed 42"
    SB_SBATCH="sbatch --job-name=${sb_job_name} \
        --output=${REPO_ROOT}/logs/${sb_job_name}_%j.out \
        --error=${REPO_ROOT}/logs/${sb_job_name}_%j.err \
        --cpus-per-task=4 --mem=16G --time=01:00:00 \
        ${PARTITION:+--partition=${PARTITION}} \
        --wrap=\"cd ${REPO_ROOT} && ${SB_CMD}\""

    if [ "${DRY_RUN}" = "1" ]; then
        echo "DRY RUN: ${sb_job_name}"
    else
        mkdir -p "${sb_run_dir}"
        JOB_ID=$(eval "${SB_SBATCH}")
        echo "SUBMITTED: ${sb_job_name} -> ${JOB_ID}"
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "============================================"
if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY RUN complete: ${JOB_COUNT} jobs WOULD be submitted (nothing sent)."
    echo "Re-run with DRY_RUN=0 to submit."
else
    echo "Submitted ${JOB_COUNT} jobs."
fi
echo "Results tree:  ${OUTPUT_ROOT_BASE}/<strategy>/"
echo "Monitor with:  squeue --me"
