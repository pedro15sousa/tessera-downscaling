# shellcheck shell=bash
# =========================================================================
# Shared definitions for scripts/experiments/*/submit.sh. Source it, do not
# run it. Every submit.sh sets SCRIPT_DIR and FOLDER first:
#
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   FOLDER="$(basename "${SCRIPT_DIR}")"
#   source "${SCRIPT_DIR}/../_lib.sh"
#
# What it provides:
#   * the data-root layout (DATA_ROOT, DATASET_DIR, TESSERA_PATH, ...), all
#     env-overridable; runs land in ${OUTPUT_ROOT} = ${DATA_ROOT}/training_runs_${FOLDER};
#   * the training hyperparameters shared by every folder of the paper;
#   * the execution knobs DRY_RUN=1 (print only) / LOCAL=1 (run in this shell
#     instead of sbatch) / TIME / PARTITION;
#   * load_flat_experiments  -- read a flat experiments.yaml into EXPERIMENTS[];
#   * require_multi_region_dataset REGION -- preflight on dataset_timestamp_global;
#   * run_job NAME gpu|cpu CMD -- dry-run / local / sbatch --wrap dispatch;
#   * run_single_region_matrix REGION -- the whole per-folder loop of the
#     five regional folders and their tessera / shuffled siblings;
#   * run_rollout_matrix -- the architecture x sweep-point x seed loop of the
#     two Norway temporal-rollout folders (structured experiments.yaml).
# =========================================================================

set -euo pipefail

: "${SCRIPT_DIR:?set SCRIPT_DIR before sourcing _lib.sh}"
: "${FOLDER:?set FOLDER before sourcing _lib.sh}"

# ---- Paths -------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATA_ROOT="${TESSERA_DATA_ROOT:-/data/weather-downscaling}"
export TESSERA_DATA_ROOT="${DATA_ROOT}"
BASE_DIR="${DATA_ROOT}"

# The paper's dataset (multi_region_snapshot_v1 layout, five regions).
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset_timestamp_global}"

# Station-validity filter used by EVERY run (baselines included): a station is
# kept iff its 2024 v1 TESSERA patch has a non-zero centre pixel and >= 50 %
# coverage. No patches are loaded for training -- filter only.
export TESSERA_PATH="${TESSERA_PATH:-${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy}"
export TESSERA_CSV="${TESSERA_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"

# Row-alignment key of every per-station vector file (latents, descriptors).
export VAE_LATENTS_CSV="${VAE_LATENTS_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"

# Per-station vector files. The paper's TESSERA arm uses the 1B-M (v2) 2017
# latents; the v1 file is the earlier generation (2024 p64 patches, VAE
# lat16_beta0.0005_grad0.5_e200) still referenced by the regional folders and
# the Norway rollout baseline folder.
export VAE_LATENTS_DIR_1BM="${BASE_DIR}/processed/vae_tessera_1B-M"
export VAE_LATENTS_PATH_1BM="${VAE_LATENTS_DIR_1BM}/station_latents_1B-M_p128_2017_crop64_lat16_grad0.5_auxon.npy"
export VAE_LATENTS_PATH_V1="${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy"
export EXTRA_DESCRIPTORS_PATH="${EXTRA_DESCRIPTORS_PATH:-${BASE_DIR}/processed/extra_descriptors.npy}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/training_runs_${FOLDER}}"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/${FOLDER}}"

# Console entry points of the package (see pyproject.toml).
TRAIN_CMD="uv run tessera-train"
EVAL_CMD="uv run tessera-evaluate"
BASELINES_CMD="uv run tessera-baselines"

# ---- Execution knobs ---------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"     # 1: print the commands, submit nothing
LOCAL="${LOCAL:-0}"         # 1: run each job in this shell (no Slurm)
TIME="${TIME:-24:00:00}"    # GPU job wall time
PARTITION="${PARTITION:-}"  # empty = cluster default

# ---- Hyperparameters (identical for every trained run of the paper) -----
# shellcheck disable=SC2206
SEEDS=(${SEEDS:-42 123 456})
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS=100
PATIENCE=10
LR="2.5e-5"
LR_WARMUP_PCT="0.05"
CNN_HIDDEN=128
CNN_LAYERS=7
MLP_HIDDEN=128
MLP_N_HIDDEN=3
NUM_WORKERS=4

MODEL_ARGS="--batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --patience ${PATIENCE} \
--lr ${LR} --lr-warmup-pct ${LR_WARMUP_PCT} --cnn-hidden ${CNN_HIDDEN} --cnn-layers ${CNN_LAYERS} \
--mlp-hidden ${MLP_HIDDEN} --mlp-n-hidden ${MLP_N_HIDDEN} --num-workers ${NUM_WORKERS}"

JOB_COUNT=0
SKIP_COUNT=0

# ---- YAML loading ------------------------------------------------------
# Fills EXPERIMENTS[] with one "name|target_args|extra_args|baseline_kind" per
# entry of a flat experiments.yaml. ${VAR} references in extra_args are
# expanded from the environment (the exports above plus whatever the calling
# submit.sh exported), so paths never live in the yaml.
load_flat_experiments() {
    [ -f "${EXPERIMENTS_YAML}" ] || { echo "ERROR: ${EXPERIMENTS_YAML} not found" >&2; exit 1; }
    mapfile -t EXPERIMENTS < <(uv run python - "${EXPERIMENTS_YAML}" <<'PYEOF'
import os, sys, yaml
with open(sys.argv[1]) as f:
    for e in yaml.safe_load(f):
        tv_arg = "--target-variables " + " ".join(e["target_variables"])
        extra = os.path.expandvars(e.get("extra_args", ""))
        print(f"{e['name']}|{tv_arg}|{extra}|{e.get('baseline_kind', '')}")
PYEOF
    )
    [ "${#EXPERIMENTS[@]}" -gt 0 ] || { echo "ERROR: no experiments in ${EXPERIMENTS_YAML}" >&2; exit 1; }
}

# ---- Preflight ---------------------------------------------------------
# Fails fast if the dataset is not usable (a warning only under DRY_RUN=1, so
# the commands can be previewed anywhere).
require_multi_region_dataset() {
    local region="$1"
    if [ ! -f "${DATASET_DIR}/metadata.json" ]; then
        echo "ERROR: ${DATASET_DIR}/metadata.json does not exist." >&2
        echo "Build the dataset first: scripts/preprocessing/preprocess_timestamp_global.py" >&2
        [ "${DRY_RUN}" = "1" ] && { echo "(continuing: DRY_RUN=1)" >&2; return 0; }
        exit 1
    fi
    uv run python - "${DATASET_DIR}/metadata.json" "${region}" <<'PYEOF'
import json, sys
md = json.load(open(sys.argv[1]))
layout = md.get("layout_version", "")
if layout != "multi_region_snapshot_v1":
    sys.exit(f"ERROR: layout_version={layout!r}, expected 'multi_region_snapshot_v1'.")
if sys.argv[2] not in md.get("regions", {}):
    sys.exit(f"ERROR: region {sys.argv[2]!r} not in {sorted(md.get('regions', {}))}.")
PYEOF
    local stats
    for stats in normalisation_stats_no_static.npz normalisation_stats.npz; do
        if [ ! -f "${DATASET_DIR}/regions/${region}/${stats}" ]; then
            echo "ERROR: ${DATASET_DIR}/regions/${region}/${stats} missing." >&2
            exit 1
        fi
    done
}

# ---- Dispatch ----------------------------------------------------------
# run_job NAME gpu|cpu CMD
#   DRY_RUN=1 -> print; LOCAL=1 -> `cd REPO_ROOT && eval CMD`; else sbatch --wrap.
run_job() {
    local name="$1" kind="$2" cmd="$3"
    local resources
    if [ "${kind}" = "gpu" ]; then
        resources="--gpus=1 --time=${TIME}"
    else
        resources="--cpus-per-task=4 --mem=16G --time=02:00:00"
    fi
    local sbatch_cmd="sbatch --job-name=${name} --output=${LOG_DIR}/${name}_%j.out --error=${LOG_DIR}/${name}_%j.err ${resources} ${PARTITION:+--partition=${PARTITION}} --wrap=\"cd ${REPO_ROOT} && ${cmd}\""
    if [ "${DRY_RUN}" = "1" ]; then
        echo "DRY RUN: ${name}"
        echo "  ${sbatch_cmd}"
    elif [ "${LOCAL}" = "1" ]; then
        echo "RUNNING (local): ${name}"
        (cd "${REPO_ROOT}" && eval "${cmd}") && echo "DONE: ${name}" || echo "FAILED: ${name}" >&2
    else
        mkdir -p "${LOG_DIR}"
        local job_id
        job_id=$(eval "${sbatch_cmd}")
        echo "SUBMITTED: ${name} -> ${job_id}"
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
}

# mkdir -p unless DRY_RUN=1 (a dry run creates nothing on disk).
ensure_dir() {
    [ "${DRY_RUN}" = "1" ] || mkdir -p "$1"
}

announce() {
    echo "============================================"
    echo "${FOLDER}"
    echo "============================================"
    echo "Data root:   ${DATA_ROOT}"
    echo "Dataset:     ${DATASET_DIR}"
    echo "Output:      ${OUTPUT_ROOT}"
    echo "YAML:        ${EXPERIMENTS_YAML}"
    echo "Seeds:       ${SEEDS[*]}"
    echo "Mode:        $([ "${DRY_RUN}" = 1 ] && echo dry-run || { [ "${LOCAL}" = 1 ] && echo local || echo sbatch; })"
    echo ""
}

summarise() {
    echo ""
    echo "============================================"
    if [ "${DRY_RUN}" = 1 ]; then
        echo "${FOLDER}: ${JOB_COUNT} jobs would be dispatched, ${SKIP_COUNT} skipped"
    else
        echo "${FOLDER}: ${JOB_COUNT} jobs dispatched, ${SKIP_COUNT} skipped"
    fi
    echo "Results in:  ${OUTPUT_ROOT}/"
    [ "${LOCAL}" = 1 ] || echo "Monitor:     squeue --me"
    echo "============================================"
}

# ---- The single-region matrix (regional, tessera and shuffled folders) ---
# For every yaml entry x seed: trained entries -> one GPU job that trains
# then evaluates on the training region; baseline_kind entries -> one CPU job
# running tessera-baselines on the SAME TESSERA-filtered station set. Entries
# whose latents file does not exist yet are skipped (re-run once it lands);
# runs with a test_summary.json are skipped.
run_single_region_matrix() {
    local region="$1"
    load_flat_experiments
    require_multi_region_dataset "${region}"
    ensure_dir "${OUTPUT_ROOT}"
    announce
    echo "Region:      ${region}"
    echo "Configs:     ${#EXPERIMENTS[@]}  (x ${#SEEDS[@]} seeds)"
    echo ""

    local common_args="--dataset-dir ${DATASET_DIR} --tessera-path ${TESSERA_PATH} \
--tessera-station-csv ${TESSERA_CSV} --train-regions ${region} ${MODEL_ARGS}"

    local experiment name target_args extra_args baseline_kind seed run_dir job latents
    for experiment in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name target_args extra_args baseline_kind <<< "${experiment}"

        latents=$(sed -n 's/.*--vae-latents-path \([^ ]*\).*/\1/p' <<< "${extra_args}")
        if [ -n "${latents}" ] && [ ! -f "${latents}" ]; then
            echo "SKIP: ${name} (latents not available: ${latents})"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi

        for seed in "${SEEDS[@]}"; do
            run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"
            job="${JOB_TAG:-}${name}_s${seed}"
            if [ -f "${run_dir}/test_summary.json" ]; then
                echo "SKIP: ${job} (already complete)"
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            fi
            if [ -n "${baseline_kind}" ]; then
                run_job "${job}" cpu "${BASELINES_CMD} --baseline ${baseline_kind} \
--dataset-dir ${DATASET_DIR} ${target_args} --train-regions ${region} \
--tessera-path ${TESSERA_PATH} --tessera-station-csv ${TESSERA_CSV} \
--min-tessera-patch-coverage 0.5 ${extra_args} --output-dir ${run_dir} --seed ${seed}"
            else
                run_job "${job}" gpu "${TRAIN_CMD} ${common_args} ${target_args} ${extra_args} \
--seed ${seed} --output-dir ${run_dir} && ${EVAL_CMD} --checkpoint ${run_dir}/best_model.pt \
--batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS}"
            fi
        done
    done
    summarise
}

# ---- The Norway rollout matrix (the two temporal-rollout folders) -------
# These folders use a structured experiments.yaml (sweep_points /
# architectures / simple_baselines) plus sidecar files built once by
# pick_probe_set.py (probe_station_ids.json) and build_rollout_schedule.py
# (rollout_schedule.json). Every run trains on Europe with the probe stations
# hidden before their activation date (--probe-active-from-file) and the
# training window cut at the sweep point (--train-end-override), then
# evaluates on ALL European stations (--region-specs-test-file {"europe":
# "all"}) so the probe / always-on / spatial-test breakdown comes out of one
# eval pass. simple_baselines run once (seed 42; no sweep -- they do not
# train) with --station-split all for the same breakdown.

# Fills ROLLOUT_ARCHS[] ("name|target_args|extra_args"), ROLLOUT_SWEEPS[]
# (labels) and ROLLOUT_SIMPLE[] ("name|baseline|target_args|extra_args").
load_rollout_experiments() {
    [ -f "${EXPERIMENTS_YAML}" ] || { echo "ERROR: ${EXPERIMENTS_YAML} not found" >&2; exit 1; }
    local section var
    for section in architectures sweep_points simple_baselines; do
        case "${section}" in
            architectures) var=ROLLOUT_ARCHS ;;
            sweep_points) var=ROLLOUT_SWEEPS ;;
            *) var=ROLLOUT_SIMPLE ;;
        esac
        mapfile -t "${var}" < <(uv run python - "${EXPERIMENTS_YAML}" "${section}" <<'PYEOF'
import os, sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
for e in spec.get(sys.argv[2], []):
    if sys.argv[2] == "sweep_points":
        print(e["label"])
        continue
    tv_arg = "--target-variables " + " ".join(e["target_variables"])
    extra = os.path.expandvars(e.get("extra_args", ""))
    if sys.argv[2] == "architectures":
        print(f"{e['name']}|{tv_arg}|{extra}")
    else:
        print(f"{e['name']}|{e['baseline']}|{tv_arg}|{extra}")
PYEOF
        )
    done
    if [ "${#ROLLOUT_ARCHS[@]}" -eq 0 ] || [ "${#ROLLOUT_SWEEPS[@]}" -eq 0 ]; then
        echo "ERROR: ${EXPERIMENTS_YAML} needs non-empty 'architectures' and 'sweep_points'." >&2
        exit 1
    fi
}

# Validates rollout_schedule.json against probe_station_ids.json, writes the
# per-sweep probe_active_from_<label>.json (left alone when present),
# train_end_overrides.json and region_specs_{train,test}.json next to the
# yaml, and fills ROLLOUT_TRAIN_END[label] from the schedule.
declare -A ROLLOUT_TRAIN_END
materialise_rollout_schedule() {
    local schedule="${SCRIPT_DIR}/rollout_schedule.json" probes="${SCRIPT_DIR}/probe_station_ids.json"
    local f
    for f in "${schedule}" "${probes}"; do
        if [ ! -f "${f}" ]; then
            echo "ERROR: ${f} missing (scripts/experiments/pick_probe_set.py, then build_rollout_schedule.py)." >&2
            exit 1
        fi
    done
    REGION_SPECS_TRAIN_JSON="${SCRIPT_DIR}/region_specs_train.json"
    REGION_SPECS_TEST_JSON="${SCRIPT_DIR}/region_specs_test.json"
    echo '{"europe":"train"}' > "${REGION_SPECS_TRAIN_JSON}"
    echo '{"europe":"all"}'   > "${REGION_SPECS_TEST_JSON}"
    local label train_end
    while IFS='=' read -r label train_end; do
        ROLLOUT_TRAIN_END["${label}"]="${train_end}"
    done < <(uv run python - "${schedule}" "${probes}" "${SCRIPT_DIR}" <<'PYEOF'
import json, sys
from pathlib import Path
schedule = json.loads(Path(sys.argv[1]).read_text())
probes = set(json.loads(Path(sys.argv[2]).read_text())["probe_station_ids"])
out_dir = Path(sys.argv[3])
md = schedule["schedule_metadata"]
print("Rollout schedule: T_0 %s, T_rollout %s months, %d probes, activation-order seed %s"
      % (md["rollout_anchor_t_0"], md["t_rollout_months"], md["n_stations"],
         md["activation_order_seed"]), file=sys.stderr)
overrides = {}
for label, sweep in schedule["sweep_points"].items():
    paf = sweep["probe_active_from"]
    if set(paf) != probes:
        sys.exit(f"ERROR: sweep {label!r}: probe_active_from keys != probe_station_ids.json")
    overrides[label] = sweep["train_end_override"]
    path = out_dir / f"probe_active_from_{label}.json"
    if not path.exists():
        path.write_text(json.dumps(paf, indent=2))
    n_online = sum(1 for v in paf.values() if not v.startswith("9999"))
    print(f"  {label:<5} train_end {overrides[label]}  {n_online:>5}/{len(paf)} probes online",
          file=sys.stderr)
(out_dir / "train_end_overrides.json").write_text(json.dumps(overrides, indent=2))
for label, train_end in overrides.items():
    print(f"{label}={train_end}")
PYEOF
    )
}

run_rollout_matrix() {
    load_rollout_experiments
    require_multi_region_dataset europe
    materialise_rollout_schedule
    ensure_dir "${OUTPUT_ROOT}"
    announce
    echo "Architectures: ${#ROLLOUT_ARCHS[@]}   Sweep points: ${#ROLLOUT_SWEEPS[@]}   Seeds: ${#SEEDS[@]}"
    echo "Trained jobs:  $(( ${#ROLLOUT_ARCHS[@]} * ${#ROLLOUT_SWEEPS[@]} * ${#SEEDS[@]} ))   References: ${#ROLLOUT_SIMPLE[@]}"
    echo ""

    local common_args="--dataset-dir ${DATASET_DIR} --tessera-path ${TESSERA_PATH} \
--tessera-station-csv ${TESSERA_CSV} --region-specs-train-file ${REGION_SPECS_TRAIN_JSON} ${MODEL_ARGS}"

    local entry name target_args extra_args sweep seed run_dir job probe_json train_end
    for entry in "${ROLLOUT_ARCHS[@]}"; do
        IFS='|' read -r name target_args extra_args <<< "${entry}"
        for sweep in "${ROLLOUT_SWEEPS[@]}"; do
            probe_json="${SCRIPT_DIR}/probe_active_from_${sweep}.json"
            train_end="${ROLLOUT_TRAIN_END[${sweep}]:-}"
            if [ -z "${train_end}" ] || [ ! -f "${probe_json}" ]; then
                echo "ERROR: sweep point ${sweep} (experiments.yaml) is not in rollout_schedule.json." >&2
                exit 1
            fi
            for seed in "${SEEDS[@]}"; do
                run_dir="${OUTPUT_ROOT}/${name}_${sweep}_seed${seed}"
                job="${JOB_TAG:-}${name}_${sweep}_s${seed}"
                if [ -f "${run_dir}/test_summary.json" ]; then
                    echo "SKIP: ${job} (already complete)"
                    SKIP_COUNT=$((SKIP_COUNT + 1))
                    continue
                fi
                run_job "${job}" gpu "${TRAIN_CMD} ${common_args} ${target_args} ${extra_args} \
--probe-active-from-file ${probe_json} --train-end-override ${train_end} \
--seed ${seed} --output-dir ${run_dir} && ${EVAL_CMD} --checkpoint ${run_dir}/best_model.pt \
--batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --region-specs-test-file ${REGION_SPECS_TEST_JSON}"
            done
        done
    done

    local baseline_kind
    for entry in "${ROLLOUT_SIMPLE[@]}"; do
        IFS='|' read -r name baseline_kind target_args extra_args <<< "${entry}"
        run_dir="${OUTPUT_ROOT}/${name}_seed42"
        job="${JOB_TAG:-}${name}"
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "SKIP: ${job} (already complete)"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        run_job "${job}" cpu "${BASELINES_CMD} --baseline ${baseline_kind} --dataset-dir ${DATASET_DIR} \
${target_args} --train-regions europe --station-split all ${extra_args} \
--output-dir ${run_dir} --seed 42"
    done
    summarise
}
