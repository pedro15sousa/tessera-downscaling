#!/bin/bash
# Submit the station-count axis data-efficiency experiments for EU 14y.
#
# For each (K_train, seed) pair, samples K_train station_ids from the
# EU 85% train pool with a per-(K, seed) numpy seed. Trains the
# specified architectures on each subset and evaluates on the 15%
# spatial-test set during the held-out test year.
#
# Pre-flight steps performed inline:
#   1. For each (sweep_point.k_train, seed) pair, writes
#      ``train_station_allowlist_K{K_train}_seed{seed}.json`` next to
#      this submit script. Existing files are left in place (so the
#      experiment is reproducible across re-runs even if the YAML is
#      edited later — delete the files explicitly to regenerate).
#      The "Kfull" sweep point doesn't get an allowlist file: those
#      runs are submitted WITHOUT --train-station-allowlist-file so
#      they use every train-split station, identical to existing
#      snapshot_14y_eu runs.
#   2. Submits one sbatch per (architecture, sweep_point, seed).
#   3. Submits one sbatch per simple_baseline (no K-sweep, no model
#      seed, default station_split=test for direct comparability).
#
# Usage (from repo root):
#   bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_eu_station_count_efficiency/submit.sh
#
# Dry run:
#   DRY_RUN=1 bash projects/tessera_downscaling/scripts/experiments/snapshot_14y_eu_station_count_efficiency/submit.sh
set -euo pipefail

# ---- Paths ----
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"

DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset_timestamp_global}"

TESSERA_PATH="${TESSERA_PATH:-${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy}"
TESSERA_CSV="${TESSERA_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH:-${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy}"
export VAE_LATENTS_PATH_LAT64="${VAE_LATENTS_PATH_LAT64:-${BASE_DIR}/processed/station_latents_lat64_l1.npy}"
export VAE_LATENTS_CSV="${VAE_LATENTS_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/training_runs_snapshot_14y_eu_station_count_efficiency}"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"
SIMPLE_BASELINE_SCRIPT="projects/tessera_downscaling/scripts/baselines/evaluate_simple_baselines.py"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"

NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-}"

# ---- Hyperparameters ----
SEEDS=(42 123 456)
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

SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"

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

if [ "${SKIP_UV_SYNC}" = "1" ]; then
    echo "SKIP_UV_SYNC=1, skipping uv sync."
else
    echo "Pre-syncing environment..."
    cd "${REPO_ROOT}"
    uv sync --group core
fi

# ---- Step 1: Generate per-(K, seed) allowlist files ----
# For each finite-K sweep point and each model seed, samples K
# station_ids from the EU train pool and writes the allowlist JSON.
# Cached: existing files are left in place.
echo "Materialising per-(K, seed) train-station allowlist JSON files..."
python3 <<PYEOF
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

spec = yaml.safe_load(Path("${EXPERIMENTS_YAML}").read_text())
pool_cfg = spec["sample_pool"]
source_region = pool_cfg["source_region"]
source_split = pool_cfg["source_spatial_split"]
seeds = [42, 123, 456]   # MUST match the SEEDS bash array above.

stations = pd.read_csv("${DATASET_DIR}/stations.csv")
mask = (
    (stations["region"] == source_region)
    & (stations["spatial_split"] == source_split)
)
pool = stations.loc[mask, "station_id"].astype(str).tolist()
n_total = len(pool)
print(f"Train-station pool: {n_total} stations from {source_region}/{source_split}")

script_dir = Path("${SCRIPT_DIR}")
for sp in spec["sweep_points"]:
    label = sp["label"]
    k_train = sp.get("k_train")
    if k_train is None:
        # "Kfull" — no allowlist needed; runs use every train station.
        continue
    if k_train > n_total:
        raise SystemExit(
            f"sweep_point label={label} requests k_train={k_train} but pool "
            f"has only {n_total} stations."
        )
    for seed in seeds:
        out_path = script_dir / f"train_station_allowlist_{label}_seed{seed}.json"
        if out_path.exists():
            print(f"  {out_path.name} exists, leaving untouched.")
            continue
        # Stable per-(K, seed) sampling seed derived via SHA-256 so that
        # accidentally reordering the bash SEEDS array won't reshuffle the
        # historical subsets. (numpy default_rng accepts arbitrary uint64
        # seeds, so SHA → first 8 bytes gives us 2^64 distinct streams.)
        key = f"{k_train}:{seed}".encode()
        sample_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = np.random.default_rng(sample_seed)
        idx = np.sort(rng.choice(n_total, size=k_train, replace=False))
        sampled_ids = [pool[i] for i in idx]
        payload = {
            "station_ids": sampled_ids,
            "k_train": k_train,
            "model_seed": seed,
            "sample_seed": sample_seed,
            "source_region": source_region,
            "source_spatial_split": source_split,
            "n_pool": n_total,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out_path.name}: {k_train} of {n_total} stations.")
PYEOF

# ---- Step 2: Region-specs files (same shape as the other experiments) ----
REGION_SPECS_TRAIN_JSON="${SCRIPT_DIR}/region_specs_train.json"
REGION_SPECS_TEST_JSON="${SCRIPT_DIR}/region_specs_test.json"
echo '{"europe":"train"}' > "${REGION_SPECS_TRAIN_JSON}"
echo '{"europe":"all"}'   > "${REGION_SPECS_TEST_JSON}"

# ---- Step 3: Load architecture + sweep_point lists from YAML ----
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
    # Emit "label|k_train_or_FULL"
    k = sp.get("k_train")
    print(f"{sp['label']}|{'FULL' if k is None else k}")
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

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

echo "============================================"
echo "Submitting snapshot_14y_eu_station_count_efficiency"
echo "============================================"
echo "Architectures: ${#ARCH_ENTRIES[@]}"
echo "Sweep points:  ${#SWEEP_ENTRIES[@]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Trained jobs:  $(( ${#ARCH_ENTRIES[@]} * ${#SWEEP_ENTRIES[@]} * ${#SEEDS[@]} ))"
echo "Simple jobs:   ${#SIMPLE_BASELINE_ENTRIES[@]}"
echo ""

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

# ---- Step 4: Submit trained jobs (architecture × sweep × seed) ----
JOB_COUNT=0
for arch_entry in "${ARCH_ENTRIES[@]}"; do
    IFS='|' read -r arch_name target_args arch_extra_args <<< "${arch_entry}"

    for sweep_entry in "${SWEEP_ENTRIES[@]}"; do
        IFS='|' read -r sweep_label k_train_raw <<< "${sweep_entry}"

        for seed in "${SEEDS[@]}"; do
            run_name="${arch_name}_${sweep_label}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${run_name}"
            job_name="${run_name}"

            if [ -f "${run_dir}/test_summary.json" ]; then
                echo "SKIP: ${job_name} (already complete)"
                continue
            fi

            mkdir -p "${run_dir}"

            # Optional allowlist arg — empty for "Kfull".
            ALLOWLIST_ARG=""
            if [ "${k_train_raw}" != "FULL" ]; then
                ALLOWLIST_FILE="${SCRIPT_DIR}/train_station_allowlist_${sweep_label}_seed${seed}.json"
                if [ ! -f "${ALLOWLIST_FILE}" ]; then
                    echo "ERROR: ${ALLOWLIST_FILE} missing — step 1 should have created it." >&2
                    exit 1
                fi
                ALLOWLIST_ARG="--train-station-allowlist-file ${ALLOWLIST_FILE}"
            fi

            TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} \
                ${COMMON_TRAIN_ARGS} \
                ${target_args} \
                ${arch_extra_args} \
                ${ALLOWLIST_ARG} \
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

            if [ "${DRY_RUN:-0}" = "1" ]; then
                echo "DRY RUN: ${job_name}"
                echo "  ${SBATCH_CMD}"
                echo ""
            else
                JOB_ID=$(eval "${SBATCH_CMD}")
                echo "SUBMITTED: ${job_name} -> ${JOB_ID}"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

# ---- Step 5: Submit simple baselines (no K-sweep, no seed dep.) ----
# Default station_split=test, matching the trained runs' eval set.
for sb_entry in "${SIMPLE_BASELINE_ENTRIES[@]}"; do
    IFS='|' read -r sb_name sb_kind sb_target_variables <<< "${sb_entry}"
    sb_run_dir="${OUTPUT_ROOT}/${sb_name}_seed42"
    sb_job_name="${sb_name}"

    if [ -f "${sb_run_dir}/test_summary.json" ]; then
        echo "SKIP: ${sb_job_name} (already complete)"
        continue
    fi

    mkdir -p "${sb_run_dir}"
    SB_CMD="${REPO_ROOT}/.venv/bin/python ${SIMPLE_BASELINE_SCRIPT} \
        --baseline ${sb_kind} \
        --dataset-dir ${DATASET_DIR} \
        --target-variables ${sb_target_variables} \
        --train-regions europe \
        --normalisation-policy ${NORMALISATION_POLICY} \
        --output-dir ${sb_run_dir} \
        --seed 42"

    SB_SBATCH="sbatch --job-name=${sb_job_name} \
        --output=${REPO_ROOT}/logs/${sb_job_name}_%j.out \
        --error=${REPO_ROOT}/logs/${sb_job_name}_%j.err \
        --cpus-per-task=4 --mem=16G --time=01:00:00 \
        ${PARTITION:+--partition=${PARTITION}} \
        --wrap=\"cd ${REPO_ROOT} && ${SB_CMD}\""

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY RUN: ${sb_job_name}"
        echo "  ${SB_SBATCH}"
        echo ""
    else
        JOB_ID=$(eval "${SB_SBATCH}")
        echo "SUBMITTED: ${sb_job_name} -> ${JOB_ID}"
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "============================================"
echo "Submitted ${JOB_COUNT} jobs to $(basename ${OUTPUT_ROOT})"
echo "============================================"
echo "Results in:    ${OUTPUT_ROOT}/"
echo "Monitor with:  squeue --me"