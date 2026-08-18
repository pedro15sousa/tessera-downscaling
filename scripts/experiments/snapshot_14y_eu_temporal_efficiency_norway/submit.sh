set -euo pipefail

# ---- Paths ----
# Defaults point at the Isambard layout (where this script normally
# runs). All paths are env-overridable so the script is also testable
# in other environments (e.g. local dev box, CI dry-runs).
REPO_ROOT="${REPO_ROOT:-/projects/u6do/pmms2/end-to-end-forecasting}"
BASE_DIR="${BASE_DIR:-${REPO_ROOT}/projects/tessera_downscaling/.tmp_output}"

# 14y multi-region dataset (same one the snapshot_14y_eu folder uses).
DATASET_DIR="${DATASET_DIR:-${BASE_DIR}/dataset_timestamp_global}"

TESSERA_PATH="${TESSERA_PATH:-${BASE_DIR}/processed/tessera_global/patch_embeddings_2024.npy}"
TESSERA_CSV="${TESSERA_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"
export VAE_LATENTS_PATH="${VAE_LATENTS_PATH:-${BASE_DIR}/processed/station_latents_lat16_grad0.5.npy}"
export VAE_LATENTS_PATH_LAT64="${VAE_LATENTS_PATH_LAT64:-${BASE_DIR}/processed/station_latents_lat64_l1.npy}"
export VAE_LATENTS_CSV="${VAE_LATENTS_CSV:-${BASE_DIR}/processed/tessera_global/station_list_filtered.csv}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_DIR}/training_runs_snapshot_14y_eu_temporal_efficiency_norway}"
TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"
SIMPLE_BASELINE_SCRIPT="projects/tessera_downscaling/scripts/baselines/evaluate_simple_baselines.py"

# Resolve YAML next to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENTS_YAML="${SCRIPT_DIR}/experiments.yaml"
PROBE_IDS_JSON="${SCRIPT_DIR}/probe_station_ids.json"

# Per-region normalisation, same as snapshot_14y_eu single-region runs.
NORMALISATION_POLICY="per_region"

# ---- Slurm settings ----
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-}"

# ---- Hyperparameters (mirror snapshot_14y_eu) ----
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

# Optional: SKIP_UV_SYNC=1 to skip the `uv sync` (useful for DRY_RUN in
# environments where uv isn't set up — the sync is only there to make
# sure the Isambard venv is fresh before submitting).
SKIP_UV_SYNC="${SKIP_UV_SYNC:-0}"

# ---- Preflight on dataset layout ----
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

if [ ! -f "${PROBE_IDS_JSON}" ]; then
    echo "ERROR: probe_station_ids.json does not exist at ${PROBE_IDS_JSON}." >&2
    echo "       Generate it from the Norway-probe selection notebook cell"     >&2
    echo "       before running this submit script. See the file-level"        >&2
    echo "       docstring for details."                                       >&2
    exit 1
fi
python3 <<PYEOF
import json
from pathlib import Path

d = json.loads(Path("${PROBE_IDS_JSON}").read_text())
required = {"probe_station_ids", "n_probe"}
missing = required - set(d)
if missing:
    raise SystemExit(
        f"probe_station_ids.json is missing required keys {sorted(missing)}. "
        f"Re-generate from the notebook cell."
    )
n = len(d["probe_station_ids"])
if n != int(d["n_probe"]):
    raise SystemExit(
        f"probe_station_ids.json: n_probe={d['n_probe']} but station_ids "
        f"list has {n} entries — file is inconsistent."
    )
if n < 1:
    raise SystemExit("probe_station_ids.json contains zero probe stations.")
# Print audit metadata — exact fields vary by selection method, so be
# tolerant of what's there.
print(f"Using hand-picked probe set ({n} stations).")
for k in ("selection_method", "bbox_lat", "bbox_lon", "elev_min_m",
          "fraction_actual", "description"):
    if k in d:
        print(f"  {k}: {d[k]}")
PYEOF

# ---- Step 2: For each sweep_point, write probe_active_from_{label}.json ----
# Generates one file per sweep_point (cached: existing files left alone).
echo "Materialising per-sweep-point active-from JSON files..."
python3 <<PYEOF
import json
from pathlib import Path
import yaml

spec = yaml.safe_load(Path("${EXPERIMENTS_YAML}").read_text())
probe_ids = json.loads(Path("${PROBE_IDS_JSON}").read_text())["probe_station_ids"]

script_dir = Path("${SCRIPT_DIR}")
for sp in spec["sweep_points"]:
    label = sp["label"]
    active_from = sp["active_from"]
    out_path = script_dir / f"probe_active_from_{label}.json"
    if out_path.exists():
        print(f"  {out_path.name} exists, leaving untouched.")
        continue
    payload = {sid: active_from for sid in probe_ids}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_path.name}: {len(payload)} entries, active_from={active_from}")
PYEOF

# ---- Step 3: Materialise region-specs JSON files (once each) ----
# Used by all training jobs (region_specs_train) and all evaluation
# jobs (region_specs_test). The test specs use "all" so evaluate.py
# can compute the probe / always_on / spatial_test breakdown in a
# single eval pass.
REGION_SPECS_TRAIN_JSON="${SCRIPT_DIR}/region_specs_train.json"
REGION_SPECS_TEST_JSON="${SCRIPT_DIR}/region_specs_test.json"
echo '{"europe":"train"}' > "${REGION_SPECS_TRAIN_JSON}"
echo '{"europe":"all"}'   > "${REGION_SPECS_TEST_JSON}"

# ---- Step 4: Load architecture + sweep_point lists from YAML ----
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

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}"

echo "============================================"
echo "Submitting snapshot_14y_eu_temporal_efficiency"
echo "============================================"
echo "Architectures: ${#ARCH_ENTRIES[@]}"
echo "Sweep points:  ${#SWEEP_ENTRIES[@]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Trained jobs:  $(( ${#ARCH_ENTRIES[@]} * ${#SWEEP_ENTRIES[@]} * ${#SEEDS[@]} ))"
echo "Simple jobs:   ${#SIMPLE_BASELINE_ENTRIES[@]}"
echo ""

# Shared training args (no region-spec — that's per-run-dir below).
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

# ---- Step 5: Submit the trained jobs (architecture × sweep × seed) ----
JOB_COUNT=0
for arch_entry in "${ARCH_ENTRIES[@]}"; do
    IFS='|' read -r arch_name target_args arch_extra_args <<< "${arch_entry}"

    for sweep_label in "${SWEEP_ENTRIES[@]}"; do
        PROBE_ACTIVE_FROM_JSON="${SCRIPT_DIR}/probe_active_from_${sweep_label}.json"
        if [ ! -f "${PROBE_ACTIVE_FROM_JSON}" ]; then
            echo "ERROR: ${PROBE_ACTIVE_FROM_JSON} missing — step 2 should have created it." >&2
            exit 1
        fi

        for seed in "${SEEDS[@]}"; do
            run_name="${arch_name}_${sweep_label}_seed${seed}"
            run_dir="${OUTPUT_ROOT}/${run_name}"
            job_name="${run_name}"

            if [ -f "${run_dir}/test_summary.json" ]; then
                echo "SKIP: ${job_name} (already complete)"
                continue
            fi

            mkdir -p "${run_dir}"

            TRAIN_CMD="${REPO_ROOT}/.venv/bin/python ${TRAIN_SCRIPT} \
                ${COMMON_TRAIN_ARGS} \
                ${target_args} \
                ${arch_extra_args} \
                --probe-active-from-file ${PROBE_ACTIVE_FROM_JSON} \
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

# ---- Step 6: Submit the simple non-trained baselines ----
# One run per simple_baseline entry — no x-sweep, no model seed, since
# era5_interp is deterministic and identical across all configurations.
# Eval region_specs is "all" to match the trained runs' subset
# breakdown, but the probe-set / always-on partition doesn't affect
# the baseline's predictions — only how the metrics are sliced.
for sb_entry in "${SIMPLE_BASELINE_ENTRIES[@]}"; do
    IFS='|' read -r sb_name sb_kind sb_target_variables <<< "${sb_entry}"
    sb_run_dir="${OUTPUT_ROOT}/${sb_name}_seed42"  # seed42 chosen by convention; baseline is deterministic
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
        --station-split all \
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
echo "Probe set:     ${PROBE_IDS_JSON}"
echo "Results in:    ${OUTPUT_ROOT}/"
echo "Monitor with:  squeue --me"