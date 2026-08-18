#!/bin/bash
# Run downscaling experiments: baseline and TESSERA variants.
#
# All experiments use the same TESSERA-filtered station set. The baseline
# does not load patches (just filters stations). TESSERA variants read
# patches on-the-fly via mmap and encode in chunks on the GPU.
#
# Multi-seed: each experiment is run with multiple seeds for variance
# estimation. Results can be compared across seeds to assess stability.
#
# Usage:
#   bash projects/tessera_downscaling/scripts/run_experiments.sh

set -euo pipefail

# ---- Configuration ----
DATASET_DIR=".tmp_output/dataset_daily"
# TESSERA_PATH=".tmp_output/processed/tessera/point_embeddings_2024.npy"
TESSERA_PATH=".tmp_output/processed/tessera/patch16_embeddings_2024.npy"
TESSERA_CSV=".tmp_output/processed/tessera/station_list_filtered.csv"
OUTPUT_ROOT=".tmp_output/training_runs"

SEEDS=(42 123 456)

# Shared hyperparameters.
COMMON_ARGS="
    --dataset-dir ${DATASET_DIR}
    --tessera-path ${TESSERA_PATH}
    --tessera-station-csv ${TESSERA_CSV}
    --batch-size 1
    --epochs 100
    --patience 10
    --lr 2.5e-5
    --cnn-hidden 128
    --cnn-layers 7
    --mlp-hidden 128
    --mlp-n-hidden 3
    --num-workers 0
"

TRAIN_SCRIPT="projects/tessera_downscaling/scripts/train.py"
EVAL_SCRIPT="projects/tessera_downscaling/scripts/evaluate.py"

# ---- Experiment definitions ----
# Each entry: "name|extra_args"
EXPERIMENTS=(
    # "tessera_cnn_dim64|--tessera-method cnn --tessera-output-dim 64"
    # "baseline_tessera_stations|"
    # "tessera_meanpool|--tessera-method meanpool"
    # "tessera_patch16_linear_dim16|--tessera-method linear --tessera-output-dim 16"
    # "tessera_patch16_cnn_dim64|--tessera-method cnn --tessera-output-dim 64"
    # "tessera_patch16_cnn_dim8|--tessera-method cnn --tessera-output-dim 8"
    # "tessera_point_linear_dim16|--tessera-method linear --tessera-output-dim 16"
    "tessera_patch16_cnn_bn_dim16|--tessera-method cnn --tessera-output-dim 16"
)

# ---- Run all experiments x seeds ----
for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name extra_args <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"

        echo ""
        echo "============================================"
        echo "Experiment: ${name} (seed=${seed})"
        echo "============================================"

        # Skip if already completed.
        if [ -f "${run_dir}/test_summary.json" ]; then
            echo "  Already complete, skipping."
            continue
        fi

        uv run --group core python ${TRAIN_SCRIPT} ${COMMON_ARGS} \
            ${extra_args} \
            --seed ${seed} \
            --output-dir ${run_dir}
    done
done

# ---- Run detailed evaluations ----
echo ""
echo "============================================"
echo "Running detailed evaluations..."
echo "============================================"

for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name extra_args <<< "${experiment}"

    for seed in "${SEEDS[@]}"; do
        run_dir="${OUTPUT_ROOT}/${name}_seed${seed}"

        if [ -f "${run_dir}/best_model.pt" ] && [ ! -f "${run_dir}/test_results.json" ]; then
            echo ""
            echo "--- Evaluating: ${name} (seed=${seed}) ---"
            uv run --group core python ${EVAL_SCRIPT} \
                --checkpoint ${run_dir}/best_model.pt
        fi
    done
done

echo ""
echo "============================================"
echo "All experiments complete."
echo "============================================"
echo "Results saved under: ${OUTPUT_ROOT}/"
ls -d ${OUTPUT_ROOT}/*/ 2>/dev/null || echo "  (no results yet)"