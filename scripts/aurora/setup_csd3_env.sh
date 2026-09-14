#!/bin/bash
# Idempotent set-up of the ROCm Python environment the Aurora latent extraction
# uses on CSD3's MI355X nodes (partition mi355x, account zea-p005-zenith-gpu).
#
# The GPU nodes carry only the amdgpu kernel driver (no /opt/rocm, no rocm-smi),
# so every ROCm user-space library comes bundled in PyTorch's ROCm wheels. The
# project's pyproject pins torch to the CUDA index, so this env is NOT the
# project's .venv: it is a plain venv under the RDS data root, populated with
# `uv pip` (project config ignored) and a ROCm torch build; every job script
# under scripts/aurora runs `${ENV_DIR}/bin/python` directly, never `uv run`.
#
# Layout (all under RDS_ROOT, default /rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling):
#   env/aurora-rocm/   the venv (python 3.12 + torch ROCm + microsoft-aurora + this repo, editable)
#   env/uv-cache/      uv's download cache      (UV_CACHE_DIR)
#   env/uv-python/     uv-managed CPython 3.12  (UV_PYTHON_INSTALL_DIR)
#   env/hf-cache/      HuggingFace cache with both Aurora checkpoints prefetched (HF_HOME)
#
# Usage (login node; safe to re-run, every step skips what is already done):
#   bash scripts/aurora/setup_csd3_env.sh              # build / refresh the env, prefetch checkpoints
#   bash scripts/aurora/setup_csd3_env.sh --gpu-check  # + a short interactive GPU job that runs the
#                                                      #   small Aurora model on random data (ROCm kernels)
#   bash scripts/aurora/setup_csd3_env.sh --tests      # + run the repo's unit tests inside the env
# Knobs (env): RDS_ROOT, ENV_DIR, TORCH_BACKEND (default rocm7.0), TORCH_OVERRIDES (default
#   "torch==2.10.0 torchvision==0.25.0"), GPU_ACCOUNT / GPU_PARTITION for --gpu-check. See _csd3.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/aurora/_csd3.sh
source "${REPO_ROOT}/scripts/aurora/_csd3.sh"

# The project pins torch==2.9.0, which PyTorch ships for ROCm only as a 6.4
# wheel -- and the ROCm 6.4 runtime rejects the MI355X (gfx950: "Unsupported
# HSA device"). The env therefore overrides the pin with the first torch
# published on the rocm7.0 index, and the matching torchvision (timm needs
# it). `--torch-backend` routes only the torch-family packages to that index;
# everything else comes from PyPI at the project's pins.
TORCH_BACKEND="${TORCH_BACKEND:-rocm7.0}"
TORCH_OVERRIDES="${TORCH_OVERRIDES:-torch==2.10.0 torchvision==0.25.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

DO_GPU_CHECK=0
DO_TESTS=0
for arg in "$@"; do
    case "${arg}" in
        --gpu-check) DO_GPU_CHECK=1 ;;
        --tests) DO_TESTS=1 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

mkdir -p "${RDS_ROOT}/env" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${HF_HOME}"
csd3_load_uv

echo "== python ${PYTHON_VERSION} (uv-managed, ${UV_PYTHON_INSTALL_DIR})"
uv python install "${PYTHON_VERSION}" --no-config

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "== creating venv ${ENV_DIR}"
    uv venv --no-config --python "${PYTHON_VERSION}" "${ENV_DIR}"
else
    echo "== venv present: ${ENV_DIR}"
fi
PY="${ENV_DIR}/bin/python"

# The project (editable) with the aurora + ingest extras, plus pytest for
# --tests. `--no-config` ignores the repo's pyproject [tool.uv] (CUDA index)
# and any user config; the override file replaces the torch pins wherever
# they appear (pyproject, microsoft-aurora, timm).
echo "== project (editable) + extras aurora, ingest; torch backend ${TORCH_BACKEND}, overrides: ${TORCH_OVERRIDES}"
printf '%s\n' ${TORCH_OVERRIDES} > "${ENV_DIR}/torch-override.txt"
uv pip install --no-config --python "${PY}" \
    --torch-backend "${TORCH_BACKEND}" --override "${ENV_DIR}/torch-override.txt" \
    -e "${REPO_ROOT}[aurora,ingest]" "pytest>=8.3.5"

echo "== sanity (login node, CPU)"
"${PY}" - <<'EOF'
import importlib.metadata as md
import torch
import torchvision
print(f"torch {torch.__version__}  hip={torch.version.hip}  cuda_build={torch.version.cuda}  torchvision {torchvision.__version__}")
assert torch.version.hip, "torch is not a ROCm build -- check TORCH_BACKEND / TORCH_OVERRIDES"
assert "rocm" in torchvision.__version__, "torchvision is not a ROCm build (must match torch)"
for pkg in ("microsoft-aurora", "timm", "xarray", "gcsfs", "tessera-downscaling"):
    print(f"{pkg:20s} {md.version(pkg)}")
import aurora  # noqa: F401  (imports the package; no model load)
from tessera_downscaling.preprocessing import aurora_latent  # noqa: F401
print("imports ok")
EOF

# Both models pin a HuggingFace revision of the checkpoint; fetch exactly that
# so the GPU jobs can run with HF_HUB_OFFLINE=1.
echo "== Aurora checkpoints -> ${HF_HOME}"
"${PY}" - <<'EOF'
from aurora import AuroraPretrained, AuroraSmallPretrained
from huggingface_hub import hf_hub_download
for cls in (AuroraPretrained, AuroraSmallPretrained):
    p = hf_hub_download(
        cls.default_checkpoint_repo, cls.default_checkpoint_name,
        revision=cls.default_checkpoint_revision,
    )
    print(f"  {cls.__name__}: {cls.default_checkpoint_name} @ {cls.default_checkpoint_revision[:12]} -> {p}")
EOF

if [ "${DO_TESTS}" = "1" ]; then
    echo "== unit tests"
    (cd "${REPO_ROOT}" && "${PY}" -m pytest -q tests)
fi

if [ "${DO_GPU_CHECK}" = "1" ]; then
    echo "== GPU check (interactive job on ${GPU_PARTITION}, QOS ${GPU_QOS_INTERACTIVE})"
    srun -A "${GPU_ACCOUNT}" -p "${GPU_PARTITION}" --qos "${GPU_QOS_INTERACTIVE}" \
        -N 1 -n 1 --gres=gpu:1 -c 8 -t 00:15:00 --job-name=aurora_gpu_check \
        --export=ALL,HF_HOME="${HF_HOME}",HF_HUB_OFFLINE=1 \
        "${PY}" "${REPO_ROOT}/scripts/aurora/_gpu_check.py"
fi

echo "== done: ${ENV_DIR}"
