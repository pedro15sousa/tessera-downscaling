# Shared CSD3 settings for the Aurora latent extraction (sourced, not run, by
# setup_csd3_env.sh, submit_aurora_latents.sh and the job scripts).
#
# Everything on the cluster lives under RDS_ROOT, laid out like the project's
# data root (DATA.md), so the extractor/verifier defaults apply unchanged:
#   datasets/dataset_timestamp_global/   metadata.json + regions/*/{lats,lons}.npy  (scp'd from the workstation)
#   ingest/processed/era5_static/        era5_static_0p25_all.nc                     (scp'd from the workstation)
#   ingest/aurora_inputs/                13-level ERA5 staging, transient per chunk  (staged here from WB2)
#   ingest/aurora/                       latents + latent_calibration.json + chunks/<id>.done markers
#   ingest/aurora_verify/                decoded-field subset for the workstation's drift check
#   env/                                 the ROCm venv, uv + HF caches (setup_csd3_env.sh)
# Override any variable in the environment before calling a script.

RDS_ROOT="${RDS_ROOT:-/rds/project/rds-FIX1FzgwKT0/pmms2/weather-downscaling}"
ENV_DIR="${ENV_DIR:-${RDS_ROOT}/env/aurora-rocm}"
PY="${ENV_DIR}/bin/python"

export TESSERA_DATA_ROOT="${RDS_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${RDS_ROOT}/env/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${RDS_ROOT}/env/uv-python}"
export HF_HOME="${HF_HOME:-${RDS_ROOT}/env/hf-cache}"

# Slurm: the project's accounts are bound to one partition each
# (sacctmgr show assoc user=$USER): GPU work on the MI355X nodes, CPU work
# (staging download, verification) on the desktop9 nodes (4 cores, 64 GB).
GPU_ACCOUNT="${GPU_ACCOUNT:-zea-p005-zenith-gpu}"
GPU_PARTITION="${GPU_PARTITION:-mi355x}"
GPU_QOS="${GPU_QOS:-gpu1}"                 # 36 h wall, 64 GPUs per user
GPU_QOS_INTERACTIVE="${GPU_QOS_INTERACTIVE:-intr}"  # 1 h, one job at a time
CPU_ACCOUNT="${CPU_ACCOUNT:-zea-p005-cpu}"
CPU_PARTITION="${CPU_PARTITION:-desktop9}"
CPU_QOS="${CPU_QOS:-cpu1}"                 # 36 h wall

UV_MODULE="${UV_MODULE:-ceuadmin/uv/0.12.5}"

# uv is not on the default PATH of the login nodes; it is provided as a module.
csd3_load_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
    module load "${UV_MODULE}"
    command -v uv >/dev/null 2>&1 || { echo "uv not found (module ${UV_MODULE})" >&2; return 1; }
}

csd3_require_env() {
    if [ ! -x "${PY}" ]; then
        echo "ROCm env missing at ${ENV_DIR}; run: bash scripts/aurora/setup_csd3_env.sh" >&2
        return 1
    fi
}
