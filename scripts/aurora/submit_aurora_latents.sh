#!/bin/bash
# Submit / drive the CSD3 Aurora latent extraction, one chunk at a time.
#
# The workstation issues these over ssh (see scripts/aurora/README_csd3.md):
#   bash scripts/aurora/submit_aurora_latents.sh --chunk <id> --mode <mode> [options]
#
# Modes (chunk ids and windows: scripts/aurora/chunks.json, `chunks.py list`):
#   status     what exists for the chunk: markers, file counts, queued jobs, quota
#   dry-run    the extractor's --dry-run for the chunk, on the login node (no job):
#              inputs present? accounting. Same check the stage job runs at its end.
#   stage      CPU array job: download the chunk's ERA5 staging window from
#              WeatherBench2 (--stage-shards sub-windows, default 4), then a
#              dependent check job that writes <marker_dir>/<id>.staged
#   smoke      GPU job: small model, 3 inits, into ingest/aurora_smoke (separate tree)
#   calibrate  GPU job: pretrained model, --calibrate N (default 50) on the chunk,
#              writes ingest/aurora/latent_calibration.json. Once, on chunk 2010,
#              before any shard.
#   shard      GPU array job: the real extraction, --n-shards contiguous
#              sub-windows (default 4, one MI355X each); needs the calibration file
#   verify     CPU job: structural verifier over the chunk; on success writes the
#              checksum manifest and <marker_dir>/<id>.done
#   pipeline   stage -> check -> [calibrate with --with-calibrate] -> shard -> verify,
#              chained with afterok dependencies; one command per chunk
#   clean      delete the chunk's inputs, latents and verify subset from RDS
#              (keeps what the next chunk shares); requires <id>.done and
#              <id>.pulled (the workstation writes .pulled after its checks)
#
# Options: --n-shards N   --stage-shards N   --calibrate-n N   --after <jobid>
#          --with-calibrate (pipeline)   --force (redo/ignore markers)   --print (show
#          the sbatch commands, submit nothing)
# Every mode is resume-safe: the downloader and extractor skip files already
# written, verify/stage-check rewrite their markers, clean is idempotent.
# Resources, accounts, RDS_ROOT and the env come from scripts/aurora/_csd3.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=scripts/aurora/_csd3.sh
source "${REPO_ROOT}/scripts/aurora/_csd3.sh"
cd "${REPO_ROOT}"

CHUNK=""
MODE=""
N_SHARDS="${N_SHARDS:-4}"
STAGE_SHARDS="${STAGE_SHARDS:-4}"
STAGE_PROCESSES="${STAGE_PROCESSES:-8}"
CALIBRATE_N="${CALIBRATE_N:-50}"
AFTER=""
WITH_CALIBRATE=0
FORCE=0
PRINT=0
# Wall times (QOS caps: gpu1 36 h, cpu1 36 h).
STAGE_TIME="${STAGE_TIME:-12:00:00}"
CHECK_TIME="${CHECK_TIME:-01:00:00}"
SMOKE_TIME="${SMOKE_TIME:-01:00:00}"
CALIBRATE_TIME="${CALIBRATE_TIME:-06:00:00}"
SHARD_TIME="${SHARD_TIME:-24:00:00}"
VERIFY_TIME="${VERIFY_TIME:-04:00:00}"
GPU_CPUS="${GPU_CPUS:-16}"

usage() { sed -n '2,36p' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --chunk) CHUNK="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --n-shards) N_SHARDS="$2"; shift 2 ;;
        --stage-shards) STAGE_SHARDS="$2"; shift 2 ;;
        --calibrate-n) CALIBRATE_N="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --with-calibrate) WITH_CALIBRATE=1; shift ;;
        --force) FORCE=1; shift ;;
        --print|-n) PRINT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ -n "${CHUNK}" ] && [ -n "${MODE}" ] || { usage >&2; exit 2; }

eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}")"
MARKER_ABS="${RDS_ROOT}/${MARKER}"
MARKER_DIR_ABS="${RDS_ROOT}/${MARKER_DIR}"
STAGING_ROOT="${RDS_ROOT}/ingest/aurora_inputs"
OUTPUT_ROOT="${RDS_ROOT}/ingest/aurora"
VERIFY_ROOT="${RDS_ROOT}/ingest/aurora_verify"
CALIBRATION="${OUTPUT_ROOT}/latent_calibration.json"
LOG_DIR="logs/aurora/${CHUNK}"
mkdir -p "${LOG_DIR}"

export CHUNK N_SHARDS STAGE_SHARDS STAGE_PROCESSES CALIBRATE_N RDS_ROOT FORCE

# --------------------------------------------------------------------------- #
# sbatch wrappers: print the command with --print, else submit and echo the id.
# --------------------------------------------------------------------------- #

_submit() {  # _submit <label> <sbatch args...>  -> prints the job id (or a placeholder)
    local label="$1"; shift
    if [ "${PRINT}" = "1" ]; then
        echo "sbatch $*" >&2
        echo "JOBID"
        return 0
    fi
    local id
    id=$(sbatch --parsable "$@")
    id="${id%%;*}"
    echo "submitted ${label}: job ${id}" >&2
    echo "${id}"
}

_dep() {  # dependency flag for an optional job id
    if [ -n "$1" ]; then echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes"; fi
}

_gpu_common() {
    echo "-A ${GPU_ACCOUNT} -p ${GPU_PARTITION} --qos ${GPU_QOS} -N 1 -n 1 --gres=gpu:1 -c ${GPU_CPUS}"
}
_cpu_common() {
    echo "-A ${CPU_ACCOUNT} -p ${CPU_PARTITION} --qos ${CPU_QOS} -N 1 -n 1 -c 4"
}

submit_stage() {  # -> id of the check job
    local dep="$1"
    local arr chk
    # shellcheck disable=SC2046
    arr=$(_submit "stage ${CHUNK} (${STAGE_SHARDS} download tasks)" $(_cpu_common) $(_dep "${dep}") \
        --job-name="aur-stage-${CHUNK}" --time="${STAGE_TIME}" --array="0-$((STAGE_SHARDS - 1))" \
        --output="${LOG_DIR}/stage_%A_%a.out" --error="${LOG_DIR}/stage_%A_%a.err" \
        --export=ALL,STAGE_STEP=download scripts/aurora/slurm/stage_aurora_inputs.sh)
    # shellcheck disable=SC2046
    chk=$(_submit "stage-check ${CHUNK}" $(_cpu_common) $(_dep "${arr}") \
        --job-name="aur-stagechk-${CHUNK}" --time="${CHECK_TIME}" \
        --output="${LOG_DIR}/stage_check_%j.out" --error="${LOG_DIR}/stage_check_%j.err" \
        --export=ALL,STAGE_STEP=check scripts/aurora/slurm/stage_aurora_inputs.sh)
    echo "${chk}"
}

submit_gpu() {  # submit_gpu <extract_mode> <dep> [array]  -> id
    local emode="$1" dep="$2" time_limit extra=()
    case "${emode}" in
        smoke) time_limit="${SMOKE_TIME}" ;;
        calibrate) time_limit="${CALIBRATE_TIME}" ;;
        shard) time_limit="${SHARD_TIME}"; extra=(--array="0-$((N_SHARDS - 1))") ;;
    esac
    local out="${LOG_DIR}/${emode}_%j.out" err="${LOG_DIR}/${emode}_%j.err"
    if [ "${emode}" = "shard" ]; then out="${LOG_DIR}/shard_%A_%a.out"; err="${LOG_DIR}/shard_%A_%a.err"; fi
    # shellcheck disable=SC2046
    _submit "${emode} ${CHUNK}" $(_gpu_common) $(_dep "${dep}") "${extra[@]}" \
        --job-name="aur-${emode}-${CHUNK}" --time="${time_limit}" \
        --output="${out}" --error="${err}" \
        --export=ALL,EXTRACT_MODE="${emode}" scripts/aurora/slurm/extract_aurora_chunk.sh
}

submit_verify() {  # -> id
    local dep="$1"
    # shellcheck disable=SC2046
    _submit "verify ${CHUNK}" $(_cpu_common) $(_dep "${dep}") \
        --job-name="aur-verify-${CHUNK}" --time="${VERIFY_TIME}" \
        --output="${LOG_DIR}/verify_%j.out" --error="${LOG_DIR}/verify_%j.err" \
        --export=ALL scripts/aurora/slurm/verify_aurora_chunk.sh
}

# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

need_inputs() {
    for f in "${RDS_ROOT}/datasets/dataset_timestamp_global/metadata.json" \
             "${RDS_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc"; do
        [ -f "${f}" ] || { echo "missing on RDS: ${f} (scp the inputs tarball to ${RDS_ROOT} first)" >&2; return 1; }
    done
    for r in ${REGIONS}; do
        [ -f "${RDS_ROOT}/datasets/dataset_timestamp_global/regions/${r}/lats.npy" ] || {
            echo "missing on RDS: datasets/dataset_timestamp_global/regions/${r}/lats.npy" >&2; return 1; }
    done
}

refuse_if_done() {
    if [ -f "${MARKER_ABS}" ] && [ "${FORCE}" != "1" ]; then
        echo "chunk ${CHUNK} already has its marker ${MARKER_ABS}; --force to redo" >&2
        exit 1
    fi
}

count_present() {  # count_present <kind> <dir-template with LEAD/REGION>  -> "present/expected"
    local kind="$1" tmpl="$2" n_exp=0 n_ok=0 d
    local stems
    stems=$(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind "${kind}")
    local n_stems
    n_stems=$(echo "${stems}" | wc -l)
    local leads="${LEADS}"
    [ "${kind}" = "encoder" ] && leads="0"
    for region in ${REGIONS}; do
        for lead in ${leads}; do
            d="${tmpl//LEAD/${lead}}"; d="${d//REGION/${region}}"
            n_exp=$((n_exp + n_stems))
            if [ -d "${d}" ]; then
                n_ok=$((n_ok + $(echo "${stems}" | sed "s|^|${d}/|; s|$|.npy|" | xargs -d '\n' -r stat -c %n 2>/dev/null | wc -l)))
            fi
        done
    done
    echo "${n_ok}/${n_exp}"
}

# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

case "${MODE}" in
    status)
        echo "chunk ${CHUNK}: valid ${VALID_START}..${VALID_END}, staging ${STAGING_START}..${STAGING_END}, ${N_SLOTS} slots"
        echo "RDS_ROOT=${RDS_ROOT}"
        for m in staged done pulled cleaned; do
            f="${MARKER_DIR_ABS}/${CHUNK}.${m}"
            if [ -f "${f}" ]; then echo "  [x] ${m}  ($(head -c 200 "${f}" | tr '\n' ' '))"; else echo "  [ ] ${m}"; fi
        done
        [ -f "${CALIBRATION}" ] && echo "  calibration: ${CALIBRATION}" || echo "  calibration: MISSING (run --mode calibrate on chunk $(python3 scripts/aurora/chunks.py list | sed -n 2p | cut -d' ' -f1))"
        echo "  env: $([ -x "${PY}" ] && echo "${ENV_DIR}" || echo 'MISSING (setup_csd3_env.sh)')"
        if [ -d "${STAGING_ROOT}" ]; then
            n_frames=$(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind staging | sed "s|^|${STAGING_ROOT}/era5_wb2_quarter_2m_temperature/data/|; s|$|.nc|" | xargs -d '\n' -r stat -c %n 2>/dev/null | wc -l)
            echo "  staging frames (2m_temperature): ${n_frames}/$(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind staging | wc -l)  problems: $(find "${STAGING_ROOT}" -name '*-problem.txt' 2>/dev/null | wc -l)"
        else
            echo "  staging: none"
        fi
        echo "  backbone files: $(count_present backbone "${OUTPUT_ROOT}/leadLEADh/REGION/latent_backbone")"
        echo "  encoder files : $(count_present encoder "${OUTPUT_ROOT}/lead0h/REGION/latent_encoder")"
        echo "  verify subset : $([ -d "${VERIFY_ROOT}" ] && find "${VERIFY_ROOT}" -name '*.nc' | wc -l || echo 0) files"
        echo "  jobs:"
        squeue --me --name="aur-stage-${CHUNK},aur-stagechk-${CHUNK},aur-smoke-${CHUNK},aur-calibrate-${CHUNK},aur-shard-${CHUNK},aur-verify-${CHUNK}" \
            -o "    %.10i %-22j %.8T %.10M %.6D %R" 2>/dev/null | tail -n +2 || true
        echo "  quota: $(df -h "${RDS_ROOT}" | awk 'NR==2 {print $3" used, "$4" free ("$5")"}')"
        ;;
    dry-run)
        csd3_require_env; need_inputs
        dtype=(); [ -f "${CALIBRATION}" ] || dtype=(--dtype float16)
        # shellcheck disable=SC2086
        "${PY}" scripts/aurora/extract_aurora_latents.py \
            --global-metadata "${RDS_ROOT}/datasets/dataset_timestamp_global/metadata.json" \
            --era5-staging-root "${STAGING_ROOT}" \
            --static-file "${RDS_ROOT}/ingest/processed/era5_static/era5_static_0p25_all.nc" \
            --output-root "${OUTPUT_ROOT}" --regions ${REGIONS} --leads ${LEADS} --split all \
            --start-date "${VALID_START}" --end-date "${VALID_END}" "${dtype[@]}" --dry-run
        ;;
    stage)
        csd3_require_env; need_inputs
        submit_stage "${AFTER}" > /dev/null
        ;;
    smoke)
        csd3_require_env; need_inputs
        submit_gpu smoke "${AFTER}" > /dev/null
        echo "smoke output: ${RDS_ROOT}/ingest/aurora_smoke (separate from the real tree)" >&2
        ;;
    calibrate)
        csd3_require_env; need_inputs
        if [ -f "${CALIBRATION}" ] && [ "${FORCE}" != "1" ]; then
            echo "calibration file exists: ${CALIBRATION}; --force to redo (all shards must then be redone too)" >&2
            exit 1
        fi
        submit_gpu calibrate "${AFTER}" > /dev/null
        ;;
    shard)
        csd3_require_env; need_inputs; refuse_if_done
        if [ ! -f "${CALIBRATION}" ] && [ -z "${AFTER}" ]; then
            echo "no calibration file (${CALIBRATION}); run --mode calibrate first, or --after its job id" >&2
            exit 1
        fi
        submit_gpu shard "${AFTER}" > /dev/null
        ;;
    verify)
        csd3_require_env; need_inputs
        submit_verify "${AFTER}" > /dev/null
        ;;
    pipeline)
        csd3_require_env; need_inputs; refuse_if_done
        if [ ! -f "${CALIBRATION}" ] && [ "${WITH_CALIBRATE}" != "1" ]; then
            echo "no calibration file (${CALIBRATION}); add --with-calibrate (chunk 2010) or calibrate first" >&2
            exit 1
        fi
        dep="${AFTER}"
        if [ -f "${MARKER_DIR_ABS}/${CHUNK}.staged" ] && [ "${FORCE}" != "1" ]; then
            echo "chunk ${CHUNK} already staged (${MARKER_DIR_ABS}/${CHUNK}.staged); skipping the stage jobs" >&2
        else
            dep=$(submit_stage "${dep}")
        fi
        if [ "${WITH_CALIBRATE}" = "1" ]; then
            dep=$(submit_gpu calibrate "${dep}")
        fi
        dep=$(submit_gpu shard "${dep}")
        dep=$(submit_verify "${dep}")
        echo "pipeline for chunk ${CHUNK} submitted; final job ${dep} writes ${MARKER_ABS}" >&2
        ;;
    clean)
        if [ "${FORCE}" != "1" ]; then
            [ -f "${MARKER_ABS}" ] || { echo "no ${MARKER_ABS}; refusing to clean an unverified chunk (--force)" >&2; exit 1; }
            [ -f "${MARKER_DIR_ABS}/${CHUNK}.pulled" ] || { echo "no ${MARKER_DIR_ABS}/${CHUNK}.pulled; the workstation writes it after its checks (--force)" >&2; exit 1; }
        fi
        list=$(mktemp)
        # Staging frames and latents the next chunk does not share; every
        # variable directory and lead/region directory via globs.
        _existing() { local f; for f in "$@"; do [ -e "${f}" ] && printf '%s\n' "${f}"; done; return 0; }
        while read -r s; do
            _existing "${STAGING_ROOT}"/era5_wb2_quarter_*/data/"${s}".nc \
                      "${STAGING_ROOT}"/era5_wb2_quarter_*/data/"${s}".nc-problem.txt
        done < <(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind staging --until-next) >> "${list}"
        while read -r s; do
            _existing "${OUTPUT_ROOT}"/lead*h/*/latent_backbone/"${s}".npy
        done < <(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind backbone) >> "${list}"
        while read -r s; do
            _existing "${OUTPUT_ROOT}"/lead0h/*/latent_encoder/"${s}".npy
        done < <(python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind encoder --until-next) >> "${list}"
        if [ -d "${VERIFY_ROOT}" ]; then
            find "${VERIFY_ROOT}" -type f -name '*.nc' | awk -v lo="${VALID_START}" -v hi="${VALID_END}" '
                { n = split($0, p, "/"); stem = substr(p[n], 1, 13); if (stem >= lo && stem <= hi) print }' >> "${list}"
        fi
        n=$(wc -l < "${list}")
        if [ "${PRINT}" = "1" ]; then
            echo "would delete ${n} files, e.g.:"; head -5 "${list}"; rm -f "${list}"; exit 0
        fi
        echo "deleting ${n} files of chunk ${CHUNK} under ${RDS_ROOT} ..."
        xargs -a "${list}" -d '\n' -r -n 200 rm -f
        rm -f "${list}"
        mkdir -p "${MARKER_DIR_ABS}"
        echo "cleaned=$(date -u +%FT%TZ) files=${n}" > "${MARKER_DIR_ABS}/${CHUNK}.cleaned"
        echo "chunk ${CHUNK} cleaned (${n} files); $(df -h "${RDS_ROOT}" | awk 'NR==2 {print $4" free"}')"
        ;;
    *)
        echo "unknown mode: ${MODE}" >&2; usage >&2; exit 2 ;;
esac
