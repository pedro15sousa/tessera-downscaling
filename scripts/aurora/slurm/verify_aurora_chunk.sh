#!/bin/bash
# Structural verification of one chunk's latents on CSD3 and the chunk's
# "done" marker (CPU job; submitted by submit_aurora_latents.sh --mode verify).
#
# Runs scripts/aurora/verify_aurora_latents.py over the chunk's valid window
# (coverage per lead/region, metadata vs the dataset grids, spot reloads).
# The physical drift check is skipped here on purpose: the original forecasts
# are not on RDS, and the verifier fails a drift comparison that has no
# originals, so --verify-root points at a path that does not exist; the
# workstation runs the drift check after each pull. On success it writes
#   <RDS_ROOT>/<marker_dir>/<chunk>.verify.txt   the verifier's output
#   <RDS_ROOT>/<marker_dir>/<chunk>.sha256       sha256 of every file of the chunk,
#                                               paths relative to RDS_ROOT (for the
#                                               workstation's `sha256sum -c` after rsync)
#   <RDS_ROOT>/<marker_dir>/<chunk>.done         the marker the workstation polls for
#
# Env from the submitter: CHUNK, RDS_ROOT.

set -euo pipefail

# sbatch runs a COPY of this script from Slurm's spool directory, so
# BASH_SOURCE cannot locate the repo: take the submitter's exported REPO_ROOT,
# then the submission directory (the submitter cd's to the repo root).
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
# shellcheck source=scripts/aurora/_csd3.sh
source "${REPO_ROOT}/scripts/aurora/_csd3.sh"
csd3_require_env
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

: "${CHUNK:?CHUNK is required}"
eval "$(python3 scripts/aurora/chunks.py show "${CHUNK}")"
OUTPUT_ROOT="${RDS_ROOT}/ingest/aurora"
VERIFY_ROOT="${RDS_ROOT}/ingest/aurora_verify"
marker_dir="${RDS_ROOT}/${MARKER_DIR}"
mkdir -p "${marker_dir}"
rm -f "${marker_dir}/${CHUNK}.done"

echo "[$(date)] verify chunk=${CHUNK} (${VALID_START} .. ${VALID_END}) host=$(hostname) job=${SLURM_JOB_ID:-?}"

set +e
"${PY}" scripts/aurora/verify_aurora_latents.py \
    --output-root "${OUTPUT_ROOT}" \
    --global-dataset "${RDS_ROOT}/datasets/dataset_timestamp_global" \
    --leads ${LEADS} --regions ${REGIONS} --split all \
    --start-date "${VALID_START}" --end-date "${VALID_END}" \
    --spot-check 20 \
    --verify-root "${RDS_ROOT}/ingest/NO_DRIFT_CHECK_ON_CSD3" \
    | tee "${marker_dir}/${CHUNK}.verify.txt"
status=${PIPESTATUS[0]}
set -e
if [ "${status}" -ne 0 ]; then
    echo "*** structural verification FAILED for chunk ${CHUNK}; no marker written"
    exit 1
fi

# ---- checksum manifest of the chunk's files (relative to RDS_ROOT) ----
echo "[$(date)] hashing the chunk's files"
list="${marker_dir}/${CHUNK}.files"
: > "${list}"

# The three per-directory sidecars. printf with one '%s/%s\n' and four
# arguments would reuse the format and emit "latent_lats.npy/latent_lons.npy",
# a path that exists nowhere, so loop instead.
sidecars() {
    local d="$1" f
    for f in latent_meta.json latent_lats.npy latent_lons.npy; do
        printf '%s/%s\n' "${d}" "${f}"
    done
}

for region in ${REGIONS}; do
    for lead in ${LEADS}; do
        d="ingest/aurora/lead${lead}h/${region}/latent_backbone"
        python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind backbone | sed "s|^|${d}/|; s|$|.npy|" >> "${list}"
        sidecars "${d}" >> "${list}"
    done
    d="ingest/aurora/lead0h/${region}/latent_encoder"
    python3 scripts/aurora/chunks.py stems "${CHUNK}" --kind encoder | sed "s|^|${d}/|; s|$|.npy|" >> "${list}"
    sidecars "${d}" >> "${list}"
done
echo "ingest/aurora/latent_calibration.json" >> "${list}"
# Decoded-field subset of the chunk (valid stems are lexicographically ordered).
if [ -d "${VERIFY_ROOT}" ]; then
    (cd "${RDS_ROOT}" && find ingest/aurora_verify -type f -name '*.nc' | sort) | awk -v lo="${VALID_START}" -v hi="${VALID_END}" '
        { n = split($0, p, "/"); stem = substr(p[n], 1, 13); if (stem >= lo && stem <= hi) print }' >> "${list}"
fi
# Only files that exist go into the manifest (the verifier already proved the
# latent set complete; the physical subset is sparse by design). stat exits
# non-zero on a missing file and xargs then exits 123, which under
# `set -o pipefail` would fail the whole job, so swallow that status here --
# the count printed below is what tells us whether anything is missing.
(cd "${RDS_ROOT}" && { xargs -a "${list}" -d '\n' -r stat -c '%n' 2>/dev/null || true; } | sort -u) > "${list}.present"
n_listed=$(wc -l < "${list}")
n_present=$(wc -l < "${list}.present")
(cd "${RDS_ROOT}" && xargs -a "${list}.present" -d '\n' -r -P "${SLURM_CPUS_PER_TASK:-4}" -n 64 sha256sum) | sort -k2 > "${marker_dir}/${CHUNK}.sha256"
rm -f "${list}" "${list}.present"
echo "manifest: ${n_present}/${n_listed} listed files present, $(wc -l < "${marker_dir}/${CHUNK}.sha256") hashed"

{
    echo "chunk=${CHUNK}"
    echo "valid_window=${VALID_START}..${VALID_END}"
    echo "verified=$(date -u +%FT%TZ)"
    echo "job=${SLURM_JOB_ID:-?}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "manifest=${MARKER_DIR}/${CHUNK}.sha256"
    echo "files=$(wc -l < "${marker_dir}/${CHUNK}.sha256")"
} > "${marker_dir}/${CHUNK}.done.tmp"
mv "${marker_dir}/${CHUNK}.done.tmp" "${marker_dir}/${CHUNK}.done"
echo "[$(date)] chunk ${CHUNK} verified: ${marker_dir}/${CHUNK}.done"
