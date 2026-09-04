#!/usr/bin/env bash
#SBATCH --job-name=trip-leak-v4
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RUN_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_INDEX_OVERLAY_ROOT:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" \
  "${TRIP_RUN_DIR}" "${TRIP_RETRIEVAL_ARCHIVE}" "${TRIP_INDEX_OVERLAY_ROOT}" \
  "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes task root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_RUN_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
TRIP_JOB_LOCAL_ROOT="${JOBFS:-/tmp}"
export TMPDIR="${TRIP_JOB_LOCAL_ROOT}/trip-${USER}-${SLURM_JOB_ID}-tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TMPDIR}" "${TRIP_RUN_DIR}"
cd "${TRIP_PROJECT_ROOT}"

"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"

pool_dir="${TRIP_JOB_LOCAL_ROOT}/trip-leak-v4-${SLURM_JOB_ID}-pool"
"${TRIP_VENV}/bin/python" scripts/build_exploration_pool_v4.py \
  --output-dir "${pool_dir}" \
  --expected-lock configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json

"${TRIP_VENV}/bin/python" scripts/run_retrieval_query_leakage_v4.py \
  --config configs/evaluation/retrieval_query_leakage_v4.json \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --index-overlay-root "${TRIP_INDEX_OVERLAY_ROOT}" \
  --query-pool-dir "${pool_dir}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}" \
  --registry-output "${TRIP_RUN_DIR}/index_image_sha_registry.jsonl" \
  --report-output "${TRIP_RUN_DIR}/leakage_audit.json"

"${TRIP_VENV}/bin/python" scripts/verify_retrieval_query_leakage_evidence_v4.py \
  --config configs/evaluation/retrieval_query_leakage_v4.json \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --index-overlay-root "${TRIP_INDEX_OVERLAY_ROOT}" \
  --query-pool-dir "${pool_dir}" \
  --registry "${TRIP_RUN_DIR}/index_image_sha_registry.jsonl" \
  --report "${TRIP_RUN_DIR}/leakage_audit.json" \
  --expected-job-id "${SLURM_JOB_ID}" \
  --expected-source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}" \
  > "${TRIP_RUN_DIR}/independent_verification.json"
