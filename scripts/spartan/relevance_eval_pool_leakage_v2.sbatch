#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-leak-v2
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_PRIMARY_INDEX_ROOT:?}"
: "${TRIP_COMPARISON_PHOTOS_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_OUTPUT_DIR:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" \
  "${TRIP_RETRIEVAL_ARCHIVE}" "${TRIP_PRIMARY_INDEX_ROOT}" \
  "${TRIP_COMPARISON_PHOTOS_ROOT}" "${TRIP_POOL_DIR}" "${TRIP_OUTPUT_DIR}" \
  "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_POOL_DIR}"
test ! -e "${TRIP_OUTPUT_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export TMPDIR="${TRIP_CACHE_ROOT}/tmp"
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
"${TRIP_VENV}/bin/python" scripts/build_automated_evidence_pool_v2.py \
  --output-dir "${TRIP_POOL_DIR}" \
  --expected-lock configs/evaluation/evidence_enhancement/automated_pool_lock_v2.json
mkdir -p "${TRIP_OUTPUT_DIR}"
"${TRIP_VENV}/bin/python" scripts/audit_retrieval_query_leakage_v2.py \
  --config configs/evaluation/automated_evidence_v2.json \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --primary-index-root "${TRIP_PRIMARY_INDEX_ROOT}" \
  --comparison-photos-root "${TRIP_COMPARISON_PHOTOS_ROOT}" \
  --query-manifest configs/evaluation/evidence_enhancement/query_manifest_v1.jsonl \
  --query-manifest "${TRIP_POOL_DIR}/search_calibration_manifest.jsonl" \
  --query-manifest "${TRIP_POOL_DIR}/search_holdout_manifest.jsonl" \
  --query-manifest "${TRIP_POOL_DIR}/vlm_manifest_weak_v3.jsonl" \
  --registry-output "${TRIP_OUTPUT_DIR}/index_image_sha_registry.jsonl" \
  --report-output "${TRIP_OUTPUT_DIR}/leakage_audit.json"
