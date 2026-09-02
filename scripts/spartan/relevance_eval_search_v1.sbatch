#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-search
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_QUERY_ASSET_DIR:?}"
: "${TRIP_OUTPUT_FILE:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_OUTPUT_FILE}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test -x "${TRIP_VENV}/bin/python"
test ! -e "${TRIP_OUTPUT_FILE}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export TMPDIR="${TRIP_CACHE_ROOT}/tmp"
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
"${TRIP_VENV}/bin/python" scripts/run_relevance_evidence.py run-search \
  --asset-dir "${TRIP_QUERY_ASSET_DIR}" \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --output "${TRIP_OUTPUT_FILE}"
