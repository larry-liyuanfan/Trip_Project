#!/usr/bin/env bash
#SBATCH --job-name=trip-dist-smoke
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RUN_DIR:?}"
: "${TRIP_MILVUS_RPM:?}"
: "${TRIP_DEPENDENCY_DIR:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_RUN_DIR}" \
  "${TRIP_MILVUS_RPM}" "${TRIP_DEPENDENCY_DIR}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_RUN_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export PYTHONDONTWRITEBYTECODE=1
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"

"${TRIP_VENV}/bin/python" scripts/smoke_distributed_milvus_runtime_v6.py \
  --config configs/evaluation/automated_evidence_v6_distributed.json \
  --milvus-rpm "${TRIP_MILVUS_RPM}" \
  --dependency-dir "${TRIP_DEPENDENCY_DIR}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --output-dir "${TRIP_RUN_DIR}"
