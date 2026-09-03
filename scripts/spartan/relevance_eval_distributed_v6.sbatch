#!/usr/bin/env bash
#SBATCH --job-name=trip-dist-v6
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=publicgpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RUN_DIR:?}"
: "${TRIP_V4_POOL_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_MILVUS_RPM:?}"
: "${TRIP_DEPENDENCY_DIR:?}"
: "${TRIP_V4_ADAPTER:?}"
: "${TRIP_V4_ADAPTER_SHA256:?}"
: "${TRIP_V5_ADAPTER:?}"
: "${TRIP_V5_ADAPTER_SHA256:?}"
: "${TRIP_V5_QUALITY_SELECTION:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_RUN_DIR}" \
  "${TRIP_V4_POOL_DIR}" "${TRIP_RETRIEVAL_ARCHIVE}" "${TRIP_MILVUS_RPM}" \
  "${TRIP_DEPENDENCY_DIR}" "${TRIP_V4_ADAPTER}" "${TRIP_V5_ADAPTER}" \
  "${TRIP_V5_QUALITY_SELECTION}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_RUN_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"

"${TRIP_VENV}/bin/python" scripts/run_distributed_milvus_cluster_v6.py run \
  --config configs/evaluation/automated_evidence_v6_distributed.json \
  --bundle-dir "${TRIP_V4_POOL_DIR}" \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --milvus-rpm "${TRIP_MILVUS_RPM}" \
  --dependency-dir "${TRIP_DEPENDENCY_DIR}" \
  --baseline-adapter "${TRIP_V4_ADAPTER}" \
  --baseline-adapter-sha256 "${TRIP_V4_ADAPTER_SHA256}" \
  --candidate-adapter "${TRIP_V5_ADAPTER}" \
  --candidate-adapter-sha256 "${TRIP_V5_ADAPTER_SHA256}" \
  --quality-selection "${TRIP_V5_QUALITY_SELECTION}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --output-dir "${TRIP_RUN_DIR}"
