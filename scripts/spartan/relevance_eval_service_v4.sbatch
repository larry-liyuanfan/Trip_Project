#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-svc-v4
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=gpu_l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_MILVUS_RPM:?}"
: "${TRIP_BASELINE_ADAPTER:?}"
: "${TRIP_BASELINE_ADAPTER_SHA256:?}"
: "${TRIP_CANDIDATE_ADAPTER:?}"
: "${TRIP_CANDIDATE_ADAPTER_SHA256:?}"
: "${TRIP_OUTPUT_FILE:?}"
: "${TRIP_RAW_OUTPUT_FILE:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_POOL_DIR}" \
  "${TRIP_RETRIEVAL_ARCHIVE}" "${TRIP_MILVUS_RPM}" "${TRIP_BASELINE_ADAPTER}" \
  "${TRIP_CANDIDATE_ADAPTER}" "${TRIP_OUTPUT_FILE}" "${TRIP_RAW_OUTPUT_FILE}" \
  "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_OUTPUT_FILE}"
test ! -e "${TRIP_RAW_OUTPUT_FILE}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export TMPDIR="${JOBFS}/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${TMPDIR}"
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
"${TRIP_VENV}/bin/python" scripts/run_http_milvus_service_benchmark_v4.py run \
  --config configs/evaluation/automated_evidence_v4.json \
  --bundle-dir "${TRIP_POOL_DIR}" \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --milvus-rpm "${TRIP_MILVUS_RPM}" \
  --jobfs "${JOBFS}" \
  --baseline-adapter "${TRIP_BASELINE_ADAPTER}" \
  --baseline-adapter-sha256 "${TRIP_BASELINE_ADAPTER_SHA256}" \
  --candidate-adapter "${TRIP_CANDIDATE_ADAPTER}" \
  --candidate-adapter-sha256 "${TRIP_CANDIDATE_ADAPTER_SHA256}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --output "${TRIP_OUTPUT_FILE}" \
  --raw-output "${TRIP_RAW_OUTPUT_FILE}"
