#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-vlm-v3
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_VLM_ROLE:?}"
: "${TRIP_OUTPUT_FILE:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" \
  "${TRIP_POOL_DIR}" "${TRIP_OUTPUT_FILE}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test -f "${TRIP_POOL_DIR}/vlm_manifest_weak_v3.jsonl"
test ! -e "${TRIP_OUTPUT_FILE}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export TMPDIR="${TRIP_CACHE_ROOT}/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
command=(
  "${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence.py
  --manifest "${TRIP_POOL_DIR}/vlm_manifest_weak_v3.jsonl"
  --asset-dir "${TRIP_POOL_DIR}"
  --role "${TRIP_VLM_ROLE}"
  --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
  --output "${TRIP_OUTPUT_FILE}"
  --max-new-tokens 256
)
if [[ "${TRIP_VLM_ROLE}" != "zero_shot" ]]; then
  : "${TRIP_ADAPTER_PATH:?adapter role requires TRIP_ADAPTER_PATH}"
  : "${TRIP_ADAPTER_SHA256:?adapter role requires TRIP_ADAPTER_SHA256}"
  command+=(--adapter-path "${TRIP_ADAPTER_PATH}" --adapter-sha256 "${TRIP_ADAPTER_SHA256}")
fi
"${command[@]}"
