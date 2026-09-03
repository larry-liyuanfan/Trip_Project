#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-vlm-v4
#SBATCH --account=punim2936
#SBATCH --partition=gpu-a100
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:MIG-1g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_SPLIT:?}"
: "${TRIP_VLM_ROLE:?}"
: "${TRIP_OUTPUT_FILE:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_POOL_DIR}" \
  "${TRIP_OUTPUT_FILE}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
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
  "${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence_v4.py
  --config configs/evaluation/automated_evidence_v4.json
  --bundle-dir "${TRIP_POOL_DIR}"
  --split "${TRIP_SPLIT}"
  --role "${TRIP_VLM_ROLE}"
  --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
  --output "${TRIP_OUTPUT_FILE}"
)
if [[ "${TRIP_VLM_ROLE}" != "zero_shot" ]]; then
  : "${TRIP_ADAPTER_PATH:?}"
  : "${TRIP_ADAPTER_SHA256:?}"
  case "${TRIP_ADAPTER_PATH}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "adapter path escapes project root" >&2; exit 2;; esac
  command+=(--adapter-path "${TRIP_ADAPTER_PATH}" --adapter-sha256 "${TRIP_ADAPTER_SHA256}")
fi
if [[ "${TRIP_SPLIT}" == "final" ]]; then
  : "${TRIP_SELECTION_RECORD:?}"
  : "${TRIP_FINAL_MARKER:?}"
  command+=(--selection-record "${TRIP_SELECTION_RECORD}" --final-marker "${TRIP_FINAL_MARKER}")
fi
"${command[@]}"
