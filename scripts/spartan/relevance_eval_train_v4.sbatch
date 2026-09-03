#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-train-v4
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_INITIAL_ADAPTER_DIR:?}"
: "${TRIP_OUTPUT_DIR:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_POOL_DIR}" \
  "${TRIP_INITIAL_ADAPTER_DIR}" "${TRIP_OUTPUT_DIR}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test -f "${TRIP_POOL_DIR}/bundle_lock.json"
test ! -e "${TRIP_OUTPUT_DIR}"
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
"${TRIP_VENV}/bin/python" scripts/train_targeted_exploration_adapter_v4.py \
  --config configs/evaluation/automated_evidence_v4.json \
  --bundle-dir "${TRIP_POOL_DIR}" \
  --initial-adapter "${TRIP_INITIAL_ADAPTER_DIR}" \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --confirm-training-lock
