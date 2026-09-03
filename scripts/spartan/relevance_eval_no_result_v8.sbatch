#!/usr/bin/env bash
#SBATCH --job-name=trip-noresult-v8
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_PRIOR_V4_POOL_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_OUTPUT_DIR:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" \
  "${TRIP_PRIOR_V4_POOL_DIR}" "${TRIP_RETRIEVAL_ARCHIVE}" "${TRIP_OUTPUT_DIR}" \
  "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes task root: ${path}" >&2; exit 2;; esac
done
test -f "${TRIP_PRIOR_V4_POOL_DIR}/search_training_manifest.jsonl"
test ! -e "${TRIP_OUTPUT_DIR}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
export HOME="${TRIP_CACHE_ROOT}/home"
export HF_HOME="${TRIP_CACHE_ROOT}/huggingface"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
TRIP_JOB_LOCAL_ROOT="${JOBFS:-/tmp}"
export TMPDIR="${TRIP_JOB_LOCAL_ROOT}/trip-${USER}-${SLURM_JOB_ID}-tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"
cd "${TRIP_PROJECT_ROOT}"

"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"

pool_dir="${TRIP_JOB_LOCAL_ROOT}/trip-no-result-v8-${SLURM_JOB_ID}-pool"
"${TRIP_VENV}/bin/python" scripts/build_no_result_stress_pool_v8.py \
  --output-dir "${pool_dir}" \
  --prior-v4-lock configs/evaluation/evidence_enhancement/exploration_pool_lock_v4.json \
  --expected-lock configs/evaluation/evidence_enhancement/no_result_stress_pool_lock_v8.json

"${TRIP_VENV}/bin/python" scripts/run_no_result_stress_v8.py \
  --config configs/evaluation/automated_evidence_v8_no_result.json \
  --bundle-dir "${pool_dir}" \
  --prior-v4-bundle-dir "${TRIP_PRIOR_V4_POOL_DIR}" \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --output-dir "${TRIP_OUTPUT_DIR}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"
