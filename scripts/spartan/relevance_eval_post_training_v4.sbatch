#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-post-v4
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_POOL_DIR:?}"
: "${TRIP_TRAINING_OUTPUT_DIR:?}"
: "${TRIP_BASELINE_RESULT_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_MILVUS_RPM:?}"
: "${TRIP_BASELINE_ADAPTER:?}"
: "${TRIP_BASELINE_ADAPTER_SHA256:?}"
: "${TRIP_POST_RUN_DIR:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_POOL_DIR}" \
  "${TRIP_TRAINING_OUTPUT_DIR}" "${TRIP_BASELINE_RESULT_DIR}" "${TRIP_RETRIEVAL_ARCHIVE}" \
  "${TRIP_MILVUS_RPM}" "${TRIP_BASELINE_ADAPTER}" "${TRIP_POST_RUN_DIR}" \
  "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_POST_RUN_DIR}"
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
mkdir -p "${TMPDIR}"
cd "${TRIP_PROJECT_ROOT}"
"${TRIP_VENV}/bin/python" scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --expected-implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}"

candidate_adapter="${TRIP_TRAINING_OUTPUT_DIR}/adapter"
candidate_model="${candidate_adapter}/adapter_model.safetensors"
training_summary="${TRIP_TRAINING_OUTPUT_DIR}/run_summary.json"
test -f "${candidate_model}"
test -f "${training_summary}"
candidate_sha256="$(sha256sum "${candidate_model}" | awk '{print $1}')"
recorded_candidate_sha256="$("${TRIP_VENV}/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["adapter_model_sha256"])' \
  "${training_summary}")"
test "${candidate_sha256}" = "${recorded_candidate_sha256}"
mkdir -p "${TRIP_POST_RUN_DIR}/vlm-development" "${TRIP_POST_RUN_DIR}/vlm-final" \
  "${TRIP_POST_RUN_DIR}/service"

candidate_development="${TRIP_POST_RUN_DIR}/vlm-development/targeted_exploration_adapter_v4.jsonl"
"${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence_v4.py \
  --config configs/evaluation/automated_evidence_v4.json \
  --bundle-dir "${TRIP_POOL_DIR}" \
  --split development \
  --role targeted_exploration_adapter_v4 \
  --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --adapter-path "${candidate_adapter}" \
  --adapter-sha256 "${candidate_sha256}" \
  --output "${candidate_development}"

development_selection="${TRIP_POST_RUN_DIR}/vlm-development/selection.json"
"${TRIP_VENV}/bin/python" scripts/score_vlm_exploration_v4.py \
  --config configs/evaluation/automated_evidence_v4.json \
  --split development \
  --result "${TRIP_BASELINE_RESULT_DIR}/zero_shot.jsonl" \
  --result "${TRIP_BASELINE_RESULT_DIR}/old_unified_adapter.jsonl" \
  --result "${TRIP_BASELINE_RESULT_DIR}/current_system_repair_checkpoint_87.jsonl" \
  --result "${candidate_development}" \
  --output "${development_selection}"

development_status="$("${TRIP_VENV}/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fixed_gates"]["status"])' \
  "${development_selection}")"
if [[ "${development_status}" == "PASS" ]]; then
  candidate_final="${TRIP_POST_RUN_DIR}/vlm-final/targeted_exploration_adapter_v4.jsonl"
  final_marker="${TRIP_POST_RUN_DIR}/vlm-final/consumption_marker.json"
  "${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence_v4.py \
    --config configs/evaluation/automated_evidence_v4.json \
    --bundle-dir "${TRIP_POOL_DIR}" \
    --split final \
    --role targeted_exploration_adapter_v4 \
    --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --adapter-path "${candidate_adapter}" \
    --adapter-sha256 "${candidate_sha256}" \
    --selection-record "${development_selection}" \
    --final-marker "${final_marker}" \
    --output "${candidate_final}"
  "${TRIP_VENV}/bin/python" scripts/score_vlm_exploration_v4.py \
    --config configs/evaluation/automated_evidence_v4.json \
    --split final \
    --result "${candidate_final}" \
    --output "${TRIP_POST_RUN_DIR}/vlm-final/metrics.json"
else
  "${TRIP_VENV}/bin/python" - "${TRIP_POST_RUN_DIR}/vlm-final/not_run.json" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
    "final_consumed": False,
    "fresh_test_used": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

"${TRIP_VENV}/bin/python" scripts/run_http_milvus_service_benchmark_v4.py run \
  --config configs/evaluation/automated_evidence_v4.json \
  --bundle-dir "${TRIP_POOL_DIR}" \
  --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
  --milvus-rpm "${TRIP_MILVUS_RPM}" \
  --jobfs "${TRIP_JOB_LOCAL_ROOT}" \
  --baseline-adapter "${TRIP_BASELINE_ADAPTER}" \
  --baseline-adapter-sha256 "${TRIP_BASELINE_ADAPTER_SHA256}" \
  --candidate-adapter "${candidate_adapter}" \
  --candidate-adapter-sha256 "${candidate_sha256}" \
  --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --output "${TRIP_POST_RUN_DIR}/service/summary.json" \
  --raw-output "${TRIP_POST_RUN_DIR}/service/raw.jsonl"

"${TRIP_VENV}/bin/python" - "${TRIP_POST_RUN_DIR}" "${development_status}" \
  "${candidate_sha256}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name != "chain_summary.json":
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        })
payload = {
    "schema_version": "relevance_eval_post_training_chain_v4",
    "status": "COMPLETED",
    "development_gate_status": sys.argv[2],
    "candidate_adapter_model_sha256": sys.argv[3],
    "final_consumed": (root / "vlm-final" / "consumption_marker.json").is_file(),
    "fresh_test_used": False,
    "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    "artifacts": files,
}
(root / "chain_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
