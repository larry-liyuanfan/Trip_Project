#!/usr/bin/env bash
#SBATCH --job-name=trip-context-v5
#SBATCH --account=punim2936
#SBATCH --partition=gpu-l40s
#SBATCH --qos=publicgpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_VENV:?}"
: "${TRIP_CACHE_ROOT:?}"
: "${TRIP_RUN_DIR:?}"
: "${TRIP_V4_ADAPTER:?}"
: "${TRIP_V4_ADAPTER_SHA256:?}"
: "${TRIP_PERF_POOL_DIR:?}"
: "${TRIP_RETRIEVAL_ARCHIVE:?}"
: "${TRIP_MILVUS_RPM:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_VENV}" "${TRIP_CACHE_ROOT}" "${TRIP_RUN_DIR}" \
  "${TRIP_V4_ADAPTER}" "${TRIP_PERF_POOL_DIR}" "${TRIP_RETRIEVAL_ARCHIVE}" \
  "${TRIP_MILVUS_RPM}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_RUN_DIR}"
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

pool_dir="${TRIP_JOB_LOCAL_ROOT}/trip-context-v5-${SLURM_JOB_ID}-pool"
training_dir="${TRIP_RUN_DIR}/training"
development_dir="${TRIP_RUN_DIR}/vlm-development"
performance_dir="${TRIP_RUN_DIR}/service"
final_dir="${TRIP_RUN_DIR}/vlm-final"

"${TRIP_VENV}/bin/python" scripts/build_context_focus_pool_v5.py \
  --output-dir "${pool_dir}" \
  --expected-lock configs/evaluation/evidence_enhancement/context_focus_pool_lock_v5.json

"${TRIP_VENV}/bin/python" scripts/train_targeted_exploration_adapter_v4.py \
  --config configs/evaluation/automated_evidence_v5.json \
  --bundle-dir "${pool_dir}" \
  --initial-adapter "${TRIP_V4_ADAPTER}" \
  --output-dir "${training_dir}" \
  --implementation-commit "${TRIP_IMPLEMENTATION_COMMIT_SHA}" \
  --confirm-training-lock

candidate_adapter="${training_dir}/adapter"
candidate_sha256="$(sha256sum "${candidate_adapter}/adapter_model.safetensors" | awk '{print $1}')"
recorded_candidate_sha256="$("${TRIP_VENV}/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["adapter_model_sha256"])' \
  "${training_dir}/run_summary.json")"
test "${candidate_sha256}" = "${recorded_candidate_sha256}"
mkdir -p "${development_dir}" "${performance_dir}" "${final_dir}"

baseline_development="${development_dir}/targeted_exploration_adapter_v4.jsonl"
candidate_development="${development_dir}/context_focus_adapter_v5.jsonl"
for specification in \
  "targeted_exploration_adapter_v4|${TRIP_V4_ADAPTER}|${TRIP_V4_ADAPTER_SHA256}|${baseline_development}" \
  "context_focus_adapter_v5|${candidate_adapter}|${candidate_sha256}|${candidate_development}"; do
  IFS='|' read -r role adapter adapter_sha output <<<"${specification}"
  "${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence_v4.py \
    --config configs/evaluation/automated_evidence_v5.json \
    --bundle-dir "${pool_dir}" \
    --split development \
    --role "${role}" \
    --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --adapter-path "${adapter}" \
    --adapter-sha256 "${adapter_sha}" \
    --output "${output}"
done

development_selection="${development_dir}/selection.json"
"${TRIP_VENV}/bin/python" scripts/score_context_focus_v5.py \
  --config configs/evaluation/automated_evidence_v5.json \
  --split development \
  --result "${baseline_development}" \
  --result "${candidate_development}" \
  --output "${development_selection}"

development_status="$("${TRIP_VENV}/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["fixed_gates"]["status"])' \
  "${development_selection}")"
performance_status="NOT_RUN_DEVELOPMENT_GATE_FAILED"
if [[ "${development_status}" == "PASS" ]]; then
  "${TRIP_VENV}/bin/python" scripts/run_http_milvus_service_benchmark_v4.py run \
    --config configs/evaluation/automated_evidence_v5.json \
    --bundle-dir "${TRIP_PERF_POOL_DIR}" \
    --retrieval-archive "${TRIP_RETRIEVAL_ARCHIVE}" \
    --milvus-rpm "${TRIP_MILVUS_RPM}" \
    --jobfs "${TRIP_JOB_LOCAL_ROOT}" \
    --baseline-adapter "${TRIP_V4_ADAPTER}" \
    --baseline-adapter-sha256 "${TRIP_V4_ADAPTER_SHA256}" \
    --candidate-adapter "${candidate_adapter}" \
    --candidate-adapter-sha256 "${candidate_sha256}" \
    --source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
    --output "${performance_dir}/summary.json" \
    --raw-output "${performance_dir}/raw.jsonl"
  performance_status="$("${TRIP_VENV}/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fixed_gates"]["status"])' \
    "${performance_dir}/summary.json")"
fi

if [[ "${development_status}" == "PASS" && "${performance_status}" == "PASS" ]]; then
  candidate_final="${final_dir}/context_focus_adapter_v5.jsonl"
  final_marker="${final_dir}/consumption_marker.json"
  "${TRIP_VENV}/bin/python" scripts/run_vlm_semantic_evidence_v4.py \
    --config configs/evaluation/automated_evidence_v5.json \
    --bundle-dir "${pool_dir}" \
    --split final \
    --role context_focus_adapter_v5 \
    --base-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --adapter-path "${candidate_adapter}" \
    --adapter-sha256 "${candidate_sha256}" \
    --selection-record "${development_selection}" \
    --final-marker "${final_marker}" \
    --output "${candidate_final}"
  "${TRIP_VENV}/bin/python" scripts/score_context_focus_v5.py \
    --config configs/evaluation/automated_evidence_v5.json \
    --split final \
    --result "${candidate_final}" \
    --output "${final_dir}/metrics.json"
else
  "${TRIP_VENV}/bin/python" - "${final_dir}/not_run.json" "${development_status}" \
    "${performance_status}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "NOT_RUN_QUALITY_OR_LATENCY_GATE_FAILED",
    "development_gate_status": sys.argv[2],
    "performance_gate_status": sys.argv[3],
    "final_consumed": False,
    "fresh_test_used": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

"${TRIP_VENV}/bin/python" - "${TRIP_RUN_DIR}" "${development_status}" \
  "${performance_status}" "${candidate_sha256}" <<'PY'
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
    "schema_version": "context_focus_chain_v5",
    "status": "COMPLETED",
    "development_gate_status": sys.argv[2],
    "performance_gate_status": sys.argv[3],
    "candidate_adapter_model_sha256": sys.argv[4],
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
