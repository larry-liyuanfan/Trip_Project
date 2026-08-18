#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-adapter-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -uo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_CONFIG:?set the versioned Week 6 config}"
: "${TRIP_EVAL_INPUT:?set locked validation JSONL}"
: "${TRIP_ADAPTER_DIR:?set verified adapter directory}"
: "${TRIP_OUTPUT_DIR:?set immutable evaluation output directory}"
: "${TRIP_HF_HOME:?set isolated model cache}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"
: "${TRIP_RUN_ID:?set a unique evaluation run ID}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TRIP_GIT_COMMIT="$(git rev-parse HEAD)"
source "${TRIP_VENV}/bin/activate"

supervisor_dir="${TRIP_OUTPUT_DIR}.supervisor"
mkdir -p "${supervisor_dir}"
control_file="${supervisor_dir}/resume.request"
state_file="${supervisor_dir}/state.json"
resume_flag=()
attempt=0

write_state() {
  local status="$1"
  local child_exit="$2"
  python -c 'import json,os,pathlib,sys,time; path=pathlib.Path(sys.argv[1]); payload={"status":sys.argv[2],"child_exit_code":int(sys.argv[3]),"attempt":int(sys.argv[4]),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"git_commit":os.environ.get("TRIP_GIT_COMMIT"),"updated_at_epoch":time.time()}; temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temp.replace(path)' \
    "${state_file}" "${status}" "${child_exit}" "${attempt}"
}

python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" check-environment || exit 20
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" validate-data \
  --scenario itinerary_planning --input "${TRIP_EVAL_INPUT}" || exit 21

while true; do
  attempt=$((attempt + 1))
  write_state "evaluating" 0
  python scripts/evaluate_week6_adapter.py run \
    --config "${TRIP_CONFIG}" \
    --eval-input "${TRIP_EVAL_INPUT}" \
    --adapter-dir "${TRIP_ADAPTER_DIR}" \
    --output-dir "${TRIP_OUTPUT_DIR}" \
    --max-samples "${TRIP_MAX_SAMPLES:-64}" \
    --max-new-tokens "${TRIP_MAX_NEW_TOKENS:-2048}" \
    --adapter-role "${TRIP_ADAPTER_ROLE:-initial}" \
    "${resume_flag[@]}"
  child_exit=$?
  if (( child_exit == 0 )); then
    write_state "completed" 0
    sync
    exit 0
  fi

  write_state "waiting_for_versioned_resume" "${child_exit}"
  while true; do
    while [[ ! -f "${control_file}" ]]; do
      sleep 15
    done
    requested_commit="$(sed -n '1p' "${control_file}")"
    current_commit="$(git rev-parse HEAD)"
    if [[ "${requested_commit}" != "${current_commit}" ]]; then
      write_state "rejected_resume_commit" "${child_exit}"
      sleep 15
      continue
    fi
    resume_flag=(--resume)
    export TRIP_GIT_COMMIT="${current_commit}"
    mv "${control_file}" "${supervisor_dir}/resume.accepted.attempt-${attempt}"
    break
  done
done
