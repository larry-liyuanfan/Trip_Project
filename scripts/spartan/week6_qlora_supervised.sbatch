#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-qlora-supervised
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -uo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_CONFIG:?set the versioned Week 6 config}"
: "${TRIP_SCENARIO:?set Week 6 scenario}"
: "${TRIP_TRAIN_INPUT:?set locked training JSONL}"
: "${TRIP_EVAL_INPUT:?set locked validation JSONL}"
: "${TRIP_OUTPUT_DIR:?set a new output directory}"
: "${TRIP_HF_HOME:?set isolated model cache}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"
: "${TRIP_RUN_ID:?set a unique formal run ID}"
: "${TRIP_INIT_ADAPTER:?set the verified source adapter}"

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
attempt=0
walltime_signal=0

write_state() {
  local status="$1"
  local child_exit="$2"
  local checkpoint="$3"
  python -c 'import json,os,pathlib,sys,time; path=pathlib.Path(sys.argv[1]); payload={"status":sys.argv[2],"child_exit_code":int(sys.argv[3]),"latest_checkpoint":sys.argv[4] or None,"attempt":int(sys.argv[5]),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"git_commit":os.environ.get("TRIP_GIT_COMMIT"),"updated_at_epoch":time.time()}; temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temp.replace(path)' \
    "${state_file}" "${status}" "${child_exit}" "${checkpoint}" "${attempt}"
}

on_walltime_signal() {
  walltime_signal=1
  write_state "walltime_signal_received" 0 "$(latest_checkpoint)"
}
trap on_walltime_signal USR1

latest_checkpoint() {
  if [[ ! -d "${TRIP_OUTPUT_DIR}" ]]; then
    return 0
  fi
  find "${TRIP_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -V | tail -n 1 | sed "s#^#${TRIP_OUTPUT_DIR}/#"
}

python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" check-environment || exit 20
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" validate-data \
  --scenario "${TRIP_SCENARIO}" --input "${TRIP_TRAIN_INPUT}" || exit 21
python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" validate-data \
  --scenario "${TRIP_SCENARIO}" --input "${TRIP_EVAL_INPUT}" || exit 22

resume_checkpoint=""
while (( walltime_signal == 0 )); do
  attempt=$((attempt + 1))
  train_args=(
    python scripts/train_week6_qlora.py --config "${TRIP_CONFIG}" train-full
    --scenario "${TRIP_SCENARIO}"
    --train-input "${TRIP_TRAIN_INPUT}"
    --eval-input "${TRIP_EVAL_INPUT}"
    --output-dir "${TRIP_OUTPUT_DIR}"
    --confirm-dataset-lock
  )
  if [[ -n "${resume_checkpoint}" ]]; then
    train_args+=(--resume-from-checkpoint "${resume_checkpoint}")
  else
    train_args+=(--init-adapter "${TRIP_INIT_ADAPTER}")
  fi

  write_state "training" 0 "${resume_checkpoint}"
  "${train_args[@]}"
  child_exit=$?
  if (( child_exit == 0 )); then
    write_state "completed" 0 "$(latest_checkpoint)"
    sync
    exit 0
  fi

  checkpoint="$(latest_checkpoint)"
  write_state "waiting_for_versioned_resume" "${child_exit}" "${checkpoint}"
  while (( walltime_signal == 0 )); do
    if [[ ! -f "${control_file}" ]]; then
      sleep 15
      continue
    fi
    requested_commit="$(sed -n '1p' "${control_file}")"
    requested_checkpoint="$(sed -n '2p' "${control_file}")"
    current_commit="$(git rev-parse HEAD)"
    if [[ "${requested_commit}" != "${current_commit}" ]]; then
      write_state "rejected_resume_commit" "${child_exit}" "${checkpoint}"
      sleep 15
      continue
    fi
    if [[ ! -d "${requested_checkpoint}" ]]; then
      write_state "rejected_resume_checkpoint" "${child_exit}" "${checkpoint}"
      sleep 15
      continue
    fi
    case "$(realpath "${requested_checkpoint}")" in
      "$(realpath "${TRIP_OUTPUT_DIR}")"/checkpoint-*) ;;
      *)
        write_state "rejected_resume_scope" "${child_exit}" "${checkpoint}"
        sleep 15
        continue
        ;;
    esac
    resume_checkpoint="${requested_checkpoint}"
    export TRIP_GIT_COMMIT="${current_commit}"
    mv "${control_file}" "${supervisor_dir}/resume.accepted.attempt-${attempt}"
    break
  done
done

write_state "walltime_exit" 0 "$(latest_checkpoint)"
sync
exit 0
