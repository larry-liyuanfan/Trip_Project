#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-final-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -uo pipefail
: "${TRIP_PROJECT_ROOT:?set isolated project root}"
: "${TRIP_VENV:?set the project-scoped Spartan virtual environment}"
: "${TRIP_HF_HOME:?set isolated model cache}"
: "${TRIP_SCENARIO:?set one final evaluation scenario}"
: "${TRIP_ADAPTER_DIR:?set the verified best adapter directory}"
: "${TRIP_EXPECTED_ADAPTER_SHA256:?set the locked adapter model SHA-256}"
: "${TRIP_RUN_ID:?set one immutable final evaluation run ID}"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
runtime_cache="outputs/week6/final_evaluation/runtime/${TRIP_RUN_ID}"
mkdir -p "${runtime_cache}/home" "${runtime_cache}/xdg" "${runtime_cache}/torch/kernels"
export HOME="${runtime_cache}/home"
export XDG_CACHE_HOME="${runtime_cache}/xdg"
export TORCH_HOME="${runtime_cache}/torch"
export PYTORCH_KERNEL_CACHE_PATH="${runtime_cache}/torch/kernels"
export HF_HOME="${TRIP_HF_HOME}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export TRIP_GIT_COMMIT="$(git rev-parse HEAD)"
source "${TRIP_VENV}/bin/activate"

supervisor_dir="outputs/week6/final_evaluation/supervisors/${TRIP_RUN_ID}"
state_file="${supervisor_dir}/state.json"
mkdir -p "${supervisor_dir}"

write_state() {
  local status="$1"
  local child_exit="$2"
  python -c 'import json,os,pathlib,sys,time; path=pathlib.Path(sys.argv[1]); payload={"status":sys.argv[2],"child_exit_code":int(sys.argv[3]),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),"git_commit":os.environ.get("TRIP_GIT_COMMIT"),"scenario":os.environ.get("TRIP_SCENARIO"),"run_id":os.environ.get("TRIP_RUN_ID"),"updated_at_epoch":time.time()}; temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8"); temp.replace(path)' \
    "${state_file}" "${status}" "${child_exit}"
}

python scripts/train_week6_qlora.py \
  --config configs/week6/qwen3_vl_8b_qlora_final300_v4.json check-environment || exit 20

write_state "evaluating" 0
python scripts/run_week6_final_evaluation.py \
  --config configs/evaluation_week6_final.yaml \
  --run-id "${TRIP_RUN_ID}" \
  --scenario "${TRIP_SCENARIO}" \
  --adapter-dir "${TRIP_ADAPTER_DIR}" \
  --expected-adapter-sha256 "${TRIP_EXPECTED_ADAPTER_SHA256}" \
  --prompt-version standardized_v2 \
  --max-input-tokens 8192
child_exit=$?
if (( child_exit != 0 )); then
  write_state "waiting_for_versioned_diagnosis" "${child_exit}"
  while true; do sleep 15; done
fi

python scripts/score_week3_evaluation.py \
  --config configs/evaluation_week6_final.yaml \
  --run-id "${TRIP_RUN_ID}"
child_exit=$?
if (( child_exit != 0 )); then
  write_state "waiting_for_scoring_diagnosis" "${child_exit}"
  while true; do sleep 15; done
fi

write_state "completed" 0
sync

