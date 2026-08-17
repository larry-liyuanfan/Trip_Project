#!/usr/bin/env bash
#SBATCH --job-name=trip-w6-fix-vision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
: "${TRIP_DEPLOY_ROOT:?set isolated Trip deployment root}"
: "${TRIP_PROJECT_ROOT:?set Trip repository root}"
: "${TRIP_VENV:?set the existing Week 6 environment}"

case "${TRIP_PROJECT_ROOT}" in
  "${TRIP_DEPLOY_ROOT}"/*) ;;
  *) echo "TRIP_PROJECT_ROOT must be below TRIP_DEPLOY_ROOT" >&2; exit 2 ;;
esac
case "${TRIP_VENV}" in
  "${TRIP_DEPLOY_ROOT}"/envs/*) ;;
  *) echo "TRIP_VENV must be below TRIP_DEPLOY_ROOT/envs" >&2; exit 2 ;;
esac
test -x "${TRIP_VENV}/bin/python"

cd "${TRIP_PROJECT_ROOT}"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PIP_NO_CACHE_DIR=1
source "${TRIP_VENV}/bin/activate"
python - <<'PY'
import importlib.util
import torch

if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
    raise SystemExit("refusing to repair an unexpected torch/CUDA environment")
if importlib.util.find_spec("torchvision") is not None:
    raise SystemExit("refusing to overwrite an existing torchvision installation")
PY
python -m pip install 'torchvision==0.23.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c 'import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__)'
sync
