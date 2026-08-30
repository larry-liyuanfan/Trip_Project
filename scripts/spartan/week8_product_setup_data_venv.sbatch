#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-data-env
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-trip-w8-data-env-%j.out
#SBATCH --error=slurm-trip-w8-data-env-%j.err

set -euo pipefail

: "${TRIP_DEPLOY_ROOT:?TRIP_DEPLOY_ROOT is required}"
: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_DATA_VENV:?TRIP_DATA_VENV is required}"

case "$TRIP_DATA_VENV" in
  "$TRIP_DEPLOY_ROOT"/envs/*) ;;
  *) echo "TRIP_DATA_VENV must be below TRIP_DEPLOY_ROOT/envs" >&2; exit 2 ;;
esac
if [[ -e "$TRIP_DATA_VENV" ]]; then
  echo "refusing to mutate an existing environment: $TRIP_DATA_VENV" >&2
  exit 2
fi

module purge
module load GCCcore/11.3.0
module load Python/3.11.3
export PIP_CACHE_DIR="$TRIP_DEPLOY_ROOT/pip-cache"
export XDG_CACHE_HOME="$TRIP_DEPLOY_ROOT/xdg-cache"
export TMPDIR="$TRIP_DEPLOY_ROOT/tmp"
mkdir -p "$TRIP_DEPLOY_ROOT/envs" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

python -m venv "$TRIP_DATA_VENV"
"$TRIP_DATA_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$TRIP_DATA_VENV/bin/python" -m pip install -r "$TRIP_PROJECT_ROOT/requirements-data.txt"
"$TRIP_DATA_VENV/bin/python" -m pip check
"$TRIP_DATA_VENV/bin/python" -c 'import pyarrow; print(pyarrow.__version__)'
