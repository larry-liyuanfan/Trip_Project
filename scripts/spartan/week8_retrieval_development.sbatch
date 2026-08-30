#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-retrieval-dev
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-trip-w8-retrieval-dev-%j.out
#SBATCH --error=slurm-trip-w8-retrieval-dev-%j.err

set -euo pipefail

: "${TRIP_PROJECT_ROOT:?TRIP_PROJECT_ROOT is required}"
: "${TRIP_SOURCE_PROJECT_ROOT:?TRIP_SOURCE_PROJECT_ROOT is required}"
: "${TRIP_RETRIEVAL_DIR:?TRIP_RETRIEVAL_DIR is required}"
: "${TRIP_RETRIEVAL_LOCK_DIR:?TRIP_RETRIEVAL_LOCK_DIR is required}"
: "${TRIP_RETRIEVAL_DEVELOPMENT_DIR:?TRIP_RETRIEVAL_DEVELOPMENT_DIR is required}"
: "${TRIP_VENV:?TRIP_VENV is required}"

cd "$TRIP_PROJECT_ROOT"
module purge
module load GCCcore/11.3.0
module load Python/3.11.3
source "$TRIP_VENV/bin/activate"

python -m unittest tests.test_week8_retrieval -v
python scripts/manage_week8_retrieval.py build-lock \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --source-project-root "$TRIP_SOURCE_PROJECT_ROOT" \
  --output-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py validate-lock \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py load-milvus-index \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py evaluate-development \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR" \
  --output-dir "$TRIP_RETRIEVAL_DEVELOPMENT_DIR" \
  --retrieval-backend auto
