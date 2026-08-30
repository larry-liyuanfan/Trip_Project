#!/usr/bin/env bash
#SBATCH --job-name=trip-w8-ret-lat-v5
#SBATCH --account=punim2936
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-trip-w8-ret-lat-v5-%j.out
#SBATCH --error=slurm-trip-w8-ret-lat-v5-%j.err

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
unset MILVUS_URI

config="configs/week8/retrieval_latency_v5.json"
resource_usage="${TRIP_RETRIEVAL_DEVELOPMENT_DIR}.process_resource_usage.txt"
mkdir -p "$(dirname "$resource_usage")"

python -m unittest tests.test_week8_retrieval -v
python scripts/manage_week8_retrieval.py build-lock \
  --config "$config" \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --source-project-root "$TRIP_SOURCE_PROJECT_ROOT" \
  --output-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py validate-lock \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR"
python scripts/manage_week8_retrieval.py load-milvus-index \
  --config "$config" \
  --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
  --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR"
/usr/bin/time -v -o "$resource_usage" \
  python scripts/manage_week8_retrieval.py evaluate-latency-development \
    --config "$config" \
    --retrieval-dir "$TRIP_RETRIEVAL_DIR" \
    --lock-dir "$TRIP_RETRIEVAL_LOCK_DIR" \
    --output-dir "$TRIP_RETRIEVAL_DEVELOPMENT_DIR" \
    --retrieval-backend milvus
