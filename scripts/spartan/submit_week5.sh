#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <benchmark|shards> <account> <h100|a100|l40s> <migration-dir>" >&2
}

[[ $# -eq 4 ]] || { usage; exit 2; }
: "${TRIP_HF_HOME:?export TRIP_HF_HOME to the isolated project model cache before submission}"
mode=$1
account=$2
profile=$3
migration_dir=$4

case "${profile}" in
  h100) partition=gpu-h100; cpus=16; memory=96G ;;
  a100) partition=gpu-a100; cpus=16; memory=96G ;;
  l40s) partition=gpu-l40s; cpus=12; memory=64G ;;
  *) usage; exit 2 ;;
esac

manifest="${migration_dir}/migration_manifest.json"
[[ -f "${manifest}" ]] || { echo "missing migration manifest" >&2; exit 3; }
project_root=$(git rev-parse --show-toplevel)

if [[ "${mode}" == "benchmark" ]]; then
  config=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["benchmark"]["config"])' "${manifest}")
  run_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_ids"]["benchmark"])' "${manifest}")
  sbatch \
    --account="${account}" --partition="${partition}" --time=01:00:00 \
    --cpus-per-task="${cpus}" --mem="${memory}" --job-name=trip-w5-benchmark \
    --export=ALL,TRIP_PROJECT_ROOT="${project_root}",TRIP_CONFIG="${config}",TRIP_RUN_ID="${run_id}" \
    scripts/spartan/week5_job.sbatch
elif [[ "${mode}" == "shards" ]]; then
  shard_count=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["shards"]))' "${manifest}")
  [[ "${shard_count}" -gt 0 ]] || { echo "no shards" >&2; exit 4; }
  sbatch \
    --account="${account}" --partition="${partition}" --time=24:00:00 \
    --cpus-per-task="${cpus}" --mem="${memory}" --array="0-$((shard_count-1))" \
    --job-name=trip-w5-shard \
    --export=ALL,TRIP_PROJECT_ROOT="${project_root}",TRIP_MIGRATION_DIR="${migration_dir}" \
    scripts/spartan/week5_job.sbatch
else
  usage
  exit 2
fi
