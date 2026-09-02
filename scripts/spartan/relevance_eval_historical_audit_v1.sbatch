#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-audit
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
: "${TRIP_SPARTAN_BASE:?}"
: "${TRIP_PROJECT_ROOT:?}"
: "${TRIP_HISTORICAL_ROOT:?}"
: "${TRIP_OUTPUT_FILE:?}"
: "${TRIP_SOURCE_MANIFEST:?}"
: "${TRIP_SOURCE_SNAPSHOT_SHA256:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_HISTORICAL_ROOT}" "${TRIP_OUTPUT_FILE}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes Iris project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_OUTPUT_FILE}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
base="${TRIP_HISTORICAL_ROOT}/outputs/system_repair"
comparison="${base}/development_comparison/system_repair_development_comparison_20260825_v4"
candidate="${base}/training/system_repair_continuation_sft_20260825_v4/development_evaluations/step-000087"
dataset="${base}/locked_data/system_repair_fresh_multitask_20260825_v2/development.jsonl"
cd "${TRIP_PROJECT_ROOT}"
python scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
python scripts/audit_system_repair_development.py \
  --dataset "${dataset}" \
  --zero-shot-raw "${comparison}/zero_shot/raw_outputs.jsonl" \
  --zero-shot-metrics "${comparison}/zero_shot/metrics.json" \
  --old-unified-adapter-raw "${comparison}/existing_adapter/raw_outputs.jsonl" \
  --old-unified-adapter-metrics "${comparison}/existing_adapter/metrics.json" \
  --current-system-repair-checkpoint-87-raw "${candidate}/raw_outputs.jsonl" \
  --current-system-repair-checkpoint-87-metrics "${candidate}/metrics.json" \
  --output "${TRIP_OUTPUT_FILE}"
