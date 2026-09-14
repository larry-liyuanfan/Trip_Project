#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-final-audit
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
: "${TRIP_IMPLEMENTATION_COMMIT_SHA:?}"
for path in "${TRIP_PROJECT_ROOT}" "${TRIP_HISTORICAL_ROOT}" "${TRIP_OUTPUT_FILE}" "${TRIP_SOURCE_MANIFEST}"; do
  case "${path}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "path escapes Iris project root: ${path}" >&2; exit 2;; esac
done
test ! -e "${TRIP_OUTPUT_FILE}"
module purge
module load GCCcore/11.3.0 Python/3.11.3
base="${TRIP_HISTORICAL_ROOT}/outputs/system_repair"
final="${base}/final_test/system_repair_fresh_test_once_20260825_v4"
data="${base}/locked_data/system_repair_fresh_multitask_20260825_v2"
cd "${TRIP_PROJECT_ROOT}"
python scripts/verify_relevance_source_snapshot.py \
  --project-root "${TRIP_PROJECT_ROOT}" \
  --manifest "${TRIP_SOURCE_MANIFEST}" \
  --expected-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}"
python scripts/audit_system_repair_final_test.py \
  --dataset "${data}/test.jsonl" \
  --raw "${final}/raw_outputs.jsonl" \
  --metrics "${final}/metrics.json" \
  --gate "${base}/gates/system_repair_final_test_gate_20260825_v4.json" \
  --consumption "${data}/system_repair_test_consumption.json" \
  --implementation-commit-sha "${TRIP_IMPLEMENTATION_COMMIT_SHA}" \
  --run-source-snapshot-sha256 "${TRIP_SOURCE_SNAPSHOT_SHA256}" \
  --output "${TRIP_OUTPUT_FILE}"
