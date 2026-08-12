#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SPARTAN_ACCOUNT:-}" ]]; then
  echo "SPARTAN_ACCOUNT is required" >&2
  exit 2
fi

echo "Account associations"
sacctmgr show assoc where account="${SPARTAN_ACCOUNT}" format=Account,User,Partition,QOS -P

echo "GPU partition state"
sinfo -p gpu-h100,gpu-a100,gpu-l40s -o '%P|%a|%l|%D|%G'

echo "Current account jobs"
squeue -A "${SPARTAN_ACCOUNT}" -o '%.18i|%.12P|%.30j|%.8u|%.2t|%.10M|%.10l|%R'

echo "Estimated starts for already submitted Trip jobs"
squeue --start -A "${SPARTAN_ACCOUNT}" -n trip-w5-benchmark,trip-w5-shard || true
