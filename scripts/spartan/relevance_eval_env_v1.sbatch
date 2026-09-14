#!/usr/bin/env bash
#SBATCH --job-name=trip-rel-env
#SBATCH --account=punim2936
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail

: "${TRIP_SPARTAN_BASE:?set TRIP_SPARTAN_BASE}"
: "${TRIP_VENV:?set TRIP_VENV}"
: "${TRIP_CACHE_ROOT:?set TRIP_CACHE_ROOT}"
: "${TRIP_SHARED_HF_HOME:?set TRIP_SHARED_HF_HOME}"

case "${TRIP_VENV}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "venv escapes project root" >&2; exit 2;; esac
case "${TRIP_CACHE_ROOT}" in "${TRIP_SPARTAN_BASE}"/*) ;; *) echo "cache escapes project root" >&2; exit 2;; esac
test ! -e "${TRIP_VENV}" || { echo "refusing to overwrite existing venv" >&2; exit 2; }
test -d "${TRIP_SHARED_HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct"

module purge
module load GCCcore/11.3.0 Python/3.11.3
mkdir -p "${TRIP_CACHE_ROOT}"/{pip,huggingface/hub,xdg,tmp,home}
export HOME="${TRIP_CACHE_ROOT}/home"
export PIP_CACHE_DIR="${TRIP_CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${TRIP_CACHE_ROOT}/xdg"
export TMPDIR="${TRIP_CACHE_ROOT}/tmp"

python -m venv "${TRIP_VENV}"
"${TRIP_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${TRIP_VENV}/bin/python" -m pip install \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"${TRIP_VENV}/bin/python" -m pip install \
  transformers==4.57.1 accelerate==1.10.1 peft==0.17.1 bitsandbytes==0.47.0 \
  qwen-vl-utils==0.0.14 Pillow==11.3.0 numpy==2.2.6 psutil==7.0.0 \
  'pymilvus[milvus_lite]==2.6.16' milvus-lite==3.2.1

ln -s "${TRIP_SHARED_HF_HOME}/hub/models--Qwen--Qwen3-VL-8B-Instruct" \
  "${TRIP_CACHE_ROOT}/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct"
"${TRIP_VENV}/bin/python" -m pip freeze
