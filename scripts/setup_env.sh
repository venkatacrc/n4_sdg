#!/usr/bin/env bash
# Run this on the 8xB200 node (b200-72) before anything else.
# Verifies the software stack meets Blackwell/B200 requirements and
# creates the directories used by the rest of the pipeline.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/raid/models}"
DATA_DIR="${DATA_DIR:-/raid/sdg_data}"
MIN_DRIVER_MAJOR=575

echo "== GPU / driver check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver first." >&2
  exit 1
fi
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
DRIVER_MAJOR=${DRIVER_VERSION%%.*}
echo "Detected driver: ${DRIVER_VERSION}"
if [ "${DRIVER_MAJOR}" -lt "${MIN_DRIVER_MAJOR}" ]; then
  echo "Driver ${DRIVER_VERSION} is older than the ${MIN_DRIVER_MAJOR}+ required for B200/Blackwell." >&2
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
echo "Detected ${GPU_COUNT} GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "${GPU_COUNT}" -ne 8 ]; then
  echo "Warning: expected 8 GPUs for TP=8, found ${GPU_COUNT}. Adjust --tensor-parallel-size / trainer.devices accordingly." >&2
fi

echo
echo "== Docker + NVIDIA Container Toolkit check =="
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Install Docker Engine first (https://docs.docker.com/engine/install/)." >&2
  exit 1
fi

echo "Running a GPU smoke test container (nvidia/cuda:12.8.0-base-ubuntu22.04) ..."
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi \
  || { echo "GPU smoke test failed - check the NVIDIA Container Toolkit install." >&2; exit 1; }

echo
echo "== Disk space check =="
df -h /raid / | sed 's/^/  /'

echo
echo "== Creating directories =="
mkdir -p "${MODELS_DIR}/Nemotron-4-340B-Instruct-hf"
mkdir -p "${MODELS_DIR}/Nemotron-4-340B-Reward"
mkdir -p "${DATA_DIR}"
echo "  ${MODELS_DIR}/Nemotron-4-340B-Instruct-hf"
echo "  ${MODELS_DIR}/Nemotron-4-340B-Reward"
echo "  ${DATA_DIR}"

echo
echo "Environment check complete. Next: scripts/download_weights.sh"
