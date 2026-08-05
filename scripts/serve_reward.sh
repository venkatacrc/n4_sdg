#!/usr/bin/env bash
# Phase 2: serve Nemotron-4-340B-Reward with NeMo-Aligner across all 8 B200s.
#
# NeMo-Aligner was archived/deprecated in May 2025, but NeMo Framework 25.02 /
# 25.02.01 is the last release that (a) still ships NeMo-Aligner and (b) added
# "Aligner on Blackwell" support. Do not substitute a newer NeMo container
# (Aligner is dropped from 25.04 onwards) or a much older one (no B200/SM100
# kernel support). Run this only after scripts/serve_instruct.sh has been
# stopped and vLLM has released the GPUs - both 340B models resident at TP=8
# simultaneously would leave too little headroom for KV cache/batching.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/raid/models}"
RM_NEMO_FILE="${RM_NEMO_FILE:-${MODELS_DIR}/Nemotron-4-340B-Reward}"
CONTAINER_TAG="${CONTAINER_TAG:-nvcr.io/nvidia/nemo:25.02.01}"
PORT="${PORT:-1424}"
TP_SIZE="${TP_SIZE:-8}"

echo "Using container: ${CONTAINER_TAG}"
echo "Reward checkpoint: ${RM_NEMO_FILE}"
echo "(Check ${MODELS_DIR}/Nemotron-4-340B-Reward after download to confirm the exact"
echo " .nemo file/dir name and adjust RM_NEMO_FILE if it differs.)"

docker run --rm --gpus all \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "${PORT}:${PORT}" \
  -v "${MODELS_DIR}:${MODELS_DIR}" \
  "${CONTAINER_TAG}" \
  bash -c "
    export HYDRA_FULL_ERROR=1
    python -u /opt/NeMo-Aligner/examples/nlp/gpt/serve_reward_model.py \
      rm_model_file=${RM_NEMO_FILE} \
      trainer.num_nodes=1 \
      trainer.devices=${TP_SIZE} \
      ++model.tensor_model_parallel_size=${TP_SIZE} \
      ++model.pipeline_model_parallel_size=1 \
      inference.inference_micro_batch_size=4 \
      inference.port=${PORT}
  "

# Smoke test once it's up (from another shell, inside a Python env with pytriton):
#   python -c "
#   from pytriton.client import ModelClient
#   import numpy as np
#   with ModelClient('localhost:1424', 'reward_model') as c:
#       out = c.infer_batch(sentences=np.array([[b'hello world!']]))
#       print(out)
#   "
