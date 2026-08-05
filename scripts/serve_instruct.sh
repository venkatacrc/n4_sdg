#!/usr/bin/env bash
# Phase 1: serve Nemotron-4-340B-Instruct with vLLM across all 8 B200s.
#
# BF16, TP=8: ~85GB/GPU for weights, leaving >90GB/GPU of the 183GB HBM3e
# per B200 for KV cache - no quantization needed for this model/context length.
#
# Requires: pip install -U vllm  (a recent release with CUDA 12.8+/Blackwell
# wheels; confirm with `python -c "import torch; print(torch.cuda.get_device_capability())"`
# reporting (10, 0) for B200 before scaling up to TP=8).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/raid/models}"
MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/Nemotron-4-340B-Instruct-hf}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name nemotron-4-340b-instruct \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port "${PORT}" \
  --trust-remote-code

# Smoke test once it's up (from another shell):
#   curl -s http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"nemotron-4-340b-instruct","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
