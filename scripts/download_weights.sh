#!/usr/bin/env bash
# Downloads the two checkpoints needed by the pipeline into /raid/models.
#
# Requires:
#   - `huggingface-cli` (pip install -U "huggingface_hub[cli]")
#   - HF_TOKEN exported in the environment (or already logged in via
#     `huggingface-cli login`) with access accepted for the NVIDIA Open
#     Model License on both nvidia/Nemotron-4-340B-Instruct and
#     nvidia/Nemotron-4-340B-Reward.
#
# Instruct model: ~682GB (BF16 safetensors, community HF conversion that
#   vLLM can load directly - see README for why the raw nvidia/.nemo
#   checkpoint is not used here).
# Reward model: ~682GB (.nemo / Megatron checkpoint, consumed by
#   NeMo-Aligner's serve_reward_model.py, not vLLM).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/raid/models}"
INSTRUCT_REPO="${INSTRUCT_REPO:-mgoin/Nemotron-4-340B-Instruct-hf}"
REWARD_REPO="${REWARD_REPO:-nvidia/Nemotron-4-340B-Reward}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Run: pip install -U \"huggingface_hub[cli]\"" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Either export HF_TOKEN=<your token> or run 'huggingface-cli login' first." >&2
fi

echo "== Downloading ${INSTRUCT_REPO} -> ${MODELS_DIR}/Nemotron-4-340B-Instruct-hf =="
huggingface-cli download "${INSTRUCT_REPO}" \
  --local-dir "${MODELS_DIR}/Nemotron-4-340B-Instruct-hf" \
  --local-dir-use-symlinks False \
  ${HF_TOKEN:+--token "${HF_TOKEN}"}

echo
echo "== Downloading ${REWARD_REPO} -> ${MODELS_DIR}/Nemotron-4-340B-Reward =="
huggingface-cli download "${REWARD_REPO}" \
  --local-dir "${MODELS_DIR}/Nemotron-4-340B-Reward" \
  --local-dir-use-symlinks False \
  ${HF_TOKEN:+--token "${HF_TOKEN}"}

echo
echo "Done. Verify sizes:"
du -sh "${MODELS_DIR}"/* 2>/dev/null || true

cat <<'EOF'

Note on licensing: both checkpoints are distributed under the NVIDIA Open
Model License (https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
Review it before generating and distributing synthetic data derived from
these models.
EOF
