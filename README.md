# n4_sdg — Nemotron-4-340B Synthetic Data Generation on 8xB200

Self-hosted synthetic-data-generation (SDG) pipeline built on NVIDIA's own
Nemotron-4-340B recipe: **Nemotron-4-340B-Instruct** generates candidate
prompts and responses, **Nemotron-4-340B-Reward** scores them on five
HelpSteer2 attributes (helpfulness, correctness, coherence, complexity,
verbosity), and the highest-scoring responses are kept for SFT/preference
training data. Orchestration uses NVIDIA's [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)
`NemotronGenerator` pipelines.

Target hardware: a single node with 8x NVIDIA B200 (183GB HBM3e each),
driver 575+/CUDA 12.8+ for Blackwell support.

## Why this design

- **NIM is not usable for this model today.** The current NIM LLM support
  matrix no longer lists `nemotron-4-340b-instruct` — it's a legacy 2024
  model dropped from the actively maintained catalog. We serve it directly
  with vLLM instead.
- **Instruct model → vLLM.** NVIDIA only ships `.nemo` weights for
  Nemotron-4-340B-Instruct, but the community HF/safetensors conversion
  [`mgoin/Nemotron-4-340B-Instruct-hf`](https://huggingface.co/mgoin/Nemotron-4-340B-Instruct-hf)
  loads directly in vLLM (native `nemotron` architecture support, vLLM PR
  [#6611](https://github.com/vllm-project/vllm/pull/6611)).
- **Reward model → NeMo-Aligner.** `nvidia/Nemotron-4-340B-Reward` has a
  custom multi-attribute head that cannot be converted to HF/vLLM. It's
  served with NeMo-Aligner's `serve_reward_model.py`, which exposes the
  model through a raw PyTriton endpoint (not OpenAI-compatible).
- **Container pin: `nvcr.io/nvidia/nemo:25.02.01`.** NeMo-Aligner was
  archived/deprecated starting with NeMo Framework 25.04. NeMo 25.02 /
  25.02.01 is the *last* release that both ships NeMo-Aligner **and** added
  Blackwell/B200 support ("Aligner on Blackwell" landed in 25.02). Anything
  newer drops Aligner; anything much older has no SM100 kernels and will
  fail to run on B200s.
- **BF16, no quantization.** The model is dense, 340B params, 4K context.
  BF16 weights are ~682GB; at tensor-parallel-8 across 8xB200 that's
  ~85GB/GPU for weights, leaving >90GB/GPU of headroom for KV cache. FP8/NVFP4
  quantization isn't necessary for memory or speed here.
- **Two sequential phases, not concurrent.** Running both 340B models at
  TP=8 at the same time would need ~170GB/GPU for weights alone, leaving too
  little room for KV cache/batching. Generation and scoring run as separate
  phases, each using the full node.

## Repository layout

```
scripts/
  setup_env.sh        Verify driver/Docker/toolkit, create storage dirs
  download_weights.sh Download both checkpoints from Hugging Face
  serve_instruct.sh    Phase 1: vLLM server for Nemotron-4-340B-Instruct
  serve_reward.sh      Phase 2: NeMo-Aligner server for Nemotron-4-340B-Reward
sdg/
  config.yaml          All tunable settings (endpoints, sampling, filtering)
  common.py             Reward-model prompt formatting + scalarization
  clients/reward_client.py  Adapter: NeMo-Aligner's PyTriton protocol -> LLMClient interface
  generate_prompts.py   Phase 1a: synthetic prompt generation (NemotronGenerator)
  generate_responses.py Phase 1b: sample N responses per prompt from vLLM
  score_and_filter.py   Phase 2: score with the reward model, keep the best
  utils.py              Config loading + JSONL helpers
requirements.txt        Orchestration-side Python deps (not needed inside
                         the vLLM or NeMo containers themselves)
```

## Setup (run once on the B200 node)

```bash
./scripts/setup_env.sh
export HF_TOKEN=<your Hugging Face token with NVIDIA Open Model License access>
./scripts/download_weights.sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -U vllm   # separate from the orchestration venv is also fine
```

Verify the downloaded reward checkpoint's actual file/directory name under
`/raid/models/Nemotron-4-340B-Reward` and set `RM_NEMO_FILE` in
`scripts/serve_reward.sh` accordingly if it differs from the default.

## Phase 1 — generate prompts and responses

1. Start the Instruct server (foreground; use `tmux`/`screen` or run in the
   background):

   ```bash
   ./scripts/serve_instruct.sh
   ```

   Wait for `Uvicorn running on http://0.0.0.0:8000`, then smoke-test:

   ```bash
   curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"nemotron-4-340b-instruct","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
   ```

2. Generate synthetic prompts (open Q&A, writing, math, Python):

   ```bash
   python -m sdg.generate_prompts
   ```

3. Sample multiple candidate responses per prompt:

   ```bash
   python -m sdg.generate_responses
   ```

4. Stop the vLLM server (`Ctrl-C` / kill the process) to free all 8 GPUs
   before starting Phase 2.

## Phase 2 — score and filter

1. Start the reward server:

   ```bash
   ./scripts/serve_reward.sh
   ```

   This can take a while to load (~682GB checkpoint across 8 GPUs). Watch
   the container logs for the PyTriton server startup message.

2. Score every sampled response and keep the best per prompt:

   ```bash
   python -m sdg.score_and_filter
   ```

   Output: `/raid/sdg_data/sft_dataset.jsonl`, each line a
   `{"messages": [...], "reward_scores": {...}, "reward_scalar": ...}`
   record. Tune `reward.min_scalar_score` and `reward.keep_top_k` in
   `sdg/config.yaml` (or re-run just this step — no need to regenerate
   responses).

3. Stop the reward container when done.

## Tuning knobs (`sdg/config.yaml`)

- `instruct.sampling` / `instruct.n_samples_per_prompt` — generation
  temperature/top_p and how many candidate responses to sample per prompt
  before filtering.
- `pipeline.*` — how many topics/subtopics/openlines per category to
  generate (controls total dataset size).
- `reward.min_scalar_score` / `reward.keep_top_k` — quality bar and how
  many responses to keep per prompt (`keep_top_k=1` for SFT-style data,
  `keep_top_k>=2` if you want preference pairs from the best/worst samples).

All values can also be overridden with environment variables, e.g.
`INSTRUCT_N_SAMPLES_PER_PROMPT=8 python -m sdg.generate_responses`.

## Licensing

Both checkpoints are distributed under the [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).
Review its terms before generating and distributing synthetic data derived
from these models.

## Known validation risks

- The exact reward-server wire protocol implemented in
  `sdg/clients/reward_client.py` mirrors NeMo-Aligner's
  `examples/nlp/data/steerlm/attribute_annotate.py` reference client as of
  this writing — re-verify against the installed container's
  `serve_reward_model.py`/`common.py` if scores look wrong.
- Pin vLLM to a version with CUDA 12.8+/Blackwell wheels; confirm
  `python -c "import torch; print(torch.cuda.get_device_capability())"`
  reports `(10, 0)` before scaling from a TP=1 smoke test to TP=8.
