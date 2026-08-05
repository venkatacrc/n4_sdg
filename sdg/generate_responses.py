"""Phase 1b: sample multiple candidate responses per prompt from the
vLLM-served Nemotron-4-340B-Instruct endpoint.

Run after generate_prompts.py and before score_and_filter.py (which
requires the reward server from serve_reward.sh to be running instead of
vLLM - see the two-phase runbook in README.md).

Usage:
    python -m sdg.generate_responses
"""

from __future__ import annotations

import logging

from nemo_curator import OpenAIClient
from openai import OpenAI
from tqdm import tqdm

from sdg.utils import load_config, read_jsonl, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    instruct_cfg = config["instruct"]

    openai_client = OpenAI(base_url=instruct_cfg["base_url"], api_key=instruct_cfg["api_key"])
    client = OpenAIClient(openai_client)

    model = instruct_cfg["model"]
    model_kwargs = dict(instruct_cfg["sampling"])
    model_kwargs["n"] = instruct_cfg["n_samples_per_prompt"]

    prompts = list(read_jsonl(config["paths"]["prompts_file"]))
    logger.info("Sampling %d responses for each of %d prompts...", model_kwargs["n"], len(prompts))

    records = []
    for item in tqdm(prompts, desc="Generating responses"):
        messages = [{"role": "user", "content": item["prompt"]}]
        try:
            responses = client.query_model(model=model, messages=messages, **model_kwargs)
        except Exception:
            logger.exception("Failed to generate responses for prompt_id=%s", item["prompt_id"])
            continue

        for sample_idx, response_text in enumerate(responses):
            records.append(
                {
                    "prompt_id": item["prompt_id"],
                    "pipeline": item["pipeline"],
                    "prompt": item["prompt"],
                    "sample_idx": sample_idx,
                    "response": response_text,
                }
            )

    write_jsonl(config["paths"]["raw_responses_file"], records)
    logger.info("Wrote %d (prompt, response) samples to %s", len(records), config["paths"]["raw_responses_file"])


if __name__ == "__main__":
    main()
