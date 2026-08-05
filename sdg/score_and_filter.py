"""Phase 2: score every sampled response with Nemotron-4-340B-Reward and
keep only the highest-quality response(s) per prompt.

Run this after serve_reward.sh is up and generate_responses.py has
produced the raw dataset. See sdg/clients/reward_client.py for why a
custom adapter (rather than nemo_curator.OpenAIClient) is needed here.

Usage:
    python -m sdg.score_and_filter
"""

from __future__ import annotations

import logging
from collections import defaultdict

from tqdm import tqdm

from sdg.clients.reward_client import NemotronAlignerRewardClient
from sdg.common import scalarize
from sdg.utils import load_config, read_jsonl, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCORING_BATCH_SIZE = 32


def main() -> None:
    config = load_config()
    reward_cfg = config["reward"]

    reward_client = NemotronAlignerRewardClient(
        host=reward_cfg["host"], port=reward_cfg["port"], model_name=reward_cfg["model_name"]
    )

    records = list(read_jsonl(config["paths"]["raw_responses_file"]))
    logger.info("Scoring %d (prompt, response) samples...", len(records))

    for start in tqdm(range(0, len(records), SCORING_BATCH_SIZE), desc="Scoring with reward model"):
        batch = records[start : start + SCORING_BATCH_SIZE]
        conversations = [
            [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
            for r in batch
        ]
        try:
            scores = reward_client.query_reward_model_batch(conversations)
        except Exception:
            logger.exception("Reward scoring failed for batch starting at %d", start)
            continue
        for record, score in zip(batch, scores):
            record["reward_scores"] = score
            record["reward_scalar"] = scalarize(score)

    by_prompt: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if "reward_scalar" in record:
            by_prompt[record["prompt_id"]].append(record)

    min_score = reward_cfg["min_scalar_score"]
    keep_top_k = reward_cfg["keep_top_k"]

    final_dataset = []
    for prompt_id, candidates in by_prompt.items():
        candidates.sort(key=lambda r: r["reward_scalar"], reverse=True)
        kept = [c for c in candidates[:keep_top_k] if c["reward_scalar"] >= min_score]
        for c in kept:
            final_dataset.append(
                {
                    "prompt_id": prompt_id,
                    "pipeline": c["pipeline"],
                    "messages": [
                        {"role": "user", "content": c["prompt"]},
                        {"role": "assistant", "content": c["response"]},
                    ],
                    "reward_scores": c["reward_scores"],
                    "reward_scalar": c["reward_scalar"],
                }
            )

    write_jsonl(config["paths"]["final_dataset_file"], final_dataset)
    logger.info(
        "Kept %d/%d prompts (%d final examples) after filtering at min_scalar_score=%s, keep_top_k=%s -> %s",
        len(by_prompt),
        len(set(r["prompt_id"] for r in records)),
        len(final_dataset),
        min_score,
        keep_top_k,
        config["paths"]["final_dataset_file"],
    )


if __name__ == "__main__":
    main()
