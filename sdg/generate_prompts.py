"""Phase 1a: generate synthetic prompts (openlines) using NeMo Curator's
NemotronGenerator against the vLLM-served Nemotron-4-340B-Instruct endpoint.

Replicates the four prompt-generation pipelines from the Nemotron-4-340B
technical report: open Q&A, writing tasks, math problems, and Python
coding problems. Run this before generate_responses.py.

Usage:
    python -m sdg.generate_prompts
"""

from __future__ import annotations

import logging

from nemo_curator import OpenAIClient
from nemo_curator.synthetic import NemotronGenerator
from openai import OpenAI

from sdg.utils import load_config, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    instruct_cfg = config["instruct"]
    pipeline_cfg = config["pipeline"]

    openai_client = OpenAI(base_url=instruct_cfg["base_url"], api_key=instruct_cfg["api_key"])
    client = OpenAIClient(openai_client)
    generator = NemotronGenerator(client)

    model = instruct_cfg["model"]
    model_kwargs = dict(instruct_cfg["sampling"])
    ignore_conversion_failure = pipeline_cfg["ignore_conversion_failure"]

    prompts: list[dict] = []

    logger.info("Generating open Q&A prompts...")
    open_qa = generator.run_open_qa_pipeline(
        n_macro_topics=pipeline_cfg["n_macro_topics"],
        n_subtopics=pipeline_cfg["n_subtopics"],
        n_openlines=pipeline_cfg["n_openlines"],
        n_revisions=pipeline_cfg["n_revisions"],
        model=model,
        model_kwargs=model_kwargs,
        ignore_conversion_failure=ignore_conversion_failure,
    )
    prompts += [{"pipeline": "open_qa", "prompt": p} for p in open_qa]
    logger.info("  -> %d prompts", len(open_qa))

    logger.info("Generating writing-task prompts...")
    writing_topics = generator.convert_response_to_yaml_list(
        generator.generate_macro_topics(n_macro_topics=pipeline_cfg["n_macro_topics"], model=model, model_kwargs=model_kwargs)[0],
        model=model,
        model_kwargs=model_kwargs,
    )
    writing_tasks = generator.run_writing_pipeline(
        topics=writing_topics,
        text_material_types=pipeline_cfg["writing_text_material_types"],
        n_openlines=pipeline_cfg["n_openlines"],
        n_revisions=pipeline_cfg["n_revisions"],
        model=model,
        model_kwargs=model_kwargs,
        ignore_conversion_failure=ignore_conversion_failure,
    )
    prompts += [{"pipeline": "writing", "prompt": p} for p in writing_tasks]
    logger.info("  -> %d prompts", len(writing_tasks))

    logger.info("Generating math prompts...")
    math_questions = generator.run_math_pipeline(
        n_macro_topics=pipeline_cfg["n_macro_topics"],
        school_level=pipeline_cfg["math_school_level"],
        n_subtopics=pipeline_cfg["n_subtopics"],
        n_openlines=pipeline_cfg["n_openlines"],
        model=model,
        model_kwargs=model_kwargs,
        ignore_conversion_failure=ignore_conversion_failure,
    )
    prompts += [{"pipeline": "math", "prompt": p} for p in math_questions]
    logger.info("  -> %d prompts", len(math_questions))

    logger.info("Generating Python coding prompts...")
    python_questions = generator.run_python_pipeline(
        n_macro_topics=pipeline_cfg["n_macro_topics"],
        n_subtopics=pipeline_cfg["n_subtopics"],
        n_openlines=pipeline_cfg["n_openlines"],
        model=model,
        model_kwargs=model_kwargs,
        ignore_conversion_failure=ignore_conversion_failure,
    )
    prompts += [{"pipeline": "python", "prompt": p} for p in python_questions]
    logger.info("  -> %d prompts", len(python_questions))

    for idx, record in enumerate(prompts):
        record["prompt_id"] = idx

    write_jsonl(config["paths"]["prompts_file"], prompts)
    logger.info("Wrote %d total prompts to %s", len(prompts), config["paths"]["prompts_file"])


if __name__ == "__main__":
    main()
