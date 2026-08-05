"""Shared constants for talking to the Nemotron-4-340B-Reward server.

These mirror NeMo-Aligner's SteerLM prompt format
(examples/nlp/data/steerlm/common.py) exactly, since the reward server
receives raw strings and does no chat templating of its own - the client
is responsible for building the same format the model was trained on.
"""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

SYSTEM_PROMPT_TEMPLATE = "<extra_id_0>System\n{value}\n"
USER_TURN_TEMPLATE = "<extra_id_1>User\n{value}\n"
ASSISTANT_TURN_TEMPLATE = "<extra_id_1>Assistant\n{value}\n"
LABEL_PREFIX = "<extra_id_2>"

# Index 0-3 (OpenAssistant attributes) are present for architectural
# compatibility with Llama2-13B-SteerLM-RM but were never trained for
# Nemotron-4-340B-Reward; index 4-8 are the trained HelpSteer2 attributes.
UNUSED_ATTRIBUTES = ["quality", "toxicity", "humor", "creativity"]
HELPSTEER_ATTRIBUTES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
ALL_STEERLM_ATTRIBUTES = UNUSED_ATTRIBUTES + HELPSTEER_ATTRIBUTES

# NVIDIA's recommended weights for collapsing the 9 raw outputs into a
# single scalar quality score via elementwise multiply-then-sum. The
# first four entries are zero because those attributes are untrained.
DEFAULT_SCALARIZATION_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.3, 0.74, 0.46, 0.47, -0.33]


def format_conversation_for_reward_model(
    messages: list[dict], system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    """Builds the raw string the reward model expects for a conversation.

    `messages` is a list of {"role": "user"|"assistant", "content": str}
    dicts, ending on the assistant turn that should be scored.
    """
    text = SYSTEM_PROMPT_TEMPLATE.format(value=system_prompt)
    for turn in messages:
        role = turn["role"]
        value = turn["content"]
        if role == "user":
            text += USER_TURN_TEMPLATE.format(value=value)
        elif role == "assistant":
            text += ASSISTANT_TURN_TEMPLATE.format(value=value)
        else:
            raise ValueError(f"Unsupported role for reward scoring: {role!r}")
    text += LABEL_PREFIX
    return text


def scalarize(attribute_scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Combines the 5 HelpSteer2 attribute scores into a single quality score."""
    default_weights = dict(zip(HELPSTEER_ATTRIBUTES, DEFAULT_SCALARIZATION_WEIGHTS[4:]))
    weights = weights or default_weights
    return sum(attribute_scores[attr] * weights[attr] for attr in HELPSTEER_ATTRIBUTES)
