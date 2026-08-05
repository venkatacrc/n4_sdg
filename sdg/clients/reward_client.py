"""Adapter that lets NeMo Curator (and our own scripts) score conversations
against a self-hosted NeMo-Aligner reward server.

NeMo-Aligner's `serve_reward_model.py` exposes the model through PyTriton
(`pytriton.client.FuturesModelClient`), not an OpenAI-compatible HTTP API.
NeMo Curator's built-in `OpenAIClient.query_reward_model` therefore cannot
talk to it directly. This module implements the same `LLMClient` interface
(`query_reward_model`) against the actual PyTriton wire protocol, so the
rest of the pipeline can use a self-hosted reward server exactly like the
hosted `nvidia/nemotron-4-340b-reward` endpoint on build.nvidia.com.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np

try:
    from nemo_curator import LLMClient
except ImportError:  # pragma: no cover - allows this module to be imported/tested without nemo_curator installed
    class LLMClient:  # type: ignore[no-redef]
        pass

from pytriton.client import FuturesModelClient

from sdg.common import HELPSTEER_ATTRIBUTES, format_conversation_for_reward_model

logger = logging.getLogger(__name__)


class NemotronAlignerRewardClient(LLMClient):
    """Talks to a `serve_reward_model.py` PyTriton server for Nemotron-4-340B-Reward."""

    def __init__(self, host: str = "localhost", port: int = 1424, model_name: str = "reward_model"):
        self.host = host
        self.port = port
        self.model_name = model_name

    def query_model(self, *, messages, model, **kwargs):
        raise NotImplementedError(
            "NemotronAlignerRewardClient only serves the reward model; use an "
            "OpenAIClient pointed at the vLLM endpoint for text generation."
        )

    def query_reward_model(self, *, messages: Iterable[dict], model: Optional[str] = None) -> dict:
        """Scores a single conversation. `model` is accepted for interface
        compatibility with `nemo_curator.LLMClient` but ignored, since the
        reward model served is fixed for the lifetime of the server."""
        return self.query_reward_model_batch([list(messages)])[0]

    def query_reward_model_batch(self, conversations: list[list[dict]]) -> list[dict]:
        """Scores many conversations in one round trip.

        Much faster than calling `query_reward_model` in a loop when
        scoring an entire generated dataset - use this from
        `score_and_filter.py`.
        """
        sentences = [format_conversation_for_reward_model(c) for c in conversations]
        sentences_np = np.char.encode(np.array(sentences)[..., np.newaxis], "utf-8")

        futures = []
        with FuturesModelClient(f"{self.host}:{self.port}", self.model_name) as client:
            for row in np.split(sentences_np, sentences_np.shape[0]):
                futures.append(client.infer_batch(sentences=row))
            results = [future.result() for future in futures]

        scored = []
        for result in results:
            raw_scores = result["rewards"].flatten().tolist()
            if len(raw_scores) != 9:
                logger.warning("Expected 9 reward floats from the server, got %d: %s", len(raw_scores), raw_scores)
            helpsteer_scores = raw_scores[-5:]
            scored.append(dict(zip(HELPSTEER_ATTRIBUTES, helpsteer_scores)))
        return scored
