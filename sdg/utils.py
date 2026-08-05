"""Small shared helpers for the SDG scripts: config loading and JSONL I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _apply_env_overrides(node: Any, prefix: str = "") -> Any:
    """Recursively overrides config values with matching environment variables.

    A leaf at config path `instruct.base_url` can be overridden with the
    env var `INSTRUCT_BASE_URL`.
    """
    if isinstance(node, dict):
        return {key: _apply_env_overrides(value, f"{prefix}{key}_") for key, value in node.items()}
    env_name = prefix.rstrip("_").upper()
    return os.environ.get(env_name, node)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _apply_env_overrides(raw)


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path | str, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_jsonl(path: Path | str, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
