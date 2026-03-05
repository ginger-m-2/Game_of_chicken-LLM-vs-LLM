# src/utils.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple


@lru_cache(maxsize=1)
def load_mbti_prompt_file(path: str | Path = "prompts/mbti_prompts.json") -> Dict:
    """
    Load the MBTI prompt JSON once and cache it for reuse.
    Path is relative to the repo root by default.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"MBTI prompt file not found at {p}. "
            "Create prompts/mbti_prompts.json (see README / setup)."
        )
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_mbti_system_prompt(mbti: str, path: str | Path = "prompts/mbti_prompts.json") -> str:
    """
    Returns the fixed system prompt text for the given MBTI type.
    """
    data = load_mbti_prompt_file(path)
    prompts = data.get("prompts", {})
    if mbti not in prompts:
        raise KeyError(f"MBTI type '{mbti}' not found in {path}.")
    return prompts[mbti]


def get_prompt_version(path: str | Path = "prompts/mbti_prompts.json") -> str:
    """
    Returns the prompt-set version string (useful to log in results).
    """
    data = load_mbti_prompt_file(path)
    meta = data.get("meta", {})
    return str(meta.get("version", "unknown"))