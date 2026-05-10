"""Resolves persona prompts for the three experimental conditions."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

VALID_CONDITIONS = {"true_persona", "neutral", "shuffled_persona"}


def validate_condition(condition: str) -> str:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid condition '{condition}'. "
            f"Expected one of: {sorted(VALID_CONDITIONS)}"
        )
    return condition


def load_prompt_text(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def load_persona_prompt(prompts_dir: Path, mbti: str) -> str:
    return load_prompt_text(prompts_dir / f"{mbti}.txt")


def load_neutral_prompt(prompts_dir: Path) -> str:
    return load_prompt_text(prompts_dir / "neutral.txt")


def make_derangement(items: list[str], rng: random.Random) -> Dict[str, str]:
    """Permutation where no item maps to itself."""
    if len(items) < 2:
        raise ValueError("Need at least 2 items to create a derangement.")

    shuffled = items[:]
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled)):
            return dict(zip(items, shuffled))


def build_shuffle_map(condition: str, rng: random.Random) -> Optional[Dict[str, str]]:
    validate_condition(condition)
    if condition == "shuffled_persona":
        return make_derangement(MBTI_TYPES, rng)
    return None


def resolve_persona(
    *,
    agent_mbti: str,
    condition: str,
    prompts_dir: Path,
    shuffle_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, Optional[str]]:
    """Returns (prompt_text, prompt_mbti). prompt_mbti is None for neutral."""
    validate_condition(condition)

    if condition == "true_persona":
        prompt_mbti = agent_mbti
        prompt_text = load_persona_prompt(prompts_dir, prompt_mbti)
        return prompt_text, prompt_mbti

    if condition == "neutral":
        prompt_text = load_neutral_prompt(prompts_dir)
        return prompt_text, None

    if condition == "shuffled_persona":
        if shuffle_map is None:
            raise ValueError("shuffle_map is required for shuffled_persona")
        prompt_mbti = shuffle_map[agent_mbti]
        prompt_text = load_persona_prompt(prompts_dir, prompt_mbti)
        return prompt_text, prompt_mbti

    raise ValueError(f"Unhandled condition: {condition}")