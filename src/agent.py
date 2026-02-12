from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI

Action = Literal["ESCALATE", "YIELD"]

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


def get_api_key() -> Optional[str]:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def _parse_action(text: str) -> Optional[Action]:
    """
    Robustly parse ESCALATE/YIELD from model output.
    Accepts outputs like "ESCALATE", "Action: YIELD", "I choose to ESCALATE."
    """
    if not text:
        return None
    t = text.strip().upper()

    # Exact match
    if t in ("ESCALATE", "YIELD"):
        return t  # type: ignore

    # Search within longer text
    m = re.search(r"\b(ESCALATE|YIELD)\b", t)
    if m:
        return m.group(1)  # type: ignore

    return None


def build_prompt(mbti: str, opponent_mbti: str, round_name: str) -> str:
    """
    Minimal MBTI-conditioning prompt. Keep it short for stability.
    """
    return (
        f"You are an AI agent with MBTI type {mbti}.\n"
        f"You are playing a one-shot Game of Chicken in round {round_name}.\n"
        f"Your opponent's MBTI type is {opponent_mbti}.\n\n"
        "Choose exactly ONE action:\n"
        "- ESCALATE\n"
        "- YIELD\n\n"
        "Respond with ONLY the single word: ESCALATE or YIELD."
    )


@dataclass
class Agent:
    name: str
    mbti: str
    profile: Dict[str, Any]
    use_llm: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = 0.7

    def decide(
        self,
        rng: random.Random,
        context: Optional[Dict[str, Any]] = None
    ) -> Action:
        """
        Decision policy:
        - If use_llm=True: call OpenAI and parse ESCALATE/YIELD
        - Else: risk-based simulated policy using profile["risk"]
        """
        context = context or {}
        opponent_mbti = context.get("opponent_mbti", "UNKNOWN")
        round_name = context.get("round", "UNKNOWN")

        # LLM mode
        if self.use_llm:
            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Put it in a .env file or environment variable."
                )

            client = OpenAI(api_key=api_key)

            prompt = build_prompt(self.mbti, opponent_mbti, round_name)

            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            raw = (resp.choices[0].message.content or "").strip()
            action = _parse_action(raw)

            # Safe fallback (never crash tournament due to model formatting)
            if action is None:
                # fallback: risk-based
                risk = float(self.profile.get("risk", 0.5))
                return "ESCALATE" if rng.random() < risk else "YIELD"

            return action

        # Simulated mode
        risk = float(self.profile.get("risk", 0.5))
        return "ESCALATE" if rng.random() < risk else "YIELD"
