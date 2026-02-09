from dotenv import load_dotenv
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

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

@dataclass(frozen=True)
class Agent:
    name: str
    mbti: str
    profile: Dict[str, Any]
    use_llm: bool = False

    def __init__(self, name, mbti, profile, use_llm):
        self.name = name
        self.mbti = mbti
        self.profile = profile
        self.use_llm = use_llm

    def decide(self, rng: random.Random, context: Optional[Dict[str, Any]] = None) -> Action:
        """
        Decision policy.
        For now: placeholder based on 'risk' in profile.
        Later: if use_llm=True, call your LLM here.
        """
        if self.use_llm:
            # LLM hook (leave disabled until you implement it)
            api_key = get_api_key()
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Use a .env file or environment variable."
                )
            # TODO: implement LLM call here and return "ESCALATE" or "YIELD"
            # For now, fall back to risk policy even if use_llm=True
            pass

        risk = float(self.profile.get("risk", 0.5))
        return "ESCALATE" if rng.random() < risk else "YIELD"