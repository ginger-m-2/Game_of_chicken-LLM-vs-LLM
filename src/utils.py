# src/utils.py
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, List


# -----------------------------
# Step 1: Persona prompt loading
# -----------------------------

@lru_cache(maxsize=1)
def load_mbti_prompt_file(path: str | Path = "prompts/mbti_prompts.json") -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"MBTI prompt file not found at {p}. "
            "Create prompts/mbti_prompts.json."
        )
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_mbti_system_prompt(mbti: str, path: str | Path = "prompts/mbti_prompts.json") -> str:
    data = load_mbti_prompt_file(path)
    prompts = data.get("prompts", {})
    if mbti not in prompts:
        raise KeyError(f"MBTI type '{mbti}' not found in {path}.")
    return str(prompts[mbti])


def get_prompt_version(path: str | Path = "prompts/mbti_prompts.json") -> str:
    data = load_mbti_prompt_file(path)
    meta = data.get("meta", {})
    return str(meta.get("version", "unknown"))


# -----------------------------
# Step 2: Machine-readable decisions
# -----------------------------

@dataclass(frozen=True)
class Decision:
    action: str                 # "ESCALATE" or "YIELD"
    reason: str                 # short explanation
    format_ok: bool             # whether we parsed valid JSON cleanly
    raw_text: str               # raw model output (audit/debug)
    used_fallback: bool         # whether fallback action was used


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced {...} JSON object from text using brace balancing.
    Returns None if no balanced object is found.
    """
    if not text:
        return None
    s = text
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_decision_json(raw_text: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Parse JSON of the form:
      {"action":"ESCALATE"|"YIELD","reason":"..."}
    Returns (action, reason, ok).
    """
    if not raw_text:
        return None, None, False

    cleaned = _strip_code_fences(raw_text)
    js = extract_first_json_object(cleaned)
    if js is None:
        return None, None, False

    try:
        obj = json.loads(js)
    except Exception:
        return None, None, False

    action = str(obj.get("action", "")).upper().strip()
    reason = str(obj.get("reason", "")).strip()

    if action not in ("ESCALATE", "YIELD"):
        return None, None, False
    if not reason:
        reason = "No reason provided."
    return action, reason, True