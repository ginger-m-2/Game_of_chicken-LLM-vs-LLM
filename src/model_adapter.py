"""Calls Gemini, falls back to a mock if the key is missing or calls fail."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Optional, Tuple

from utils import _strip_code_fences, extract_first_json_object


VALID_ACTIONS = {"YIELD", "DRIVE", "SWERVE", "STRAIGHT"}
ACTION_ALIASES = {"ESCALATE": "DRIVE", "STRAIGHT": "DRIVE", "SWERVE": "YIELD"}

_GEMINI_CLIENT = None  # lazily initialized
_GEMINI_INIT_FAILED = False
_FALLBACK_WARNED = False
_SEEN_ERRORS: set[str] = set()
_FATAL_ERROR_TOKENS = ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")

_LAST_CALL_TIME: float = 0.0


def _target_interval_seconds() -> float:
    try:
        rpm = float(os.environ.get("GEMINI_RPM", "8"))
    except ValueError:
        rpm = 8.0
    if rpm <= 0:
        return 0.0
    return 60.0 / rpm


def _throttle() -> None:
    global _LAST_CALL_TIME
    interval = _target_interval_seconds()
    if interval <= 0:
        _LAST_CALL_TIME = time.monotonic()
        return
    now = time.monotonic()
    elapsed = now - _LAST_CALL_TIME
    if elapsed < interval:
        time.sleep(interval - elapsed)
    _LAST_CALL_TIME = time.monotonic()


def _normalize_gemini_model(model_name: str) -> str:
    if not model_name or not model_name.lower().startswith(("gemini", "models/gemini")):
        return "gemini-2.5-flash"
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def _get_gemini_client():
    global _GEMINI_CLIENT, _GEMINI_INIT_FAILED

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    if _GEMINI_INIT_FAILED:
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _GEMINI_INIT_FAILED = True
        return None

    try:
        from google import genai

        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        return _GEMINI_CLIENT
    except Exception as exc:
        _GEMINI_INIT_FAILED = True
        print(f"[model_adapter] Gemini client init failed: {exc}", file=sys.stderr)
        return None


def _warn_fallback_once(reason: str) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    print(
        f"[model_adapter] Using deterministic mock model ({reason}). "
        "Results will NOT reflect real LLM behavior. "
        "Set GEMINI_API_KEY to enable real Gemini calls.",
        file=sys.stderr,
    )


def extract_action(raw_text: str) -> str:
    """Pull DRIVE or YIELD out of raw text. Defaults to YIELD."""
    text = (raw_text or "").strip().upper()

    for action in VALID_ACTIONS:
        if action in text:
            return ACTION_ALIASES.get(action, action)

    return "YIELD"


def parse_reasoned_response(raw_text: str) -> Tuple[str, Optional[str], bool]:
    """Parse {"action": ..., "reason": ...}. Returns (action, reason, parsed_ok)."""
    if not raw_text:
        return "YIELD", None, False

    cleaned = _strip_code_fences(raw_text)
    js = extract_first_json_object(cleaned)

    if js is not None:
        try:
            obj = json.loads(js)
            raw_action = str(obj.get("action", "")).upper().strip()
            action = ACTION_ALIASES.get(raw_action, raw_action)
            reason = str(obj.get("reason", "")).strip() or None
            if action in {"DRIVE", "YIELD"}:
                return action, reason, True
        except Exception:
            pass

    return extract_action(raw_text), None, False


def build_game_prompt(
    *,
    persona_prompt: str,
    opponent_last_action: Optional[str],
) -> str:
    opp = opponent_last_action if opponent_last_action is not None else "NONE"

    return f"""{persona_prompt}

Game context:
- This is a strategic head-to-head game of Chicken.
- Your opponent's last action: {opp}
- DRIVE means escalate / hold course. YIELD means back down / swerve.
- Mutual DRIVE is the worst outcome. Unilateral DRIVE wins.

Choose exactly one action from: YIELD or DRIVE.

Respond with a single JSON object on one line, with no surrounding text or
code fences, in exactly this shape:
{{"action": "DRIVE", "reason": "<one short sentence explaining your choice in character>"}}

The "action" field must be either "DRIVE" or "YIELD". Keep "reason" under 25 words.
"""


def _mock_action(full_prompt: str, seed: int, temperature: float) -> Tuple[str, Optional[str]]:
    """Our determinsitc "model" using arithemtic. Returns (action, reason=None)."""
    rng = random.Random(seed)
    score = sum(ord(c) for c in full_prompt[:300]) + seed + int(temperature*100)
    raw = "YIELD" if (score + rng.randint(0, 9))%2 == 0 else "DRIVE"
    return extract_action(raw), None


_RETRY_BACKOFFS = (15.0, 30.0, 60.0, 120.0)
_RETRYABLE_TOKENS = (
    "RESOURCE_EXHAUSTED", "429",
    "UNAVAILABLE", "503",
    "DEADLINE_EXCEEDED", "504",
    "INTERNAL", "500",
)


def _is_retryable_error(msg: str) -> bool:
    return any(token in msg for token in _RETRYABLE_TOKENS)


def _gemini_action(
    *,
    client,
    model_name: str,
    full_prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Optional[Tuple[str, Optional[str]]]:
    global _GEMINI_INIT_FAILED

    try:
        from google.genai import types
    except Exception as exc:
        head = str(exc).splitlines()[0][:200]
        if head not in _SEEN_ERRORS:
            _SEEN_ERRORS.add(head)
            print(f"[model_adapter] google-genai SDK import failed: {head}", file=sys.stderr)
        return None

    # Gemini seed is INT32, tournament seeds are UINT32
    clamped_seed = seed % (2**31 - 1)

    # Disable Gemini 2.5 "thinking" tokens so they don't eat max_output_tokens
    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "seed": clamped_seed,
    }
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except (AttributeError, TypeError):
        pass

    config = types.GenerateContentConfig(**config_kwargs)
    resolved_model = _normalize_gemini_model(model_name)

    attempt = 0
    while True:
        _throttle()
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=full_prompt,
                config=config,
            )
            text = getattr(response, "text", None) or ""
            action, reason, _ = parse_reasoned_response(text)
            return action, reason
        except Exception as exc:
            msg = str(exc)
            head = msg.splitlines()[0][:200]

            if _is_retryable_error(msg) and attempt < len(_RETRY_BACKOFFS):
                wait = _RETRY_BACKOFFS[attempt]
                attempt += 1
                if head not in _SEEN_ERRORS:
                    _SEEN_ERRORS.add(head)
                    print(
                        f"[model_adapter] Transient Gemini error ({head}). "
                        f"Backing off {wait:.0f}s and retrying.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[model_adapter] Transient error again, retry {attempt}, "
                        f"sleeping {wait:.0f}s.",
                        file=sys.stderr,
                    )
                time.sleep(wait)
                continue

            # Non-rate-limit error, or retries exhausted
            if head not in _SEEN_ERRORS:
                _SEEN_ERRORS.add(head)
                print(f"[model_adapter] Gemini call failed: {head}", file=sys.stderr)
            if any(token in msg for token in _FATAL_ERROR_TOKENS):
                _GEMINI_INIT_FAILED = True
            return None


def generate_action(
    *,
    model_name: str,
    persona_prompt: str,
    opponent_last_action: Optional[str],
    seed: int,
    temperature: float,
    max_tokens: int,
    adapter_template: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Returns (action, reason). Reason is None if mock or unparseable."""
    full_prompt = build_game_prompt(
        persona_prompt=persona_prompt,
        opponent_last_action=opponent_last_action,
    )

    if os.environ.get("MOCK_MODEL") == "1":
        _warn_fallback_once("MOCK_MODEL=1")
        return _mock_action(full_prompt, seed, temperature)

    if _GEMINI_INIT_FAILED:
        _warn_fallback_once("Gemini permanently disabled this run")
        return _mock_action(full_prompt, seed, temperature)

    client = _get_gemini_client()
    if client is None:
        _warn_fallback_once("GEMINI_API_KEY missing or SDK unavailable")
        return _mock_action(full_prompt, seed, temperature)

    result = _gemini_action(
        client=client,
        model_name=model_name,
        full_prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    if result is None:
        _warn_fallback_once("Gemini call failed at runtime")
        return _mock_action(full_prompt, seed, temperature)
    return result
