"""
model_adapter.py

Single place to connect model inference to the tournament runner.

Behavior:
- If GEMINI_API_KEY is set, calls Gemini via google-genai and parses the
  response into a normalized action.
- If the key is missing or the SDK call fails, falls back to a deterministic
  mock so the pipeline can still run for smoke tests. A warning is printed
  once so this never goes unnoticed.

The mock can also be forced explicitly via MOCK_MODEL=1, which is useful for
unit tests and offline development.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Optional


VALID_ACTIONS = {"YIELD", "DRIVE", "SWERVE", "STRAIGHT"}

_GEMINI_CLIENT = None  # lazily initialized
_GEMINI_INIT_FAILED = False
_FALLBACK_WARNED = False
_SEEN_ERRORS: set[str] = set()
_FATAL_ERROR_TOKENS = ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")


def _normalize_gemini_model(model_name: str) -> str:
    """
    Accepts either a Gemini model name or a non-Gemini name. Returns a valid Gemini model id.
    """
    if not model_name or not model_name.lower().startswith(("gemini", "models/gemini")):
        return "gemini-2.5-flash-lite"
    if model_name.startswith("models/"):
        return model_name.split("/", 1)[1]
    return model_name


def _get_gemini_client():
    """
    Returns None if initialization fails (missing key, missing SDK, etc.).
    """
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
    except Exception as exc:  # ImportError, auth errors, etc.
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
    """
    Normalize model output into one of the expected actions.
    Defaults to YIELD if no recognized token is present.
    """
    text = (raw_text or "").strip().upper()

    for action in VALID_ACTIONS:
        if action in text:
            return action

    return "YIELD"


def build_game_prompt(
    *,
    persona_prompt: str,
    opponent_last_action: Optional[str],
) -> str:
    """
    Compose the full prompt sent to the model.
    """
    opp = opponent_last_action if opponent_last_action is not None else "NONE"

    return f"""{persona_prompt}

Game context:
- This is a strategic head-to-head game of Chicken.
- Your opponent's last action: {opp}
- DRIVE means escalate / hold course. YIELD means back down / swerve.
- Mutual DRIVE is the worst outcome. Unilateral DRIVE wins.

Choose exactly one action from:
YIELD
DRIVE

Return only the chosen action as a single word, with no punctuation or explanation.
"""


def _mock_action(full_prompt: str, seed: int, temperature: float) -> str:
    """Deterministic-ish smoke-test fallback."""
    rng = random.Random(seed)
    score = sum(ord(c) for c in full_prompt[:300]) + seed + int(temperature * 100)
    raw = "YIELD" if (score + rng.randint(0, 9)) % 2 == 0 else "DRIVE"
    return extract_action(raw)


def _gemini_action(
    *,
    client,
    model_name: str,
    full_prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Optional[str]:
    """
    Call Gemini once. Returns the normalized action, or None on failure
    (so the caller can fall back to the mock).
    """
    try:
        from google.genai import types

        # Gemini's seed field is INT32; the tournament uses UINT32 seeds.
        clamped_seed = seed % (2**31 - 1)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            seed=clamped_seed,
        )
        response = client.models.generate_content(
            model=_normalize_gemini_model(model_name),
            contents=full_prompt,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        return extract_action(text)
    except Exception as exc:
        global _GEMINI_INIT_FAILED
        msg = str(exc)
        # Print each distinct error message at most once, with a short head.
        head = msg.splitlines()[0][:200]
        if head not in _SEEN_ERRORS:
            _SEEN_ERRORS.add(head)
            print(f"[model_adapter] Gemini call failed: {head}", file=sys.stderr)
        # Short-circuit the rest of the run on unrecoverable auth errors.
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
) -> str:
    """
    Main model hook used by the tournament runner.
    Returns a normalized action string ('DRIVE' or 'YIELD').
    """
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

    action = _gemini_action(
        client=client,
        model_name=model_name,
        full_prompt=full_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    if action is None:
        _warn_fallback_once("Gemini call failed at runtime")
        return _mock_action(full_prompt, seed, temperature)
    return action
