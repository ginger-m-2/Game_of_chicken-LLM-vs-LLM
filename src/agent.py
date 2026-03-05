# src/agent.py
from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List

from utils import (
    Decision,
    parse_decision_json,
    get_mbti_system_prompt,
    get_prompt_version,
)

Action = Literal["ESCALATE", "YIELD"]
Method = Literal["neutral", "prompt", "lora"]

MBTI_DIMENSIONS: Dict[str, Dict[str, int]] = {
    "ISTJ": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 1, "F": 0, "J": 1, "P": 0},
    "ISFJ": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 1, "P": 0},
    "INFJ": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 0, "F": 1, "J": 1, "P": 0},
    "INTJ": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 1, "F": 0, "J": 1, "P": 0},
    "ISTP": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 1, "F": 0, "J": 0, "P": 1},
    "ISFP": {"E": 0, "I": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1},
    "INFP": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 0, "F": 1, "J": 0, "P": 1},
    "INTP": {"E": 0, "I": 1, "N": 1, "S": 0, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESTP": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESFP": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1},
    "ENFP": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 0, "F": 1, "J": 0, "P": 1},
    "ENTP": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 1, "F": 0, "J": 0, "P": 1},
    "ESTJ": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 1, "F": 0, "J": 1, "P": 0},
    "ESFJ": {"E": 1, "I": 0, "N": 0, "S": 1, "T": 0, "F": 1, "J": 1, "P": 0},
    "ENFJ": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 0, "F": 1, "J": 1, "P": 0},
    "ENTJ": {"E": 1, "I": 0, "N": 1, "S": 0, "T": 1, "F": 0, "J": 1, "P": 0},
}


@dataclass(frozen=True)
class AgentConfig:
    method: Method
    mbti: str
    model_name: str = "llama3:8b"
    adapter_model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 80

    # Step 1 persona prompt set path (used when method=="prompt")
    prompt_path: str = "prompts/mbti_prompts.json"


class Agent:
    def __init__(self, cfg: AgentConfig):
        if cfg.mbti not in MBTI_DIMENSIONS:
            raise ValueError(
                f"Unknown MBTI type '{cfg.mbti}'. Expected one of: {sorted(MBTI_DIMENSIONS.keys())}"
            )
        self.cfg = cfg

    # -----------------------------
    # Step 2: Machine-readable API
    # -----------------------------
    def act_json(
        self,
        *,
        opponent: AgentConfig,
        rng: random.Random,
        context: Optional[Dict] = None,
    ) -> Decision:
        """
        Return a machine-readable decision object.

        Enforces strict JSON output. If parsing fails, retries once with a
        correction instruction. If it still fails, uses a fallback action.
        """
        # Optional fast test mode
        if os.getenv("DRY_RUN", "0") == "1":
            a = "ESCALATE" if rng.random() < 0.5 else "YIELD"
            return Decision(
                action=a,
                reason="DRY_RUN stub decision.",
                format_ok=True,
                raw_text=f'{{"action":"{a}","reason":"DRY_RUN stub decision."}}',
                used_fallback=False,
            )

        prompt = self._build_prompt(opponent=opponent, context=context)

        model_to_use = (
            self.cfg.adapter_model_name
            if self.cfg.method == "lora" and self.cfg.adapter_model_name
            else self.cfg.model_name
        )

        inference_seed = rng.randrange(2**31)

        raw = self._query_model(
            model=model_to_use,
            prompt=prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            seed=inference_seed,
        )

        action, reason, ok = parse_decision_json(raw)

        # Retry once with a format-only correction
        if not ok:
            correction = (
                "FORMAT ERROR: Return ONLY valid JSON with keys action and reason.\n"
                "Valid actions: ESCALATE or YIELD.\n"
                "No markdown, no code fences, no extra keys.\n"
                "Example: {\"action\":\"YIELD\",\"reason\":\"One short sentence.\"}\n"
            )
            raw2 = self._query_model(
                model=model_to_use,
                prompt=prompt + "\n\n" + correction,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                seed=inference_seed,
            )
            action, reason, ok = parse_decision_json(raw2)
            raw = raw2

        if ok and action is not None and reason is not None:
            return Decision(action=action, reason=reason, format_ok=True, raw_text=raw, used_fallback=False)

        # Fallback (keep runs alive; mark clearly)
        fb = "ESCALATE" if rng.random() < 0.5 else "YIELD"
        return Decision(
            action=fb,
            reason="Fallback used (model did not return valid decision JSON).",
            format_ok=False,
            raw_text=raw,
            used_fallback=True,
        )

    # Backwards-compatible API used by your current runner
    def act(
        self,
        *,
        opponent: AgentConfig,
        rng: random.Random,
        context: Optional[Dict] = None,
    ) -> Action:
        d = self.act_json(opponent=opponent, rng=rng, context=context)
        return d.action  # type: ignore[return-value]

    # -----------------------------
    # Updated prompt builder (Step 2)
    # -----------------------------
    def _build_prompt(self, *, opponent: AgentConfig, context: Optional[Dict] = None) -> str:
        """
        Construct a prompt that enforces machine-readable JSON output:
          {"action":"ESCALATE"|"YIELD","reason":"<short sentence>"}

        Prompt-only agents receive a fixed MBTI persona template from
        prompts/mbti_prompts.json (Step 1).
        """
        persona = ""
        if self.cfg.method == "prompt":
            persona_text = get_mbti_system_prompt(self.cfg.mbti, self.cfg.prompt_path)
            version = get_prompt_version(self.cfg.prompt_path)
            persona = f"{persona_text}\n(Prompt set version: {version})\n\n"

        # Context (proposal-friendly)
        ctx_lines: List[str] = []
        if context:
            if "dice_self" in context:
                ctx_lines.append(f"Your dice roll: {context['dice_self']}")
            if "dice_opp" in context:
                ctx_lines.append(f"Opponent dice roll: {context['dice_opp']}")
            if "opp_last_action" in context and context["opp_last_action"] is not None:
                ctx_lines.append(f"Opponent last action: {context['opp_last_action']}")

        ctx_block = ""
        if ctx_lines:
            ctx_block = "Context:\n" + "\n".join(ctx_lines) + "\n\n"

        instructions = (
            "You are playing a one-shot Game of Chicken.\n"
            "Choose exactly one action.\n"
            "Valid actions: ESCALATE or YIELD.\n\n"
            "Return ONLY valid JSON with exactly two keys: action and reason.\n"
            "Do NOT output markdown or code fences.\n"
            "Examples:\n"
            "{\"action\":\"ESCALATE\",\"reason\":\"I will pressure the opponent.\"}\n"
            "{\"action\":\"YIELD\",\"reason\":\"Mutual escalation is too costly.\"}\n"
        )

        return persona + ctx_block + instructions

    # -----------------------------
    # Model call (unchanged)
    # -----------------------------
    def _query_model(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        base_cmd: List[str] = [
            "ollama",
            "run",
            model,
            "--temperature",
            str(temperature),
            "--num-predict",
            str(max_tokens),
        ]

        if seed is not None:
            cmd_with_seed = base_cmd + ["--seed", str(seed)]
            out = self._run_ollama(cmd_with_seed, prompt)
            if out is not None:
                return out

        out2 = self._run_ollama(base_cmd, prompt)
        return out2 if out2 is not None else ""

    def _run_ollama(self, cmd: List[str], prompt: str) -> Optional[str]:
        try:
            result = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            return result.stdout.decode("utf-8").strip()
        except Exception:
            return None