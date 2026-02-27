# src/agent.py
"""
agent.py

Defines the core LLM agent abstraction used in the Game of Chicken experiments.

This module implements the AgentConfig data structure (which specifies
conditioning method, MBTI type, and model parameters) and the Agent class,
which generates strategic decisions ("ESCALATE" or "YIELD") during gameplay.

The Agent class supports three conditioning modes:
    - neutral: no persona prompt, no adapter
    - prompt: MBTI personality injected via system prompt
    - lora: personality embedded via fine-tuned adapter weights

This file contains no experiment orchestration or logging logic.
It is responsible solely for personality conditioning and decision generation.
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List

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

EXPECTED_ESCALATION_BIAS: Dict[str, float] = {
    "ENTJ": 0.65, "ESTJ": 0.60, "ENTP": 0.60, "INTJ": 0.58,
    "ESTP": 0.57, "INTP": 0.55, "ENFJ": 0.52, "ENFP": 0.50,
    "ISTJ": 0.48, "ISTP": 0.48, "INFJ": 0.45, "INFP": 0.42,
    "ESFJ": 0.40, "ISFJ": 0.35, "ESFP": 0.35, "ISFP": 0.35,
}

@dataclass(frozen=True)
class AgentConfig:
    method: Method
    mbti: str
    model_name: str = "llama3:8b"
    adapter_model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 50


class Agent:
    def __init__(self, cfg: AgentConfig):
        if cfg.mbti not in MBTI_DIMENSIONS:
            raise ValueError(
                f"Unknown MBTI type '{cfg.mbti}'. Expected one of: {sorted(MBTI_DIMENSIONS.keys())}"
            )
        self.cfg = cfg
        self.traits = MBTI_DIMENSIONS[cfg.mbti]
        self.expected_bias = EXPECTED_ESCALATION_BIAS.get(cfg.mbti, 0.5)

    def act(
        self,
        *,
        opponent: AgentConfig,
        rng: random.Random,
        context: Optional[Dict] = None,
    ) -> Action:
        prompt = self._build_prompt(opponent)

        model_to_use = (
            self.cfg.adapter_model_name
            if self.cfg.method == "lora" and self.cfg.adapter_model_name
            else self.cfg.model_name
        )

        # Derive a deterministic inference seed from the caller-provided RNG
        # (so each game is reproducible if game_seed is reproducible).
        inference_seed = rng.randrange(2**31)

        raw = self._query_model(
            model=model_to_use,
            prompt=prompt,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            seed=inference_seed,
        )

        action = self._parse_action(raw)
        if action is None:
            action = "ESCALATE" if rng.random() < 0.5 else "YIELD"
        return action

    def _build_prompt(self, opponent: AgentConfig) -> str:
        persona = ""
        if self.cfg.method == "prompt":
            persona = (
                f"You are an AI agent with MBTI type {self.cfg.mbti}.\n"
                f"Trait bits: {self.traits}\n"
                "Behave consistently with this personality in strategic decisions.\n\n"
            )

        game_rules = (
            "You are playing the Game of Chicken.\n"
            "Choose exactly one action: ESCALATE or YIELD.\n"
            "Return ONLY the single word ESCALATE or YIELD.\n"
        )
        return persona + game_rules

    def _query_model(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        """
        Calls Ollama via CLI.

        Reproducibility note:
          - We *try* to pass --seed if supported by your Ollama build.
          - If not supported, we retry without --seed.
        """
        base_cmd: List[str] = [
            "ollama",
            "run",
            model,
            "--temperature",
            str(temperature),
            "--num-predict",
            str(max_tokens),
        ]

        # Try with --seed first (if provided)
        if seed is not None:
            cmd_with_seed = base_cmd + ["--seed", str(seed)]
            out = self._run_ollama(cmd_with_seed, prompt)
            if out is not None:
                return out

        # Fallback: no seed
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
            # If an unknown flag (like --seed) is used, some builds return nonzero.
            if result.returncode != 0:
                return None
            return result.stdout.decode("utf-8").strip()
        except Exception:
            return None

    def _parse_action(self, text: str) -> Optional[Action]:
        t = (text or "").upper()
        if "ESCALATE" in t:
            return "ESCALATE"
        if "YIELD" in t:
            return "YIELD"
        return None