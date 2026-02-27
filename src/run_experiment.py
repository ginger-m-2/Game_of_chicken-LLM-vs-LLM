from __future__ import annotations

import json
import random
from pathlib import Path
from itertools import product, combinations

from agent import Agent, AgentConfig, Method
from chicken import payoff

MBTI_16 = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
]

def stable_seed(master_seed: int, a_method: str, a_mbti: str, b_method: str, b_mbti: str, rep: int) -> int:
    """
    Deterministic seed for each game, independent of iteration order.
    """
    key = f"{master_seed}|{a_method}:{a_mbti}|{b_method}:{b_mbti}|{rep}"
    acc = 2166136261
    for ch in key:
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc

def build_agents(
    mbti_types: list[str],
    methods: list[Method],
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> list[Agent]:
    agents: list[Agent] = []
    for method, mbti in product(methods, mbti_types):
        cfg = AgentConfig(
            method=method,
            mbti=mbti,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        agents.append(Agent(cfg))
    return agents

def generate_matchups(agents: list[Agent]) -> list[tuple[Agent, Agent]]:
    return list(combinations(agents, 2))

def run(
    *,
    mbti_types: list[str],
    methods: list[Method],
    repeats_per_matchup: int,
    seed: int,                 # master seed
    out_jsonl: Path,
    model_name: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 50,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    agents = build_agents(
        mbti_types,
        methods,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    matchups = generate_matchups(agents)

    with out_jsonl.open("w", encoding="utf-8") as f:
        # meta line for reproducibility + analysis
        f.write(json.dumps({
            "record_type": "meta",
            "master_seed": seed,
            "methods": methods,
            "mbti_types": mbti_types,
            "repeats_per_matchup": repeats_per_matchup,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_agents": len(agents),
            "num_matchups": len(matchups),
        }) + "\n")

        game_id = 0
        for (a, b) in matchups:
            for r in range(repeats_per_matchup):
                game_seed = stable_seed(seed, a.cfg.method, a.cfg.mbti, b.cfg.method, b.cfg.mbti, r)
                grng = random.Random(game_seed)

                action_a = a.act(opponent=b.cfg, rng=grng, context=None)
                action_b = b.act(opponent=a.cfg, rng=grng, context=None)
                pa, pb = payoff(action_a, action_b)

                row = {
                    "record_type": "game",
                    "game_id": game_id,
                    "repeat": r,
                    "master_seed": seed,
                    "game_seed": game_seed,

                    "a_method": a.cfg.method,
                    "a_mbti": a.cfg.mbti,
                    "b_method": b.cfg.method,
                    "b_mbti": b.cfg.mbti,

                    "action_a": action_a,
                    "action_b": action_b,
                    "payoff_a": pa,
                    "payoff_b": pb,
                }
                f.write(json.dumps(row) + "\n")
                game_id += 1

    print(f"Wrote {game_id} games to {out_jsonl}")

if __name__ == "__main__":
    run(
        mbti_types=MBTI_16,
        methods=["neutral", "prompt", "lora"],
        repeats_per_matchup=20,
        seed=42,
        out_jsonl=Path("data/results/results.jsonl"),
        model_name="llama3:8b",
        temperature=0.7,
        max_tokens=50,
    )