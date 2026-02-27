from __future__ import annotations

import json
import random
from pathlib import Path
from itertools import product, combinations
from typing import Optional

from agent import Agent, AgentConfig, Method
from chicken import payoff

MBTI_16 = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ",
]


def build_agents(
    mbti_types: list[str],
    methods: list[Method],
    *,
    model_name: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 50,
    seed: int = 42,
    shuffle_prompt_personas: bool = False,
) -> list[Agent]:
    """
    Builds one Agent per (method, true_mbti).

    If shuffle_prompt_personas is True:
      - prompt agents get a persona_mbti that is randomly permuted
      - but we keep true_mbti for logging (so analysis can check the control)
    """
    rng = random.Random(seed)

    persona_map = {t: t for t in mbti_types}
    if shuffle_prompt_personas:
        perm = mbti_types[:]
        rng.shuffle(perm)
        persona_map = {true_t: perm[i] for i, true_t in enumerate(mbti_types)}

    agents: list[Agent] = []
    for method, true_mbti in product(methods, mbti_types):
        persona_mbti = persona_map[true_mbti] if method == "prompt" else true_mbti

        cfg = AgentConfig(
            method=method,
            mbti=persona_mbti,          # used for prompt persona injection if method=="prompt"
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            # adapter_model_name can be set here if you use it (optional):
            # adapter_model_name=(f"lora-{true_mbti}" if method=="lora" else None),
        )
        agent = Agent(cfg)

        # attach true/persona labels for logging convenience (non-invasive)
        agent.true_mbti = true_mbti          # type: ignore[attr-defined]
        agent.persona_mbti = persona_mbti    # type: ignore[attr-defined]

        agents.append(agent)

    return agents


def generate_matchups(agents: list[Agent]) -> list[tuple[Agent, Agent]]:
    return list(combinations(agents, 2))


def run(
    *,
    mbti_types: list[str],
    methods: list[Method],
    repeats_per_matchup: int,
    seed: int,
    out_jsonl: Path,
    model_name: str = "llama3:8b",
    temperature: float = 0.7,
    max_tokens: int = 50,
    max_matchups: Optional[int] = None,
    shuffle_prompt_personas: bool = False,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    agents = build_agents(
        mbti_types,
        methods,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        shuffle_prompt_personas=shuffle_prompt_personas,
    )
    matchups = generate_matchups(agents)

    # Optional sampling to keep runs manageable while developing
    if max_matchups is not None and 0 < max_matchups < len(matchups):
        idxs = list(range(len(matchups)))
        rng.shuffle(idxs)
        keep = set(idxs[:max_matchups])
        matchups = [m for i, m in enumerate(matchups) if i in keep]

    with out_jsonl.open("w", encoding="utf-8") as f:
        # meta header (helps analysis)
        f.write(json.dumps({
            "record_type": "meta",
            "seed": seed,
            "methods": methods,
            "mbti_types": mbti_types,
            "repeats_per_matchup": repeats_per_matchup,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "shuffle_prompt_personas": shuffle_prompt_personas,
            "num_agents": len(agents),
            "num_matchups": len(matchups),
        }) + "\n")

        game_id = 0
        for (a, b) in matchups:
            for r in range(repeats_per_matchup):
                game_seed = rng.randrange(1_000_000_000)
                grng = random.Random(game_seed)

                action_a = a.act(opponent=b.cfg, rng=grng, context=None)
                action_b = b.act(opponent=a.cfg, rng=grng, context=None)

                pa, pb = payoff(action_a, action_b)

                row = {
                    "record_type": "game",
                    "game_id": game_id,
                    "repeat": r,
                    "seed": seed,
                    "game_seed": game_seed,

                    "a_method": a.cfg.method,
                    "a_mbti_true": getattr(a, "true_mbti", a.cfg.mbti),
                    "a_mbti_persona": getattr(a, "persona_mbti", a.cfg.mbti),

                    "b_method": b.cfg.method,
                    "b_mbti_true": getattr(b, "true_mbti", b.cfg.mbti),
                    "b_mbti_persona": getattr(b, "persona_mbti", b.cfg.mbti),

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
        repeats_per_matchup=20,             # start smaller for dev
        seed=42,
        out_jsonl=Path("data/results/results.jsonl"),
        temperature=0.7,
        max_tokens=50,
        max_matchups=300,                   # optional cap for dev; remove for full run
        shuffle_prompt_personas=False,      # set True to run the control condition
    )