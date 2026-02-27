"""
tournament.py

Implements an optional single-elimination tournament mode.

Agents are paired in bracket rounds, winners advance, and
a final champion is determined.

This module is separate from the repeated-measures experimental
framework and is intended for exploratory or illustrative use.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import yaml

from agent import Agent
from chicken import payoff


# =========================
# YAML loader (local)
# =========================

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# Data structures
# =========================

@dataclass
class MatchRow:
    run_id: int
    round: str
    agent_a: str
    agent_b: str
    mbti_a: str
    mbti_b: str
    action_a: str
    action_b: str
    payoff_a: float
    payoff_b: float
    winner: str
    mode: str
    model: str
    temperature: float


# =========================
# Agent construction
# =========================

def build_agents(
    mbti_profiles_path: str,
    use_llm: bool,
    model: str,
    temperature: float,
    mode: str = "mbti",
) -> List[Agent]:
    profiles = load_yaml(mbti_profiles_path)

    agents: List[Agent] = []

    # Expect profiles YAML to be { "INTJ": {...}, "ENTP": {...}, ... }
    for i, (mbti, profile) in enumerate(profiles.items()):
        if mode == "neutral":
            profile = {"risk": 0.5}

        agents.append(
            Agent(
                name=f"Agent_{i}",
                mbti=mbti,
                profile=profile,
                use_llm=use_llm,
                model=model,
                temperature=temperature,
            )
        )

    return agents


# =========================
# One tournament round
# =========================

def run_round(
    round_name: str,
    agents: List[Agent],
    rng: random.Random,
    run_id: int,
    mode: str,
    model: str,
    temperature: float,
) -> Tuple[List[Agent], List[MatchRow]]:
    winners: List[Agent] = []
    rows: List[MatchRow] = []

    if len(agents) % 2 != 0:
        raise ValueError("Number of agents must be even.")

    for i in range(0, len(agents), 2):
        a = agents[i]
        b = agents[i + 1]

        action_a = a.decide(rng, context={"round": round_name, "opponent_mbti": b.mbti})
        action_b = b.decide(rng, context={"round": round_name, "opponent_mbti": a.mbti})

        payoff_a, payoff_b = payoff(action_a, action_b)

        # Decide winner by payoff; break ties randomly
        if payoff_a > payoff_b:
            winner_agent = a
        elif payoff_b > payoff_a:
            winner_agent = b
        else:
            winner_agent = a if rng.random() < 0.5 else b

        winners.append(winner_agent)

        rows.append(
            MatchRow(
                run_id=run_id,
                round=round_name,
                agent_a=a.name,
                agent_b=b.name,
                mbti_a=a.mbti,
                mbti_b=b.mbti,
                action_a=action_a,
                action_b=action_b,
                payoff_a=payoff_a,
                payoff_b=payoff_b,
                winner=winner_agent.mbti,
                mode=mode,
                model=model,
                temperature=temperature,
            )
        )

    return winners, rows


# =========================
# Single tournament (engine)
# =========================

def run_tournament(
    rng: random.Random,
    mbti_profiles_path: str,
    use_llm: bool,
    model: str,
    temperature: float,
    mode: str,
    run_id: int,
) -> Tuple[str, List[MatchRow]]:
    agents = build_agents(
        mbti_profiles_path=mbti_profiles_path,
        use_llm=use_llm,
        model=model,
        temperature=temperature,
        mode=mode,
    )

    rng.shuffle(agents)

    all_rows: List[MatchRow] = []
    current = agents

    for rn in ["R16", "QF", "SF", "F"]:
        current, rows = run_round(
            rn, current, rng, run_id, mode, model, temperature
        )
        all_rows.extend(rows)

    champion = current[0].mbti
    return champion, all_rows


# =========================
# Core experiment runner from cfg dict
# =========================

def _run_experiment_with_cfg(cfg: dict, mbti_profiles_path: str) -> str:
    seed0 = int(cfg.get("seed", 42))
    num_runs = int(cfg.get("num_runs", 15))
    use_llm = bool(cfg.get("use_llm", False))
    model = str(cfg.get("model", "models/gemini-2.5-flash"))
    temperature = float(cfg.get("temperature", 0.7))
    mode = str(cfg.get("mode", "mbti"))
    out_path = str(cfg.get("out_path", "data/results.jsonl"))

    all_rows: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []

    for run_id in range(num_runs):
        rng = random.Random(seed0 + run_id)

        champion, rows = run_tournament(
            rng=rng,
            mbti_profiles_path=mbti_profiles_path,
            use_llm=use_llm,
            model=model,
            temperature=temperature,
            mode=mode,
            run_id=run_id,
        )

        all_rows.extend(asdict(r) for r in rows)
        meta.append(
            {
                "run_id": run_id,
                "seed": seed0 + run_id,
                "champion": champion,
                "use_llm": use_llm,
                "model": model,
                "temperature": temperature,
                "mode": mode,
            }
        )

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write JSONL
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # Write meta JSON
    meta_path = out_path.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return f"[{mode}] Finished {num_runs} runs. Wrote {out_path} and {meta_path}"


# =========================
# Public runners
# =========================

def run_experiment(
    tournament_cfg_path: str = "config/tournament.yaml",
    mbti_profiles_path: str = "config/mbti_profiles.yaml",
    out_path: str = "data/results.jsonl",
) -> str:
    cfg = load_yaml(tournament_cfg_path)
    cfg["out_path"] = str(cfg.get("out_path", out_path))
    return _run_experiment_with_cfg(cfg, mbti_profiles_path)


def run_both_conditions(
    tournament_cfg_path: str = "config/tournament.yaml",
    mbti_profiles_path: str = "config/mbti_profiles.yaml",
    out_dir: str = "data",
) -> str:
    cfg = load_yaml(tournament_cfg_path)

    cfg_mbti = dict(cfg)
    cfg_mbti["mode"] = "mbti"
    cfg_mbti["out_path"] = f"{out_dir}/results_mbti.jsonl"

    cfg_neutral = dict(cfg)
    cfg_neutral["mode"] = "neutral"
    cfg_neutral["out_path"] = f"{out_dir}/results_neutral.jsonl"

    msg1 = _run_experiment_with_cfg(cfg_mbti, mbti_profiles_path)
    msg2 = _run_experiment_with_cfg(cfg_neutral, mbti_profiles_path)

    return msg1 + "\n" + msg2