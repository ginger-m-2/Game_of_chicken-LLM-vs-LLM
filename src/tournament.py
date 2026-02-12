from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from agent import Agent, MBTI_TYPES
from chicken import payoff, winner_from_actions


@dataclass
class MatchRow:
    round_name: str
    agent_a: str
    mbti_a: str
    action_a: str
    payoff_a: int
    agent_b: str
    mbti_b: str
    action_b: str
    payoff_b: int
    winner: str
    winner_mbti: str


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_agents(mbti_profiles_path: str, use_llm: bool, model: str, temperature: float) -> List[Agent]:
    profiles = load_yaml(mbti_profiles_path)
    agents: List[Agent] = []
    for t in MBTI_TYPES:
        agents.append(
            Agent(
                name=f"Agent_{t}",
                mbti=t,
                profile=profiles.get(t, {}),
                use_llm=use_llm,
                model=model,
                temperature=temperature,
            )
        )
    return agents


def run_round(round_name: str, agents: List[Agent], rng: random.Random) -> Tuple[List[Agent], List[MatchRow]]:
    assert len(agents) % 2 == 0
    winners: List[Agent] = []
    rows: List[MatchRow] = []

    for i in range(0, len(agents), 2):
        a = agents[i]
        b = agents[i + 1]

        action_a = a.decide(rng, context={"round": round_name, "opponent_mbti": b.mbti})
        action_b = b.decide(rng, context={"round": round_name, "opponent_mbti": a.mbti})

        pa, pb = payoff(action_a, action_b)
        win_idx = winner_from_actions(action_a, action_b, rng)
        winner = a if win_idx == 0 else b
        winners.append(winner)

        rows.append(
            MatchRow(
                round_name=round_name,
                agent_a=a.name,
                mbti_a=a.mbti,
                action_a=action_a,
                payoff_a=pa,
                agent_b=b.name,
                mbti_b=b.mbti,
                action_b=action_b,
                payoff_b=pb,
                winner=winner.name,
                winner_mbti=winner.mbti,
            )
        )

    return winners, rows


def write_results_csv(rows: List[MatchRow], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "round",
            "agent_a", "mbti_a", "action_a", "payoff_a",
            "agent_b", "mbti_b", "action_b", "payoff_b",
            "winner", "winner_mbti",
        ])
        for r in rows:
            w.writerow([
                r.round_name,
                r.agent_a, r.mbti_a, r.action_a, r.payoff_a,
                r.agent_b, r.mbti_b, r.action_b, r.payoff_b,
                r.winner, r.winner_mbti,
            ])


def run_tournament(
    tournament_cfg_path: str = "config/tournament.yaml",
    mbti_profiles_path: str = "config/mbti_profiles.yaml",
    out_csv: str = "data/results.csv",
) -> str:
    cfg = load_yaml(tournament_cfg_path)
    seed = int(cfg.get("seed", 42))
    use_llm = bool(cfg.get("use_llm", False))
    model = str(cfg.get("model", "gpt-4o-mini"))
    temperature = float(cfg.get("temperature", 0.7))

    rng = random.Random(seed)

    agents = build_agents(mbti_profiles_path, use_llm=use_llm, model=model, temperature=temperature)
    rng.shuffle(agents)  # randomized bracket seeding

    all_rows: List[MatchRow] = []

    # 16 -> 8 -> 4 -> 2 -> 1
    round_names = ["R16", "QF", "SF", "F"]
    current = agents
    for rn in round_names:
        current, rows = run_round(rn, current, rng)
        all_rows.extend(rows)

    champion = current[0]
    write_results_csv(all_rows, out_csv)

    return f"Champion: {champion.mbti} ({champion.name}). Results saved to {out_csv}"
