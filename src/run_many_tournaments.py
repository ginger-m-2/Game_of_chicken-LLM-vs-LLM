"""
run_many_tournaments.py

Runs many MBTI tournaments with support for:
- true_persona
- neutral
- shuffled_persona

Writes JSONL records:
- meta
- match
- champion
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mbti_conditions import (
    MBTI_TYPES,
    build_shuffle_map,
    resolve_persona,
    validate_condition,
)
from model_adapter import generate_action


@dataclass
class AgentSpec:
    agent_mbti: str
    prompt_text: str
    prompt_mbti: Optional[str]
    method: str = "prompt"


def bracket_pairs(items: list[AgentSpec]) -> list[tuple[AgentSpec, AgentSpec]]:
    if len(items) % 2 != 0:
        raise ValueError("Bracket requires an even number of agents.")
    return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]


def next_round_name(current_size: int) -> str:
    mapping = {
        16: "R16",
        8: "QF",
        4: "SF",
        2: "F",
    }
    return mapping.get(current_size, f"R{current_size}")


def decide_winner(
    *,
    a_action: str,
    b_action: str,
    rng: random.Random,
) -> str:
    """
    Match winner logic.

    Current rule:
    - DRIVE beats YIELD
    - same action => random tiebreak
    """
    if a_action == "DRIVE" and b_action == "YIELD":
        return "a"
    if b_action == "DRIVE" and a_action == "YIELD":
        return "b"

    return "a" if rng.randint(0, 1) == 0 else "b"


def write_jsonl_record(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record) + "\n")
    # Flush after every record so the file is observable mid-run.
    # Cheap relative to the cost of a Gemini call.
    handle.flush()


def build_agents(
    *,
    prompts_dir: Path,
    condition: str,
    shuffle_map: Optional[Dict[str, str]],
) -> list[AgentSpec]:
    agents: list[AgentSpec] = []
    for agent_mbti in MBTI_TYPES:
        prompt_text, prompt_mbti = resolve_persona(
            agent_mbti=agent_mbti,
            condition=condition,
            prompts_dir=prompts_dir,
            shuffle_map=shuffle_map,
        )
        agents.append(
            AgentSpec(
                agent_mbti=agent_mbti,
                prompt_text=prompt_text,
                prompt_mbti=prompt_mbti,
                method="prompt",
            )
        )
    return agents


def run_single_tournament(
    *,
    tournament_id: int,
    condition: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    master_seed: int,
    output_handle,
    prompts_dir: Path,
    adapter_template: Optional[str] = None,
) -> str:
    validate_condition(condition)

    tournament_seed = master_seed + tournament_id
    rng = random.Random(tournament_seed)

    shuffle_map = build_shuffle_map(condition, rng)
    agents = build_agents(
        prompts_dir=prompts_dir,
        condition=condition,
        shuffle_map=shuffle_map,
    )

    write_jsonl_record(
        output_handle,
        {
            "record_type": "meta",
            "tournament_id": tournament_id,
            "master_seed": master_seed,
            "tournament_seed": tournament_seed,
            "condition": condition,
            "method": "prompt",
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "adapter_template": adapter_template,
            "num_agents": len(agents),
            "shuffle_map": shuffle_map,
        },
    )

    round_agents = agents[:]
    match_global_id = 0
    opponent_last_actions: dict[str, Optional[str]] = {mbti: None for mbti in MBTI_TYPES}

    while len(round_agents) > 1:
        round_name = next_round_name(len(round_agents))
        winners: list[AgentSpec] = []

        for a, b in bracket_pairs(round_agents):
            match_seed = rng.randint(0, 2**32 - 1)
            match_rng = random.Random(match_seed)

            prev_opp_last_action_a = opponent_last_actions.get(a.agent_mbti)
            prev_opp_last_action_b = opponent_last_actions.get(b.agent_mbti)

            a_action, a_reason = generate_action(
                model_name=model_name,
                persona_prompt=a.prompt_text,
                opponent_last_action=prev_opp_last_action_a,
                seed=match_seed + 1,
                temperature=temperature,
                max_tokens=max_tokens,
                adapter_template=adapter_template,
            )

            b_action, b_reason = generate_action(
                model_name=model_name,
                persona_prompt=b.prompt_text,
                opponent_last_action=prev_opp_last_action_b,
                seed=match_seed + 2,
                temperature=temperature,
                max_tokens=max_tokens,
                adapter_template=adapter_template,
            )

            winner_side = decide_winner(
                a_action=a_action,
                b_action=b_action,
                rng=match_rng,
            )

            winner = a if winner_side == "a" else b
            winners.append(winner)

            write_jsonl_record(
                output_handle,
                {
                    "record_type": "match",
                    "tournament_id": tournament_id,
                    "condition": condition,
                    "round": round_name,
                    "match_id": match_global_id,
                    "seed": match_seed,
                    "a_method": a.method,
                    "a_mbti": a.agent_mbti,
                    "a_prompt_mbti": a.prompt_mbti,
                    "b_method": b.method,
                    "b_mbti": b.agent_mbti,
                    "b_prompt_mbti": b.prompt_mbti,
                    "opp_last_action_a": prev_opp_last_action_a,
                    "opp_last_action_b": prev_opp_last_action_b,
                    "action_a": a_action,
                    "action_b": b_action,
                    "reason_a": a_reason,
                    "reason_b": b_reason,
                    "winner": winner.agent_mbti,
                },
            )

            opponent_last_actions[a.agent_mbti] = b_action
            opponent_last_actions[b.agent_mbti] = a_action

            match_global_id += 1

        round_agents = winners

    champion = round_agents[0]

    write_jsonl_record(
        output_handle,
        {
            "record_type": "champion",
            "tournament_id": tournament_id,
            "condition": condition,
            "champion_mbti": champion.agent_mbti,
            "champion_prompt_mbti": champion.prompt_mbti,
            "champion_method": champion.method,
        },
    )

    return champion.agent_mbti


ProgressCallback = Callable[[int, int, str, str], None]


def run_many_tournaments(
    *,
    n_tournaments: int,
    output_path: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    master_seed: int,
    prompts_dir: Path,
    condition: str,
    adapter_template: Optional[str] = None,
    on_tournament_complete: Optional[ProgressCallback] = None,
) -> Counter:
    validate_condition(condition)

    champion_counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for tournament_id in range(n_tournaments):
            champion_mbti = run_single_tournament(
                tournament_id=tournament_id,
                condition=condition,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                master_seed=master_seed,
                output_handle=f,
                prompts_dir=prompts_dir,
                adapter_template=adapter_template,
            )
            champion_counts[champion_mbti] += 1

            if on_tournament_complete is not None:
                on_tournament_complete(
                    tournament_id + 1, n_tournaments, condition, champion_mbti
                )

    return champion_counts


def run_all_conditions(
    *,
    n_tournaments: int,
    output_path: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    master_seed: int,
    prompts_dir: Path,
    conditions: list[str],
    adapter_template: Optional[str] = None,
    on_tournament_complete: Optional[ProgressCallback] = None,
) -> Counter:
    for condition in conditions:
        validate_condition(condition)

    overall_counts: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for condition in conditions:
            for tournament_id in range(n_tournaments):
                champion_mbti = run_single_tournament(
                    tournament_id=tournament_id,
                    condition=condition,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    master_seed=master_seed,
                    output_handle=f,
                    prompts_dir=prompts_dir,
                    adapter_template=adapter_template,
                )
                overall_counts[(condition, champion_mbti)] += 1

                if on_tournament_complete is not None:
                    on_tournament_complete(
                        tournament_id + 1, n_tournaments, condition, champion_mbti
                    )

    return overall_counts