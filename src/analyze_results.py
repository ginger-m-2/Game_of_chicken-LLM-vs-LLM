"""
analyze_results.py
Usage:
    python src/analyze_results.py results.jsonl

Computes:
  1. Champion frequency by condition
  2. Action frequency and DRIVE rate by condition
  3. Mutual-drive rate by condition
  4. Chi-square test on (action x condition) for true_persona vs neutral
  5. Cross-tournament behavioral variance per condition
  6. MBTI dimension breakdown (DRIVE rate and win rate)
  7. Shuffled-condition prompt-vs-identity comparison
  8. Prompt assignment summary
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, List, Optional, Tuple


_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None or os.environ.get("FORCE_COLOR") == "1"


def _h(text: str) -> str:
    return f"\033[1;36m{text}\033[0m" if _COLOR else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _COLOR else text


def _hi(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _COLOR else text


DIMENSION_LABELS = {
    "E_I": ("E", "I"),
    "N_S": ("N", "S"),
    "T_F": ("T", "F"),
    "J_P": ("J", "P"),  
}
DIMENSION_POSITIONS = {"E_I": 0, "N_S": 1, "T_F": 2, "J_P": 3}


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else (100.0 * n / d)


def safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else n / d


# per condition aggregation primitives

def matches_by_condition(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") == "match":
            out[row["condition"]].append(row)
    return dict(out)


def champions_by_condition(rows: List[dict]) -> Dict[str, Counter]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row.get("record_type") == "champion":
            out[row["condition"]][row["champion_mbti"]] += 1
    return dict(out)


# 1. Champion frequency

def section_champion_frequency(champions: Dict[str, Counter]) -> str:
    lines = [_h("=== Champion Frequency by Condition ===")]
    for condition in sorted(champions):
        counter = champions[condition]
        total = sum(counter.values())
        lines.append("")
        lines.append(f"{condition} (n={total})")
        for mbti, count in counter.most_common():
            lines.append(f"  {mbti}: {count} ({pct(count, total):.1f}%)")
    return "\n".join(lines)


# 2 & 3. Action frequency, DRIVE rate, mutual-DRIVE rate

def drive_yield_counts(matches: List[dict]) -> Tuple[int, int]:
    drive = 0
    yield_ = 0
    for m in matches:
        for a in (m["action_a"], m["action_b"]):
            if a == "DRIVE":
                drive += 1
            elif a == "YIELD":
                yield_ += 1
    return drive, yield_


def mutual_drive_count(matches: List[dict]) -> int:
    return sum(1 for m in matches if m["action_a"] == "DRIVE" and m["action_b"] == "DRIVE")


def section_action_summary(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== Action Summary by Condition ===")]
    lines.append("")
    lines.append(f"{'condition':<20} {'n_actions':>10} {'drive_rate':>11} {'mutual_drive_rate':>18}")
    for condition in sorted(by_condition):
        matches = by_condition[condition]
        drive, yield_ = drive_yield_counts(matches)
        total_actions = drive + yield_
        drive_rate = safe_div(drive, total_actions)
        mutual = mutual_drive_count(matches)
        mutual_rate = safe_div(mutual, len(matches))
        lines.append(
            f"{condition:<20} {total_actions:>10} {_hi(f'{drive_rate:>11.4f}')} {_hi(f'{mutual_rate:>18.4f}')}"
        )
    return "\n".join(lines)


# 4. Chi-square: action x condition for true_persona vs neutral

def chi_square_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float, int]:
    """2x2 chi-square test. Returns (chi2, p, df=1)."""
    total = a + b + c + d
    if total == 0:
        return 0.0, 1.0, 1

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    chi2 = 0.0
    for observed, row, col in (
        (a, row1, col1),
        (b, row1, col2),
        (c, row2, col1),
        (d, row2, col2),
    ):
        expected = (row * col) / total
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected

    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p, 1


def try_scipy_chi2(table: List[List[int]]) -> Optional[Tuple[float, float, int]]:
    try:
        from scipy.stats import chi2_contingency
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        return float(chi2), float(p), int(dof)
    except Exception:
        return None


def section_chi_square(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== Chi-square: DRIVE vs YIELD across conditions ===")]
    lines.append("")
    lines.append("Pairwise tests with df=1 (each comparison is a 2x2 contingency table).")
    lines.append("")
    lines.append(f"{'comparison':<40} {'chi2':>10} {'p':>12}")

    conditions = sorted(by_condition)
    pairs = [
        (conditions[i], conditions[j])
        for i in range(len(conditions))
        for j in range(i + 1, len(conditions))
    ]

    for c1, c2 in pairs:
        d1, y1 = drive_yield_counts(by_condition[c1])
        d2, y2 = drive_yield_counts(by_condition[c2])
        scipy_result = try_scipy_chi2([[d1, y1], [d2, y2]])
        if scipy_result is not None:
            chi2, p, _ = scipy_result
        else:
            chi2, p, _ = chi_square_2x2(d1, y1, d2, y2)
        label = f"{c1} vs {c2}"
        lines.append(f"{label:<40} {_hi(f'{chi2:>10.4f}')} {_hi(f'{p:>12.4g}')}")

    return "\n".join(lines)


# 5. Cross-tournament variance of per-agent DRIVE rate

def per_agent_per_tournament_drive_rates(
    matches: List[dict],
) -> Dict[str, Dict[int, Tuple[int, int]]]:
    """
    Returns: agent_mbti -> tournament_id -> (drive_count, total_actions)
    """
    out: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for m in matches:
        for side in ("a", "b"):
            mbti = m[f"{side}_mbti"]
            action = m[f"action_{side}"]
            tid = m["tournament_id"]
            entry = out[mbti][tid]
            entry[1] += 1
            if action == "DRIVE":
                entry[0] += 1
    return {mbti: {tid: tuple(v) for tid, v in d.items()} for mbti, d in out.items()}


def section_consistency(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== Cross-Tournament Behavioral Consistency ===")]
    lines.append("")
    lines.append("For each agent we compute its DRIVE rate within each tournament,")
    lines.append("then take the population variance across tournaments. We report")
    lines.append("the mean of those per-agent variances, by condition. Lower means")
    lines.append("the condition produces more stable behavior.")
    lines.append("")
    lines.append(f"{'condition':<20} {'n_agents':>10} {'mean_within_agent_variance':>28}")

    for condition in sorted(by_condition):
        per_agent = per_agent_per_tournament_drive_rates(by_condition[condition])
        per_agent_variances: List[float] = []
        for _mbti, tournaments in per_agent.items():
            rates = [
                safe_div(drive, total)
                for (drive, total) in tournaments.values()
                if total > 0
            ]
            if len(rates) >= 2:
                per_agent_variances.append(pvariance(rates))
        mean_var = mean(per_agent_variances) if per_agent_variances else 0.0
        lines.append(
            f"{condition:<20} {len(per_agent):>10} {_hi(f'{mean_var:>28.6f}')}"
        )

    return "\n".join(lines)


# 6. MBTI dimension breakdown

def per_agent_action_counts(
    matches: List[dict],
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    agent_mbti -> (drive_count, yield_count, wins, total_matches)
    """
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for m in matches:
        for side in ("a", "b"):
            mbti = m[f"{side}_mbti"]
            action = m[f"action_{side}"]
            row = out[mbti]
            if action == "DRIVE":
                row[0] += 1
            elif action == "YIELD":
                row[1] += 1
            row[3] += 1
        winner = m.get("winner")
        if winner is not None:
            out[winner][2] += 1
    return {mbti: tuple(v) for mbti, v in out.items()}


def section_dimension_breakdown(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== MBTI Dimension Breakdown (DRIVE rate & win rate) ===")]
    lines.append("")
    lines.append("For each MBTI dimension we group the 16 agents into the two sides")
    lines.append("(like E vs I) and aggregate every action they took and every match")
    lines.append("they played, by condition.")

    for condition in sorted(by_condition):
        per_agent = per_agent_action_counts(by_condition[condition])
        lines.append("")
        lines.append(f"-- condition: {condition} --")
        lines.append(
            f"{'dimension':<10} {'side':<5} {'n_agents':>9} "
            f"{'drive_rate':>11} {'win_rate':>10} {'n_matches':>10}"
        )
        for dim_key, (high, low) in DIMENSION_LABELS.items():
            pos = DIMENSION_POSITIONS[dim_key]
            for side_letter in (high, low):
                drive_total = 0
                yield_total = 0
                wins_total = 0
                matches_total = 0
                n_agents = 0
                for mbti, (d, y, w, t) in per_agent.items():
                    if mbti[pos] == side_letter:
                        drive_total += d
                        yield_total += y
                        wins_total += w
                        matches_total += t
                        n_agents += 1
                action_total = drive_total + yield_total
                drive_rate = safe_div(drive_total, action_total)
                win_rate = safe_div(wins_total, matches_total)
                lines.append(
                    f"{dim_key:<10} {side_letter:<5} {n_agents:>9} "
                    f"{_hi(f'{drive_rate:>11.4f}')} {_hi(f'{win_rate:>10.4f}')} {matches_total:>10}"
                )

    return "\n".join(lines)


# 7. Shuffled condition: prompt vs identity

def section_shuffled_prompt_vs_identity(by_condition: Dict[str, List[dict]]) -> str:
    matches = by_condition.get("shuffled_persona", [])
    lines = [_h("=== Shuffled Persona: prompt-keyed vs identity-keyed DRIVE rate ===")]
    if not matches:
        lines.append("")
        lines.append("(no shuffled_persona matches present)")
        return "\n".join(lines)

    by_prompt: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    by_identity: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for m in matches:
        for side in ("a", "b"):
            agent = m[f"{side}_mbti"]
            prompt = m.get(f"{side}_prompt_mbti")
            action = m[f"action_{side}"]
            if prompt is not None:
                row = by_prompt[prompt]
                row[1] += 1
                if action == "DRIVE":
                    row[0] += 1
            row2 = by_identity[agent]
            row2[1] += 1
            if action == "DRIVE":
                row2[0] += 1

    def variance_of_rates(grouped: Dict[str, List[int]]) -> float:
        rates = [safe_div(d, t) for (d, t) in grouped.values() if t > 0]
        return pvariance(rates) if len(rates) >= 2 else 0.0

    var_prompt = variance_of_rates(by_prompt)
    var_identity = variance_of_rates(by_identity)

    lines.append("")
    lines.append(
        "If behavior follows prompt content, between-group variance of DRIVE rate"
    )
    lines.append("should be larger when grouped by the assigned prompt MBTI than when")
    lines.append("grouped by the agent's intrinsic identity MBTI.")
    lines.append("")
    lines.append(f"{'grouping':<25} {'between-group variance':>25}")
    lines.append(f"{'by prompt MBTI':<25} {_hi(f'{var_prompt:>25.6f}')}")
    lines.append(f"{'by agent identity':<25} {_hi(f'{var_identity:>25.6f}')}")
    ratio = safe_div(var_prompt, var_identity) if var_identity > 0 else float("inf")
    lines.append("")
    lines.append(f"Ratio (prompt / identity): {_hi(f'{ratio:.3f}')}")

    return "\n".join(lines)


# 8. Prompt assignment summary (sanity check)

def section_prompt_assignment(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== Prompt Assignment Sanity Check ===")]
    for condition in sorted(by_condition):
        counts: Counter = Counter()
        for m in by_condition[condition]:
            for side in ("a", "b"):
                agent = m[f"{side}_mbti"]
                prompt = m.get(f"{side}_prompt_mbti")
                if prompt is None:
                    counts["null_prompt"] += 1
                elif prompt == agent:
                    counts["aligned"] += 1
                else:
                    counts["misaligned"] += 1
        total = sum(counts.values())
        lines.append("")
        lines.append(f"{condition} (n={total} agent appearances)")
        for label, count in counts.most_common():
            lines.append(f"  {label}: {count} ({pct(count, total):.1f}%)")
    return "\n".join(lines)


# Reasoning-trace coverage

def section_reasoning_coverage(by_condition: Dict[str, List[dict]]) -> str:
    lines = [_h("=== Reasoning Trace Coverage ===")]
    lines.append("")
    lines.append("Fraction of agent appearances that include a parsed 'reason' string.")
    lines.append("Coverage < 100% indicates the model returned non-JSON output for those")
    lines.append("matches (or that the run used the deterministic mock backend).")
    lines.append("")
    lines.append(f"{'condition':<20} {'with_reason':>12} {'total':>8} {'coverage':>10}")
    for condition in sorted(by_condition):
        with_reason = 0
        total = 0
        for m in by_condition[condition]:
            for side in ("a", "b"):
                total += 1
                if m.get(f"reason_{side}"):
                    with_reason += 1
        coverage = safe_div(with_reason, total)
        lines.append(f"{condition:<20} {with_reason:>12} {total:>8} {coverage:>10.4f}")
    return "\n".join(lines)


# Entry point

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/analyze_results.py <results.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = load_jsonl(path)
    by_cond = matches_by_condition(rows)
    champs = champions_by_condition(rows)

    sections = [
        section_champion_frequency(champs),
        section_action_summary(by_cond),
        section_chi_square(by_cond),
        section_consistency(by_cond),
        section_dimension_breakdown(by_cond),
        section_shuffled_prompt_vs_identity(by_cond),
        section_prompt_assignment(by_cond),
        section_reasoning_coverage(by_cond),
    ]
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
