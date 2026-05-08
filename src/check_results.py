from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python src/check_results.py <results.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = load_jsonl(path)

    by_type = Counter(row.get("record_type") for row in rows)
    print("Record counts by type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")

    metas = [r for r in rows if r.get("record_type") == "meta"]
    matches = [r for r in rows if r.get("record_type") == "match"]
    champs = [r for r in rows if r.get("record_type") == "champion"]

    if not metas:
        raise ValueError("No meta records found.")
    if not matches:
        raise ValueError("No match records found.")
    if not champs:
        raise ValueError("No champion records found.")

    tournaments_by_condition = defaultdict(set)
    champs_by_condition = Counter()

    for meta in metas:
        condition = meta["condition"]
        tournaments_by_condition[condition].add(meta["tournament_id"])

        if condition == "shuffled_persona":
            shuffle_map = meta.get("shuffle_map")
            if not shuffle_map:
                raise ValueError("shuffled_persona meta missing shuffle_map")
            bad = [k for k, v in shuffle_map.items() if k == v]
            if bad:
                raise ValueError(f"shuffle_map is not a derangement; fixed points: {bad}")

    for row in matches:
        condition = row["condition"]
        a_mbti = row["a_mbti"]
        b_mbti = row["b_mbti"]
        a_prompt_mbti = row.get("a_prompt_mbti")
        b_prompt_mbti = row.get("b_prompt_mbti")

        if condition == "neutral":
            if a_prompt_mbti is not None or b_prompt_mbti is not None:
                raise ValueError("neutral condition should have null prompt MBTI values")

        if condition == "true_persona":
            if a_prompt_mbti != a_mbti:
                raise ValueError(f"true_persona mismatch: {a_mbti} got {a_prompt_mbti}")
            if b_prompt_mbti != b_mbti:
                raise ValueError(f"true_persona mismatch: {b_mbti} got {b_prompt_mbti}")

        if condition == "shuffled_persona":
            if a_prompt_mbti == a_mbti:
                raise ValueError(f"shuffled_persona fixed point for {a_mbti}")
            if b_prompt_mbti == b_mbti:
                raise ValueError(f"shuffled_persona fixed point for {b_mbti}")

    for row in champs:
        champs_by_condition[row["condition"]] += 1

    print("\nTournament counts by condition:")
    for condition, tids in sorted(tournaments_by_condition.items()):
        print(f"  {condition}: {len(tids)} tournaments")

    print("\nChampion counts by condition:")
    for condition, count in sorted(champs_by_condition.items()):
        print(f"  {condition}: {count}")

    print("\nValidation passed.")
    

if __name__ == "__main__":
    main()