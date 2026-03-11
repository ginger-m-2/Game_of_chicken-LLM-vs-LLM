# src/main.py
"""
main.py

Command-line entry point for running MBTI Game of Chicken tournaments.

This script parses CLI arguments (model backend, conditioning method, number of
tournaments, random seed, output path, generation parameters) and dispatches
execution to the tournament runner defined in tournament.py.

Typical usage:
- Dry run (no LLM calls):
    DRY_RUN=1 python src/main.py --tournaments 1

- Real run with local Ollama model:
    python src/main.py --method prompt --model llama3:8b --tournaments 1

Outputs:
- Writes a JSONL file containing:
    - one meta record
    - one champion record per tournament
    - one match record per match
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tournament import write_tournament_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MBTI Game of Chicken single-elimination tournament(s)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b",
        help="Model name to use (e.g., llama3:8b for Ollama).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="prompt",
        choices=["neutral", "prompt", "lora"],
        help="Conditioning method to use for all 16 agents.",
    )
    parser.add_argument(
        "--tournaments",
        type=int,
        default=1,
        help="Number of tournaments to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for reproducibility.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/results/tournaments.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate per decision.",
    )
    parser.add_argument(
        "--adapter_template",
        type=str,
        default=None,
        help=(
            "Optional template for LoRA-served model names. "
            "Example: 'lora-{MBTI}' or 'llama3:8b-{mbti}'."
        ),
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running MBTI Game of Chicken tournament(s)...")
    print(f"  model         : {args.model}")
    print(f"  method        : {args.method}")
    print(f"  tournaments   : {args.tournaments}")
    print(f"  seed          : {args.seed}")
    print(f"  output        : {out_path}")
    print(f"  temperature   : {args.temperature}")
    print(f"  max_tokens    : {args.max_tokens}")
    print(f"  adapter_templ : {args.adapter_template}")

    write_tournament_jsonl(
        out_path=out_path,
        master_seed=args.seed,
        method=args.method,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_tournaments=args.tournaments,
        adapter_template=args.adapter_template,
    )

    print("Tournament run complete.")


if __name__ == "__main__":
    main()