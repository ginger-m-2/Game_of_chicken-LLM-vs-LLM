"""
CLI entrypoint for the MBTI tournament project.

Commands:
- run-many-tournaments
- run-all-conditions
- summarize

Examples:
    python src/main.py run-many-tournaments \
        --n-tournaments 10 \
        --condition true_persona

    python src/main.py run-all-conditions --n-tournaments 10

    python src/main.py summarize data/results/true_persona_20260424-131200.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from cli_utils import (
    default_output_path,
    fail,
    format_champion_table,
    ok,
    progress_line,
    validate_setup,
    warn,
)
from mbti_conditions import VALID_CONDITIONS
from run_many_tournaments import run_all_conditions, run_many_tournaments


DEFAULT_RESULTS_DIR = Path("data/results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MBTI tournament experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n-tournaments", type=int, default=1)
    common.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSONL path. Defaults to {DEFAULT_RESULTS_DIR}/<label>_<timestamp>.jsonl",
    )
    common.add_argument("--model-name", type=str, default="gemini-2.5-flash-lite")
    common.add_argument("--temperature", type=float, default=0.7)
    common.add_argument("--max-tokens", type=int, default=80)
    common.add_argument("--master-seed", type=int, default=42)
    common.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("prompts"),
        help="Directory containing MBTI persona prompt files and neutral.txt",
    )
    common.add_argument(
        "--adapter-template",
        type=str,
        default=None,
        help="Optional adapter template name/path used by your model wrapper.",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment and prompts, then exit without running the tournament.",
    )
    common.add_argument(
        "--rpm",
        type=float,
        default=None,
        help=(
            "Max Gemini requests per minute. Defaults to 8 (safe for the free tier's 10 RPM cap). "
            "Set higher if you have paid quota; set to 0 to disable throttling."
        ),
    )

    p_many = subparsers.add_parser(
        "run-many-tournaments",
        parents=[common],
        help="Run one condition across many tournaments.",
    )
    p_many.add_argument(
        "--condition",
        type=str,
        default="true_persona",
        choices=sorted(VALID_CONDITIONS),
        help="Experimental condition to run.",
    )

    p_all = subparsers.add_parser(
        "run-all-conditions",
        parents=[common],
        help="Run true_persona, neutral, and shuffled_persona into one JSONL file.",
    )
    p_all.add_argument(
        "--conditions",
        nargs="+",
        default=["true_persona", "neutral", "shuffled_persona"],
        choices=sorted(VALID_CONDITIONS),
        help="List of conditions to run.",
    )

    p_summarize = subparsers.add_parser(
        "summarize",
        help="Summarize a results JSONL file (wraps analyze_results).",
    )
    p_summarize.add_argument("results", type=Path, help="Path to a results JSONL file.")
    p_summarize.add_argument(
        "--plots",
        action="store_true",
        help="Also save report figures next to the results file under data/analysis/<basename>/",
    )
    p_summarize.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Override the output directory for figures. Defaults to data/analysis/<results basename>/",
    )

    return parser


def resolve_output_path(output: Path | None, label: str) -> Path:
    if output is not None:
        return output
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return default_output_path(DEFAULT_RESULTS_DIR, label)


def preflight(prompts_dir: Path) -> bool:
    problems = validate_setup(prompts_dir)
    if problems:
        print(fail("Preflight checks failed:"), file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return False
    print(ok("Preflight checks passed (API key + prompt files)."))
    return True


def make_progress_printer(total_for_condition: int):
    def _cb(current: int, total: int, condition: str, champion: str) -> None:
        line = progress_line(current, total_for_condition, condition, champion)
        print(line, flush=True)

    return _cb


def main() -> None:
    load_dotenv(override=True)
    parser = build_parser()
    args = parser.parse_args()

    # Propagate --rpm to the model adapter via env var (read lazily there).
    if getattr(args, "rpm", None) is not None:
        import os as _os

        _os.environ["GEMINI_RPM"] = str(args.rpm)

    if args.command == "summarize":
        import analyze_results

        sys.argv = ["analyze_results.py", str(args.results)]
        analyze_results.main()

        if args.plots:
            import plots as plots_mod

            output_dir = args.plots_dir or (
                Path("data/analysis") / args.results.stem
            )
            generated = plots_mod.make_all_plots(args.results, output_dir)
            print()
            print(ok(f"Wrote {len(generated)} figure(s) to {output_dir}"))
            for p in generated:
                print(f"  {p}")
        return

    if not preflight(args.prompts_dir):
        sys.exit(1)

    if args.dry_run:
        print(warn("Dry-run: skipping tournament execution."))
        return

    if args.command == "run-many-tournaments":
        output_path = resolve_output_path(args.output, args.condition)
        print(ok(f"Writing results to {output_path}"))
        counts = run_many_tournaments(
            n_tournaments=args.n_tournaments,
            output_path=output_path,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            master_seed=args.master_seed,
            prompts_dir=args.prompts_dir,
            condition=args.condition,
            adapter_template=args.adapter_template,
            on_tournament_complete=make_progress_printer(args.n_tournaments),
        )
        print()
        print(format_champion_table(counts, title="=== Champion Counts ==="))
        return

    if args.command == "run-all-conditions":
        output_path = resolve_output_path(args.output, "all_conditions")
        print(ok(f"Writing results to {output_path}"))
        counts = run_all_conditions(
            n_tournaments=args.n_tournaments,
            output_path=output_path,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            master_seed=args.master_seed,
            prompts_dir=args.prompts_dir,
            conditions=args.conditions,
            adapter_template=args.adapter_template,
            on_tournament_complete=make_progress_printer(args.n_tournaments),
        )
        print()
        print(format_champion_table(counts, title="=== Champion Counts by Condition ==="))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
