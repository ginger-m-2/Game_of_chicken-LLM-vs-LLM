"""CLI helpers: colors, preflight checks, summary table."""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


ANSI_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


def color(text: str, name: str) -> str:
    if not ANSI_ENABLED or name not in _COLORS:
        return text
    return f"{_COLORS[name]}{text}{_COLORS['reset']}"


def ok(text: str) -> str:
    return color(f"[OK] {text}", "green")


def warn(text: str) -> str:
    return color(f"[!]  {text}", "yellow")


def fail(text: str) -> str:
    return color(f"[X]  {text}", "red")


REQUIRED_ENV_VARS = ("GEMINI_API_KEY",)


def validate_setup(prompts_dir: Path) -> list[str]:
    """Returns a list of problems; empty means all good."""
    problems: list[str] = []

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            problems.append(
                f"Missing environment variable {var}. "
                f"Add it to your .env file (e.g. {var}=your_key_here)."
            )

    if not prompts_dir.exists():
        problems.append(f"Prompts directory not found: {prompts_dir}")
    else:
        missing = [
            mbti
            for mbti in _MBTI_TYPES
            if not (prompts_dir / f"{mbti}.txt").exists()
        ]
        if missing:
            problems.append(
                f"Missing persona prompt files in {prompts_dir}: {', '.join(missing)}"
            )
        if not (prompts_dir / "neutral.txt").exists():
            problems.append(f"Missing neutral prompt file: {prompts_dir / 'neutral.txt'}")

    return problems


_MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


def default_output_path(base_dir: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / f"{label}_{timestamp}.jsonl"


def format_champion_table(counts: Counter, title: Optional[str] = None) -> str:
    """Render champion counts as a ranked table."""
    if not counts:
        return color("(no results)", "dim")

    sample_key = next(iter(counts))
    grouped: dict[str, Counter] = {}
    if isinstance(sample_key, tuple):
        for (condition, mbti), n in counts.items():
            grouped.setdefault(condition, Counter())[mbti] = n
    else:
        grouped["results"] = counts

    lines: list[str] = []
    if title:
        lines.append(color(title, "bold"))

    for group_name, sub in grouped.items():
        total = sum(sub.values())
        header = f"{group_name} (n={total})"
        lines.append("")
        lines.append(color(header, "cyan"))
        lines.append(color(f"{'MBTI':<6} {'Wins':>5}  {'Win %':>6}", "dim"))
        for mbti, n in sub.most_common():
            pct = 0.0 if total == 0 else (100.0 * n / total)
            row = f"{mbti:<6} {n:>5}  {pct:>5.1f}%"
            lines.append(row)

    return "\n".join(lines)


def progress_line(current: int, total: int, label: str, champion: str) -> str:
    bar_width = 20
    filled = 0 if total == 0 else int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"[{bar}] {current:>3}/{total} {label} "
        f"champion={color(champion, 'green')}"
    )
