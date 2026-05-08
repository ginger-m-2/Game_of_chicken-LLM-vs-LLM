"""
plots.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import (
    DIMENSION_LABELS,
    DIMENSION_POSITIONS,
    champions_by_condition,
    load_jsonl,
    matches_by_condition,
    per_agent_action_counts,
    safe_div,
)


MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Figure 1: DRIVE rate per MBTI dimension, grouped by condition

def plot_drive_rate_by_dimension(
    by_condition: Dict[str, List[dict]],
    output_path: Path,
) -> None:
    conditions = sorted(by_condition)
    if not conditions:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    axes_flat = axes.flatten()

    bar_width = 0.8 / len(conditions)
    x_positions = np.arange(2)  # two sides per dimension

    for ax, (dim_key, (high, low)) in zip(axes_flat, DIMENSION_LABELS.items()):
        pos = DIMENSION_POSITIONS[dim_key]

        for ci, condition in enumerate(conditions):
            per_agent = per_agent_action_counts(by_condition[condition])

            rates = []
            for side_letter in (high, low):
                drive_total = 0
                action_total = 0
                for mbti, (d, y, _w, _t) in per_agent.items():
                    if mbti[pos] == side_letter:
                        drive_total += d
                        action_total += d + y
                rates.append(safe_div(drive_total, action_total))

            offset = (ci - (len(conditions) - 1) / 2) * bar_width
            ax.bar(x_positions + offset, rates, width=bar_width, label=condition)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([high, low])
        ax.set_title(dim_key.replace("_", " vs "))
        ax.set_ylim(0, 1)
        ax.set_ylabel("DRIVE rate")
        ax.grid(axis="y", linestyle=":", linewidth=0.5)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(conditions), frameon=False)
    fig.suptitle("DRIVE rate by MBTI dimension and condition")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Figure 2: Champion frequencies per MBTI, grouped by condition

def plot_champion_frequencies(
    champions: Dict[str, "object"],  # condition -> Counter
    output_path: Path,
) -> None:
    conditions = sorted(champions)
    if not conditions:
        return

    x = np.arange(len(MBTI_TYPES))
    bar_width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, condition in enumerate(conditions):
        counter = champions[condition]
        heights = [counter.get(mbti, 0) for mbti in MBTI_TYPES]
        offset = (ci - (len(conditions) - 1) / 2) * bar_width
        ax.bar(x + offset, heights, width=bar_width, label=condition)

    ax.set_xticks(x)
    ax.set_xticklabels(MBTI_TYPES, rotation=45, ha="right")
    ax.set_ylabel("Tournaments won")
    ax.set_title("Champion frequency per MBTI by condition")
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Figure 3: DRIVE rate heatmap (16 MBTIs x N conditions)

def plot_agent_drive_heatmap(
    by_condition: Dict[str, List[dict]],
    output_path: Path,
) -> None:
    conditions = sorted(by_condition)
    if not conditions:
        return

    matrix = np.zeros((len(MBTI_TYPES), len(conditions)))
    for ci, condition in enumerate(conditions):
        per_agent = per_agent_action_counts(by_condition[condition])
        for ri, mbti in enumerate(MBTI_TYPES):
            d, y, _w, _t = per_agent.get(mbti, (0, 0, 0, 0))
            matrix[ri, ci] = safe_div(d, d + y)

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(conditions), 8))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(MBTI_TYPES)))
    ax.set_yticklabels(MBTI_TYPES)
    ax.set_title("Per-agent DRIVE rate by condition")

    # Annotate each cell with the rate.
    for ri in range(len(MBTI_TYPES)):
        for ci in range(len(conditions)):
            value = matrix[ri, ci]
            text_color = "white" if value < 0.5 else "black"
            ax.text(ci, ri, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("DRIVE rate")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Entry

def make_all_plots(results_path: Path, output_dir: Path) -> List[Path]:
    rows = load_jsonl(results_path)
    by_cond = matches_by_condition(rows)
    champs = champions_by_condition(rows)

    _ensure_dir(output_dir)

    paths: List[Path] = []
    p1 = output_dir / "drive_rate_by_dimension.png"
    plot_drive_rate_by_dimension(by_cond, p1)
    paths.append(p1)

    p2 = output_dir / "champion_frequencies.png"
    plot_champion_frequencies(champs, p2)
    paths.append(p2)

    p3 = output_dir / "agent_drive_heatmap.png"
    plot_agent_drive_heatmap(by_cond, p3)
    paths.append(p3)

    return paths
