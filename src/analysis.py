"""
analysis.py

Post-experiment analysis

Supports:
    - Escalation/win-rate analysis per MBTI type
    - Trait-dimension comparisons (E/I, S/N, T/F, J/P)
    - Statistical tests across conditions and dimensions
    - Champion frequency analysis
    - Reasoning trace keyword analysis (if reason field present)
    - Visualizations saved to data/analysis/

Usage:
    python src/analysis.py
    python src/analysis.py --out-dir data/analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed — chi-squared tests will be skipped.")
    print("         Run: pip install scipy\n")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

DIMENSIONS = ["EI", "SN", "TF", "JP"]
DIM_LABELS = {"EI": "E / I", "SN": "S / N", "TF": "T / F", "JP": "J / P"}

# Escalation keywords for reasoning trace analysis
ESCALATION_KEYWORDS = [
    "escalate", "drive", "aggressive", "dominate", "win", "force",
    "risk", "bluff", "bold", "pressure", "threat", "challenge",
]
YIELD_KEYWORDS = [
    "yield", "cooperate", "safe", "avoid", "mutual", "loss", "harm",
    "cautious", "rational", "worst", "catastrophe", "swerve",
]

# Default file paths (tries these first, then fallbacks)
DEFAULT_FILES = {
    "true_persona":     "true_test.jsonl",
    "neutral":          "neutral_test.jsonl",
    "shuffled_persona": "shuffled_test.jsonl",
}
FALLBACK_FILES = {
    "true_persona": "data/results_mbti.jsonl",
    "neutral":      "data/results_neutral.jsonl",
}

def load_jsonl(path: str) -> pd.DataFrame:
    """Load all records from a JSONL file into a DataFrame."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def _is_escalate(action: str) -> bool:
    """Normalize DRIVE/ESCALATE → True, YIELD → False."""
    return str(action).upper() in ("ESCALATE", "DRIVE")


def _payoff(esc_a: bool, esc_b: bool) -> tuple[int, int]:
    """Return (payoff_a, payoff_b) for a match."""
    if esc_a and esc_b:
        return (-10, -10)
    if esc_a:
        return (5, -5)
    if esc_b:
        return (-5, 5)
    return (0, 0)


def load_matches(path: str) -> pd.DataFrame:
    """
    Load match records from a JSONL file, computing derived columns:
      - escalate_a / escalate_b  (bool)
      - payoff_a / payoff_b      (int, computed from actions)
      - a_EI, a_SN, a_TF, a_JP  (MBTI dimension poles for agent A)
      - b_EI, b_SN, b_TF, b_JP  (MBTI dimension poles for agent B)
    """
    df = load_jsonl(path)
    matches = df[df["record_type"] == "match"].copy().reset_index(drop=True)

    matches["escalate_a"] = matches["action_a"].apply(_is_escalate)
    matches["escalate_b"] = matches["action_b"].apply(_is_escalate)

    payoffs = matches.apply(
        lambda r: _payoff(r["escalate_a"], r["escalate_b"]), axis=1
    )
    matches["payoff_a"] = payoffs.apply(lambda x: x[0])
    matches["payoff_b"] = payoffs.apply(lambda x: x[1])

    for side, col in [("a", "a_mbti"), ("b", "b_mbti")]:
        for i, dim in enumerate(DIMENSIONS):
            matches[f"{side}_{dim}"] = matches[col].apply(
                lambda x, i=i: str(x)[i] if isinstance(x, str) and len(x) == 4 else "?"
            )

    return matches


def load_champions(path: str) -> pd.DataFrame:
    """Load only champion records from a JSONL file."""
    df = load_jsonl(path)
    return df[df["record_type"] == "champion"].copy().reset_index(drop=True)


def resolve_paths() -> dict[str, str]:
    """Find available result files and return {condition: path}."""
    resolved = {}
    for cond, path in DEFAULT_FILES.items():
        if Path(path).exists():
            resolved[cond] = path
        elif cond in FALLBACK_FILES and Path(FALLBACK_FILES[cond]).exists():
            resolved[cond] = FALLBACK_FILES[cond]
    return resolved


# ---------------------------------------------------------------------------
# Per-MBTI metrics
# ---------------------------------------------------------------------------

def escalation_by_mbti(df: pd.DataFrame) -> pd.DataFrame:
    """Escalation rate per MBTI type."""
    a = df[["a_mbti", "escalate_a"]].rename(columns={"a_mbti": "mbti", "escalate_a": "escalated"})
    b = df[["b_mbti", "escalate_b"]].rename(columns={"b_mbti": "mbti", "escalate_b": "escalated"})
    long = pd.concat([a, b], ignore_index=True)
    out = (
        long.groupby("mbti")
        .agg(n=("escalated", "count"), n_escalate=("escalated", "sum"))
        .reset_index()
    )
    out["escalation_rate"] = out["n_escalate"] / out["n"]
    return out.sort_values("escalation_rate", ascending=False).reset_index(drop=True)


def payoff_by_mbti(df: pd.DataFrame) -> pd.DataFrame:
    """Mean payoff per MBTI type."""
    a = df[["a_mbti", "payoff_a"]].rename(columns={"a_mbti": "mbti", "payoff_a": "payoff"})
    b = df[["b_mbti", "payoff_b"]].rename(columns={"b_mbti": "mbti", "payoff_b": "payoff"})
    long = pd.concat([a, b], ignore_index=True)
    out = (
        long.groupby("mbti")
        .agg(n=("payoff", "count"), mean_payoff=("payoff", "mean"), std_payoff=("payoff", "std"))
        .reset_index()
    )
    return out.sort_values("mean_payoff", ascending=False).reset_index(drop=True)


def win_rate_by_mbti(df: pd.DataFrame) -> pd.DataFrame:
    """Win rate (wins / appearances) per MBTI type."""
    app_a = df["a_mbti"].value_counts()
    app_b = df["b_mbti"].value_counts()
    appearances = app_a.add(app_b, fill_value=0).rename("appearances")
    wins = df["winner"].value_counts().rename("wins")
    out = pd.concat([appearances, wins], axis=1).fillna(0).reset_index()
    out.columns = ["mbti", "appearances", "wins"]
    out["win_rate"] = out["wins"] / out["appearances"]
    return out.sort_values("win_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MBTI dimension analysis
# ---------------------------------------------------------------------------

def escalation_by_dimension(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    For each MBTI dimension (EI, SN, TF, JP), compute escalation rate
    for each pole (e.g., E vs I).
    """
    results = {}
    for dim in DIMENSIONS:
        a = df[[f"a_{dim}", "escalate_a"]].rename(
            columns={f"a_{dim}": "pole", "escalate_a": "escalated"}
        )
        b = df[[f"b_{dim}", "escalate_b"]].rename(
            columns={f"b_{dim}": "pole", "escalate_b": "escalated"}
        )
        long = pd.concat([a, b], ignore_index=True)
        out = (
            long.groupby("pole")
            .agg(n=("escalated", "count"), n_escalate=("escalated", "sum"))
            .reset_index()
        )
        out["escalation_rate"] = out["n_escalate"] / out["n"]
        results[dim] = out.sort_values("pole")
    return results


def win_rate_by_dimension(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    For each MBTI dimension, compute win rate for each pole.
    A win is counted when the agent with that pole won its match.
    """
    results = {}
    for dim in DIMENSIONS:
        rows = []
        for pole in sorted(df[f"a_{dim}"].dropna().unique()):
            sub_a = df[df[f"a_{dim}"] == pole]
            sub_b = df[df[f"b_{dim}"] == pole]
            wins_a = (sub_a["winner"] == sub_a["a_mbti"]).sum()
            wins_b = (sub_b["winner"] == sub_b["b_mbti"]).sum()
            total = len(sub_a) + len(sub_b)
            wins = int(wins_a + wins_b)
            rows.append({
                "pole": pole,
                "matches": total,
                "wins": wins,
                "win_rate": wins / total if total else 0,
            })
        results[dim] = pd.DataFrame(rows).sort_values("pole").reset_index(drop=True)
    return results


def dimension_chi_squared(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    For each MBTI dimension, chi-squared test of whether escalation rate
    differs significantly between the two poles.
    """
    if not HAS_SCIPY:
        print("  Skipped (scipy not installed).")
        return pd.DataFrame()

    rows = []
    for dim in DIMENSIONS:
        a_col = df[[f"a_{dim}", "escalate_a"]].rename(columns={f"a_{dim}": "pole", "escalate_a": "esc"})
        b_col = df[[f"b_{dim}", "escalate_b"]].rename(columns={f"b_{dim}": "pole", "escalate_b": "esc"})
        long = pd.concat([a_col, b_col], ignore_index=True)
        poles = sorted(long["pole"].dropna().unique())
        if len(poles) != 2:
            continue
        g0 = long[long["pole"] == poles[0]]["esc"]
        g1 = long[long["pole"] == poles[1]]["esc"]
        ct = np.array([
            [int(g0.sum()), int(len(g0) - g0.sum())],
            [int(g1.sum()), int(len(g1) - g1.sum())],
        ])
        chi2, p, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
        rows.append({
            "dimension": dim,
            f"{poles[0]}_esc_rate": round(float(g0.mean()), 4),
            f"{poles[1]}_esc_rate": round(float(g1.mean()), 4),
            "chi2": round(chi2, 4),
            "p_value": round(p, 4),
            "significant_p05": p < 0.05,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------

def compare_conditions(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Top-level metrics table across all conditions."""
    rows = []
    for cond, df in dfs.items():
        n = len(df)
        esc_rate = (df["escalate_a"].sum() + df["escalate_b"].sum()) / (2 * n)
        mutual_esc = (df["escalate_a"] & df["escalate_b"]).mean()
        mutual_yield = (~df["escalate_a"] & ~df["escalate_b"]).mean()
        rows.append({
            "condition": cond,
            "n_matches": n,
            "escalation_rate": round(esc_rate, 4),
            "mutual_escalation_rate": round(mutual_esc, 4),
            "mutual_yield_rate": round(mutual_yield, 4),
        })
    return pd.DataFrame(rows)


def condition_pairwise_chi_squared(dfs: dict[str, pd.DataFrame]) -> None:
    """Chi-squared test for escalation rate between each pair of conditions."""
    if not HAS_SCIPY:
        print("  Skipped (scipy not installed).")
        return
    keys = list(dfs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c1, c2 = keys[i], keys[j]
            esc1 = pd.concat([dfs[c1]["escalate_a"], dfs[c1]["escalate_b"]])
            esc2 = pd.concat([dfs[c2]["escalate_a"], dfs[c2]["escalate_b"]])
            ct = np.array([
                [int(esc1.sum()), int(len(esc1) - esc1.sum())],
                [int(esc2.sum()), int(len(esc2) - esc2.sum())],
            ])
            chi2, p, _, _ = scipy_stats.chi2_contingency(ct, correction=False)
            sig = "** SIGNIFICANT **" if p < 0.05 else "not significant"
            print(f"  {c1} vs {c2}: chi2={chi2:.3f}, p={p:.4f}  ({sig})")


def champion_distribution(champ_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Champion frequency (count + rate) per MBTI type per condition."""
    rows = []
    for cond, df in champ_dfs.items():
        counts = df["champion_mbti"].value_counts()
        total = len(df)
        for mbti, n in counts.items():
            rows.append({
                "condition": cond,
                "mbti": mbti,
                "champion_count": n,
                "champion_rate": round(n / total, 4),
            })
    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "champion_count"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Reasoning trace analysis
# ---------------------------------------------------------------------------

def reasoning_trace_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Keyword analysis of 'reason_a' / 'reason_b' fields (if present).
    Returns None if neither field exists in the DataFrame.
    """
    has_a = "reason_a" in df.columns
    has_b = "reason_b" in df.columns
    if not has_a and not has_b:
        return None

    rows = []
    for side, mbti_col, esc_col, reason_col in [
        ("a", "a_mbti", "escalate_a", "reason_a"),
        ("b", "b_mbti", "escalate_b", "reason_b"),
    ]:
        if reason_col not in df.columns:
            continue
        sub = df[[mbti_col, esc_col, reason_col]].dropna(subset=[reason_col])
        for _, row in sub.iterrows():
            reason = str(row[reason_col]).lower()
            esc_kw = sum(1 for kw in ESCALATION_KEYWORDS if kw in reason)
            yld_kw = sum(1 for kw in YIELD_KEYWORDS if kw in reason)
            rows.append({
                "mbti": row[mbti_col],
                "escalated": bool(row[esc_col]),
                "esc_keywords": esc_kw,
                "yield_keywords": yld_kw,
                "lean": "escalation" if esc_kw > yld_kw else ("yield" if yld_kw > esc_kw else "neutral"),
            })

    if not rows:
        return None

    trace_df = pd.DataFrame(rows)
    return (
        trace_df.groupby("mbti")
        .agg(
            n=("escalated", "count"),
            escalation_rate=("escalated", "mean"),
            avg_esc_keywords=("esc_keywords", "mean"),
            avg_yield_keywords=("yield_keywords", "mean"),
        )
        .reset_index()
        .sort_values("escalation_rate", ascending=False)
    )


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, out_dir: Optional[str], filename: str) -> None:
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(out_dir) / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_escalation_by_mbti(dfs: dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """Grouped bar chart: escalation rate per MBTI type, one group per condition."""
    all_mbti = sorted(set(
        m for df in dfs.values()
        for m in pd.concat([df["a_mbti"], df["b_mbti"]]).unique()
    ))
    conditions = list(dfs.keys())
    n = len(conditions)
    x = np.arange(len(all_mbti))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cond in enumerate(conditions):
        rates = escalation_by_mbti(dfs[cond]).set_index("mbti")["escalation_rate"]
        vals = [float(rates.get(m, 0)) for m in all_mbti]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cond)

    ax.set_xticks(x)
    ax.set_xticklabels(all_mbti, rotation=45, ha="right")
    ax.set_ylabel("Escalation Rate")
    ax.set_title("Escalation Rate by MBTI Type and Condition")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, out_dir, "escalation_by_mbti.png")


def plot_dimension_escalation(dfs: dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """2×2 subplot: escalation rate per dimension pole per condition."""
    conditions = list(dfs.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, dim in zip(axes.flat, DIMENSIONS):
        pole_rates = {}
        for cond, df in dfs.items():
            by_dim = escalation_by_dimension(df)[dim].set_index("pole")["escalation_rate"]
            pole_rates[cond] = by_dim

        poles = sorted(set(p for pr in pole_rates.values() for p in pr.index))
        x = np.arange(len(poles))
        width = 0.8 / len(conditions)

        for i, cond in enumerate(conditions):
            vals = [float(pole_rates[cond].get(p, 0)) for p in poles]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cond)

        ax.set_xticks(x)
        ax.set_xticklabels(poles)
        ax.set_title(f"Dimension: {DIM_LABELS[dim]}")
        ax.set_ylabel("Escalation Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=7)

    plt.suptitle("Escalation Rate by MBTI Dimension Pole", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, out_dir, "dimension_escalation.png")


def plot_win_rate_by_dimension(dfs: dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """2×2 subplot: win rate per dimension pole per condition."""
    conditions = list(dfs.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, dim in zip(axes.flat, DIMENSIONS):
        pole_wr = {}
        for cond, df in dfs.items():
            wr = win_rate_by_dimension(df)[dim].set_index("pole")["win_rate"]
            pole_wr[cond] = wr

        poles = sorted(set(p for pr in pole_wr.values() for p in pr.index))
        x = np.arange(len(poles))
        width = 0.8 / len(conditions)

        for i, cond in enumerate(conditions):
            vals = [float(pole_wr[cond].get(p, 0)) for p in poles]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cond)

        ax.set_xticks(x)
        ax.set_xticklabels(poles)
        ax.set_title(f"Dimension: {DIM_LABELS[dim]}")
        ax.set_ylabel("Win Rate")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=7)

    plt.suptitle("Win Rate by MBTI Dimension Pole", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, out_dir, "dimension_win_rate.png")


def plot_champion_heatmap(champ_dfs: dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """Heatmap: champion rate by MBTI type (rows) and condition (columns)."""
    all_mbti = sorted(set(m for df in champ_dfs.values() for m in df["champion_mbti"].unique()))
    conditions = list(champ_dfs.keys())
    data = np.zeros((len(all_mbti), len(conditions)))

    for j, cond in enumerate(conditions):
        counts = champ_dfs[cond]["champion_mbti"].value_counts()
        total = len(champ_dfs[cond])
        for i, m in enumerate(all_mbti):
            data[i, j] = counts.get(m, 0) / total if total else 0

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 2), max(6, len(all_mbti) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticks(range(len(all_mbti)))
    ax.set_yticklabels(all_mbti)
    plt.colorbar(im, ax=ax, label="Champion Rate")
    ax.set_title("Champion Rate by MBTI Type and Condition")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "champion_heatmap.png")


def plot_payoff_by_mbti(dfs: dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """Grouped bar chart: mean payoff per MBTI type per condition."""
    all_mbti = sorted(set(
        m for df in dfs.values()
        for m in pd.concat([df["a_mbti"], df["b_mbti"]]).unique()
    ))
    conditions = list(dfs.keys())
    n = len(conditions)
    x = np.arange(len(all_mbti))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cond in enumerate(conditions):
        payoffs = payoff_by_mbti(dfs[cond]).set_index("mbti")["mean_payoff"]
        vals = [float(payoffs.get(m, 0)) for m in all_mbti]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=cond)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(all_mbti, rotation=45, ha="right")
    ax.set_ylabel("Mean Payoff")
    ax.set_title("Mean Payoff by MBTI Type and Condition")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, out_dir, "payoff_by_mbti.png")

def summarize_results(path: str) -> None:
    """Print a summary for a single results file (legacy entry point)."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        return
    df = load_matches(path)
    condition = df["condition"].iloc[0] if "condition" in df.columns and len(df) else p.name

    print(f"\n{'='*50}")
    print(f"SUMMARY: {condition}  ({len(df)} matches)")
    print(f"{'='*50}")
    esc_rate = (df["escalate_a"].sum() + df["escalate_b"].sum()) / (2 * len(df))
    print(f"Escalation rate:        {esc_rate:.3f}")
    print(f"Mutual escalation rate: {(df['escalate_a'] & df['escalate_b']).mean():.3f}")
    print(f"Mutual yield rate:      {(~df['escalate_a'] & ~df['escalate_b']).mean():.3f}")
    print("\nPer-MBTI escalation rates:")
    print(escalation_by_mbti(df).to_string(index=False))
    print("\nPer-MBTI win rates:")
    print(win_rate_by_mbti(df).to_string(index=False))
    print("\nPer-MBTI mean payoffs:")
    print(payoff_by_mbti(df).to_string(index=False))

def main(out_dir: Optional[str] = "data/analysis") -> None:
    paths = resolve_paths()
    if not paths:
        print("No result files found. Run experiments first.")
        print("Expected files:", list(DEFAULT_FILES.values()))
        return

    print(f"Found condition files: {list(paths.keys())}")

    dfs: dict[str, pd.DataFrame] = {}
    champ_dfs: dict[str, pd.DataFrame] = {}
    for cond, path in paths.items():
        dfs[cond] = load_matches(path)
        champ_dfs[cond] = load_champions(path)

    # ------------------------------------------------------------------
    # 1. Per-condition summaries
    # ------------------------------------------------------------------
    for cond, df in dfs.items():
        n = len(df)
        esc_rate = (df["escalate_a"].sum() + df["escalate_b"].sum()) / (2 * n)
        print(f"\n{'='*55}")
        print(f"CONDITION: {cond}  ({n} matches)")
        print(f"{'='*55}")
        print(f"Escalation rate:        {esc_rate:.3f}")
        print(f"Mutual escalation rate: {(df['escalate_a'] & df['escalate_b']).mean():.3f}")
        print(f"Mutual yield rate:      {(~df['escalate_a'] & ~df['escalate_b']).mean():.3f}")
        print("\nPer-MBTI escalation rates:")
        print(escalation_by_mbti(df).to_string(index=False))
        print("\nPer-MBTI win rates:")
        print(win_rate_by_mbti(df).to_string(index=False))
        print("\nPer-MBTI mean payoffs:")
        print(payoff_by_mbti(df).to_string(index=False))

    # ------------------------------------------------------------------
    # 2. MBTI dimension analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("MBTI DIMENSION ANALYSIS — ESCALATION RATE")
    print(f"{'='*55}")
    for cond, df in dfs.items():
        print(f"\n  --- {cond} ---")
        for dim, ddf in escalation_by_dimension(df).items():
            row = ddf.set_index("pole")["escalation_rate"]
            poles = sorted(row.index)
            if len(poles) == 2:
                p0, p1 = poles
                print(f"    {dim}: {p0}={row[p0]:.3f}  {p1}={row[p1]:.3f}  (diff={abs(row[p0]-row[p1]):.3f})")

    print(f"\n{'='*55}")
    print("MBTI DIMENSION ANALYSIS — WIN RATE")
    print(f"{'='*55}")
    for cond, df in dfs.items():
        print(f"\n  --- {cond} ---")
        for dim, ddf in win_rate_by_dimension(df).items():
            row = ddf.set_index("pole")["win_rate"]
            poles = sorted(row.index)
            if len(poles) == 2:
                p0, p1 = poles
                print(f"    {dim}: {p0}={row[p0]:.3f}  {p1}={row[p1]:.3f}  (diff={abs(row[p0]-row[p1]):.3f})")

    # Chi-squared for each condition
    for cond, df in dfs.items():
        print(f"\n  --- Chi-squared (dimension escalation, {cond}) ---")
        chi_df = dimension_chi_squared(df, label=cond)
        if not chi_df.empty:
            print(chi_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Condition comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("CONDITION COMPARISON")
    print(f"{'='*55}")
    comp = compare_conditions(dfs)
    print(comp.to_string(index=False))
    print("\nPairwise chi-squared tests (escalation rate):")
    condition_pairwise_chi_squared(dfs)

    # ------------------------------------------------------------------
    # 4. Champion analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("CHAMPION ANALYSIS")
    print(f"{'='*55}")
    champ_table = champion_distribution(champ_dfs)
    for cond in champ_table["condition"].unique():
        print(f"\n  {cond}:")
        sub = champ_table[champ_table["condition"] == cond]
        print(sub[["mbti", "champion_count", "champion_rate"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Reasoning trace (if reason fields are logged)
    # ------------------------------------------------------------------
    for cond, df in dfs.items():
        trace = reasoning_trace_analysis(df)
        if trace is not None:
            print(f"\n{'='*55}")
            print(f"REASONING TRACE ANALYSIS: {cond}")
            print(f"{'='*55}")
            print(trace.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Visualizations
    # ------------------------------------------------------------------
    if out_dir:
        print(f"\n{'='*55}")
        print(f"GENERATING VISUALIZATIONS → {out_dir}/")
        print(f"{'='*55}")
        plot_escalation_by_mbti(dfs, out_dir)
        plot_dimension_escalation(dfs, out_dir)
        plot_win_rate_by_dimension(dfs, out_dir)
        plot_payoff_by_mbti(dfs, out_dir)
        if any(len(df) > 0 for df in champ_dfs.values()):
            plot_champion_heatmap(champ_dfs, out_dir)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Game of Chicken results.")
    parser.add_argument(
        "--out-dir",
        default="data/analysis",
        help="Directory to save plots (default: data/analysis). Pass '' to show interactively.",
    )
    args = parser.parse_args()
    main(out_dir=args.out_dir or None)