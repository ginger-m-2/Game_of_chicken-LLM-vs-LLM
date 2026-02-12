from __future__ import annotations
from dotenv import load_dotenv
import os
import pandas as pd

def summarize_results(csv_path: str = "data/results.csv") -> None:
    df = pd.read_csv(csv_path)

    win_counts = df["winner_mbti"].value_counts().rename_axis("mbti").reset_index(name="wins")
    print("\nWin counts by MBTI:")
    print(win_counts.to_string(index=False))

    # Trait-level quick summaries (E/I, T/F, J/P)
    df["E_I"] = df["winner_mbti"].str[0]
    df["S_N"] = df["winner_mbti"].str[1]
    df["T_F"] = df["winner_mbti"].str[2]
    df["J_P"] = df["winner_mbti"].str[3]

    print("\nWins by trait:")
    for col in ["E_I", "S_N", "T_F", "J_P"]:
        counts = df[col].value_counts()
        print(f"{col}:\n{counts.to_string()}\n")