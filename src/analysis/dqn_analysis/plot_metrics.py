import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "dqn_results.jsonl"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def load_frames():
    records = [json.loads(line) for line in DATA_PATH.open("r", encoding="utf-8") if line.strip()]
    match_rows = []
    turn_rows = []
    for rec in records:
        match_rows.append(
            {
                "match_id": rec["match_id"],
                "players": tuple(rec["players"]),
                "winner": rec["winner"],
                "turns": rec["turns"],
            }
        )
        for step in rec["steps"]:
            turn_rows.append(
                {
                    "match_id": rec["match_id"],
                    "player": step["player"],
                    "forced_win_available": step["forced_win_available"],
                    "forced_win_success": step["forced_win_success"],
                    "forced_block_available": step["forced_block_available"],
                    "forced_block_success": step["forced_block_success"],
                    "decision_time_ms": step["decision_time_ms"],
                }
            )
    return pd.DataFrame(match_rows), pd.DataFrame(turn_rows)


def plot_forced(forced_df):
    forced_df[["win_rate", "block_rate"]].plot(kind="bar", ylim=(0, 1.05))
    plt.xticks(rotation=0)
    plt.ylabel("Ratio")
    plt.title("Forced Win / Block Success")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "forced_rates.png")
    plt.close()


def plot_turns(match_df):
    match_df.groupby(match_df["players"].astype(str))["turns"].mean().plot(kind="bar")
    plt.ylabel("Avg Turns")
    plt.title("Average Turns per Pairing")
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "avg_turns.png")
    plt.close()


def main():
    match_df, turn_df = load_frames()
    forced = turn_df.groupby("player").agg(
        {
            "forced_win_available": "sum",
            "forced_win_success": "sum",
            "forced_block_available": "sum",
            "forced_block_success": "sum",
        }
    )
    forced["win_rate"] = forced["forced_win_success"] / forced["forced_win_available"]
    forced["block_rate"] = forced["forced_block_success"] / forced["forced_block_available"]

    plot_forced(forced)
    plot_turns(match_df)


if __name__ == "__main__":
    main()
