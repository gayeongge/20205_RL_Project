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


def annotate_bars(bars, fmt="{:.2f}", offset=6):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax = bar.axes
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def annotate_forced_rates(bars, rates):
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        label = "N/A" if np.isnan(rate) else f"{rate * 100:.0f}%"
        ax = bar.axes
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def set_xtick_style(ax, rotation=20):
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")


def format_pair_label(pair):
    if isinstance(pair, str):
        return pair
    first, second = pair
    return f"{first} vs\n{second}"


def plot_forced(forced_df):
    players = forced_df.index.tolist()
    x = np.arange(len(players))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    win_rates = forced_df["win_rate"].fillna(0).to_numpy()
    block_rates = forced_df["block_rate"].fillna(0).to_numpy()
    win_bars = ax.bar(x - width / 2, win_rates, width, label="Forced Win")
    block_bars = ax.bar(x + width / 2, block_rates, width, label="Forced Block")
    annotate_forced_rates(win_bars, forced_df["win_rate"].tolist())
    annotate_forced_rates(block_bars, forced_df["block_rate"].tolist())
    ax.set_xticks(x)
    ax.set_xticklabels(players)
    set_xtick_style(ax, rotation=25)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_title("Forced Win / Block Success")
    ax.legend()
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "forced_rates.png")
    plt.close(fig)


def plot_turns(match_df):
    labeled = match_df.copy()
    labeled["pair_label"] = labeled["players"].apply(format_pair_label)
    avg_turns = labeled.groupby("pair_label")["turns"].agg(["mean", "std"]).rename(columns={"mean": "avg", "std": "std"})
    x = np.arange(len(avg_turns.index))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, avg_turns["avg"], yerr=avg_turns["std"], capsize=5)
    annotate_bars(bars, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(avg_turns.index)
    ax.set_ylabel("Average Turns")
    ax.set_title("Average Turns per Pairing")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "avg_turns.png")
    plt.close(fig)


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
    forced["win_rate"] = forced.apply(
        lambda row: row["forced_win_success"] / row["forced_win_available"] if row["forced_win_available"] else np.nan,
        axis=1,
    )
    forced["block_rate"] = forced.apply(
        lambda row: row["forced_block_success"] / row["forced_block_available"] if row["forced_block_available"] else np.nan,
        axis=1,
    )

    plot_forced(forced)
    plot_turns(match_df)


if __name__ == "__main__":
    main()
