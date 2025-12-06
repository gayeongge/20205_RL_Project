import argparse
import json
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "league_results.jsonl"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
SUMMARY_PATH = Path(__file__).resolve().parent / "summary_page.md"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_records(limit=None) -> List[Dict]:
    records = []
    with DATA_PATH.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
            if limit and len(records) >= limit:
                break
    return records


def compute_role_table(records: List[Dict]) -> pd.DataFrame:
    stats = defaultdict(lambda: {
        "start_games": 0,
        "start_wins": 0,
        "start_draws": 0,
        "second_games": 0,
        "second_wins": 0,
        "second_draws": 0,
    })
    for rec in records:
        first, second = rec["players"]
        winner = rec["winner"]
        stats[first]["start_games"] += 1
        stats[second]["second_games"] += 1
        if winner == first:
            stats[first]["start_wins"] += 1
        elif winner == second:
            stats[second]["second_wins"] += 1
        else:
            stats[first]["start_draws"] += 1
            stats[second]["second_draws"] += 1
    frame = pd.DataFrame.from_dict(stats, orient="index")
    frame["start_win_rate"] = frame["start_wins"] / frame["start_games"]
    frame["second_win_rate"] = frame["second_wins"] / frame["second_games"]
    return frame


def build_dataframe(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for step in rec["steps"]:
            rows.append({
                "match_id": rec["match_id"],
                "players": tuple(rec["players"]),
                "starter": rec["starter"],
                "winner": rec["winner"],
                "turns": rec["turns"],
                "player": step["player"],
                "action": step["action"],
                "forced_win_available": step["forced_win_available"],
                "forced_win_success": step["forced_win_success"],
                "forced_block_available": step["forced_block_available"],
                "forced_block_success": step["forced_block_success"],
                "central_move": step["central_move"],
                "decision_time_ms": step.get("decision_time_ms", np.nan),
                "search_nodes": step.get("search_nodes", np.nan),
                "search_depth": step.get("search_depth", np.nan),
                "playable_actions": step.get("playable_actions", np.nan),
            })
    return pd.DataFrame(rows)


def annotate_bars(bars, fmt="{:.2f}"):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        plt.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def annotate_forced_rates(bars, rates):
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        if np.isnan(rate):
            label = "N/A"
        else:
            label = f"{rate * 100:.0f}%"
        plt.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def set_xtick_style(rotation=20):
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment("right")


def plot_forced(forced: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(forced.index))
    width = 0.35
    win_rates = forced["win_rate"].fillna(0).to_numpy()
    block_rates = forced["block_rate"].fillna(0).to_numpy()
    win_bars = plt.bar(x - width / 2, win_rates, width=width, label="Forced Win")
    block_bars = plt.bar(x + width / 2, block_rates, width=width, label="Forced Block")
    annotate_forced_rates(win_bars, forced["win_rate"].tolist())
    annotate_forced_rates(block_bars, forced["block_rate"].tolist())
    plt.xticks(x, forced.index)
    set_xtick_style(rotation=25)
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.title("Forced Win / Block Response Rate")
    plt.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "forced_response.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_turns(agg_turns: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    x = np.arange(len(agg_turns.index))
    bars = plt.bar(x, agg_turns["avg_turns"], yerr=agg_turns["std_turns"], capsize=5)
    annotate_bars(bars, fmt="{:.1f}")
    plt.xticks(x, agg_turns.index)
    set_xtick_style(rotation=25)
    plt.ylabel("Turns")
    plt.title("Average Game Length (+/- 1 std)")
    plt.tight_layout()
    path = PLOTS_DIR / "average_turns.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_search_cost(agg_search: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    plt.scatter(agg_search["search_nodes"], agg_search["decision_time_ms"], s=120)
    offset_cycle = cycle([(5, 6), (-5, 6), (5, -6), (-5, -6), (0, 10), (0, -10)])
    for player, row in agg_search.iterrows():
        dx, dy = next(offset_cycle)
        plt.annotate(
            player,
            (row["search_nodes"], row["decision_time_ms"]),
            textcoords="offset points",
            xytext=(dx, dy),
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
            fontsize=8,
        )
    plt.xscale("log")
    plt.xlabel("Average Search Nodes (log)")
    plt.ylabel("Average Decision Time (ms)")
    plt.title("Search Cost vs Decision Time")
    plt.tight_layout()
    path = PLOTS_DIR / "search_cost.png"
    plt.savefig(path)
    plt.close()
    return path


def main(limit: int):
    records = load_records(limit)
    if not records:
        raise SystemExit("No valid records found")
    role_table = compute_role_table(records)
    df = build_dataframe(records)

    agg_turns = df.groupby("player")["turns"].agg(["mean", "std"]).rename(columns={"mean": "avg_turns", "std": "std_turns"})

    forced = df.groupby("player").apply(lambda g: pd.Series({
        "win_available": g["forced_win_available"].sum(),
        "win_success": g["forced_win_success"].sum(),
        "block_available": g["forced_block_available"].sum(),
        "block_success": g["forced_block_success"].sum(),
    }))
    forced["win_rate"] = forced.apply(lambda r: r["win_success"] / r["win_available"] if r["win_available"] else np.nan, axis=1)
    forced["block_rate"] = forced.apply(lambda r: r["block_success"] / r["block_available"] if r["block_available"] else np.nan, axis=1)

    agg_search = df.groupby("player").agg({
        "decision_time_ms": "mean",
        "search_nodes": "mean",
    })

    forced_plot = plot_forced(forced)
    turns_plot = plot_turns(agg_turns)
    search_plot = plot_search_cost(agg_search)

    summary_lines = [
        "# League Metrics Summary",
        "",
        "## Start vs Second Win Table",
        role_table[["start_games", "start_wins", "start_win_rate", "second_games", "second_wins", "second_win_rate"]].to_string(),
        "",
        "## Forced Move Response",
        forced[["win_available", "win_success", "win_rate", "block_available", "block_success", "block_rate"]].to_string(),
        "",
        "## Average Turns",
        agg_turns.to_string(),
        "",
        "## Search Cost",
        agg_search.to_string(),
        "",
        "## Plots",
        f"- Forced response: {forced_plot.relative_to(Path.cwd())}",
        f"- Average turns: {turns_plot.relative_to(Path.cwd())}",
        f"- Search cost: {search_plot.relative_to(Path.cwd())}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print("Saved plots and summary page")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to read")
    args = parser.parse_args()
    main(args.limit)
