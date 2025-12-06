import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "dqn_results.jsonl"


def load_records():
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing results file: {DATA_PATH}")
    records = []
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_tables(records):
    match_rows = []
    turn_rows = []
    for rec in records:
        match_rows.append(
            {
                "match_id": rec["match_id"],
                "players": tuple(rec["players"]),
                "starter": rec["starter"],
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
                    "decision_time_ms": step.get("decision_time_ms", np.nan),
                }
            )
    match_df = pd.DataFrame(match_rows)
    turn_df = pd.DataFrame(turn_rows)
    return match_df, turn_df


def summarize():
    records = load_records()
    match_df, turn_df = build_tables(records)

    win_table = match_df.groupby(["players", "winner"]).size().unstack(fill_value=0)
    start_table = (
        match_df.groupby(["starter", "winner"]).size().unstack(fill_value=0)
    )
    avg_turns = match_df.groupby(match_df["players"].apply(tuple))["turns"].mean()

    forced = turn_df.groupby("player").agg(
        {
            "forced_win_available": "sum",
            "forced_win_success": "sum",
            "forced_block_available": "sum",
            "forced_block_success": "sum",
            "decision_time_ms": "mean",
        }
    )
    forced["win_rate"] = forced["forced_win_success"] / forced["forced_win_available"]
    forced["block_rate"] = forced["forced_block_success"] / forced["forced_block_available"]

    print("=== Win table ===")
    print(win_table)
    print("\n=== Starter outcomes ===")
    print(start_table)
    print("\n=== Average turns per pairing ===")
    print(avg_turns)
    print("\n=== Forced-move stats ===")
    print(forced)


if __name__ == "__main__":
    summarize()
