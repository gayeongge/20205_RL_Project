import argparse
import importlib
import itertools
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

DQN_MODULES = [
    {"key": "dqn", "module": "DQN.drl_dqn"},
    {"key": "double", "module": "DQN.drl_double"},
    {"key": "dueling", "module": "DQN.drl_dueling"},
]

DEFAULT_GAMES_PER_ORDER = 5
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "dqn_results.jsonl"


@dataclass
class AgentWrapper:
    name: str
    fn: Callable

    def __call__(self, obs, cfg):
        return self.fn(obs, cfg)


def extract_board(obs) -> List[int]:
    board = getattr(obs, "board", None)
    if board is None:
        board = obs["board"]
    return list(board)


def find_playable_columns(board: List[int], cfg: SimpleNamespace) -> List[int]:
    return [c for c in range(cfg.columns) if board[c] == 0]


def drop_piece(board: List[int], col: int, mark: int, cfg: SimpleNamespace) -> Optional[List[int]]:
    new_board = board.copy()
    for r in range(cfg.rows - 1, -1, -1):
        idx = r * cfg.columns + col
        if new_board[idx] == 0:
            new_board[idx] = mark
            return new_board
    return None


def check_win(board: List[int], mark: int, cfg: SimpleNamespace) -> bool:
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    for r in range(rows):
        offset = r * cols
        for c in range(cols - n + 1):
            if all(board[offset + c + i] == mark for i in range(n)):
                return True
    for c in range(cols):
        for r in range(rows - n + 1):
            if all(board[(r + i) * cols + c] == mark for i in range(n)):
                return True
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(board[(r + i) * cols + (c + i)] == mark for i in range(n)):
                return True
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(board[(r - i) * cols + (c + i)] == mark for i in range(n)):
                return True
    return False


def find_winning_moves(board: List[int], mark: int, cfg: SimpleNamespace) -> List[int]:
    wins = []
    for col in find_playable_columns(board, cfg):
        next_board = drop_piece(board, col, mark, cfg)
        if next_board is not None and check_win(next_board, mark, cfg):
            wins.append(col)
    return wins


def build_wrappers() -> Dict[str, AgentWrapper]:
    wrappers = {}
    for spec in DQN_MODULES:
        module = importlib.import_module(spec["module"])
        wrappers[spec["key"]] = AgentWrapper(spec["key"], getattr(module, "my_agent"))
    return wrappers


def build_turn_logs(order: List[str], histories: Dict[str, List[Dict]], cfg: SimpleNamespace) -> List[Dict]:
    indices = {name: 0 for name in order}
    steps: List[Dict] = []
    total_moves = sum(len(histories[name]) for name in order)
    for turn in range(total_moves):
        player_name = order[turn % 2]
        if indices[player_name] >= len(histories[player_name]):
            break
        entry = histories[player_name][indices[player_name]]
        indices[player_name] += 1
        board = entry["board"]
        action = entry["action"]
        mark = entry["mark"]
        win_moves = find_winning_moves(board, mark, cfg)
        block_moves = find_winning_moves(board, 3 - mark, cfg)
        forced_win = bool(win_moves)
        took_win = forced_win and action in win_moves
        forced_block = bool(block_moves)
        blocked = False
        if forced_block and (action in block_moves or took_win):
            blocked = True
        turn_log = {
            "turn_index": len(steps),
            "player": player_name,
            "action": action,
            "forced_win_available": forced_win,
            "forced_win_success": took_win,
            "forced_block_available": forced_block,
            "forced_block_success": blocked,
            "decision_time_ms": entry.get("decision_time_ms"),
        }
        steps.append(turn_log)
    return steps


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DQN variants")
    parser.add_argument("--games-per-order", type=int, default=DEFAULT_GAMES_PER_ORDER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def next_match_id(path: Path) -> int:
    if not path.exists():
        return 0
    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip():
                last = line.strip()
    if not last:
        return 0
    try:
        data = json.loads(last)
        return int(data.get("match_id", 0)) + 1
    except Exception:
        return 0


def main():
    args = parse_args()
    random.seed(args.seed)
    env = make("connectx", debug=False)
    cfg = SimpleNamespace(
        rows=int(env.configuration.rows),
        columns=int(env.configuration.columns),
        inarow=int(env.configuration.inarow),
    )
    wrappers = build_wrappers()
    matchups = list(itertools.combinations([spec["key"] for spec in DQN_MODULES], 2))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    match_counter = next_match_id(output_path) if args.append else 0

    with output_path.open(mode, encoding="utf-8") as fh:
        for a, b in matchups:
            for order in [(a, b), (b, a)]:
                for game_idx in range(args.games_per_order):
                    env.reset()
                    histories: Dict[str, List[Dict]] = {name: [] for name in order}

                    def wrapped_agent(name):
                        agent = wrappers[name]

                        def play(obs, configuration):
                            start = time.perf_counter()
                            action = agent(obs, configuration)
                            duration = (time.perf_counter() - start) * 1000.0
                            histories[name].append(
                                {
                                    "board": extract_board(obs),
                                    "mark": obs.mark if hasattr(obs, "mark") else obs["mark"],
                                    "action": int(action),
                                    "decision_time_ms": duration,
                                }
                            )
                            return action

                        return play

                    env.run([wrapped_agent(order[0]), wrapped_agent(order[1])])
                    result = env.state
                    rewards_raw = [result[0].reward, result[1].reward]
                    rewards = [r if r is not None else 0 for r in rewards_raw]
                    if rewards[0] > rewards[1]:
                        winner = order[0]
                    elif rewards[1] > rewards[0]:
                        winner = order[1]
                    else:
                        winner = "draw"
                    steps = build_turn_logs(list(order), histories, cfg)
                    record = {
                        "match_id": match_counter,
                        "players": order,
                        "starter": order[0],
                        "game_index": game_idx,
                        "winner": winner,
                        "rewards": rewards,
                        "turns": len(steps),
                        "steps": steps,
                    }
                    fh.write(json.dumps(record) + "\n")
                    match_counter += 1

    print(f"Saved {match_counter} games to {output_path}")


if __name__ == "__main__":
    main()
