import argparse
import itertools
import importlib
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
SUBMISSION_DIR = SRC_DIR / "submission_files"
if SUBMISSION_DIR.exists() and str(SUBMISSION_DIR) not in sys.path:
    sys.path.append(str(SUBMISSION_DIR))

AGENT_SPECS = [
    {"key": "one_head", "module": "submission_simple_greedy_baseline", "label": "Simple Greedy"},
    {"key": "n_head", "module": "submission_n_step_lookahead", "label": "N-Step Lookahead"},
    {"key": "mtdf", "module": "submission_mtdf_negamax_strategic", "label": "MTD(f) Negamax"},
    {"key": "alphazero", "module": "submission_alphazero_mcts_gap_defense", "label": "AlphaZero Gap Defense"},
    {"key": "dqn_double", "module": "submission_drl_double", "label": "Double DQN"},
]

DEFAULT_GAMES_PER_ORDER = 1
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "league_results.jsonl"


@dataclass
class Instrumentation:
    pre: Optional[Callable] = None
    post: Optional[Callable] = None


class AgentWrapper:
    def __init__(self, name: str, fn: Callable, instrumentation: Optional[Instrumentation] = None):
        self.name = name
        self.fn = fn
        self.instrumentation = instrumentation or Instrumentation()
        self.history: List[Dict] = []

    def reset_history(self) -> None:
        self.history.clear()

    def __call__(self, obs, cfg):
        board = extract_board(obs)
        meta = {}
        if self.instrumentation.pre:
            meta.update(self.instrumentation.pre(obs, cfg))
        start = time.perf_counter()
        action = self.fn(obs, cfg)
        meta["decision_time_ms"] = (time.perf_counter() - start) * 1000.0
        if self.instrumentation.post:
            post_meta = self.instrumentation.post()
            if post_meta:
                meta.update(post_meta)
        self.history.append(
            {
                "board": board,
                "mark": extract_mark(obs),
                "action": int(action),
                "meta": meta,
            }
        )
        return int(action)


def extract_board(obs) -> List[int]:
    board = getattr(obs, "board", None)
    if board is None:
        board = obs["board"]
    return list(board)


def extract_mark(obs) -> int:
    mark = getattr(obs, "mark", None)
    if mark is None:
        mark = obs["mark"]
    return int(mark)


def get_playable_columns(board: List[int], cfg: SimpleNamespace) -> List[int]:
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
            window = board[offset + c : offset + c + n]
            if all(val == mark for val in window):
                return True
    for c in range(cols):
        for r in range(rows - n + 1):
            window = [board[(r + i) * cols + c] for i in range(n)]
            if all(val == mark for val in window):
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
    winning = []
    for col in get_playable_columns(board, cfg):
        next_board = drop_piece(board, col, mark, cfg)
        if next_board is not None and check_win(next_board, mark, cfg):
            winning.append(col)
    return winning


def build_instrumentation(key: str, module) -> Instrumentation:
    if key == "one_head":
        def pre(obs, cfg):
            board = extract_board(obs)
            playable = get_playable_columns(board, cfg)
            return {"search_nodes": len(playable), "search_depth": 1, "playable_count": len(playable)}
        return Instrumentation(pre=pre)

    if key == "dqn_double":
        def pre(obs, cfg):
            board = extract_board(obs)
            playable = get_playable_columns(board, cfg)
            return {"search_nodes": 0, "search_depth": 0, "playable_count": len(playable)}
        return Instrumentation(pre=pre)

    if key == "n_head":
        counter = {"nodes": 0, "depth": 0}
        if hasattr(module, "minimax_alpha_beta"):
            original = module.minimax_alpha_beta

            def wrapped(grid, cur_player, me, depth, alpha, beta, cfg):
                counter["nodes"] += 1
                explored = module.MAX_DEPTH - depth
                if explored > counter["depth"]:
                    counter["depth"] = explored
                return original(grid, cur_player, me, depth, alpha, beta, cfg)

            module.minimax_alpha_beta = wrapped
        else:
            original = module.minimax
            max_depth = getattr(module, "LOOKAHEAD_DEPTH", 0)

            def wrapped(grid, cur_player, me, depth, cfg):
                counter["nodes"] += 1
                explored = max_depth - depth if max_depth else 0
                if explored > counter["depth"]:
                    counter["depth"] = explored
                return original(grid, cur_player, me, depth, cfg)

            module.minimax = wrapped

        def post():
            data = {"search_nodes": counter["nodes"], "search_depth": counter["depth"], "playable_count": None}
            counter["nodes"] = 0
            counter["depth"] = 0
            return data

        return Instrumentation(post=post)

    if key == "mtdf":
        counter = {"nodes": 0, "depth": 0}
        state = {"root": 0}
        original_negamax = module.negamax
        original_mtdf = module.mtdf_search

        def negamax_wrapper(grid, depth, alpha, beta, player, cfg, start_time):
            counter["nodes"] += 1
            root_depth = state.get("root", depth)
            explored = max(0, root_depth - depth)
            if explored > counter["depth"]:
                counter["depth"] = explored
            return original_negamax(grid, depth, alpha, beta, player, cfg, start_time)

        def mtdf_wrapper(grid, guess, depth, player, cfg, start_time):
            if depth > state.get("root", 0):
                state["root"] = depth
            return original_mtdf(grid, guess, depth, player, cfg, start_time)

        module.negamax = negamax_wrapper
        module.mtdf_search = mtdf_wrapper

        def post():
            data = {"search_nodes": counter["nodes"], "search_depth": counter["depth"], "playable_count": None}
            counter["nodes"] = 0
            counter["depth"] = 0
            state["root"] = 0
            return data

        return Instrumentation(post=post)

    if key == "alphazero":
        stats = {"simulations": 0}
        original_search = module.mcts_search

        def search_wrapper(grid, player, cfg, num_simulations=module.MCTS_SIMULATIONS):
            stats["simulations"] = num_simulations
            return original_search(grid, player, cfg, num_simulations=num_simulations)

        module.mcts_search = search_wrapper

        def post():
            return {"search_nodes": stats.get("simulations", 0), "search_depth": None, "playable_count": None}

        return Instrumentation(post=post)

    return Instrumentation()


def build_wrappers() -> Dict[str, AgentWrapper]:
    wrappers = {}
    for spec in AGENT_SPECS:
        module = importlib.import_module(f"submission_files.{spec['module']}")
        instrumentation = build_instrumentation(spec["key"], module)
        wrappers[spec["key"]] = AgentWrapper(spec["key"], getattr(module, "agent"), instrumentation)
    return wrappers


def central_columns(cfg: SimpleNamespace) -> set:
    center = cfg.columns // 2
    cols = {center}
    if center - 1 >= 0:
        cols.add(center - 1)
    if center + 1 < cfg.columns:
        cols.add(center + 1)
    return cols


def build_turn_logs(order: List[str], histories: Dict[str, List[Dict]], cfg: SimpleNamespace, central_cols: set) -> List[Dict]:
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
            "central_move": action in central_cols,
            "playable_actions": len(get_playable_columns(board, cfg)),
        }
        turn_log.update(entry["meta"])
        steps.append(turn_log)
    return steps


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-play league among submission agents")
    parser.add_argument("--games-per-order", type=int, default=DEFAULT_GAMES_PER_ORDER, help="Number of games per start order for each pairing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Path to store JSONL results")
    parser.add_argument("--pair", action="append", help="Limit to specific agent pairs, e.g. one_head,n_head (repeatable)")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of overwrite")
    return parser.parse_args()


def determine_matchups(pair_args: Optional[List[str]]) -> List[tuple]:
    all_pairs = list(itertools.combinations([spec["key"] for spec in AGENT_SPECS], 2))
    if not pair_args:
        return all_pairs
    selected = set()
    for entry in pair_args:
        tokens = [token.strip() for token in entry.split(",") if token.strip()]
        if len(tokens) != 2:
            raise SystemExit(f"Invalid pair specification: {entry}")
        selected.add(frozenset(tokens))
    filtered = [pair for pair in all_pairs if frozenset(pair) in selected]
    if not filtered:
        raise SystemExit("No valid matchups selected")
    return filtered


def next_match_id(path: Path) -> int:
    if not path.exists():
        return 0
    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            record = line.strip()
            if record:
                last = record
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
    central_cols = central_columns(cfg)
    wrappers = build_wrappers()
    matchups = determine_matchups(args.pair)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    match_counter = next_match_id(output_path) if args.append else 0

    with output_path.open(mode, encoding="utf-8") as fh:
        for a, b in matchups:
            for order in [(a, b), (b, a)]:
                for game_idx in range(args.games_per_order):
                    env.reset()
                    for name in order:
                        wrappers[name].reset_history()
                    env.run([wrappers[order[0]], wrappers[order[1]]])
                    histories = {name: list(wrappers[name].history) for name in order}
                    result = env.state
                    rewards_raw = [result[0].reward, result[1].reward]
                    rewards = [r if r is not None else 0 for r in rewards_raw]
                    if rewards[0] > rewards[1]:
                        winner = order[0]
                    elif rewards[1] > rewards[0]:
                        winner = order[1]
                    else:
                        winner = "draw"
                    steps = build_turn_logs(list(order), histories, cfg, central_cols)
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
