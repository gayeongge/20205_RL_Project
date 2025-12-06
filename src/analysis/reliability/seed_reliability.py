import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List

from kaggle_environments import evaluate

# src/analysis/reliability -> src
SRC_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = SRC_ROOT / "submission_files"
DEFAULT_OUTPUT = Path(__file__).with_name("results.json")
DEFAULT_EPISODES = 20
DEFAULT_SEEDS = [0, 1, 2, 42, 999]

AGENT_SPECS = [
    {"label": "Greedy (1-step)", "filename": "submission_simple_greedy_baseline.py"},
    {"label": "N-step Lookahead", "filename": "submission_n_step_lookahead.py"},
    {"label": "MTD(f) Strategic", "filename": "submission_mtdf_negamax_strategic.py"},
    {"label": "Double DQN", "filename": "submission_drl_double.py"},
    {"label": "AlphaZero MCTS", "filename": "submission_alphazero_mcts_gap_defense.py"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConnectX agent reliability across random seeds")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Games per seed")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit random seeds (e.g. --seeds 0 1 2 42)",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=None,
        help="If provided (and --seeds omitted), sample this many seeds via RNG",
    )
    parser.add_argument("--base-seed", type=int, default=42, help="RNG seed when sampling seeds")
    parser.add_argument("--opponent", type=str, default="random", help="Opponent agent spec")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save JSON summary",
    )
    return parser.parse_args()


def resolve_agent_path(filename: str) -> Path:
    path = SUBMISSION_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find agent file: {path}")
    return path


def determine_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return sorted(dict.fromkeys(args.seeds))
    if args.seed_count:
        if args.seed_count <= 0:
            raise ValueError("--seed-count must be positive")
        random.seed(args.base_seed)
        upper = 10_000
        if args.seed_count > upper:
            raise ValueError(f"--seed-count must be <= {upper}")
        return sorted(random.sample(range(upper), args.seed_count))
    return list(DEFAULT_SEEDS)


def evaluate_agent(agent_path: Path, opponent: str, seed: int, episodes: int) -> Dict[str, float]:
    rewards = evaluate(
        "connectx",
        [str(agent_path), opponent],
        num_episodes=episodes,
        configuration={"randomSeed": seed},
    )
    wins = sum(1 for r in rewards if r[0] == 1)
    draws = sum(1 for r in rewards if r[0] == 0)
    losses = episodes - wins - draws
    win_rate = wins / episodes
    return {
        "seed": seed,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": win_rate,
    }


def summarize_win_rates(win_rates: List[float]) -> Dict[str, float]:
    if not win_rates:
        return {"mean": math.nan, "std": math.nan, "ci95": math.nan}
    mean_val = sum(win_rates) / len(win_rates)
    if len(win_rates) > 1:
        std_val = statistics.stdev(win_rates)
        ci95 = 1.96 * std_val / math.sqrt(len(win_rates))
    else:
        std_val = 0.0
        ci95 = 0.0
    return {"mean": mean_val, "std": std_val, "ci95": ci95}


def run_experiment(args: argparse.Namespace) -> List[Dict]:
    seeds = determine_seeds(args)
    print(f"Evaluating {len(AGENT_SPECS)} agents across seeds {seeds}")
    summary: List[Dict] = []

    for spec in AGENT_SPECS:
        agent_path = resolve_agent_path(spec["filename"])
        seed_results = []
        for seed in seeds:
            try:
                result = evaluate_agent(agent_path, args.opponent, seed, args.episodes)
                seed_results.append(result)
                print(f"  - {spec['label']} @ seed {seed}: win_rate={result['win_rate']:.3f}")
            except Exception as exc:
                print(f"[WARN] {spec['label']} failed on seed {seed}: {exc}")
        win_rates = [item["win_rate"] for item in seed_results]
        stats = summarize_win_rates(win_rates)
        summary.append(
            {
                "agent": spec["label"],
                "file": spec["filename"],
                "opponent": args.opponent,
                "episodes_per_seed": args.episodes,
                "seeds": seeds,
                "seed_results": seed_results,
                "mean_win_rate": stats["mean"],
                "std_win_rate": stats["std"],
                "ci95": stats["ci95"],
            }
        )
        print(f"    Avg win rate: {stats['mean']:.3f} (std={stats['std']:.3f}, ci95=±{stats['ci95']:.3f})")
    return summary


def main():
    args = parse_args()
    if not SUBMISSION_DIR.exists():
        raise FileNotFoundError(f"Submission directory not found: {SUBMISSION_DIR}")

    summary = run_experiment(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
