import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize seed reliability results")
    parser.add_argument(
        "--results",
        type=str,
        default=str(Path(__file__).with_name("results.json")),
        help="Path to JSON produced by seed_reliability.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).with_name("seed_reliability.png")),
        help="Where to save the visualization (PNG)",
    )
    return parser.parse_args()


def load_results(path: Path) -> Tuple[List[Dict], List[Dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = []
    seed_rows = []
    for entry in data:
        summary.append(
            {
                "agent": entry["agent"],
                "mean": entry["mean_win_rate"],
                "ci95": entry["ci95"],
                "std": entry["std_win_rate"],
            }
        )
        for seed_entry in entry.get("seed_results", []):
            seed_rows.append(
                {
                    "agent": entry["agent"],
                    "seed": seed_entry["seed"],
                    "win_rate": seed_entry["win_rate"],
                }
            )
    return summary, seed_rows


def visualize(summary: List[Dict], seed_rows: List[Dict], output_path: Path) -> None:
    agents = [item["agent"] for item in summary]
    colors = plt.cm.viridis_r([i / max(1, len(agents) - 1) for i in range(len(agents))])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax_seed, ax_summary = axes
    for color, agent in zip(colors, agents):
        rows = [row for row in seed_rows if row["agent"] == agent]
        seeds = [row["seed"] for row in rows]
        win_rates = [row["win_rate"] for row in rows]
        ax_seed.plot(
            seeds,
            win_rates,
            marker="o",
            linestyle="-",
            label=agent,
            color=color,
        )
    ax_seed.set_title("Seed-wise Win Rates")
    ax_seed.set_xlabel("Random Seed")
    ax_seed.set_ylabel("Win Rate")
    ax_seed.set_ylim(0.0, 1.05)
    ax_seed.grid(True, linestyle="--", alpha=0.4)
    ax_seed.legend(fontsize="small")

    means = [item["mean"] for item in summary]
    ci95 = [item["ci95"] for item in summary]
    y_positions = range(len(agents))
    ax_summary.barh(
        y_positions,
        means,
        xerr=ci95,
        color=colors,
        capsize=6,
        alpha=0.9,
    )
    ax_summary.set_yticks(list(y_positions))
    ax_summary.set_yticklabels(agents)
    ax_summary.set_xlim(0.0, 1.05)
    ax_summary.set_xlabel("Mean Win Rate")
    ax_summary.set_title("Mean ± 95% CI")
    ax_summary.grid(True, axis="x", linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Seed Reliability vs Random ConnectX Opponent", fontsize=14, weight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def main():
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    summary, seed_rows = load_results(results_path)
    visualize(summary, seed_rows, Path(args.output))


if __name__ == "__main__":
    main()
