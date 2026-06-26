#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sweep_2026_06_26_0242/simulations.csv")
    parser.add_argument("--out", default="belief_optimism_bias.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required = {"true_prob", "mean_true_belief"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Add true_prob and mean_true_belief in Julia first.")

    # Average across stochastic reps/design points with similar true_prob.
    # Rounding avoids tiny floating differences from the design.
    df["true_prob_bin"] = df["true_prob"].round(2)

    grouped = (
        df.groupby("true_prob_bin")
        .agg(
            mean_belief=("mean_true_belief", "mean"),
            sd_belief=("mean_true_belief", "std"),
            n=("mean_true_belief", "size"),
        )
        .reset_index()
    )

    grouped["se_belief"] = grouped["sd_belief"] / grouped["n"] ** 0.5

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Perfect calibration line.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Non-emergent behaviour")

    # Mean belief with uncertainty.
    ax.errorbar(
        grouped["true_prob_bin"],
        grouped["mean_belief"],
        yerr=1.96 * grouped["se_belief"],
        fmt="o",
        capsize=3,
        markersize=4,
        linewidth=1,
        label="Mean agent belief",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("True probability of truthful messages")
    ax.set_ylabel("Mean final believed truth probability")
    ax.set_title("Emergent optimism bias in belief formation")

    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    plt.close(fig)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()