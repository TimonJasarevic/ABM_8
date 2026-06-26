#!/usr/bin/env python3
"""
Plot deterministic Sobol S1 and ST indices as dots with error bars.

Input:
    results/sa_indices.csv

Fallback input:
    sobol/results/sa_indices.csv

Output:
    sa_sobol_deterministic_dotgrid.png

Structure:
    x-axis: measured outputs
    colors: parameters
    marker shape:
        circle = S1
        square = ST
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_ORDER = [
    "veracity_differential",
    "avg_verify_rate",
    "avg_payoff",
    "sen_welfare",
]

OUTPUT_LABELS = {
    "veracity_differential": "Veracity\ndifferential",
    "avg_verify_rate": "Verification\nrate",
    "avg_payoff": "Average\npayoff",
    "sen_welfare": "SEN\nwelfare",
}

FACTOR_ORDER = [
    "v_cost",
    "loss",
    "tpr",
    "fpr",
    "p_fake",
    "log10_lambda",
    "loss_aversion",
]

FACTOR_LABELS = {
    "v_cost": "Verification cost",
    "loss": "Misinformation loss",
    "tpr": "True positive rate",
    "fpr": "False positive rate",
    "p_fake": "Fake-news probability",
    "log10_lambda": r"Rationality ($\log_{10}\lambda$)",
    "loss_aversion": "Loss aversion",
}


def find_results_dir(path_str: str) -> Path:
    """
    Find the directory containing sa_indices.csv.
    """
    user_path = Path(path_str)

    if (user_path / "sa_indices.csv").exists():
        return user_path

    fallback = Path("sobol/results")

    if (fallback / "sa_indices.csv").exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find sa_indices.csv in {user_path} or {fallback}."
    )


def load_indices(results_dir: Path) -> pd.DataFrame:
    """
    Load and validate sa_indices.csv.
    """
    path = results_dir / "sa_indices.csv"
    df = pd.read_csv(path)

    required_columns = {
        "output",
        "component",
        "factor",
        "S1",
        "S1_conf",
        "ST",
        "ST_conf",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    return df


def plot_deterministic(df: pd.DataFrame, results_dir: Path) -> None:
    """
    Plot deterministic S1 and ST for all outputs and all parameters.
    """
    df = df[df["component"] == "deterministic"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    x_base = np.arange(len(OUTPUT_ORDER))

    # Spread parameters within each output group.
    factor_offsets = np.linspace(-0.33, 0.33, len(FACTOR_ORDER))

    # Put S1 and ST close together for the same parameter.
    metric_offsets = {
        "S1": -0.018,
        "ST": 0.018,
    }

    marker_map = {
        "S1": "o",
        "ST": "s",
    }

    conf_map = {
        "S1": "S1_conf",
        "ST": "ST_conf",
    }

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    factor_colors = {
        factor: colors[i % len(colors)]
        for i, factor in enumerate(FACTOR_ORDER)
    }

    for factor_index, factor in enumerate(FACTOR_ORDER):
        factor_df = df[df["factor"] == factor]

        for metric in ["S1", "ST"]:
            xs = []
            ys = []
            yerrs = []

            for output_index, output in enumerate(OUTPUT_ORDER):
                row = factor_df[factor_df["output"] == output]

                if row.empty:
                    continue

                x = (
                    x_base[output_index]
                    + factor_offsets[factor_index]
                    + metric_offsets[metric]
                )

                xs.append(x)
                ys.append(float(row[metric].iloc[0]))
                yerrs.append(float(row[conf_map[metric]].iloc[0]))

            ax.errorbar(
                xs,
                ys,
                yerr=yerrs,
                fmt=marker_map[metric],
                linestyle="none",
                markersize=5.5,
                capsize=3,
                elinewidth=1.0,
                markeredgewidth=0.9,
                color=factor_colors[factor],
                ecolor=factor_colors[factor],
                alpha=0.95,
            )

    ax.axhline(
        0,
        linewidth=0.8,
        color="black",
        alpha=0.8,
    )

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [OUTPUT_LABELS[o] for o in OUTPUT_ORDER],
        fontsize=10,
    )

    ax.set_ylim(-0.10, 1.05)
    ax.set_xlim(-0.60, len(OUTPUT_ORDER) - 0.40)

    ax.set_xlabel("Measured output", fontsize=11)
    ax.set_ylabel("Sobol index", fontsize=11)

    ax.set_title(
        "Deterministic Sobol sensitivity by output",
        fontsize=14,
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Legend for parameter colors.
    factor_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=factor_colors[factor],
            label=FACTOR_LABELS[factor],
            markersize=6.5,
        )
        for factor in FACTOR_ORDER
    ]

    # Legend for marker types.
    metric_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="black",
            label=r"$S_1$",
            markersize=6.5,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color="black",
            label=r"$S_T$",
            markersize=6.5,
        ),
    ]

    # Parameter legend below the plot.
    fig.legend(
        handles=factor_handles,
        title="Parameter",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # S1 and ST legend below the parameter legend.
    fig.legend(
        handles=metric_handles,
        title="Index",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    # Reserve bottom space for both legends.
    fig.tight_layout(rect=(0, 0.22, 1, 1))

    out_path = results_dir / "sa_sobol_deterministic_dotgrid.png"

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        default="results",
        help="Directory containing sa_indices.csv. Default: results",
    )

    args = parser.parse_args()

    results_dir = find_results_dir(args.results)
    df = load_indices(results_dir)

    plot_deterministic(df, results_dir)


if __name__ == "__main__":
    main()