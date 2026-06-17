"""Small plotting helpers for experiment output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_factorial_results(csv_path: str | Path, output_dir: str | Path) -> list[Path]:
    data = pd.read_csv(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for outcome, label in [
        ("mean_cascade_size", "Mean false-message cascade size"),
        ("mean_persistence", "Mean persistence steps"),
        ("mean_cooperation_rate", "Mean verification rate"),
    ]:
        grouped = (
            data.groupby(["seed_mode", "hub_strategy"], as_index=False)[outcome]
            .mean()
            .pivot(index="seed_mode", columns="hub_strategy", values=outcome)
        )
        ax = grouped.plot(kind="bar", figsize=(8, 5))
        ax.set_xlabel("Seed location")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(title="Hub strategy")
        figure = ax.get_figure()
        figure.tight_layout()
        path = output / f"{outcome}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        created.append(path)

    return created
