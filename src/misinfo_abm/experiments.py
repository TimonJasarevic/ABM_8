"""Factorial experiments and result aggregation."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from misinfo_abm.config import ModelConfig
from misinfo_abm.model import MisinformationModel


def summarize_false_episodes(data: pd.DataFrame) -> dict[str, float]:
    false_data = data[data["message_is_false"]].copy()
    if false_data.empty:
        return {
            "mean_cascade_size": 0.0,
            "mean_peak_prevalence": 0.0,
            "mean_persistence": 0.0,
            "mean_recovery_time": 0.0,
            "mean_cooperation_rate": 0.0,
            "mean_belief_error": 0.0,
        }
    return {
        "mean_cascade_size": float(false_data["cascade_size"].mean()),
        "mean_peak_prevalence": float(false_data["peak_false_prevalence"].mean()),
        "mean_persistence": float(false_data["persistence_steps"].mean()),
        "mean_recovery_time": float(false_data["recovery_time"].mean()),
        "mean_cooperation_rate": float(false_data["cooperation_rate"].mean()),
        "mean_belief_error": float(false_data["belief_error"].mean()),
    }


def run_factorial(
    base_config: ModelConfig,
    replications: int = 20,
    include_payoff_comparison: bool = True,
) -> pd.DataFrame:
    """Run the minimum experiment requested by the project feedback."""
    seed_modes = ["hub", "random"]
    hub_strategies = ["cooperative", "non_verifying"]
    payoff_modes = ["normalized", "accumulated"] if include_payoff_comparison else ["normalized"]

    rows: list[dict[str, object]] = []
    scenario_index = 0
    for seed_mode, hub_strategy, payoff_mode in product(
        seed_modes, hub_strategies, payoff_modes
    ):
        scenario_index += 1
        for replication in range(replications):
            run_seed = base_config.seed + 10_000 * scenario_index + replication
            config = base_config.with_updates(
                seed_mode=seed_mode,
                hub_strategy=hub_strategy,
                payoff_mode=payoff_mode,
                seed=run_seed,
            )
            model = MisinformationModel(config)
            model.run_model()
            summary = summarize_false_episodes(model.episode_dataframe())
            rows.append(
                {
                    "replication": replication,
                    "seed": run_seed,
                    "seed_mode": seed_mode,
                    "hub_strategy": hub_strategy,
                    "payoff_mode": payoff_mode,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def save_factorial_results(data: pd.DataFrame, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path
