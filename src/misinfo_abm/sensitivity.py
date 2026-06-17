"""Morris screening for the most important behavioural parameters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from misinfo_abm.config import ModelConfig
from misinfo_abm.experiments import summarize_false_episodes
from misinfo_abm.model import MisinformationModel

PROBLEM = {
    "num_vars": 6,
    "names": [
        "verification_cost",
        "false_penalty_per_neighbor",
        "correction_probability",
        "forgetting_probability",
        "loss_aversion",
        "decision_sensitivity",
    ],
    "bounds": [
        [0.05, 0.80],
        [0.05, 0.80],
        [0.05, 0.60],
        [0.00, 0.20],
        [1.00, 3.00],
        [0.50, 6.00],
    ],
}


def run_morris(
    base_config: ModelConfig,
    trajectories: int = 20,
    replications: int = 5,
    output: str | Path = "results/morris_indices.csv",
) -> pd.DataFrame:
    """Run Morris screening on mean false-message cascade size.

    With six parameters, 20 trajectories produce 140 design points. Five stochastic
    replications give 700 model runs. Lower these values during debugging only.
    """
    samples = morris_sample.sample(
        PROBLEM,
        N=trajectories,
        num_levels=4,
        optimal_trajectories=None,
        seed=base_config.seed,
    )
    outputs = np.zeros(samples.shape[0], dtype=float)

    for sample_index, values in enumerate(samples):
        scores: list[float] = []
        updates = dict(zip(PROBLEM["names"], values, strict=True))
        for replication in range(replications):
            config = replace(
                base_config,
                **updates,
                seed=base_config.seed + sample_index * 1_000 + replication,
            )
            config.validate()
            model = MisinformationModel(config)
            model.run_model()
            summary = summarize_false_episodes(model.episode_dataframe())
            scores.append(summary["mean_cascade_size"])
        outputs[sample_index] = float(np.mean(scores))

    indices = morris_analyze.analyze(
        PROBLEM,
        samples,
        outputs,
        num_levels=4,
        print_to_console=False,
        seed=base_config.seed,
    )
    result = pd.DataFrame(
        {
            "parameter": PROBLEM["names"],
            "mu": indices["mu"],
            "mu_star": indices["mu_star"],
            "sigma": indices["sigma"],
            "mu_star_conf": indices["mu_star_conf"],
        }
    ).sort_values("mu_star", ascending=False)

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
