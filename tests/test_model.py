from __future__ import annotations

import pandas as pd

from misinfo_abm import MisinformationModel, ModelConfig


def small_config(**updates: object) -> ModelConfig:
    base = ModelConfig(
        n_agents=40,
        m_links=2,
        n_episodes=4,
        max_steps_per_episode=8,
        false_message_probability=1.0,
        seed=123,
    )
    return base.with_updates(**updates)


def test_scale_free_graph_and_hubs() -> None:
    model = MisinformationModel(small_config())
    assert model.graph.number_of_nodes() == 40
    assert model.graph.number_of_edges() > 0
    assert len(model.hub_nodes) == 2


def test_hub_seed_really_uses_hub() -> None:
    model = MisinformationModel(small_config(seed_mode="hub"))
    assert model.current_seed_node in model.hub_nodes


def test_random_seed_is_non_hub() -> None:
    model = MisinformationModel(small_config(seed_mode="random"))
    assert model.current_seed_node not in model.hub_nodes


def test_model_completes_and_returns_metrics() -> None:
    model = MisinformationModel(small_config())
    model.run_model()
    data = model.episode_dataframe()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 4
    assert {
        "cascade_size",
        "peak_false_prevalence",
        "persistence_steps",
        "cooperation_rate",
        "belief_error",
    }.issubset(data.columns)


def test_reproducibility() -> None:
    cfg = small_config(seed=999)
    first = MisinformationModel(cfg)
    second = MisinformationModel(cfg)
    first.run_model()
    second.run_model()
    pd.testing.assert_frame_equal(first.episode_dataframe(), second.episode_dataframe())
