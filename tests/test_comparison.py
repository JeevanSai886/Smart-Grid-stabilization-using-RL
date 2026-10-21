"""Run these locally with pytest; no executed-test claim accompanies the source ZIP."""
import numpy as np
import pandas as pd
import pytest
from smart_grid.config import load_config
from smart_grid.controllers.registry import ORDER
from smart_grid.controllers.rule_based import IdleController
from smart_grid.plotting import matched_cells, overall_summary, METRICS


def test_training_budget_has_exact_rollouts():
    c = load_config()
    assert c["training"]["total_timesteps"] == 200000
    assert c["training"]["total_timesteps"] % c["training"]["n_steps"] == 0
    assert c["training"]["n_steps"] % c["training"]["batch_size"] == 0


def test_idle_never_requests_correction():
    action = IdleController().predict(np.zeros(408, dtype=np.float32))
    np.testing.assert_array_equal(action, [2, 0])


def test_all_five_are_required_in_each_comparison_cell():
    frame = pd.DataFrame([{"controller": key, "episode_seed": 101, "delay_seconds": 0} for key in ORDER])
    matched_cells(frame, ["episode_seed", "delay_seconds"])
    with pytest.raises(ValueError):
        matched_cells(frame.iloc[:-1], ["episode_seed", "delay_seconds"])


def test_scenario_sd_does_not_count_repeated_delays_as_new_seeds():
    rows = []
    for key in ORDER:
        for seed, value in [(101, 0.0), (202, 10.0)]:
            for delay in [0, 1, 3, 5, 10]:
                rows.append({"controller": key, "episode_seed": seed, "delay_seconds": delay,
                             **{metric: value for metric in METRICS}})
    _, summary = overall_summary(pd.DataFrame(rows))
    for key in ORDER:
        assert summary.loc[key, "cumulative_reward_count"] == 2
        assert summary.loc[key, "cumulative_reward_std"] == pytest.approx(np.sqrt(50))

