import copy
import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from smart_grid.config import load_config
from smart_grid.envs import SmartGridEnv
from smart_grid.envs.smart_grid_env import AGE_INDEX, FRAME_SIZE
from smart_grid.metrics import summarize_episode


def config():
    c = load_config()
    c["environment"]["episode_steps"] = 20
    return c


def test_gym_contract_seeded_reset_and_finite_episode():
    env = SmartGridEnv(config=config())
    check_env(env, warn=True)
    first, _ = env.reset(seed=7)
    transition_a = env.step([3, 0])
    second, _ = env.reset(seed=7)
    transition_b = env.step([3, 0])
    np.testing.assert_allclose(first, second, atol=1e-6)
    np.testing.assert_allclose(transition_a[0], transition_b[0], atol=1e-6)
    assert transition_a[1] == pytest.approx(transition_b[1], abs=1e-6)
    env.action_space.seed(3)
    for _ in range(19):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert 0.2 <= info["soc"] <= 0.9
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step([2, 0])


def test_controller_cannot_see_current_measurements_and_age_ablation():
    c = config()
    c["delay"].update(mode="fixed", fixed_steps=5, packet_loss_probability=0)
    visible, hidden = SmartGridEnv(config=c), SmartGridEnv(config=c, hide_age=True)
    a, _ = visible.reset(seed=10)
    b, _ = hidden.reset(seed=10)
    initial_measurements = a.reshape(-1, FRAME_SIZE)[-1, :46].copy()
    a = visible.step([4, 2])[0].reshape(-1, FRAME_SIZE)
    b = hidden.step([4, 2])[0].reshape(-1, FRAME_SIZE)
    np.testing.assert_equal(a[-1, :46], initial_measurements)
    assert a[-1, AGE_INDEX] > 0 and b[-1, AGE_INDEX] == 0
    a[:, AGE_INDEX] = 0
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_delay_does_not_change_exogenous_scenario():
    a = config()
    b = copy.deepcopy(a)
    b["delay"].update(mode="fixed", fixed_steps=10)
    first, second = SmartGridEnv(config=a), SmartGridEnv(config=b)
    first.reset(seed=99)
    second.reset(seed=99)
    np.testing.assert_equal(first.episode_profiles, second.episode_profiles)
    np.testing.assert_equal(first.source_capacities, second.source_capacities)
    assert first.battery.soc == second.battery.soc


def test_protection_outage_recovery_and_energy_accounting():
    c = config()
    c["frequency"]["source_max_electrical_mw"] = 0.001
    c["environment"]["restart_delay_s"] = 3
    env = SmartGridEnv(config=c)
    env.reset(seed=1)
    initial_soc = env.battery.soc
    trace = []
    for _ in range(6):
        trace.append(env.step([2, 0])[-1])
    frame = pd.DataFrame(trace)
    assert frame.blackout.all()
    assert frame.blackout_start.sum() == 2
    assert env.battery.soc == initial_soc
    metrics = summarize_episode(frame)
    assert metrics["blackout_duration_s"] == 6
    assert metrics["unserved_mwh"] == pytest.approx(metrics["demand_mwh"])
    assert metrics["shed_mwh"] == 0
    assert metrics["solver_failures"] == 0


def test_grid_connected_frequency_is_imposed():
    c = config()
    c["environment"]["mode"] = "grid_connected"
    env = SmartGridEnv(config=c)
    env.reset(seed=12)
    for _ in range(3):
        info = env.step([2, 0])[-1]
        assert info["frequency_deviation_hz"] == 0


def test_fractional_actions_rejected():
    env = SmartGridEnv(config=config())
    env.reset(seed=1)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step([1.5, 0])


def test_solver_shutdown_is_distinguished_from_physical_trip(monkeypatch):
    env = SmartGridEnv(config=config())
    env.reset(seed=1)
    monkeypatch.setattr(env.feeder, "_run", lambda: False)
    info = env.step([2, 0])[-1]
    assert info["solver_failure"] and info["blackout"]
    assert info["blackout_reason"] == "solver_shutdown"
    assert not info["voltage_violation"]
    assert info["served_mw"] == 0


def test_reward_components_reconstruct_return():
    env = SmartGridEnv(config=config())
    env.reset(seed=77)
    _, reward, _, _, info = env.step([4, 1])
    assert reward == pytest.approx(sum(v for k, v in info.items() if k.startswith("reward_")))
