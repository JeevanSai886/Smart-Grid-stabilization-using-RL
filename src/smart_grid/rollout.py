"""Shared, seeded evaluation: policies receive observations only."""
import pandas as pd
from smart_grid.envs import SmartGridEnv
from smart_grid.metrics import summarize_episode


def run_episode(env, predict, seed, reset_policy=None):
    observation, reset_info = env.reset(seed=int(seed))
    if reset_policy is not None:
        reset_policy()
    records = []
    while True:
        action = predict(observation)
        observation, _, terminated, truncated, info = env.step(action)
        records.append({**info, "battery_action": int(action[0]), "shedding_action": int(action[1])})
        if terminated or truncated:
            break
    trace = pd.DataFrame(records)
    metrics = {"episode_seed": int(seed), "start_timestamp": reset_info["start_timestamp"],
               **summarize_episode(trace)}
    return metrics, trace


def assess(profiles, config, predict, seeds, hide_age=False, reset_policy=None):
    env = SmartGridEnv(profiles, config, hide_age=hide_age)
    results = []
    try:
        for seed in seeds:
            result, _ = run_episode(env, predict, seed, reset_policy)
            results.append(result)
    finally:
        env.close()
    return pd.DataFrame(results)

