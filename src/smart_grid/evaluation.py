"""Matched test evaluation and fixed baseline references for learning curves."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from smart_grid.config import ROOT, save_json, sha256
from smart_grid.controllers.registry import AGENTS, ORDER, LEARNED, agent_id, require_five
from smart_grid.controllers.rule_based import IdleController, RuleBasedController
from smart_grid.data.preprocess import read_profiles
from smart_grid.envs import SmartGridEnv
from smart_grid.logging import writer_for, log_metrics, log_trace
from smart_grid.metrics import aggregate_results
from smart_grid.rollout import run_episode


def baseline(controller, config, base_mw):
    if controller == "rule_based":
        return RuleBasedController(config, base_mw)
    if controller == "idle_agent":
        return IdleController()
    raise ValueError(f"Not a baseline: {controller}")


def validation_baselines(config, run_dir, data_dir=None):
    """Measure baselines once, then reuse that explicitly labeled reference.

    These controllers have no changing trainable parameters. Repeating their
    same measured reference at checkpoint x-coordinates is not new training.
    """
    run = Path(run_dir)
    profiles = read_profiles(Path(data_dir or ROOT / "data/processed") / "validation.csv")
    total, frequency = int(config["training"]["total_timesteps"]), int(config["training"]["evaluation_frequency"])
    checkpoints = sorted(set(range(0, total + 1, frequency)) | {total})
    all_rows = []
    for controller in ("rule_based", "idle_agent"):
        env = SmartGridEnv(profiles, config)
        policy = baseline(controller, config, env.feeder.base_total_mw)
        rows = []
        try:
            for seed in config["training"]["validation_seeds"]:
                values, _ = run_episode(env, policy.predict, seed, policy.reset)
                rows.append({"controller": controller, **values})
        finally:
            env.close()
        measured = pd.DataFrame(rows)
        reference_path = run / "baseline_validation" / f"{controller}.csv"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        measured.to_csv(reference_path, index=False)
        writer = writer_for(run / "tensorboard/agents", controller)
        try:
            for step in checkpoints:
                reference = measured.copy()
                reference["training_timesteps"] = step
                reference["reference_reused"] = step != 0
                reference["reference_source"] = str(reference_path.relative_to(run))
                all_rows.append(reference)
                log_metrics(writer, "validation", reference, step)
        finally:
            writer.close()
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(run / "baseline_validation_curve.csv", index=False)
    return result


def assemble_validation_comparison(run_dir):
    run = Path(run_dir)
    files = sorted((run / "models").glob("*/validation_episodes.csv"))
    if len(files) != len(LEARNED):
        raise ValueError("Three completed PPO validation histories are required.")
    parts = [pd.read_csv(run / "baseline_validation_curve.csv")]
    parts.extend(pd.read_csv(file) for file in files)
    frame = pd.concat(parts, ignore_index=True)
    require_five(frame)
    cells = ["training_timesteps", "episode_seed"]
    if frame.duplicated(["controller", *cells]).any():
        raise ValueError("Duplicate validation observations.")
    reference = set(map(tuple, frame[frame.controller == ORDER[0]][cells].to_numpy()))
    for controller in ORDER:
        if set(map(tuple, frame[frame.controller == controller][cells].to_numpy())) != reference:
            raise ValueError("Validation checkpoints/scenario seeds are not matched.")
    frame.to_csv(run / "validation_comparison.csv", index=False)
    return frame


def evaluate(config: dict, model_paths: dict[str, str | Path], output_dir: str | Path,
             data_dir: str | Path | None = None) -> pd.DataFrame:
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir or ROOT / "data/processed")
    test_file = data_dir / "test.csv"
    profiles = read_profiles(test_file)
    torch.set_num_threads(int(config["training"]["torch_threads"]))
    normalized = {}
    for name, path in model_paths.items():
        key = agent_id(name)
        if key in normalized:
            raise ValueError(f"More than one model for {key}; use one matched training seed per run.")
        normalized[key] = Path(path)
    if set(normalized) != set(LEARNED):
        raise ValueError(f"All three trained PPO models are required: {LEARNED}")
    models = {}
    for controller, path in normalized.items():
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            raise ValueError(f"Model training is incomplete: {controller}")
        if metadata["variant"] != AGENTS[controller]["variant"]:
            raise ValueError("Model label and recorded variant disagree.")
        for section in ("battery", "frequency", "reward"):
            if config[section] != metadata["config"][section]:
                raise ValueError(f"Physical/reward configuration mismatch: {section}")
        for key, value in config["environment"].items():
            if key not in {"episode_steps", "fast_fluctuation"} and metadata["config"]["environment"].get(key) != value:
                raise ValueError(f"Environment configuration mismatch: {key}")
        if metadata["data_provenance"]["partitions"]["test"]["sha256"] != sha256(test_file):
            raise ValueError("Test partition differs from the saved training split.")
        models[controller] = (PPO.load(path, device="cpu"), metadata)
    scenario = config["evaluation"]
    if len(scenario["seeds"]) != scenario["episodes"]:
        raise ValueError("Provide exactly one seed for each test episode.")
    writers = {key: writer_for(out.parent / "tensorboard/agents", key) for key in ORDER}
    records = []
    try:
        for delay in scenario["delay_steps"]:
            for fluctuation in scenario["fluctuation_scales"]:
                for controller in ORDER:
                    c = copy.deepcopy(config)
                    c["environment"].update(episode_steps=scenario["episode_steps"], fast_fluctuation=fluctuation)
                    c["delay"].update(mode="fixed", fixed_steps=int(delay), packet_loss_probability=0.0,
                                      max_steps=max(int(delay), c["delay"]["max_steps"]))
                    hide_age = models[controller][1]["hide_age"] if controller in models else False
                    env = SmartGridEnv(profiles, c, hide_age=hide_age)
                    policy = None if controller in models else baseline(controller, c, env.feeder.base_total_mw)
                    if policy is None:
                        model = models[controller][0]
                        predict = lambda observation: model.predict(observation, deterministic=True)[0]
                        reset_policy = None
                    else:
                        predict, reset_policy = policy.predict, policy.reset
                    cell_rows = []
                    try:
                        for episode, seed in enumerate(scenario["seeds"]):
                            measured, trace = run_episode(env, predict, seed, reset_policy)
                            seconds = float(delay * c["environment"]["control_dt_s"])
                            row = {"controller": controller, "variant": AGENTS[controller]["variant"],
                                   "training_seed": models[controller][1]["training_seed"] if controller in models else -1,
                                   "delay_steps": int(delay), "delay_seconds": seconds,
                                   "fluctuation_scale": float(fluctuation), "episode": episode, **measured}
                            records.append(row)
                            cell_rows.append(row)
                            path = out / "traces" / f"{controller}_d{delay}_f{fluctuation:g}_seed{seed}.csv"
                            path.parent.mkdir(exist_ok=True)
                            trace.to_csv(path, index=False)
                            tag = f"d{delay}_f{fluctuation:g}_seed{seed}"
                            log_trace(writers[controller], trace, tag, config["logging"]["trace_stride"])
                    finally:
                        env.close()
                    # Step is the configured delay in control steps. The default
                    # control step is 1 second; tags never pretend to be training.
                    log_metrics(writers[controller], f"test_delay/fluctuation_{fluctuation:g}",
                                pd.DataFrame(cell_rows), delay)
                    print(f"Evaluated {AGENTS[controller]['label']}: delay={seconds:g} s", flush=True)
        frame = pd.DataFrame(records)
        require_five(frame)
        frame.to_csv(out / "episodes.csv", index=False)
        aggregate_results(frame).to_csv(out / "summary.csv", index=False)
        for controller in ORDER:
            # Average repeated delay cells within each seed before descriptive SD.
            per_seed = frame[frame.controller == controller].groupby("episode_seed").mean(numeric_only=True).reset_index()
            log_metrics(writers[controller], "test_overall", per_seed, config["training"]["total_timesteps"])
        save_json(out / "metadata.json", {
            "config": config, "test_sha256": sha256(test_file),
            "models": {key: {"path": str(Path("models") / path.parent.name / path.name),
                             "sha256": sha256(path)} for key, path in normalized.items()},
            "fairness": "Same test seeds, exogenous profiles, initial SOC, horizon, physics, reward and actual delay; no test-based selection",
            "uncertainty": "Descriptive scenario SD, not independent-training-seed confidence intervals",
            "tensorboard_step_units": {"validation": "training timesteps", "test_delay": "control steps (1 second by default)",
                                       "test_overall": "training budget", "grid": "episode control step"},
        })
        return frame
    finally:
        for writer in writers.values():
            writer.close()

