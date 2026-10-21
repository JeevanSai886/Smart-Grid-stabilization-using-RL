"""CPU PPO training with fixed, matched validation cases and exact rollout budgets."""
from __future__ import annotations

import copy
import json
from importlib.metadata import version
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from smart_grid.config import ROOT, save_json, sha256
from smart_grid.controllers.registry import AGENTS, agent_id
from smart_grid.data.preprocess import read_profiles
from smart_grid.envs import SmartGridEnv
from smart_grid.logging import writer_for, log_metrics
from smart_grid.rollout import assess

VARIANTS = ("aware", "no_age", "zero_delay")


class ProgressCallback(BaseCallback):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
        self.started = time.perf_counter()

    def _on_step(self):
        if self.n_calls % self.interval == 0:
            print(f"  PPO steps={self.num_timesteps:,}; elapsed={time.perf_counter()-self.started:.1f}s", flush=True)
        infos = self.locals.get("infos", [])
        if infos:
            for key in ("observation_age_s", "frequency_peak_deviation_hz", "soc", "shed_mw"):
                self.logger.record_mean(f"grid/{key}", float(infos[0][key]))
        return True


class MatchedValidationCallback(BaseCallback):
    def __init__(self, profiles, config, output, controller, hide_age, writer):
        super().__init__()
        self.profiles, self.config, self.output = profiles, config, output
        self.controller, self.hide_age, self.writer = controller, hide_age, writer
        self.frequency = int(config["training"]["evaluation_frequency"])
        self.seeds = list(config["training"]["validation_seeds"])
        self.last_checked = None
        self.best_mean = -np.inf
        self.best_step = None
        self.frames, self.timesteps, self.returns, self.lengths = [], [], [], []

    def _evaluate(self):
        rows = assess(self.profiles, self.config,
                      lambda observation: self.model.predict(observation, deterministic=True)[0],
                      self.seeds, hide_age=self.hide_age)
        rows["controller"] = self.controller
        rows["training_timesteps"] = self.num_timesteps
        rows["reference_reused"] = False
        self.frames.append(rows)
        pd.concat(self.frames, ignore_index=True).to_csv(self.output / "validation_episodes.csv", index=False)
        self.timesteps.append(int(self.num_timesteps))
        self.returns.append(rows.cumulative_reward.to_numpy())
        self.lengths.append(rows.duration_s.to_numpy() / self.config["environment"]["control_dt_s"])
        destination = self.output / "validation"
        destination.mkdir(exist_ok=True)
        np.savez(destination / "evaluations.npz", timesteps=self.timesteps,
                 results=self.returns, ep_lengths=self.lengths)
        log_metrics(self.writer, "validation", rows, self.num_timesteps)
        mean = float(rows.cumulative_reward.mean())
        # Step zero is a learning-curve reference, not a trained candidate.
        if self.num_timesteps > 0 and mean > self.best_mean:
            self.best_mean, self.best_step = mean, int(self.num_timesteps)
            self.model.save(self.output / "best_model")
        self.last_checked = self.num_timesteps

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # At this boundary, the preceding PPO gradient updates have finished.
        if self.last_checked is None or self.num_timesteps - self.last_checked >= self.frequency:
            self._evaluate()

    def _on_training_end(self):
        if self.last_checked != self.num_timesteps:
            self._evaluate()


def train(config: dict, output_dir: str | Path, variant: str = "aware",
          data_dir: str | Path | None = None, resume: str | Path | None = None,
          tensorboard_root: str | Path | None = None) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown PPO variant {variant}")
    if resume is not None:
        raise ValueError("This matched comparison requires fresh training. Choose a fresh output folder; resume is disabled.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir or ROOT / "data/processed")
    train_path, validation_path = data_dir / "train.csv", data_dir / "validation.csv"
    train_data, validation_data = read_profiles(train_path), read_profiles(validation_path)
    t = config["training"]
    total = int(t["total_timesteps"])
    if total <= 0 or total % t["n_steps"] or t["n_steps"] % t["batch_size"]:
        raise ValueError("The total must divide into complete PPO rollouts; batch_size must divide n_steps.")
    if t["evaluation_frequency"] % t["n_steps"]:
        raise ValueError("evaluation_frequency must be a multiple of n_steps.")
    if len(t["validation_seeds"]) != t["evaluation_episodes"]:
        raise ValueError("Provide one explicit validation seed per evaluation episode.")
    training_config = copy.deepcopy(config)
    if variant == "zero_delay":
        training_config["delay"].update(mode="fixed", fixed_steps=0, packet_loss_probability=0.0)
    # All validation comparisons use the original, common delayed environment.
    validation_config = copy.deepcopy(config)
    hide_age = variant == "no_age"
    seed = int(config["seed"])
    set_random_seed(seed, using_cuda=False)
    torch.set_num_threads(int(t["torch_threads"]))
    controller = agent_id(variant)
    shared_logs = Path(tensorboard_root or output.parent.parent / "tensorboard/agents")
    writer = writer_for(shared_logs, controller)
    train_env = DummyVecEnv([lambda: Monitor(
        SmartGridEnv(train_data, training_config, hide_age=hide_age),
        filename=str(output / "train.monitor.csv"))])
    train_env.seed(seed)
    model = PPO("MlpPolicy", train_env, device="cpu", seed=seed, verbose=0,
                learning_rate=t["learning_rate"], n_steps=t["n_steps"], batch_size=t["batch_size"],
                n_epochs=t["n_epochs"], gamma=t["gamma"], gae_lambda=t["gae_lambda"],
                clip_range=t["clip_range"], ent_coef=t["ent_coef"],
                policy_kwargs={"net_arch": t["net_arch"]})
    # Optimizer diagnostics belong to PPO only. Shared TensorBoard uses common
    # physical metrics for five controllers; no baseline optimizer losses are invented.
    model.set_logger(configure(str(output / "optimizer_logs"), ["csv"]))
    validation = MatchedValidationCallback(validation_data, validation_config, output,
                                           controller, hide_age, writer)
    checkpoint = CheckpointCallback(save_freq=t["checkpoint_frequency"],
                                    save_path=str(output / "checkpoints"), name_prefix=variant)
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    metadata = {
        "variant": variant, "controller": controller, "training_seed": seed,
        "config": training_config, "validation_config": validation_config,
        "hide_age": hide_age, "status": "training", "device": "cpu",
        "requested_timesteps": total, "data_train_sha256": sha256(train_path),
        "data_validation_sha256": sha256(validation_path), "data_provenance": provenance,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("pandapower", "numpy", "torch", "stable-baselines3", "gymnasium")},
        "selection": "Best mean on fixed matched validation cases after updates; final candidate included; test data unused",
    }
    save_json(output / "metadata.json", metadata)
    started = time.perf_counter()
    print(f"Training {AGENTS[controller]['label']}: {total:,} steps", flush=True)
    try:
        model.learn(total_timesteps=total, callback=[checkpoint, validation, ProgressCallback(t["n_steps"])],
                    progress_bar=False)
        if model.num_timesteps != total:
            raise RuntimeError("Actual training steps differ from the requested budget.")
        model.save(output / "final_model")
        metadata.update(status="complete", actual_timesteps=int(model.num_timesteps),
                        training_seconds=time.perf_counter() - started,
                        selected_model="best_model.zip", selected_checkpoint_timesteps=validation.best_step,
                        best_validation_reward=validation.best_mean)
        save_json(output / "metadata.json", metadata)
    except BaseException:
        model.save(output / "interrupted_model")
        metadata.update(status="interrupted", actual_timesteps=int(model.num_timesteps))
        save_json(output / "metadata.json", metadata)
        raise
    finally:
        writer.close()
        train_env.close()
    return output / metadata["selected_model"]

