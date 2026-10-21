"""Validated configuration and reproducible output metadata."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    with (ROOT / "configs/default.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            config = merge(config, yaml.safe_load(handle) or {})
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    e, b, d, f = (c[k] for k in ("environment", "battery", "delay", "frequency"))
    if e["control_dt_s"] <= 0 or e["episode_steps"] < 2 or e["history_length"] < 1:
        raise ValueError("Positive time step, at least two episode steps and nonempty history required")
    if e["mode"] not in {"islanded", "grid_connected"}:
        raise ValueError("mode must be islanded or grid_connected")
    if not 0 <= b["soc_min"] < b["soc_max"] <= 1:
        raise ValueError("Invalid SOC limits")
    if not b["soc_min"] <= b["initial_soc_min"] <= b["initial_soc_max"] <= b["soc_max"]:
        raise ValueError("Initial SOC must lie inside battery limits")
    if min(b["capacity_mwh"], b["max_power_mw"]) <= 0:
        raise ValueError("Battery capacity and power must be positive")
    if any(not 0 < b[k] <= 1 for k in ("charge_efficiency", "discharge_efficiency")):
        raise ValueError("Efficiencies must lie in (0, 1]")
    if d["mode"] not in {"fixed", "stochastic"} or min(d["max_steps"], d["fixed_steps"]) < 0:
        raise ValueError("Invalid delay mode or delay bound")
    for key in ("packet_loss_probability", "enter_congestion_probability", "leave_congestion_probability"):
        if not 0 <= d[key] <= 1:
            raise ValueError(f"{key} must be a probability")
    if min(f["inertia_s"], f["source_rating_mva"], f["integration_dt_s"], f["governor_time_constant_s"]) <= 0:
        raise ValueError("Frequency model constants must be positive")
    fractions = c["data"]["train_fraction"] + c["data"]["validation_fraction"]
    if not 0 < c["data"]["train_fraction"] < fractions < 1:
        raise ValueError("Invalid chronological data split")


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

