"""The same physically meaningful scalar tags for all five controllers."""
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from smart_grid.controllers.registry import agent_id

IDENTIFIERS = {"training_seed", "episode_seed", "episode", "delay_steps", "delay_seconds",
               "fluctuation_scale", "training_timesteps", "reference_reused"}
GRID_SIGNALS = ("observation_age_s", "frequency_peak_deviation_hz", "frequency_deviation_hz",
                "voltage_min_pu", "voltage_max_pu", "soc", "battery_power_mw", "shed_mw",
                "unserved_mw", "line_loading_max", "reward", "blackout")


def writer_for(root, controller):
    return SummaryWriter(log_dir=str(Path(root) / agent_id(controller)))


def log_metrics(writer, phase, rows, step):
    for key in rows.select_dtypes(include=[np.number]).columns:
        if key in IDENTIFIERS:
            continue
        values = rows[key].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite measured metric: {key}")
        writer.add_scalar(f"{phase}/{key}/mean", float(values.mean()), int(step))
        if len(values) > 1:
            writer.add_scalar(f"{phase}/{key}/std", float(values.std(ddof=1)), int(step))
    writer.flush()


def log_trace(writer, trace, scenario, stride):
    stride = max(1, int(stride))
    indices = sorted(set(range(0, len(trace), stride)) | {len(trace) - 1})
    for index in indices:
        row = trace.iloc[index]
        for key in GRID_SIGNALS:
            writer.add_scalar(f"grid/{scenario}/{key}", float(row[key]), int(row["step"]))
    writer.flush()

