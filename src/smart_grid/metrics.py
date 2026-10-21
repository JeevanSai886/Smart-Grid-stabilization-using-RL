from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_episode(trace: pd.DataFrame) -> dict:
    hours = trace["dt_s"] / 3600
    violations = trace[["voltage_violation", "frequency_violation", "overload_violation"]].any(axis=1)
    duration = float(trace.dt_s.sum())
    demand_mwh = float((trace.demand_mw * hours).sum())
    # Frequency is undefined during blackout: retain finite storage placeholders
    # for the existing logger, and expose eligible duration for consumers.
    energized = ~trace.blackout.astype(bool)
    energized_duration = float(trace.loc[energized, "dt_s"].sum())
    frequency = trace.loc[energized, "frequency_deviation_hz"].to_numpy(dtype=float)
    weights = trace.loc[energized, "dt_s"].to_numpy(dtype=float)
    voltage = np.maximum(np.abs(trace.voltage_min_pu.to_numpy() - 1.0),
                         np.abs(trace.voltage_max_pu.to_numpy() - 1.0))
    all_weights = trace.dt_s.to_numpy(dtype=float)
    mean_frequency = float(np.average(np.abs(frequency), weights=weights)) if energized_duration else 0.0
    rms_frequency = float(np.sqrt(np.average(frequency ** 2, weights=weights))) if energized_duration else 0.0
    peak_frequency = float(trace.loc[energized, "frequency_peak_deviation_hz"].max()) if energized_duration else 0.0
    # Contiguous violation runs are events; violation_steps counts sampled intervals.
    violation_events = int((violations & ~violations.shift(fill_value=False)).sum())
    return {"mean_absolute_frequency_deviation_hz": mean_frequency,
            "rms_frequency_deviation_hz": rms_frequency,
            "peak_energized_frequency_deviation_hz": peak_frequency,
            "energized_duration_s": energized_duration,
            "mean_voltage_deviation_pu": float(np.average(voltage, weights=all_weights)),
            "rms_voltage_deviation_pu": float(np.sqrt(np.average(voltage ** 2, weights=all_weights))),
            "peak_voltage_deviation_pu": float(voltage.max()),
            "cumulative_reward": float(trace.reward.sum()), "duration_s": duration,
            "stability_violation_steps": int(violations.sum()), "stability_violation_events": violation_events,
            "stability_violation_duration_s": float(trace.loc[violations, "dt_s"].sum()),
            "voltage_violation_steps": int(trace.voltage_violation.sum()),
            "frequency_violation_steps": int(trace.frequency_violation.sum()),
            "overload_violation_steps": int(trace.overload_violation.sum()),
            "blackout_events": int(trace.blackout_start.sum()),
            "blackout_duration_s": float((trace.blackout * trace.dt_s).sum()),
            "blackout_events_per_hour": float(trace.blackout_start.sum() * 3600 / duration),
            "solver_failures": int(trace.solver_failure.sum()),
            "shed_mwh": float((trace.shed_mw * hours).sum()),
            "unserved_mwh": float((trace.unserved_mw * hours).sum()),
            "demand_mwh": demand_mwh,
            "served_fraction": 1 - float((trace.unserved_mw * hours).sum()) / max(demand_mwh, 1e-9),
            "battery_cycles": float(trace.battery_cycles.iloc[-1]),
            "losses_mwh": float((trace.losses_mw * hours).sum()),
            "curtailed_mwh": float((trace.curtailed_mw * hours).sum()),
            "peak_frequency_deviation_hz": float(trace.frequency_peak_deviation_hz.max()),
            "mean_observation_age_s": float(trace.observation_age_s.mean()),
            "mean_voltage_min_pu": float(trace.voltage_min_pu.mean()),
            "mean_soc": float(trace.soc.mean()),
            "mean_shed_mw": float(trace.shed_mw.mean()),
            "mean_battery_power_mw": float(trace.battery_power_mw.mean()),
            "stable_fraction": float((trace.stable * trace.dt_s).sum() / duration)}


def aggregate_results(episodes: pd.DataFrame) -> pd.DataFrame:
    groups = ["controller", "delay_steps", "fluctuation_scale"]
    metrics = [c for c in episodes.select_dtypes(include=[np.number]).columns
               if c not in {*groups, "episode_seed", "training_seed", "episode"}]
    grouped = episodes.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    return grouped.fillna(0).reset_index()

