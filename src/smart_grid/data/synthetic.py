"""Statistically structured profiles; these are synthetic, not field measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def correlated_noise(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) fluctuations with stationary initialization."""
    innovations = rng.normal(0.0, np.sqrt(1 - rho * rho), n)
    innovations[0] = rng.normal()
    return lfilter([1.0], [1.0, -rho], innovations)


def generate_profiles(days: int = 30, interval_seconds: int = 300,
                      seed: int = 42, start: str = "2018-01-01") -> pd.DataFrame:
    if days < 3 or interval_seconds <= 0:
        raise ValueError("At least three days and a positive sampling interval required")
    rng = np.random.default_rng(seed)
    n = int(days * 86400 // interval_seconds)
    index = pd.date_range(start, periods=n, freq=pd.Timedelta(seconds=interval_seconds), tz="UTC")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    daylight = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    cloud = np.clip(0.78 + 0.18 * correlated_noise(rng, n, 0.96), 0.12, 1.0)
    solar = np.clip(daylight * cloud, 0, 1)
    wind = np.clip(0.42 + 0.17 * correlated_noise(rng, n, 0.985), 0.02, 0.98)
    morning = 0.19 * np.exp(-0.5 * ((hour - 8) / 1.7) ** 2)
    evening = 0.36 * np.exp(-0.5 * ((hour - 19) / 2.3) ** 2)
    weekend = np.where(index.dayofweek >= 5, 0.94, 1.0)
    demand = (0.81 + morning + evening) * weekend
    demand += 0.035 * correlated_noise(rng, n, 0.85)
    frame = pd.DataFrame({"solar_cf": solar, "wind_cf": wind,
                          "load_factor": np.clip(demand, 0.55, 1.5)}, index=index)
    frame.index.name = "timestamp"
    return frame

