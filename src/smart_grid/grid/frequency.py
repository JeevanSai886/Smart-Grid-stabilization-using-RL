"""Single-area swing/governor approximation coupled to AC electrical demand.

M * d(df)/dt = P_mechanical - P_electrical - D * df.
M = 2 * H * S_rated / f_nominal, in MW*s/Hz.
The AC reference source supplies electrical power; finite rotor energy and the
bounded prime mover account for its imbalance. It is not a free energy source.
This is a quasi-static network / aggregate dynamic model, not transient AC DAE.
"""
from __future__ import annotations

import math
import numpy as np


class FrequencyModel:
    def __init__(self, config: dict):
        self.config = config
        self.deviation = 0.0
        self.mechanical_mw = 0.0
        self.reference_mw = 0.0

    def reset(self, electrical_mw: float, capacity_mw: float) -> None:
        self.deviation = 0.0
        self.mechanical_mw = float(np.clip(electrical_mw, 0, capacity_mw))
        self.reference_mw = self.mechanical_mw

    def advance(self, electrical_mw: float, capacity_mw: float, dt_s: float) -> dict:
        c = self.config
        n = max(1, math.ceil(dt_s / c["integration_dt_s"]))
        h = dt_s / n
        inertia = 2 * c["inertia_s"] * c["source_rating_mva"] / c["nominal_hz"]
        damping = c["damping_mw_per_hz"]
        self.mechanical_mw = min(self.mechanical_mw, capacity_mw)
        peak = abs(self.deviation)
        for _ in range(n):
            target = np.clip(self.reference_mw - c["governor_gain_mw_per_hz"] * self.deviation, 0, capacity_mw)
            delta = (target - self.mechanical_mw) * (1 - math.exp(-h / c["governor_time_constant_s"]))
            limit = c["source_ramp_mw_per_s"] * h
            self.mechanical_mw = float(np.clip(self.mechanical_mw + np.clip(delta, -limit, limit), 0, capacity_mw))
            imbalance = self.mechanical_mw - electrical_mw
            # Exact linear frequency update within each substep for fixed Pm.
            if damping > 0:
                equilibrium = imbalance / damping
                self.deviation = equilibrium + (self.deviation - equilibrium) * math.exp(-damping * h / inertia)
            else:
                self.deviation += h * imbalance / inertia
            peak = max(peak, abs(self.deviation))
        return {"peak_deviation_hz": float(peak), "deviation_hz": float(self.deviation),
                "mechanical_mw": self.mechanical_mw}
