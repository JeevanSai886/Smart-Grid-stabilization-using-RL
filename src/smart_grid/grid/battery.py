from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Battery:
    """Positive grid-side power discharges; negative grid-side power charges."""
    config: dict
    soc: float = 0.6
    throughput_mwh: float = 0.0

    def dispatch(self, requested_mw: float, dt_s: float) -> float:
        if dt_s <= 0 or not np.isfinite(requested_mw):
            raise ValueError("Finite power and positive dt required")
        c = self.config
        hours = dt_s / 3600
        available = max(0, self.soc - c["soc_min"]) * c["capacity_mwh"]
        room = max(0, c["soc_max"] - self.soc) * c["capacity_mwh"]
        discharge_limit = min(c["max_power_mw"], available * c["discharge_efficiency"] / hours)
        charge_limit = min(c["max_power_mw"], room / (c["charge_efficiency"] * hours))
        power = float(np.clip(requested_mw, -charge_limit, discharge_limit))
        stored_change = (-power / c["discharge_efficiency"] if power >= 0
                         else -power * c["charge_efficiency"]) * hours
        self.soc = float(np.clip(self.soc + stored_change / c["capacity_mwh"], c["soc_min"], c["soc_max"]))
        # Avoid a tiny floating-point energy remainder at an exhausted boundary.
        for bound in (c["soc_min"], c["soc_max"]):
            if abs(self.soc - bound) < 1e-12:
                self.soc = bound
        self.throughput_mwh += abs(stored_change)
        return power

    @property
    def equivalent_cycles(self) -> float:
        return self.throughput_mwh / (2 * self.config["capacity_mwh"])
