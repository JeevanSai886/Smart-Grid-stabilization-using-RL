from __future__ import annotations

import numpy as np
from smart_grid.envs.smart_grid_env import BATTERY_LEVELS, FRAME_SIZE


class RuleBasedController:
    """Threshold controller with hysteresis and the same stale measurements as PPO."""
    def __init__(self, config: dict, base_load_mw: float):
        self.config = config
        self.base = base_load_mw
        self.shedding = 0

    def reset(self) -> None:
        self.shedding = 0

    def predict(self, observation: np.ndarray) -> np.ndarray:
        m = np.asarray(observation).reshape(-1, FRAME_SIZE)[-1]
        solar, wind, demand = m[:3] * self.base
        minimum_voltage, loading = m[4], m[6]
        deviation, soc = m[7] * 5, m[8]
        mechanical, capacity = m[10] * self.base, m[11] * self.base
        if m[12] > 0.5:
            return np.array([2, 0], dtype=np.int64)
        maximum = self.config["battery"]["max_power_mw"]
        # The 3% loss allowance is a declared heuristic, not privileged power flow.
        required = demand * 1.03 - solar - wind
        battery_request = required - mechanical - 1.2 * deviation
        if minimum_voltage < 0.945 or loading > 1.0:
            battery_request = max(battery_request, maximum)
        if minimum_voltage > 1.055:
            battery_request = min(battery_request, -maximum)
        if soc <= self.config["battery"]["soc_min"] + 0.025:
            battery_request = min(battery_request, 0)
        if soc >= self.config["battery"]["soc_max"] - 0.025:
            battery_request = max(battery_request, 0)
        battery_index = int(np.argmin(abs(BATTERY_LEVELS * maximum - battery_request)))
        if deviation < -1.5 or minimum_voltage < 0.87:
            self.shedding = 2
        elif deviation < -0.85 or minimum_voltage < 0.91 or required > capacity + maximum:
            self.shedding = max(self.shedding, 1)
        elif deviation > -0.2 and minimum_voltage > 0.955 and required < capacity + maximum * 0.5:
            self.shedding = 0
        return np.array([battery_index, self.shedding], dtype=np.int64)


class IdleController:
    def reset(self) -> None:
        pass

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return np.array([2, 0], dtype=np.int64)

