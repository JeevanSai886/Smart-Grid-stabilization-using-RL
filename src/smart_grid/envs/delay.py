"""Packet-level sensor latency with congestion, dropout, and reordering."""
from __future__ import annotations

import heapq
import numpy as np


class DelayBuffer:
    def __init__(self, config: dict, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.queue = []
        self.latest = None
        self.latest_timestamp = -1
        self.congested = False
        self.dropped = 0

    def reset(self, initial: np.ndarray) -> None:
        self.queue.clear()
        self.latest = initial.copy()
        # Explicit warm-start measurement at episode start; no future padding.
        self.latest_timestamp = 0
        self.congested = False
        self.dropped = 0

    def submit(self, timestamp: int, value: np.ndarray, delay_steps: int | None = None) -> None:
        c = self.config
        if delay_steps is None:
            # Consume this stream independently from exogenous grid randomness.
            transition = self.rng.random()
            if self.congested:
                self.congested = not (transition < c["leave_congestion_probability"])
            else:
                self.congested = transition < c["enter_congestion_probability"]
            if self.rng.random() < c["packet_loss_probability"]:
                self.dropped += 1
                return
            if c["mode"] == "fixed":
                delay_steps = c["fixed_steps"]
            else:
                mean = c["congestion_mean_steps"] if self.congested else c["base_mean_steps"]
                delay_steps = min(c["max_steps"], int(self.rng.poisson(mean)))
        if delay_steps < 0:
            raise ValueError("Packets cannot arrive before measurement")
        heapq.heappush(self.queue, (timestamp + int(delay_steps), timestamp, value.copy()))

    def receive(self, now: int) -> tuple[np.ndarray, int]:
        if self.latest is None:
            raise RuntimeError("Reset delay buffer before receiving")
        while self.queue and self.queue[0][0] <= now:
            _, timestamp, value = heapq.heappop(self.queue)
            if timestamp > self.latest_timestamp:
                self.latest_timestamp = timestamp
                self.latest = value
        return self.latest.copy(), now - self.latest_timestamp
