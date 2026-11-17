"""Simulated lead-trailer RTK GNSS receiver (u-blox ZED-F9P class).

Truth position/velocity (world frame) -> + slowly varying correlated bias (Gauss-Markov) + white noise
-> latency -> sample-and-hold at the receiver rate. Outage windows (trees, lost corrections) mark the fix invalid and
hold the last good output. Defaults: RTK-fixed ~1.4 cm horizontal, velocity ~0.05 m/s, 10 Hz, 100 ms latency.
"""
from collections import deque
from typing import NamedTuple

import numpy as np


class GNSSReading(NamedTuple):
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    valid: bool


class GNSSModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.cfg.gnss_seed)
        self.bias = np.zeros(2)
        self._buf: deque = deque()
        self._t_next = 0.0
        self._t_prev = None
        self._held = GNSSReading(0.0, 0.0, 0.0, 0.0, 0.0, False)

    def _in_outage(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.cfg.gnss_outages_s)

    def measure(self, t: float, pos_xy, vel_xy) -> GNSSReading:
        c = self.cfg
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        # Correlated position bias (Gauss-Markov), stationary std = gnss_bias_m
        if c.gnss_bias_m > 0 and dt > 0:
            a = np.exp(-dt / c.gnss_bias_tau_s)
            self.bias = a * self.bias + np.sqrt(1 - a * a) * c.gnss_bias_m * self.rng.normal(size=2)
        self._buf.append((t, np.asarray(pos_xy, float)[:2].copy(), np.asarray(vel_xy, float)[:2].copy()))
        while len(self._buf) > 1 and self._buf[1][0] <= t - c.gnss_latency_s:
            self._buf.popleft()
        if t + 1e-12 < self._t_next:
            return self._held
        self._t_next = t + 1.0 / c.gnss_rate_hz
        if self._in_outage(t):
            self._held = self._held._replace(valid=False)
            return self._held
        _, p, v = self._buf[0]
        p = p + self.bias + self.rng.normal(0.0, c.gnss_pos_noise_m, 2)
        v = v + self.rng.normal(0.0, c.gnss_vel_noise_mps, 2)
        self._held = GNSSReading(float(p[0]), float(p[1]), float(v[0]), float(v[1]), float(np.hypot(*v)), True)
        return self._held
