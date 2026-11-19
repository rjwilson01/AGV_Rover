"""Simulated load cell: one-pole low-pass, Gaussian noise, clipping and quantisation (signed force, N)."""
from __future__ import annotations

import numpy as np


class LoadCell:
    def __init__(self, cfg, channel: int = 0):
        self.noise, self.res, self.rng_n = cfg.loadcell_noise_n, cfg.loadcell_resolution_n, cfg.loadcell_range_n
        self.bw = cfg.loadcell_bw_hz
        self._seed = cfg.loadcell_seed + 1000 * channel
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)
        self._y = None
        self._t = None

    def measure(self, t: float, force_n: float) -> float:
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        if self._y is None or self.bw <= 0:
            self._y = float(force_n)
        else:
            a = 1.0 - np.exp(-2.0 * np.pi * self.bw * dt)
            self._y += a * (float(force_n) - self._y)
        v = self._y + float(self.rng.normal(0.0, self.noise))
        v = float(np.clip(v, -self.rng_n, self.rng_n))
        return round(v / self.res) * self.res if self.res > 0 else v
