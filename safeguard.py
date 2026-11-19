"""Follower stall guard: if the lead is commanded to move but isn't (blocked, stuck, collided), the followers stop
instead of driving on open-loop and jack-knifing against it. Release is gated on the lead moving again."""


class FollowerGuard:
    def __init__(self, cfg):
        self.enabled = cfg.guard_enabled
        self.stall_ratio = cfg.guard_stall_ratio
        self.release_ratio = cfg.guard_release_ratio
        self.stall_time = cfg.guard_stall_time_s
        self.min_cmd = cfg.guard_min_cmd_mps
        self.ramp_s = cfg.guard_ramp_s
        self.reset()

    def reset(self) -> None:
        self.scale = 1.0
        self.stalled = False
        self._t_low = 0.0
        self._t_prev = None

    def update(self, t: float, v_cmd: float, v_meas: float) -> float:
        """v_cmd, v_meas: lead's commanded and measured speed (m/s, >= 0). Returns follower speed scale 0..1."""
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        if not self.enabled:
            return 1.0
        if v_cmd < self.min_cmd:
            self._t_low = 0.0                      # not asked to move: nothing to judge (lead stopping is normal)
            if self.stalled and v_cmd < self.min_cmd * 0.5:
                self.stalled = False
        elif v_meas < self.stall_ratio * v_cmd:
            self._t_low += dt
            if self._t_low >= self.stall_time:
                self.stalled = True
        else:
            self._t_low = 0.0
            if self.stalled and v_meas > self.release_ratio * v_cmd:
                self.stalled = False
        step = dt / self.ramp_s if self.ramp_s > 0 else 1.0
        self.scale = max(0.0, self.scale - step) if self.stalled else min(1.0, self.scale + step)
        return self.scale


class RetensionGuard:
    """Compression is a failure mode: when a hitch is being PUSHED, slow the trailer behind it at once.

    The resting sum of the four cable tensions of each hitch is learned over the first `retension_baseline_s`. Cables are
    pre-tensioned, so a push (compression through the bar) unloads them: the sum falls below the baseline. Once it is more
    than `retension_thresh_n` below, that hitch's scale drops immediately by `retension_gain_per_n` per N beyond the
    threshold (floor `retension_min_scale`), and recovers at (1 - min)/release_s per second. A trailer's wheel scale is the
    product of the scales of all hitches ahead of it.
    """

    def __init__(self, cfg, n_hitches: int):
        self.enabled = cfg.retension_guard
        self.n = n_hitches
        self.base_s, self.thresh = cfg.retension_baseline_s, cfg.retension_thresh_n
        self.gain, self.min_scale, self.release_s = cfg.retension_gain_per_n, cfg.retension_min_scale, cfg.retension_release_s
        self.reset()

    def reset(self) -> None:
        self.baseline = [None] * self.n
        self._acc = [[] for _ in range(self.n)]
        self.scale = [1.0] * self.n
        self.deficit = [0.0] * self.n
        self._t_prev = None

    def update(self, t: float, cable_sums) -> list[float]:
        """cable_sums: measured sum of the four cable tensions per hitch (N). Returns one scale per hitch."""
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        if not self.enabled:
            return self.scale
        for j in range(self.n):
            s = float(cable_sums[j])
            if self.baseline[j] is None:
                self._acc[j].append(s)
                if t >= self.base_s:
                    self.baseline[j] = sum(self._acc[j]) / len(self._acc[j])
                continue
            self.deficit[j] = max(0.0, (self.baseline[j] - self.thresh) - s)
            target = max(self.min_scale, 1.0 - self.gain * self.deficit[j]) if self.deficit[j] > 0.0 else 1.0
            if target < self.scale[j]:
                self.scale[j] = target                                   # attack: immediately
            else:
                rate = (1.0 - self.min_scale) / self.release_s if self.release_s > 0 else 1.0
                self.scale[j] = min(target, self.scale[j] + rate * dt)   # release: gradually
        return self.scale

    def trailer_scale(self, i: int) -> float:
        """Wheel scale for trailer index i (0 = lead): product over hitches 0..i-1."""
        out = 1.0
        for j in range(min(i, self.n)):
            out *= self.scale[j]
        return out
