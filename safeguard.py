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
