"""Lead-trailer position and heading from the sensors a real vehicle would have (GNSS, a yaw gyro, wheel odometry).

`controller.step()` normally gets the simulator's TRUE position and heading. With `[nav] gnss_in_loop = true` it gets this
estimate instead, so GNSS noise, bias, latency, outages and gyro drift show up in the path tracking and the tail swing.

  position   predicted between fixes by dead reckoning (signed lead speed along the estimated heading), pulled towards each new
             GNSS fix (velocity-propagated across the receiver latency) with gain `pos_gain`.
  heading    integrated from the lead's gyro (yaw rate), pulled towards the GNSS course over ground (from the GNSS velocity,
             flipped when reversing) whenever the speed is at least `min_course_speed_mps`; time constant `course_tau_s`.
             Without the gyro (`use_gyro = false`) only the course is used, so it lags in turns and is noisy when slow.
             The gyro's zero-rate offset is learned from the course error, so it is already trimmed when a fix is lost.
  outage     no fix: dead reckoning on odometry and the gyro (still drifts: the offset estimate is imperfect, and odometry
             ignores wheel slip).

Start-up alignment is cheated: the first call is initialised from the true pose (a real vehicle would align while driving).
Course over ground equals the heading only when the vehicle is not skidding sideways; skid is not modelled here.
The antenna is assumed to sit at the lead chassis centre.
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class LeadNav:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.pos = None
        self.hdg = 0.0
        self._t = 0.0
        self._fix_t = None
        self.fix_valid = False
        self.gyro_bias = 0.0                                   # estimated gyro zero-rate offset, rad/s

    def update(self, t: float, gnss, gyro_z, v_hint: float, truth_pos, truth_hdg) -> tuple[np.ndarray, float]:
        """-> (position estimate [x, y], heading estimate rad). gnss: GNSSReading or None; gyro_z rad/s or None;
        v_hint: signed lead speed from wheel odometry (m/s)."""
        c = self.cfg
        if self.pos is None:                                   # start-up alignment from the true pose
            self.pos = np.asarray(truth_pos, float)[:2].copy()
            self.hdg = float(truth_hdg)
            self._t = t
            self._fix_t = None if gnss is None else gnss.t
            return self.pos.copy(), self.hdg
        dt = max(t - self._t, 0.0)
        self._t = t
        if c.nav_use_gyro and gyro_z is not None:
            self.hdg = wrap(self.hdg + (float(gyro_z) - self.gyro_bias) * dt)
        self.pos = self.pos + v_hint * dt * np.array([math.cos(self.hdg), math.sin(self.hdg)])
        self.fix_valid = bool(gnss is not None and gnss.valid)
        if self.fix_valid and gnss.t != self._fix_t:
            dt_fix = (gnss.t - self._fix_t) if self._fix_t is not None else 1.0 / max(c.gnss_rate_hz, 1e-6)
            dt_fix = max(dt_fix, 1e-3)
            self._fix_t = gnss.t
            age = max(t - gnss.t, 0.0) + c.gnss_latency_s
            p = np.array([gnss.x, gnss.y]) + np.array([gnss.vx, gnss.vy]) * age
            self.pos = self.pos + c.nav_pos_gain * (p - self.pos)
            if gnss.speed >= c.nav_min_course_speed_mps:
                course = math.atan2(gnss.vy, gnss.vx)
                if v_hint < 0.0:
                    course += math.pi                          # driving backwards: the course is the reverse of the heading
                k = 1.0 - math.exp(-dt_fix / max(c.nav_course_tau_s, 1e-6))
                err = wrap(course - self.hdg)
                self.hdg = wrap(self.hdg + k * err)
                if c.nav_use_gyro and gyro_z is not None:          # learn the gyro offset (critically damped PI): it keeps the
                    tau = max(c.nav_course_tau_s, 1e-6)             # heading right through an outage
                    self.gyro_bias = float(np.clip(self.gyro_bias - err * dt_fix / (4.0 * tau * tau), -0.1, 0.1))
        return self.pos.copy(), self.hdg
