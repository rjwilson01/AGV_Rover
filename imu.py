"""Simulated per-trailer 6-axis IMU (Bosch BMI088: separate accelerometer and gyroscope dies).

MuJoCo supplies the TRUE specific force (accelerometer, includes gravity: +g on z at rest) and angular velocity
(gyro) at a site on each trailer. This model degrades them the way the part would, in this order:
  true signal -> (+ broadband vibration on the accelerometer) -> one-pole low-pass at the sensor bandwidth
  -> + turn-on bias (random per axis, per run) + gyro g-sensitivity -> + white noise (density x sqrt(bandwidth))
  -> clip to the selected full-scale range -> quantise to 16 bit -> sample-and-hold at the output data rate.

Defaults are the BMI088 datasheet (rev 1.8) figures: accel noise 160/160/190 ug/rtHz (X/Y/Z, +-3 g range), zero-g
offset 20 mg, bandwidth up to 280 Hz (245 Hz Z); gyro noise 0.014 dps/rtHz, zero-rate offset +-1 dps,
g-sensitivity 0.1 dps/g, bandwidth 47..523 Hz. Frame: the trailer body frame (x forward, y left, z up).
"""
from typing import NamedTuple, Sequence

import numpy as np

G = 9.80665


class IMUReading(NamedTuple):
    acc: tuple[float, float, float]     # m/s^2, specific force in the trailer frame (at rest: (0, 0, +9.8))
    gyro: tuple[float, float, float]    # rad/s, angular velocity in the trailer frame


def tilt_from_accel(acc: Sequence[float]) -> tuple[float, float]:
    """(pitch, roll) in radians from a quasi-static accelerometer reading (same convention as quat_to_euler)."""
    ax, ay, az = acc
    return float(np.arctan2(-ax, np.hypot(ay, az))), float(np.arctan2(ay, az))


class IMUModel:
    def __init__(self, cfg, num_sensors: int | None = None):
        self.cfg = cfg
        self.n = cfg.num_trailers if num_sensors is None else num_sensors
        self.dt_out = 1.0 / cfg.imu_rate_hz
        self.acc_density = np.asarray(cfg.imu_acc_noise_ug_rthz, dtype=float) * 1e-6 * G    # (m/s^2)/rtHz
        self.gyro_density = np.deg2rad(cfg.imu_gyro_noise_dps_rthz)                         # (rad/s)/rtHz
        self.acc_lsb = cfg.imu_acc_range_g * G / 32768.0
        self.gyro_lsb = np.deg2rad(cfg.imu_gyro_range_dps) / 32768.0
        self.reset()

    def reset(self) -> None:
        rng = np.random.default_rng(self.cfg.imu_seed)
        self.rng = rng
        # Turn-on bias, fixed for the run: uniform within the datasheet offset limits
        self.acc_bias = rng.uniform(-1.0, 1.0, (self.n, 3)) * self.cfg.imu_acc_offset_mg * 1e-3 * G
        self.gyro_bias = rng.uniform(-1.0, 1.0, (self.n, 3)) * np.deg2rad(self.cfg.imu_gyro_offset_dps)
        self._acc_f = None
        self._gyro_f = None
        self._t_prev = None
        self._t_next_sample = 0.0
        self._held = [IMUReading((0.0, 0.0, G), (0.0, 0.0, 0.0)) for _ in range(self.n)]

    @staticmethod
    def _alpha(dt: float, bw_hz: float) -> float:
        return 1.0 - float(np.exp(-2.0 * np.pi * bw_hz * dt))

    def measure(self, t: float, true_acc: Sequence[Sequence[float]], true_gyro: Sequence[Sequence[float]]) -> list[IMUReading]:
        acc = np.asarray(true_acc, dtype=float).reshape(self.n, 3)
        gyro = np.asarray(true_gyro, dtype=float).reshape(self.n, 3)
        cfg = self.cfg

        if cfg.imu_vibration_g_rms > 0.0:
            acc = acc + self.rng.normal(0.0, cfg.imu_vibration_g_rms * G, acc.shape)

        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        if self._acc_f is None:
            self._acc_f, self._gyro_f = acc.copy(), gyro.copy()
        else:
            self._acc_f += self._alpha(dt, cfg.imu_acc_bw_hz) * (acc - self._acc_f)
            self._gyro_f += self._alpha(dt, cfg.imu_gyro_bw_hz) * (gyro - self._gyro_f)

        if t + 1e-12 < self._t_next_sample:                       # between output samples: hold the last reading
            return list(self._held)
        self._t_next_sample = t + self.dt_out

        a = self._acc_f + self.acc_bias
        a = a + self.rng.normal(0.0, 1.0, a.shape) * self.acc_density * np.sqrt(cfg.imu_acc_bw_hz)
        g = self._gyro_f + self.gyro_bias + np.deg2rad(cfg.imu_gyro_g_sens_dps_per_g) * (self._acc_f / G)
        g = g + self.rng.normal(0.0, 1.0, g.shape) * self.gyro_density * np.sqrt(cfg.imu_gyro_bw_hz)

        a = np.clip(a, -cfg.imu_acc_range_g * G, cfg.imu_acc_range_g * G)
        g = np.clip(g, -np.deg2rad(cfg.imu_gyro_range_dps), np.deg2rad(cfg.imu_gyro_range_dps))
        a = np.round(a / self.acc_lsb) * self.acc_lsb
        g = np.round(g / self.gyro_lsb) * self.gyro_lsb
        self._held = [IMUReading(tuple(map(float, a[i])), tuple(map(float, g[i]))) for i in range(self.n)]
        return list(self._held)
