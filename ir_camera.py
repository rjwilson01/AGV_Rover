"""Simulated IR camera: relative roll/pitch/yaw between neighbouring trailers.

The real robot has one IR camera per joint that reports the orientation of the next
trailer relative to the trailer carrying the camera. Here that reading is derived from
the world-frame orientations of the two trailer bodies (not from the ball-joint state),
then degraded with optional noise, quantisation and latency so controllers only ever
see what the physical sensor would give them.

Angle convention matches quat_to_euler(): yaw (about Z), pitch (about Y), roll (about X),
applied in ZYX order, all in radians.
"""

from collections import deque
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from BaseSnakeController import quat_to_euler
from config import AGVConfig

QUAT_NORM_EPS = 1e-9


class IRReading(NamedTuple):
    """Relative orientation of trailer j+1 in the frame of trailer j (radians)."""

    yaw: float
    pitch: float
    roll: float


def _normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < QUAT_NORM_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conj(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b for [w, x, y, z] quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def relative_euler(q_parent, q_child) -> IRReading:
    """Euler angles of the child body expressed in the parent body's frame."""
    q_rel = quat_mul(quat_conj(_normalize(q_parent)), _normalize(q_child))
    yaw, pitch, roll = quat_to_euler(q_rel)
    return IRReading(yaw, pitch, roll)


class IRCameraModel:
    """One simulated IR camera per joint. Effects are applied noise -> quantisation -> latency."""

    def __init__(self, cfg: AGVConfig):
        self.num_joints = cfg.num_joints
        self.noise_std_rad = float(np.deg2rad(max(0.0, cfg.ir_noise_std_deg)))
        self.quant_rad = float(np.deg2rad(max(0.0, cfg.ir_quantization_deg)))
        self.latency_s = max(0.0, cfg.ir_latency_s)
        self.seed = cfg.ir_seed
        self.reset()

    def reset(self) -> None:
        """Clear the latency buffer and restart the noise sequence (call on sim reset)."""
        self._rng = np.random.default_rng(self.seed)
        self._buffer: deque[tuple[float, list[IRReading]]] = deque()

    def measure(self, t: float, trailer_quats: Sequence) -> list[IRReading]:
        """Return one IRReading per joint.

        t: simulation time in seconds.
        trailer_quats: world-frame [w, x, y, z] orientation of trailer_1 ... trailer_N, in order.
        """
        if len(trailer_quats) != self.num_joints + 1:
            raise ValueError(
                f"expected {self.num_joints + 1} trailer quaternions, got {len(trailer_quats)}"
            )

        sensed = [
            self._degrade(relative_euler(trailer_quats[j], trailer_quats[j + 1]))
            for j in range(self.num_joints)
        ]
        if self.latency_s <= 0.0:
            return sensed

        self._buffer.append((float(t), sensed))
        target_t = float(t) - self.latency_s
        # Keep the newest sample that is old enough; until history builds up, hold the first sample.
        while len(self._buffer) > 1 and self._buffer[1][0] <= target_t:
            self._buffer.popleft()
        return list(self._buffer[0][1])

    def _degrade(self, reading: IRReading) -> IRReading:
        vals = np.array(reading, dtype=float)
        if self.noise_std_rad > 0.0:
            vals = vals + self._rng.normal(0.0, self.noise_std_rad, size=3)
        if self.quant_rad > 0.0:
            vals = np.round(vals / self.quant_rad) * self.quant_rad
        return IRReading(float(vals[0]), float(vals[1]), float(vals[2]))
