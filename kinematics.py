from collections import deque
import numpy as np
from config import AGVConfig

# Numerical & algorithm constants
PATH_HISTORY_MAX_SAMPLES = 4000
MIN_LOOKAHEAD_SQ_M2 = 0.10
STRAIGHT_LINE_YAW_EPS = 1e-4
NUMERIC_EPS = 1e-9


def calc_cable_len(
    yaw: float,
    pitch: float,
    roll: float,
    y_sign: float,
    z_sign: float,
    cfg: AGVConfig,
) -> float:
    """Compute tendon length between adjacent trailer anchor plates for a given joint orientation."""
    L_bar = cfg.cable_joint_to_plate_x_m
    W_y = cfg.cable_anchor_y_m
    H_z = cfg.cable_anchor_z_m

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy * cp) * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    wy = (sy * cp) * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    wz = (-sp) * bx + (cp * sr) * by + (cp * cr) * bz

    return float(np.linalg.norm([wx + L_bar, wy - by, wz - bz]))


def quat_to_euler(q) -> tuple[float, float, float]:
    """Convert [w, x, y, z] quaternion to (yaw, pitch, roll) in radians."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw), float(pitch), float(roll)


class SnakePathController:
    def __init__(self, cfg: AGVConfig):
        self.cfg = cfg
        self.waypoints = np.asarray(cfg.waypoints, dtype=float)
        self.num_trailers = cfg.num_trailers
        self.num_joints = cfg.num_trailers - 1
        self.joint_delays_m = cfg.joint_delays_m()
        self.max_joint_yaw_rad = np.deg2rad(cfg.max_joint_yaw_deg)

        self.path_history: deque[tuple[float, float]] = deque(maxlen=PATH_HISTORY_MAX_SAMPLES)
        self.reset()

    def reset(self) -> None:
        """Reset all runtime path-tracking state (call on simulator reset)."""
        self.dist_traveled = 0.0
        self.prev_pos = None
        self.path_history.clear()
        self.last_lookahead = None
        self.last_joint_yaws = [0.0] * self.num_joints

    def _get_lookahead_point(self, pos_xy, lookahead_dist: float):
        best_dist = float("inf")
        best_seg_idx = 0
        best_proj = self.waypoints[0]

        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))
            t = np.clip(np.dot(pos_xy - a, ab) / (ab_len_sq + NUMERIC_EPS), 0.0, 1.0)
            proj = a + t * ab
            d = float(np.linalg.norm(pos_xy - proj))
            if d < best_dist:
                best_dist = d
                best_seg_idx = i
                best_proj = proj

        rem = lookahead_dist
        curr_pt = best_proj
        idx = best_seg_idx

        while idx < len(self.waypoints) - 1:
            seg_end = self.waypoints[idx + 1]
            seg_vec = seg_end - curr_pt
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len >= rem:
                return curr_pt + (seg_vec / (seg_len + NUMERIC_EPS)) * rem, False
            rem -= seg_len
            curr_pt = seg_end
            idx += 1

        return self.waypoints[-1].copy(), True

    def _lookup_yaw(self, delay_m: float) -> float:
        target_d = self.dist_traveled - delay_m
        for d, y in reversed(self.path_history):
            if d <= target_d:
                return y
        return 0.0

    def step(
        self,
        pos_xy,
        lead_hdg_rad: float,
        v_center: float | None = None,
        lookahead_dist: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """
        Compute wheel angular velocities (2 per trailer) and target joint yaws (N-1 joints).
        Returns (wheel_vels, joint_yaws).
        """
        v_cmd = self.cfg.v_center_mps if v_center is None else v_center
        ld_cmd = self.cfg.lookahead_dist_m if lookahead_dist is None else lookahead_dist
        num_wheels = 2 * self.num_trailers

        if self.prev_pos is not None:
            self.dist_traveled += float(np.linalg.norm(pos_xy - self.prev_pos))
        self.prev_pos = pos_xy.copy()

        target_pt, at_goal = self._get_lookahead_point(pos_xy, ld_cmd)
        self.last_lookahead = target_pt.copy()

        # Hold final joint angles at goal so locked wheels don't fight straightening cables
        if at_goal and np.linalg.norm(target_pt - pos_xy) < self.cfg.goal_tol_m:
            return [0.0] * num_wheels, list(self.last_joint_yaws)

        # Transform target into lead AGV local coordinates
        dx, dy = target_pt - pos_xy
        c, s = np.cos(-lead_hdg_rad), np.sin(-lead_hdg_rad)
        x_loc = c * dx - s * dy
        y_loc = s * dx + c * dy

        # Pure Pursuit curvature -> capstan joint yaw
        ld_sq = max(x_loc**2 + y_loc**2, MIN_LOOKAHEAD_SQ_M2)
        kappa = float(np.clip((2.0 * y_loc) / ld_sq, -self.cfg.max_curvature, self.cfg.max_curvature))
        target_yaw = float(
            np.clip(self.cfg.yaw_gain * kappa, -self.max_joint_yaw_rad, self.max_joint_yaw_rad)
        )
        self.path_history.append((self.dist_traveled, target_yaw))

        joint_yaws = [self._lookup_yaw(delay) for delay in self.joint_delays_m]

        # Effective steering angle at each axle:
        # Axle 0 (lead): joint_yaws[0]
        # Middle axles i: average of front and rear joints 0.5 * (joint_yaws[i-1] + joint_yaws[i])
        # Last axle (N-1): joint_yaws[-1]
        axle_yaws = []
        for i in range(self.num_trailers):
            if i == 0:
                axle_yaws.append(joint_yaws[0])
            elif i == self.num_trailers - 1:
                axle_yaws.append(joint_yaws[-1])
            else:
                axle_yaws.append(0.5 * (joint_yaws[i - 1] + joint_yaws[i]))

        wheel_vels = []
        for psi in axle_yaws:
            if abs(psi) < STRAIGHT_LINE_YAW_EPS:
                wl = wr = v_cmd / self.cfg.wheel_radius_m
            else:
                turn_radius = self.cfg.half_drawbar_len_m / np.tan(psi / 2.0)
                wl = (v_cmd * (1.0 + self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
                wr = (v_cmd * (1.0 - self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
            wheel_vels.extend([wl, wr])

        self.last_joint_yaws = list(joint_yaws)
        return wheel_vels, joint_yaws
