from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
import importlib
import numpy as np
from config import AGVConfig

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
    rear_bar_m: float | None = None,
    front_bar_m: float | None = None,
) -> float:
    """Compute spatial tendon length for unequal/telescoping rear and front half-drawbars."""
    r_bar = cfg.nominal_rear_bar_m if rear_bar_m is None else rear_bar_m
    f_bar = cfg.nominal_front_bar_m if front_bar_m is None else front_bar_m

    l_rear_plate = r_bar - cfg.chassis_half_len_m
    l_front_plate = f_bar - cfg.chassis_half_len_m
    W_y = cfg.cable_anchor_y_m
    H_z = cfg.cable_anchor_z_m

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    bx, by, bz = l_front_plate, y_sign * W_y, z_sign * H_z
    wx = (cy * cp) * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    wy = (sy * cp) * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    wz = (-sp) * bx + (cp * sr) * by + (cp * cr) * bz

    return float(np.linalg.norm([wx + l_rear_plate, wy - by, wz - bz]))


def quat_to_euler(q) -> tuple[float, float, float]:
    """Convert [w, x, y, z] quaternion to (yaw, pitch, roll) in radians."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw), float(pitch), float(roll)


class BaseSnakeController(ABC):
    """Abstract base class defining the pure-virtual interface for articulated snake controllers."""

    def __init__(self, cfg: AGVConfig):
        self.cfg = cfg
        self.waypoints = np.asarray(cfg.waypoints, dtype=float)
        self.num_trailers = cfg.num_trailers
        self.num_joints = cfg.num_trailers - 1
        self.yaw_min_rad = np.deg2rad(cfg.yaw_angle_range_deg[0])
        self.yaw_max_rad = np.deg2rad(cfg.yaw_angle_range_deg[1])
        # Per-bar tube yaw limits (used when cfg.bar_yaw_control is on)
        self.rear_yaw_min_rad = np.deg2rad(cfg.rear_bar_yaw_range_deg[0])
        self.rear_yaw_max_rad = np.deg2rad(cfg.rear_bar_yaw_range_deg[1])
        self.front_yaw_min_rad = np.deg2rad(cfg.front_bar_yaw_range_deg[0])
        self.front_yaw_max_rad = np.deg2rad(cfg.front_bar_yaw_range_deg[1])

        self.path_history: deque[tuple[float, float]] = deque(maxlen=PATH_HISTORY_MAX_SAMPLES)
        self.reset()

    def reset(self) -> None:
        self.dist_traveled = 0.0
        self.prev_pos = None
        self.path_history.clear()
        self.last_lookahead = None
        self.last_joint_yaws = [0.0] * self.num_joints
        self.last_ir_readings = [(0.0, 0.0, 0.0)] * self.num_joints
        # Commanded tube yaw per hitch: (rear bar of trailer j, front bar of trailer j+1), radians.
        # Their sum plus the ball yaw equals the commanded trailer-to-trailer yaw in last_joint_yaws.
        self.last_bar_yaws = [(0.0, 0.0)] * self.num_joints
        self.last_bar_lengths = [
            (self.cfg.nominal_rear_bar_m, self.cfg.nominal_front_bar_m)
            for _ in range(self.num_joints)
        ]

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

    def _lookup_curvature(self, delay_m: float) -> float:
        target_d = self.dist_traveled - delay_m
        for d, kappa in reversed(self.path_history):
            if d <= target_d:
                return kappa
        return 0.0

    def _compute_pure_pursuit_curvature(self, pos_xy, lead_hdg_rad: float, target_pt) -> float:
        dx, dy = target_pt - pos_xy
        c, s = np.cos(-lead_hdg_rad), np.sin(-lead_hdg_rad)
        x_loc = c * dx - s * dy
        y_loc = s * dx + c * dy
        ld_sq = max(x_loc**2 + y_loc**2, MIN_LOOKAHEAD_SQ_M2)
        return float(np.clip((2.0 * y_loc) / ld_sq, -self.cfg.max_curvature, self.cfg.max_curvature))

    @abstractmethod
    def step(
        self,
        pos_xy,
        lead_hdg_rad: float,
        v_center: float | None = None,
        lookahead_dist: float | None = None,
        ir_readings: Sequence[tuple[float, float, float]] | None = None,
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Returns (wheel_vels, joint_yaws, bar_lengths[(rear_m, front_m), ...]).

        ir_readings: one (yaw, pitch, roll) in radians per joint, as reported by the
        simulated IR camera (relative orientation of trailer j+1 in the frame of trailer j).
        """
        ...

    def _store_ir(self, ir_readings: Sequence[tuple[float, float, float]] | None) -> None:
        """Keep the latest IR camera readings so control laws can use them."""
        if ir_readings is not None:
            self.last_ir_readings = [tuple(float(v) for v in r) for r in ir_readings]

    def get_telemetry(self) -> dict[str, float]:
        """Return controller-specific telemetry channels for PlotJuggler/CSV."""
        return {}


def create_controller(cfg: AGVConfig) -> BaseSnakeController:
    """Dynamically import <ClassName>.py and instantiate <ClassName>(cfg)."""
    cls_name = cfg.controller_class
    module = importlib.import_module(cls_name)
    controller_cls = getattr(module, cls_name)
    return controller_cls(cfg)
