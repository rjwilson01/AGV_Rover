from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
import importlib
import numpy as np
from bar_geometry import pivot_x_m, warn_once
from planner import chain_curvature_limit, path_curvature, plan_path
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
        _g = self.waypoints[-1] - self.waypoints[-2]
        self.goal_pt = self.waypoints[-1].copy()                      # the REAL end of the path (the plan adds a run-out past it)
        self.goal_dir = _g / max(float(np.linalg.norm(_g)), 1e-9)
        self.max_curvature = cfg.max_curvature
        self.plan = None
        if cfg.planner_enabled:
            k_chain = chain_curvature_limit(cfg, cfg.planner_margin)
            self.plan = plan_path(self.waypoints, k_chain, cfg.planner_spacing_m, cfg.planner_radius_cap_m, cfg.planner_runout_m)
            self.waypoints = self.plan.points
            if k_chain > 0:
                self.max_curvature = min(cfg.max_curvature, k_chain)
            for w in self.plan.warnings:
                warn_once("planner", w)
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
        self.follow_override = None  # sandbox/view only: curvature every follower sees (None = normal)
        self.drive = None            # optional open-loop (v, kappa) profile for impulse/step-response testing
        self.sim_time = 0.0
        self._build_reference()
        self.reset()

    def _build_reference(self) -> None:
        """Idealised reference path: the smooth planned path with its curvature as a function of arc length.

        Followers take their curvature from this (at the lead's smoothed progress along it) instead of replaying the
        lead's own, wiggly, steering history.
        """
        _seg = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.path_s = np.concatenate([[0.0], np.cumsum(_seg)])
        self.goal_s = float(self.path_s[int(np.argmin(np.linalg.norm(self.waypoints - self.goal_pt, axis=1)))])
        self.ref_s = None
        self.ref_kappa = None
        if not (self.cfg.follow_ideal_path and self.plan is not None and len(self.waypoints) >= 3):
            return
        pts = self.waypoints
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        kap = np.zeros(len(pts))
        kap[1:-1] = path_curvature(pts)
        step = float(np.median(seg)) if len(seg) else 0.05
        n = max(1, int(round(self.cfg.ref_smooth_m / max(step, 1e-6))))
        if n > 1:
            kern = np.ones(n) / n
            pad = np.pad(kap, (n // 2, n - 1 - n // 2), mode="edge")
            kap = np.convolve(pad, kern, mode="valid")
        self.ref_s, self.ref_kappa = s, kap

    def _ground_limited_bars(self, j: int, r_cmd: float, f_cmd: float) -> tuple[float, float]:
        """Shorten the bars when the hitch is pitched so far that a bar end would touch the ground.

        The bars lie at about axle height h above the ground, so a bar of length L tilted by the hitch pitch theta drops
        L*sin|theta| at its far end. Keep that drop below h minus a margin (bar_ground_margin_m); never below the minimum
        length. Uses the IR pitch of this hitch (either sign: the end that drops depends on which trailer is higher).
        """
        if not self.cfg.bar_pitch_limit or j >= len(self.last_ir_readings):
            return r_cmd, f_cmd
        s = abs(float(np.sin(self.last_ir_readings[j][1])))
        if s < 1e-3:
            return r_cmd, f_cmd
        h = self.cfg.wheel_radius_m - self.cfg.wheel_z_offset_m - self.cfg.bar_ground_margin_m
        r_min, _ = self.cfg.rear_bar_range_m
        f_min, _ = self.cfg.front_bar_range_m
        return max(r_min, min(r_cmd, h / s)), max(f_min, min(f_cmd, h / s))

    def _use_ref(self) -> bool:
        return self.ref_kappa is not None and self.drive is None

    def _advance_reference(self) -> None:
        """Lead's progress s_ref along the ideal path: monotone and rate-limited, so a wiggling lead cannot jerk it."""
        if self.ref_s is None:
            return
        ds = max(self.dist_traveled - self._ref_last_dist, 0.0)
        self._ref_last_dist = self.dist_traveled
        s_target = min(self.s_proj, float(self.ref_s[-1]))
        self.s_ref = max(self.s_ref, min(s_target, self.s_ref + self.cfg.ref_progress_gain * ds + 1e-9))

    def _ref_curvature(self, delay_m: float) -> float:
        if self.ref_s is None:
            return 0.0
        s = self.s_ref - delay_m
        if s <= 0.0 or s >= float(self.ref_s[-1]):
            return 0.0                              # straight before the start / after the end
        return float(np.interp(s, self.ref_s, self.ref_kappa))

    def _record_dev(self, kappa: float) -> None:
        """Remember how far the lead's (shaped) curvature departs from the ideal path's, low-passed over distance.

        Followers add this back at their delay, so when the lead has to correct an off-path error (e.g. after cutting a
        corner) the chain bends the way the lead actually did, instead of the way the ideal path alone says.
        """
        if self.ref_s is None:
            return
        dev = float(kappa) - self._ref_curvature(0.0)
        ds = max(self.dist_traveled - self._dev_last_dist, 0.0)
        self._dev_last_dist = self.dist_traveled
        a = 1.0 - float(np.exp(-ds / max(self.cfg.ref_dev_filter_m, 1e-6)))
        self._dev_f += a * (dev - self._dev_f)
        lim = self.cfg.ref_dev_max
        self.dev_history.append((self.dist_traveled, float(np.clip(self._dev_f, -lim, lim))))

    def _lookup_dev(self, delay_m: float) -> float:
        target_d = self.dist_traveled - delay_m
        for d, v in reversed(self.dev_history):
            if d <= target_d:
                return v
        return 0.0

    def _follow_curvature(self, delay_m: float) -> float:
        if self.follow_override is not None:
            return float(self.follow_override)
        if not self._use_ref():
            return self._lookup_curvature(delay_m)
        k = self._ref_curvature(delay_m) + self.cfg.ref_dev_gain * self._lookup_dev(delay_m)
        return float(np.clip(k, -self.max_curvature, self.max_curvature))

    def _drive_cmd(self):
        return self.drive(self.sim_time) if self.drive is not None else None

    def reset(self) -> None:
        self.s_ref = 0.0
        self.s_proj = 0.0
        self.dev_history = deque(maxlen=PATH_HISTORY_MAX_SAMPLES)
        self._dev_f = 0.0
        self._dev_last_dist = 0.0
        self._ref_last_dist = 0.0
        self.dist_traveled = 0.0
        self.prev_pos = None
        self.path_history.clear()
        self.kappa_raw = 0.0
        self.kappa_shaped = 0.0
        self._shape_last_dist = 0.0
        self.last_lookahead_m = self.cfg.lookahead_dist_m
        self.last_lookahead = None
        self.last_joint_yaws = [0.0] * self.num_joints
        self.last_ir_readings = [(0.0, 0.0, 0.0)] * self.num_joints
        self.last_follow_states = []
        # Commanded tube yaw per hitch: (rear bar of trailer j, front bar of trailer j+1), radians.
        # Their sum plus the ball yaw equals the commanded trailer-to-trailer yaw in last_joint_yaws.
        self.last_bar_yaws = [(0.0, 0.0)] * self.num_joints
        self.last_bar_lengths = [
            (self.cfg.nominal_rear_bar_m, self.cfg.nominal_front_bar_m)
            for _ in range(self.num_joints)
        ]

    def _get_lookahead_point(self, pos_xy, lookahead_dist: float):
        A = self.waypoints[:-1]
        AB = self.waypoints[1:] - A
        ab_len_sq = np.einsum("ij,ij->i", AB, AB)
        t_proj = np.clip(np.einsum("ij,ij->i", pos_xy - A, AB) / (ab_len_sq + NUMERIC_EPS), 0.0, 1.0)
        proj_all = A + t_proj[:, None] * AB
        dists = np.linalg.norm(pos_xy - proj_all, axis=1)
        best_seg_idx = int(np.argmin(dists))
        best_proj = proj_all[best_seg_idx]
        if True:
            self.s_proj = float(self.path_s[best_seg_idx] + t_proj[best_seg_idx] * np.sqrt(ab_len_sq[best_seg_idx]))

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

    def _passed_end(self, pos_xy) -> bool:
        """True once the vehicle has progressed to the real end of the path (arc length, so a path that ends above its
        start, like the U-turn, is not mistaken for 'already there')."""
        if self.s_proj >= self.goal_s - 1e-6:
            return True
        return self.s_proj > 0.5 * self.goal_s and \
            float(np.linalg.norm(np.asarray(pos_xy) - self.goal_pt)) < self.cfg.goal_tol_m

    def _lookahead_for(self, v_cmd: float, requested: float | None = None) -> float:
        """Pure-pursuit lookahead grows with speed (time-based) so the steering loop stays stable despite actuator lag."""
        base = self.cfg.lookahead_dist_m if requested is None else requested
        self.last_lookahead_m = max(base, self.cfg.lookahead_gain_s * abs(v_cmd))
        return self.last_lookahead_m

    def _shape_curvature(self, kappa_raw: float) -> float:
        """Low-pass and slew-limit the pursuit curvature in the DISTANCE domain (so it does nothing when stopped).

        The bar-yaw commands are predicted from this curvature, so a twitchy value slams the hitches back and forth.
        """
        self.kappa_raw = float(kappa_raw)
        ds = max(self.dist_traveled - self._shape_last_dist, 0.0)
        self._shape_last_dist = self.dist_traveled
        c = self.cfg
        if ds <= 0.0:
            return self.kappa_shaped
        target = self.kappa_raw
        if c.curv_filter_m > 0.0:
            a = 1.0 - float(np.exp(-ds / c.curv_filter_m))
            target = self.kappa_shaped + a * (self.kappa_raw - self.kappa_shaped)
        if c.curv_rate_limit_per_m > 0.0:
            step = c.curv_rate_limit_per_m * ds
            target = self.kappa_shaped + float(np.clip(target - self.kappa_shaped, -step, step))
        self.kappa_shaped = float(target)
        return self.kappa_shaped

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
        return float(np.clip((2.0 * y_loc) / ld_sq, -self.max_curvature, self.max_curvature))

    @abstractmethod
    def step(
        self,
        pos_xy,
        lead_hdg_rad: float,
        v_center: float | None = None,
        lookahead_dist: float | None = None,
        ir_readings: Sequence[tuple[float, float, float]] | None = None,
        imu_readings=None,
    ) -> tuple[list[float], list[float], list[tuple[float, float]]]:
        """Returns (wheel_vels, joint_yaws, bar_lengths[(rear_m, front_m), ...]).

        ir_readings: one (yaw, pitch, roll) in radians per joint, as reported by the
        simulated IR camera (relative orientation of trailer j+1 in the frame of trailer j).
        """
        ...

    def _ir_wheel_vels(self, v_cmd: float, kappa: float, bar_lengths, kappa_follow: float | None = None,
                       bar_yaws=None) -> list[float]:
        """Wheel speeds for every trailer from the hitch kinematics and the MEASURED joint yaws.

        Planar model: trailer j has speed v_j and yaw rate w_j; its rear hitch point is L_r behind its axle, and the
        next trailer's front hitch point is L_f ahead of its axle. With psi = measured yaw of trailer j+1 relative to j
        (the IR camera yaw), matching the hitch point's velocity on both sides gives
            v_{j+1} = v_j cos(psi) - w_j L_r sin(psi)
            w_{j+1} = -(v_j sin(psi) + w_j L_r cos(psi)) / L_f
        The lead trailer follows the pure-pursuit curvature: w_0 = v_cmd * kappa. w is counter-clockwise positive, so in a
        left turn the RIGHT wheel is faster.

        With yaw tubes (bar_yaws given: commanded (rear, front) tube yaw per hitch) the hitch point is the BALL, which is
        off the trailer axis whenever a tube is yawed: in trailer j's frame it is b = (-p - s_r cos(psi_r), -s_r sin(psi_r)),
        in trailer j+1's frame a = (p + s_f cos(psi_f), -s_f sin(psi_f)), with p the tube pivot distance and s the
        pivot-to-ball reach. Matching the ball's velocity on both sides (and no lateral slip of trailer j+1's axle) gives
        the same recursion with those lever arms. With psi_r = psi_f = 0 it reduces exactly to the formulas above.
        """
        v, w = float(v_cmd), float(v_cmd) * float(kappa)
        states = [(v, w)]
        if kappa_follow is not None:
            w = float(v_cmd) * float(kappa_follow)       # followers chain from the idealised path, not the lead's wiggle
        w_lim = float(self.cfg.follower_max_yaw_rate)
        for j in range(self.num_joints):
            psi = self.last_ir_readings[j][0]
            l_r, l_f = bar_lengths[j]
            s, c = np.sin(psi), np.cos(psi)
            if bar_yaws is not None and self.cfg.wheel_use_tube_geometry:
                p = pivot_x_m(self.cfg)
                psi_r, psi_f = bar_yaws[j]
                s_r, s_f = l_r - p, l_f - p
                bx, by = -p - s_r * np.cos(psi_r), -s_r * np.sin(psi_r)          # ball in trailer j frame
                ax, ay = p + s_f * np.cos(psi_f), -s_f * np.sin(psi_f)           # ball in trailer j+1 frame
                vbx, vby = v - w * by, w * bx                                    # ball velocity, trailer j frame
                vbx2, vby2 = vbx * c + vby * s, -vbx * s + vby * c               # ... in trailer j+1 frame
                w_next = vby2 / max(ax, NUMERIC_EPS)
                v_next = vbx2 + w_next * ay
            else:
                v_next = v * c - w * l_r * s
                w_next = -(v * s + w * l_r * c) / max(l_f, NUMERIC_EPS)
            sgn = 1.0 if v_cmd >= 0.0 else -1.0                  # reversing: the followers must roll backwards too
            v = sgn * float(np.clip(sgn * v_next, self.cfg.follower_min_speed_frac * abs(v_cmd),
                                                  self.cfg.follower_max_speed_frac * abs(v_cmd)))
            w = float(np.clip(w_next, -w_lim, w_lim))
            states.append((v, w))
        self.last_follow_states = states
        half_track = 0.5 * self.cfg.track_width_m
        wheel_vels: list[float] = []
        for v_i, w_i in states:
            # w_i is CCW-positive (left turn). Both wheels spin about +y, so the RIGHT wheel is the faster one in a
            # left turn. Wheel order is (left, right).
            wheel_vels.extend([(v_i - w_i * half_track) / self.cfg.wheel_radius_m,
                               (v_i + w_i * half_track) / self.cfg.wheel_radius_m])
        return wheel_vels

    def _store_ir(self, ir_readings: Sequence[tuple[float, float, float]] | None) -> None:
        """Keep the latest IR camera readings so control laws can use them."""
        if ir_readings is not None:
            self.last_ir_readings = [tuple(float(v) for v in r) for r in ir_readings]

    def get_telemetry(self) -> dict[str, float]:
        """Return controller-specific telemetry channels for PlotJuggler/CSV."""
        metrics: dict[str, float] = {
            "lead/kappa_raw": round(self.kappa_raw, 4),
            "lead/kappa_cmd": round(self.kappa_shaped, 4),
            "lead/lookahead_m": round(self.last_lookahead_m, 3),
            "ref/s_m": round(self.s_ref, 3),
            "ref/kappa": round(self._ref_curvature(0.0), 4),
        }
        for idx, (v_i, w_i) in enumerate(self.last_follow_states, start=1):
            metrics[f"trailer{idx}/v_cmd_mps"] = round(float(v_i), 4)
            metrics[f"trailer{idx}/omega_cmd_rad_s"] = round(float(w_i), 4)
        return metrics


def create_controller(cfg: AGVConfig) -> BaseSnakeController:
    """Dynamically import <ClassName>.py and instantiate <ClassName>(cfg)."""
    cls_name = cfg.controller_class
    module = importlib.import_module(cls_name)
    controller_cls = getattr(module, cls_name)
    return controller_cls(cfg)
