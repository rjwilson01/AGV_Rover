import numpy as np
from collections import deque

def calc_cable_len(yaw, pitch, roll, y_sign, z_sign, L_bar=0.35, W_y=0.24, H_z=0.165):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy * cp) * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    wy = (sy * cp) * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    wz = (-sp) * bx + (cp * sr) * by + (cp * cr) * bz
    return float(np.linalg.norm([wx + L_bar, wy - by, wz - bz]))

def quat_to_euler(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(yaw), float(pitch), float(roll)


class SnakePathController:
    def __init__(self, waypoints, l_seg=0.80, track_w=0.64, wheel_r=0.18):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.L_SEG = l_seg
        self.TRACK_W = track_w
        self.WHEEL_R = wheel_r
        self.dist_traveled = 0.0
        self.prev_pos = None
        self.path_history = deque(maxlen=4000)
        self.last_lookahead = None



    def _get_lookahead_point(self, pos_xy, lookahead_dist):
        # 1. Find closest point on any waypoint segment
        best_dist = float("inf")
        best_seg_idx = 0
        best_proj = self.waypoints[0]

        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))
            t = np.clip(np.dot(pos_xy - a, ab) / (ab_len_sq + 1e-9), 0.0, 1.0)
            proj = a + t * ab
            d = float(np.linalg.norm(pos_xy - proj))
            if d < best_dist:
                best_dist = d
                best_seg_idx = i
                best_proj = proj

        # 2. Walk forward along the path from best_proj by lookahead_dist
        rem = lookahead_dist
        curr_pt = best_proj
        idx = best_seg_idx
        while idx < len(self.waypoints) - 1:
            seg_end = self.waypoints[idx + 1]
            seg_vec = seg_end - curr_pt
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len >= rem:
                return curr_pt + (seg_vec / (seg_len + 1e-9)) * rem, False
            rem -= seg_len
            curr_pt = seg_end
            idx += 1

        return self.waypoints[-1].copy(), True

    def _lookup_yaw(self, delay_m):
        target_d = self.dist_traveled - delay_m
        for d, y in reversed(self.path_history):
            if d <= target_d:
                return y
        return 0.0

    def step(self, pos_xy, lead_hdg_rad, v_center=1.15, lookahead_dist=1.4):
        if self.prev_pos is not None:
            self.dist_traveled += float(np.linalg.norm(pos_xy - self.prev_pos))
        self.prev_pos = pos_xy.copy()

        target_pt, at_goal = self._get_lookahead_point(pos_xy, lookahead_dist)
        self.last_lookahead = target_pt.copy()
        if at_goal and np.linalg.norm(target_pt - pos_xy) < 0.25:
            return [0.0] * 6, 0.0, 0.0

        # Transform target into lead AGV local coordinates
        dx, dy = target_pt - pos_xy
        c, s = np.cos(-lead_hdg_rad), np.sin(-lead_hdg_rad)
        x_loc = c * dx - s * dy
        y_loc = s * dx + c * dy

        # Pure pursuit curvature: y_loc > 0 means target is to the LEFT (+Y).
        # Because trailers trail behind (-X), turning LEFT requires negative joint yaw (y1 < 0)
        # and slower left wheels (wl < wr).
        # y_loc > 0 (target to the LEFT) -> kappa > 0 -> y1 > 0 -> wr > wl (turns LEFT!)
        # y_loc > 0 means target is to the LEFT (+Y).
        # In our capstan geometry, y1 < 0 pulls LEFT cables (turns LEFT / +Y).
        Ld_sq = max(x_loc**2 + y_loc**2, 0.1)
        kappa = float(np.clip((2.0 * y_loc) / Ld_sq, -0.42, 0.42))
        target_yaw = float(np.clip(-1.60 * kappa, -np.deg2rad(38.0), np.deg2rad(38.0)))

        self.path_history.append((self.dist_traveled, target_yaw))
        y1 = self._lookup_yaw(0.40)
        y2 = self._lookup_yaw(2.00)

        # Differential wheel speeds matching capstan turn direction:
        # When psi < 0 (turning LEFT / +Y), R < 0 -> wl < wr (left wheel slower, right wheel faster!)
        wheel_vels = []
        for psi in (y1, 0.5 * (y1 + y2), y2):
            if abs(psi) < 1e-4:
                wl = wr = v_center / self.WHEEL_R
            else:
                R = self.L_SEG / np.tan(psi / 2.0)
                wl = (v_center * (1.0 + self.TRACK_W / (2.0 * R))) / self.WHEEL_R
                wr = (v_center * (1.0 - self.TRACK_W / (2.0 * R))) / self.WHEEL_R
            wheel_vels.extend([wl, wr])
        return wheel_vels, y1, y2
