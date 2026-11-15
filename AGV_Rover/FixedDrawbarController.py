import numpy as np
from BaseSnakeController import BaseSnakeController, STRAIGHT_LINE_YAW_EPS


class FixedDrawbarController(BaseSnakeController):
    """Controller for fixed-length drawbars with symmetric or asymmetric yaw limits."""

    def step(self, pos_xy, lead_hdg_rad: float, v_center=None, lookahead_dist=None):
        v_cmd = self.cfg.v_center_mps if v_center is None else v_center
        ld_cmd = self.cfg.lookahead_dist_m if lookahead_dist is None else lookahead_dist
        num_wheels = 2 * self.num_trailers

        if self.prev_pos is not None:
            self.dist_traveled += float(np.linalg.norm(pos_xy - self.prev_pos))
        self.prev_pos = pos_xy.copy()

        target_pt, at_goal = self._get_lookahead_point(pos_xy, ld_cmd)
        self.last_lookahead = target_pt.copy()

        if at_goal and np.linalg.norm(target_pt - pos_xy) < self.cfg.goal_tol_m:
            return [0.0] * num_wheels, list(self.last_joint_yaws), list(self.last_bar_lengths)

        kappa = self._compute_pure_pursuit_curvature(pos_xy, lead_hdg_rad, target_pt)
        self.path_history.append((self.dist_traveled, kappa))

        r_bar = self.cfg.nominal_rear_bar_m
        f_bar = self.cfg.nominal_front_bar_m
        seg_pitch = r_bar + f_bar

        joint_yaws = []
        for j in range(self.num_joints):
            delay_m = max(0.0, j * seg_pitch + r_bar - self.cfg.actuator_lead_offset_m)
            k_j = self._lookup_curvature(delay_m)
            raw_yaw = self.cfg.yaw_gain * k_j
            joint_yaws.append(float(np.clip(raw_yaw, self.yaw_min_rad, self.yaw_max_rad)))

        axle_yaws = [
            joint_yaws[0] if i == 0
            else joint_yaws[-1] if i == self.num_trailers - 1
            else 0.5 * (joint_yaws[i - 1] + joint_yaws[i])
            for i in range(self.num_trailers)
        ]

        wheel_vels = []
        eff_len = 0.5 * seg_pitch
        for psi in axle_yaws:
            if abs(psi) < STRAIGHT_LINE_YAW_EPS:
                wl = wr = v_cmd / self.cfg.wheel_radius_m
            else:
                turn_radius = eff_len / np.tan(psi / 2.0)
                wl = (v_cmd * (1.0 + self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
                wr = (v_cmd * (1.0 - self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
            wheel_vels.extend([wl, wr])

        self.last_joint_yaws = list(joint_yaws)
        return wheel_vels, joint_yaws, list(self.last_bar_lengths)
