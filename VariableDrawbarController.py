import numpy as np
from BaseSnakeController import BaseSnakeController, NUMERIC_EPS, STRAIGHT_LINE_YAW_EPS


class VariableDrawbarController(BaseSnakeController):
    """
    Telescoping front/rear drawbars [L_min, L_max] with asymmetric angle bounds.
    Shortens drawbars toward L_min in sharp curves so the train bends tighter
    without saturating asymmetric joint stops, and extends toward L_max on straights.
    """

    def step(self, pos_xy, lead_hdg_rad: float, v_center=None, lookahead_dist=None, ir_readings=None):
        self._store_ir(ir_readings)
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

        r_min, r_max = self.cfg.rear_bar_range_m
        f_min, f_max = self.cfg.front_bar_range_m

        joint_yaws = []
        bar_lengths = []
        bar_yaws = []
        cum_dist_m = 0.0

        for j in range(self.num_joints):
            delay_m = max(0.0, cum_dist_m + 0.5 * (r_min + r_max) - self.cfg.actuator_lead_offset_m)
            k_j = self._lookup_curvature(delay_m)

            psi_at_max = self.cfg.yaw_gain * (np.arctan(r_max * k_j) + np.arctan(f_max * k_j))
            yaw_limit = self.yaw_max_rad if psi_at_max >= 0.0 else abs(self.yaw_min_rad)

            if yaw_limit > STRAIGHT_LINE_YAW_EPS:
                tightness = float(np.clip(abs(psi_at_max) / yaw_limit, 0.0, 1.0))
            else:
                tightness = 0.0

            r_cmd = r_max - tightness * (r_max - r_min)
            f_cmd = f_max - tightness * (f_max - f_min)

            raw_yaw = self.cfg.yaw_gain * (np.arctan(r_cmd * k_j) + np.arctan(f_cmd * k_j))
            psi_cmd = float(np.clip(raw_yaw, self.yaw_min_rad, self.yaw_max_rad))

            # Split the joint yaw between the two bar tubes: each arctan term is that bar's own
            # steady-state angle. If the joint limit clipped the total, shrink both bars equally.
            # Each tube is then limited to its own range; whatever is left over (psi - rear - front)
            # is taken by the passive ball joint.
            yaw_r = self.cfg.yaw_gain * float(np.arctan(r_cmd * k_j))
            yaw_f = self.cfg.yaw_gain * float(np.arctan(f_cmd * k_j))
            if abs(raw_yaw) > NUMERIC_EPS and abs(raw_yaw) > abs(psi_cmd):
                scale = psi_cmd / raw_yaw
                yaw_r *= scale
                yaw_f *= scale
            bar_yaws.append((
                float(np.clip(yaw_r, self.rear_yaw_min_rad, self.rear_yaw_max_rad)),
                float(np.clip(yaw_f, self.front_yaw_min_rad, self.front_yaw_max_rad)),
            ))

            joint_yaws.append(psi_cmd)
            bar_lengths.append((r_cmd, f_cmd))
            cum_dist_m += (r_cmd + f_cmd)

        wheel_vels = []
        for i in range(self.num_trailers):
            if i == 0:
                psi = joint_yaws[0]
                eff_len = bar_lengths[0][0]
            elif i == self.num_trailers - 1:
                psi = joint_yaws[-1]
                eff_len = bar_lengths[-1][1]
            else:
                psi = 0.5 * (joint_yaws[i - 1] + joint_yaws[i])
                eff_len = 0.5 * (bar_lengths[i - 1][1] + bar_lengths[i][0])

            if abs(psi) < STRAIGHT_LINE_YAW_EPS:
                wl = wr = v_cmd / self.cfg.wheel_radius_m
            else:
                turn_radius = eff_len / np.tan(psi / 2.0)
                wl = (v_cmd * (1.0 + self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
                wr = (v_cmd * (1.0 - self.cfg.track_width_m / (2.0 * turn_radius))) / self.cfg.wheel_radius_m
            wheel_vels.extend([wl, wr])

        self.last_joint_yaws = list(joint_yaws)
        self.last_bar_lengths = list(bar_lengths)
        self.last_bar_yaws = list(bar_yaws)
        return wheel_vels, joint_yaws, bar_lengths
    
    def get_telemetry(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for idx, (r_cmd, f_cmd) in enumerate(self.last_bar_lengths, start=1):
            metrics[f"joint{idx}/rear_bar_cmd_m"] = round(float(r_cmd), 4)
            metrics[f"joint{idx}/front_bar_cmd_m"] = round(float(f_cmd), 4)
        return metrics
