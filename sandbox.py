"""Physics + the REAL controller, driven by manual inputs, for --view.

Each frame:
  1. the IR camera model reads the true trailer orientations (as in a normal run);
  2. the real controller step() runs with an open-loop lead (speed and curvature are your set-points) and, optionally,
     a fixed curvature that every follower sees (the path ahead);
  3. per algorithm switch (see manual.py) the controller's output OR your manual set-point is written to the actuators;
  4. the hitch cables are sized exactly as in main.py.
So what you see here is what main.py would command from the same inputs. (main.py's actuator code is mirrored here, not
shared: if you change one, change the other.)
"""
from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from bar_geometry import ball_ypr_from_trailer_relative, trailer_cable_len
from BaseSnakeController import calc_cable_len, create_controller, quat_to_euler
from ir_camera import IRCameraModel, relative_euler
from manual import ManualControls

CORNER_SIGNS = (("tl", 1.0, 1.0), ("tr", -1.0, 1.0), ("bl", 1.0, -1.0), ("br", -1.0, -1.0))   # same as main.py


class Sandbox:
    def __init__(self, cfg, model, data):
        self.cfg = cfg
        self.model, self.data = model, data
        self.ctl = ManualControls(cfg)
        self.base_cfg = cfg
        self.cfg = replace(cfg, **self.ctl.cfg_overrides())
        self.controller = create_controller(self.cfg)
        self.controller.drive = self._drive
        self.ir = IRCameraModel(self.cfg)
        self.actuators = {model.actuator(i).name for i in range(model.nu)}
        self.joints = {model.joint(i).name for i in range(model.njnt)}
        self.sim_time = 0.0
        self.states = []
        self._last_flags = dict(self.ctl.flags)
        self.baseline = [None] * max(cfg.num_trailers - 1, 0)      # resting cable sum per hitch (tared at 0.5 s or by the button)
        self._cable_sum = [0.0] * max(cfg.num_trailers - 1, 0)

    # open-loop lead: speed and curvature from the set-points
    def _drive(self, t):
        return self.ctl.get("lead_speed"), self.ctl.get("lead_kappa")

    def _sync_cfg(self) -> None:
        if self.ctl.flags != self._last_flags:
            self._last_flags = dict(self.ctl.flags)
            self.cfg = replace(self.base_cfg, **self.ctl.cfg_overrides())
            self.controller.cfg = self.cfg
            self.ir = IRCameraModel(self.cfg)

    def apply_cfg(self, cfg) -> None:
        """Live parameter edit: swap in a new config without rebuilding the model (the controller restarts; the physics goes on).
        Parameters that change the model itself need a reload (configsession.py keeps those at their built values)."""
        from manual import FLAGS
        self.base_cfg = cfg
        for name, _, _, field in FLAGS:                       # a switch that mirrors a config field follows the config
            if field is not None and name != "ball_hold_ir" and hasattr(cfg, field):
                self.ctl.flags[name] = bool(getattr(cfg, field))
        self.cfg = replace(cfg, **self.ctl.cfg_overrides())
        self.controller = create_controller(self.cfg)
        self.controller.drive = self._drive
        self.ir = IRCameraModel(self.cfg)
        self._last_flags = dict(self.ctl.flags)

    # -- one frame ------------------------------------------------------------------------------------------------
    def step(self) -> None:
        self._sync_cfg()
        cfg, data, model, ctl = self.cfg, self.data, self.model, self.ctl
        t = float(data.time)
        mujoco.mj_kinematics(model, data)
        quats = [data.body(f"trailer_{i}").xquat for i in range(1, cfg.num_trailers + 1)]
        ir = self.ir.measure(t, quats)
        self._last_ir = ir
        pos_xy = np.array(data.body("trailer_1").xpos[:2], dtype=float)
        hdg, _, _ = quat_to_euler(quats[0])
        self.controller.sim_time = t
        self.controller.follow_override = ctl.get("follow_kappa")
        wheel_vels, joint_yaws, bar_lengths = self.controller.step(pos_xy, hdg, ir_readings=ir)
        sgn = cfg.joint_yaw_sign
        joint_yaws = [sgn * y for y in joint_yaws]
        bar_yaws = [(sgn * a, sgn * b) for a, b in self.controller.last_bar_yaws]
        self.states = list(self.controller.last_follow_states)

        # --- wheels
        if ctl.flags["wheel_law"]:
            for name, w in zip(self._wheel_names(), wheel_vels):
                data.actuator(name).ctrl = w
        else:
            v = ctl.get("wheel_speed") / cfg.wheel_radius_m
            for name in self._wheel_names():
                data.actuator(name).ctrl = v

        # --- hitches
        rad = np.deg2rad
        _, r_max = cfg.rear_bar_range_m
        _, f_max = cfg.front_bar_range_m
        for j in range(1, cfg.num_trailers):
            k = j - 1
            r_cmd, f_cmd = bar_lengths[k] if ctl.flags["bar_len_law"] else (ctl.get(f"h{j}_rear"), ctl.get(f"h{j}_front"))
            if ctl.flags["bar_yaw_law"]:
                y_cmd, (psi_r, psi_f) = joint_yaws[k], bar_yaws[k]
            else:
                y_cmd = rad(ctl.get(f"h{j}_yaw"))
                psi_r = float(np.clip(y_cmd / 2.0, rad(cfg.rear_bar_yaw_range_deg[0]), rad(cfg.rear_bar_yaw_range_deg[1])))
                psi_f = float(np.clip(y_cmd / 2.0, rad(cfg.front_bar_yaw_range_deg[0]), rad(cfg.front_bar_yaw_range_deg[1])))
            rear_act, front_act = f"act_slide_rear_{j}", f"act_slide_front_{j + 1}"
            if rear_act in self.actuators:
                data.actuator(rear_act).ctrl = r_max - r_cmd
            if front_act in self.actuators:
                data.actuator(front_act).ctrl = f_max - f_cmd
            if self.ctl.bars:
                psi_r_m = float(data.joint(f"yaw_rear_{j}").qpos.item())
                psi_f_m = float(data.joint(f"yaw_front_{j + 1}").qpos.item())
                psi_r_c, psi_f_c = (psi_r_m, psi_f_m) if cfg.cable_use_measured_tubes else (psi_r, psi_f)
                r_c, f_c = r_cmd, f_cmd
                if cfg.cable_use_measured_tubes:
                    if rear_act in self.actuators and f"slide_rear_{j}" in self.joints:
                        r_c = r_max - float(data.joint(f"slide_rear_{j}").qpos.item())
                    if front_act in self.actuators and f"slide_front_{j + 1}" in self.joints:
                        f_c = f_max - float(data.joint(f"slide_front_{j + 1}").qpos.item())
                y_ball = y_cmd - psi_r_c - psi_f_c
                if ctl.flags["ball_hold_ir"]:
                    _, p_ball, r_ball = ball_ypr_from_trailer_relative((ir[k].yaw, ir[k].pitch, ir[k].roll), psi_r_m, psi_f_m)
                else:
                    p_ball, r_ball = rad(ctl.get(f"h{j}_pitch")), rad(ctl.get(f"h{j}_roll"))
                data.actuator(f"act_yaw_rear_{j}").ctrl = psi_r
                data.actuator(f"act_yaw_front_{j + 1}").ctrl = psi_f
                for code, ys, zs in CORNER_SIGNS:
                    data.actuator(f"cap_t{j}_t{j + 1}_{code}").ctrl = (
                        trailer_cable_len(psi_r_c, psi_f_c, (y_ball, p_ball, r_ball), ys, zs, cfg, r_c, f_c)
                        - cfg.cable_preload_offset_m)
            else:
                p_ball, r_ball = rad(ctl.get(f"h{j}_pitch")), rad(ctl.get(f"h{j}_roll"))
                for code, ys, zs in CORNER_SIGNS:
                    data.actuator(f"cap_t{j}_t{j + 1}_{code}").ctrl = (
                        calc_cable_len(-y_cmd, -p_ball, r_ball, ys, zs, cfg, rear_bar_m=r_cmd, front_bar_m=f_cmd)
                        - cfg.cable_preload_offset_m)
        ctl.readout = self.readout_text(ir, bar_lengths, bar_yaws, joint_yaws)

    # -- live force diagram data ---------------------------------------------------------------------------------
    def force_snapshot(self, j: int) -> dict | None:
        """Forces on hitch j (1-based) from the simulator state: cable tensions, bar axial force, wheel thrust."""
        from forceview import make_snapshot
        cfg, data, model = self.cfg, self.data, self.model
        if not (1 <= j < cfg.num_trailers):
            return None
        cable = []
        for code, _, _ in CORNER_SIGNS:
            cable.append(abs(float(data.actuator(f"cap_t{j}_t{j + 1}_{code}").force.item())))
        bar = None
        if cfg.bar_yaw_control:
            try:
                mujoco.mj_rnePostConstraint(model, data)
                b_id = model.body(f"t{j}_rear_slider").id
                axis = np.asarray(data.xmat[b_id]).reshape(3, 3)[:, 0]
                bar = cfg.bar_force_sign * float(np.dot(np.asarray(data.cfrc_int[b_id])[3:], axis))
            except Exception:
                bar = None
        r = cfg.wheel_radius_m
        thrust = lambda i: (float(data.actuator(f"m_t{i}_l").force.item()) + float(data.actuator(f"m_t{i}_r").force.item())) / r
        s = sum(cable)
        self._cable_sum[j - 1] = s
        if self.baseline[j - 1] is None and float(data.time) >= cfg.retension_baseline_s:
            self.baseline[j - 1] = s
        yaw = 0.0
        if self.states and hasattr(self, "_last_ir"):
            yaw = float(self._last_ir[j - 1].yaw)
        return make_snapshot(j, cable, bar, thrust(j), thrust(j + 1), self.baseline[j - 1], cfg.retension_thresh_n, yaw)

    def tare(self) -> None:
        """Take the current cable sums as the resting values (do this with the hitch relaxed)."""
        self.baseline = list(self._cable_sum)

    def _wheel_names(self):
        return [f"m_t{i}_{s}" for i in range(1, self.cfg.num_trailers + 1) for s in ("l", "r")]

    # -- live commanded-vs-measured table -------------------------------------------------------------------------
    def readout_text(self, ir, bar_lengths, bar_yaws, joint_yaws) -> str:
        cfg, data = self.cfg, self.data
        _, r_max = cfg.rear_bar_range_m
        _, f_max = cfg.front_bar_range_m
        deg = np.rad2deg
        lines = []
        for j in range(1, cfg.num_trailers):
            k = j - 1
            r_m = r_max - float(data.joint(f"slide_rear_{j}").qpos.item()) if f"slide_rear_{j}" in self.joints else r_max
            f_m = f_max - float(data.joint(f"slide_front_{j + 1}").qpos.item()) if f"slide_front_{j + 1}" in self.joints else f_max
            lines.append(f"HITCH {j}   IR yaw {deg(ir[k].yaw):+6.1f} deg  pitch {deg(ir[k].pitch):+5.1f}  roll {deg(ir[k].roll):+5.1f}")
            lines.append(f"  bars   rear  cmd {bar_lengths[k][0]:5.3f} / meas {r_m:5.3f} m    front cmd {bar_lengths[k][1]:5.3f} / meas {f_m:5.3f} m")
            if self.ctl.bars:
                tr = float(data.joint(f"yaw_rear_{j}").qpos.item())
                tf = float(data.joint(f"yaw_front_{j + 1}").qpos.item())
                lines.append(f"  tubes  rear  cmd {deg(bar_yaws[k][0]):+6.1f} / meas {deg(tr):+6.1f} deg   front cmd {deg(bar_yaws[k][1]):+6.1f} / meas {deg(tf):+6.1f} deg")
            tens = []
            for code, _, _ in CORNER_SIGNS:
                nm = f"cap_t{j}_t{j + 1}_{code}"
                try:
                    tens.append(f"{code} {float(data.actuator(nm).ctrl.item()):.3f}/{float(data.tendon(f't{j}_t{j + 1}_{code}').length.item()):.3f}")
                except Exception:
                    tens.append(f"{code} {float(data.actuator(nm).ctrl.item()):.3f}/-")
            lines.append("  cables (cmd/actual m): " + "  ".join(tens))
            if self.ctl.bars:
                # Does the cable-length formula match the model? Evaluate it at the MEASURED pose (tube yaws, bar lengths, ball
                # angles from the IR camera) and compare with the real tendon length. ~0 mm means the formula includes the angles
                # correctly; a systematic error means the formula and the model disagree.
                try:
                    tr_m = float(data.joint(f"yaw_rear_{j}").qpos.item())
                    tf_m = float(data.joint(f"yaw_front_{j + 1}").qpos.item())
                    ball = ball_ypr_from_trailer_relative((ir[k].yaw, ir[k].pitch, ir[k].roll), tr_m, tf_m)
                    errs = []
                    for code, ys, zs in CORNER_SIGNS:
                        f_len = trailer_cable_len(tr_m, tf_m, ball, ys, zs, cfg, r_m, f_m)
                        a_len = float(data.tendon(f"t{j}_t{j + 1}_{code}").length.item())
                        errs.append(f"{code} {1000.0 * (f_len - a_len):+.1f}")
                    lines.append("  formula check (formula - actual tendon, mm, at the measured pose): " + "  ".join(errs))
                except Exception:
                    pass
        lines.append("WHEELS (cmd rad/s)  L / R")
        for i in range(1, cfg.num_trailers + 1):
            l = float(data.actuator(f"m_t{i}_l").ctrl.item())
            r = float(data.actuator(f"m_t{i}_r").ctrl.item())
            st = f"   law: v {self.states[i - 1][0]:+.3f} m/s  w {self.states[i - 1][1]:+.3f} rad/s" if i - 1 < len(self.states) else ""
            lines.append(f"  trailer {i}: {l:+7.2f} / {r:+7.2f}{st}")
        return "\n".join(lines)
