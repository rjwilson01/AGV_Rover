"""Sign-convention self-check: short open-loop manoeuvres in the real MuJoCo model, with a PASS/FAIL verdict each.

Run:  python main.py VariableDrawbarController --selfcheck [--tests <files>]   (headless, ~1 minute)
Writes the verdicts and the time series into sim_log.csv (as '# ' lines plus a table) so `runtest` pushes them.

Conventions being verified (x forward, y left, z up, yaw counter-clockwise positive):
  1. hinge       : positive yaw command on the bar hinges -> IR relative yaw (trailer j+1 in j) > 0, follower displaced to -y
  2. chain       : controller wheel law + controller hinge command together: kappa > 0 (left) turns the lead left and
                   gives a NEGATIVE follower yaw; kappa < 0 mirrors it. (A locked straight chain cannot turn, so the
                   hinges are commanded to the matching steady-state pose.)
  3. consistency : the controller's hinge command for a left turn is negative
"""
import csv
from pathlib import Path

import numpy as np
import mujoco

from bar_geometry import trailer_cable_len
from BaseSnakeController import calc_cable_len, create_controller, quat_to_euler
from ir_camera import relative_euler
from xml_builder import generate_mujoco_xml

CORNERS = (("tl", 1.0, 1.0), ("tr", -1.0, 1.0), ("bl", 1.0, -1.0), ("br", -1.0, -1.0))


class _Rig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(generate_mujoco_xml(cfg))
        self.data = mujoco.MjData(self.model)
        self.n = cfg.num_trailers
        self.caps = [f"cap_t{j}_t{j + 1}_{c}" for j in range(1, self.n) for c, _, _ in CORNERS]
        self.t = 0.0
        self.rows = []
        self.set_hinges(0.0, 0.0)

    def set_hinges(self, psi_r: float, psi_f: float) -> None:
        cfg, d = self.cfg, self.data
        r_max, f_max = cfg.rear_bar_range_m[1], cfg.front_bar_range_m[1]
        for j in range(1, self.n):
            if cfg.bar_yaw_control:
                d.actuator(f"act_yaw_rear_{j}").ctrl = psi_r
                d.actuator(f"act_yaw_front_{j + 1}").ctrl = psi_f
                for k, (_, ys, zs) in enumerate(CORNERS):
                    d.actuator(self.caps[4 * (j - 1) + k]).ctrl = (
                        trailer_cable_len(psi_r, psi_f, (0.0, 0.0, 0.0), ys, zs, cfg, r_max, f_max) - cfg.cable_preload_offset_m)
            else:
                for k, (_, ys, zs) in enumerate(CORNERS):
                    d.actuator(self.caps[4 * (j - 1) + k]).ctrl = (
                        calc_cable_len(0.0, 0.0, 0.0, ys, zs, cfg) - cfg.cable_preload_offset_m)

    def wheels(self, left: float, right: float, followers=None) -> None:
        for i in range(1, self.n + 1):
            l, r = (left, right) if followers is None or i == 1 else followers
            self.data.actuator(f"m_t{i}_l").ctrl = l
            self.data.actuator(f"m_t{i}_r").ctrl = r

    def quats(self):
        mujoco.mj_kinematics(self.model, self.data)
        return [self.data.body(f"trailer_{i}").xquat.copy() for i in range(1, self.n + 1)]

    def lead_hdg(self) -> float:
        return quat_to_euler(self.data.joint("root_jnt").qpos[3:7])[0]

    def ir_yaw(self, j: int = 0) -> float:
        q = self.quats()
        return float(relative_euler(q[j], q[j + 1])[0])

    def step(self, phase: str, secs: float, hook=None, log_every=25) -> None:
        n = int(secs / self.model.opt.timestep)
        for k in range(n):
            if hook:
                hook(self.t)
            mujoco.mj_step(self.model, self.data)
            self.t += self.model.opt.timestep
            if k % log_every == 0:
                q = self.quats()
                p1, p2 = self.data.body("trailer_1").xpos, self.data.body("trailer_2").xpos
                h = self.lead_hdg()
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                self.rows.append({
                    "phase": phase, "t": round(self.t, 3),
                    "lead_hdg_deg": round(float(np.degrees(h)), 3),
                    "gyro_z_dps": round(float(np.degrees(self.data.sensor("t1_gyro").data[2])), 3) if self.cfg.imu_enabled else "",
                    "ir_yaw_deg": round(float(np.degrees(relative_euler(q[0], q[1])[0])), 3),
                    "t2_lateral_in_lead_frame_m": round(float(-np.sin(h) * dx + np.cos(h) * dy), 3),
                    "lead_x_m": round(float(self.data.joint("root_jnt").qpos[0]), 3),
                    "lead_y_m": round(float(self.data.joint("root_jnt").qpos[1]), 3),
                    "lead_speed_mps": round(float(np.hypot(*self.data.joint("root_jnt").qvel[:2])), 3),
                })


def _ramp(t, secs=1.0):
    return min(1.0, t / secs)


def run_selfcheck(cfg, out_path: Path) -> bool:
    verdicts, ok_all = [], True

    def verdict(name, passed, detail):
        nonlocal ok_all
        ok_all &= bool(passed)
        verdicts.append(f"{'PASS' if passed else 'FAIL'} | {name} | {detail}")
        print(verdicts[-1])

    rows = []

    # 1+2. hinge: positive yaw command -> positive IR yaw (peak response), follower displaced to -y.
    #      Wheels are held (zero speed command), so judge the PEAK in the first 1.5 s, not the end value.
    if cfg.bar_yaw_control:
        for sign in (+1.0, -1.0):
            rig = _Rig(cfg)
            rig.step("1_hinge_settle", 0.5)
            n0 = len(rig.rows)
            rig.set_hinges(0.2 * sign, 0.2 * sign)
            rig.step(f"1_hinge_{'plus' if sign > 0 else 'minus'}", 1.5)
            seg = rig.rows[n0:]
            yaws = np.array([r["ir_yaw_deg"] for r in seg])
            lats = np.array([r["t2_lateral_in_lead_frame_m"] for r in seg])
            peak_yaw = float(yaws[np.argmax(np.abs(yaws))])
            peak_lat = float(lats[np.argmax(np.abs(lats))])
            verdict(f"hinge: {'+' if sign > 0 else '-'}yaw command gives {'+' if sign > 0 else '-'} IR relative yaw",
                    peak_yaw * sign > 3.0, f"peak IR yaw {peak_yaw:+.1f} deg after a {sign * 46:+.0f} deg total command")
            verdict(f"hinge: {'+' if sign > 0 else '-'}yaw swings the follower to the {'RIGHT (-y)' if sign > 0 else 'LEFT (+y)'}",
                    peak_lat * sign < -0.02, f"peak follower lateral offset in the lead frame {peak_lat:+.3f} m")
            rows += rig.rows
    else:
        verdicts.append("SKIP | hinge tests need bar_yaw_control (Variable build)")
        print(verdicts[-1])

    # 3. whole chain, both turn directions: controller wheel law + controller hinge command (physical sign) together.
    #    Hinges are commanded to the steady-state pose for the curvature, so the chain is free to follow the circle.
    for kappa in (+0.15, -0.15):
        rig = _Rig(cfg)
        ctrl = create_controller(cfg)
        bars = [(cfg.rear_bar_range_m[1], cfg.front_bar_range_m[1])] * (rig.n - 1)
        psi_total = cfg.joint_yaw_sign * cfg.yaw_gain * (np.arctan(bars[0][0] * kappa) + np.arctan(bars[0][1] * kappa))
        rig.set_hinges(0.5 * psi_total, 0.5 * psi_total)

        def drive(t, ctrl=ctrl, rig=rig, bars=bars, kappa=kappa):
            q = rig.quats()
            ctrl.last_ir_readings = [tuple(float(v) for v in relative_euler(q[j], q[j + 1])) for j in range(rig.n - 1)]
            wv = ctrl._ir_wheel_vels(1.0 * _ramp(t), kappa, bars)
            for i in range(rig.n):
                rig.data.actuator(f"m_t{i + 1}_l").ctrl = wv[2 * i]
                rig.data.actuator(f"m_t{i + 1}_r").ctrl = wv[2 * i + 1]

        h0 = rig.lead_hdg()
        name = f"3_chain_kappa_{kappa:+.2f}"
        rig.step(name, 5.0, hook=drive)
        dh3 = float(np.degrees(np.unwrap([h0, rig.lead_hdg()])[1] - h0))
        last = [r for r in rig.rows if r["phase"] == name][-40:]
        ir_end = float(np.mean([r["ir_yaw_deg"] for r in last]))
        want = np.degrees(kappa * 1.0 * 4.5)
        verdict(f"chain: kappa {kappa:+.2f} turns the lead {'LEFT' if kappa > 0 else 'RIGHT'}",
                dh3 * kappa > 0 and abs(dh3) > 0.4 * abs(want),
                f"heading change {dh3:+.1f} deg in 5 s (ideal about {want:+.0f})")
        verdict(f"chain: kappa {kappa:+.2f} gives IR yaw {'NEGATIVE' if kappa > 0 else 'POSITIVE'} (follower yaw is negative in a left turn)",
                ir_end * kappa < 0 and abs(ir_end) > 8.0,
                f"IR yaw at the end {ir_end:+.1f} deg (hinge command {np.degrees(psi_total):+.1f} deg)")
        rows += rig.rows

    # 4. controller hinge command sign for a left turn vs what the hinge physically does
    psi = cfg.yaw_gain * (np.arctan(cfg.rear_bar_range_m[1] * 0.15) + np.arctan(cfg.front_bar_range_m[1] * 0.15))
    cmd = cfg.joint_yaw_sign * psi
    verdict("consistency: controller hinge command for a left turn is NEGATIVE (follower yaw is negative in a left turn)", cmd < 0,
            f"yaw_gain {cfg.yaw_gain:+.2f} x joint_yaw_sign {cfg.joint_yaw_sign:+.0f} -> command {np.degrees(cmd):+.1f} deg for kappa +0.15")

    verdicts.append(f"RESULT | {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'} | build_tag see header")
    print(verdicts[-1])

    fields = ["phase", "t", "lead_hdg_deg", "gyro_z_dps", "ir_yaw_deg", "t2_lateral_in_lead_frame_m", "lead_x_m", "lead_y_m", "lead_speed_mps"]
    with open(out_path, "w", newline="") as f:
        f.write("# selfcheck: sign conventions measured in MuJoCo\n")
        for v in verdicts:
            f.write(f"# {v}\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_path}")
    return ok_all
