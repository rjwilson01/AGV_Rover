"""Sign-convention self-check: short open-loop manoeuvres in the real MuJoCo model, with a PASS/FAIL verdict each.

Run:  python main.py VariableDrawbarController --selfcheck [--tests <files>]   (headless, ~1 minute)
Writes the verdicts and the time series into sim_log.csv (as '# ' lines plus a table) so `runtest` pushes them.

Conventions being verified (x forward, y left, z up, yaw counter-clockwise positive):
  1. wheels      : right wheel faster than left  -> vehicle turns LEFT (heading increases), gyro z > 0
  2. hinge       : positive yaw command on the bar hinges -> IR relative yaw (trailer j+1 in j) > 0, follower displaced to -y
  3. controller  : kappa > 0 (target on the left) -> controller wheel commands -> lead heading increases
  4. consistency : follower yaw in a left turn is negative, and the controller's hinge command for kappa > 0 is negative
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

    # 1. wheels: right faster -> left turn
    rig = _Rig(cfg)
    h0 = rig.lead_hdg()
    rig.step("1_wheels_right_faster", 4.0, hook=lambda t: rig.wheels(2.0 * _ramp(t), 3.5 * _ramp(t)))
    dh = float(np.degrees(np.unwrap([h0, rig.lead_hdg()])[1] - h0))
    gz = [r["gyro_z_dps"] for r in rig.rows if r["gyro_z_dps"] != ""]
    gmean = float(np.mean(gz[len(gz) // 2:])) if gz else float("nan")
    verdict("wheels: right wheel faster turns the vehicle LEFT", dh > 5.0 and (np.isnan(gmean) or gmean > 0),
            f"heading change {dh:+.1f} deg (want > 0), gyro z mean {gmean:+.1f} dps (want > 0)")
    rows += rig.rows

    # 2. hinge: positive yaw command -> positive IR yaw, follower displaced to -y
    if cfg.bar_yaw_control:
        rig = _Rig(cfg)
        rig.step("2_hinge_settle", 0.5)
        rig.set_hinges(0.2, 0.2)
        rig.step("2_hinge_plus", 3.0)
        yaw = float(np.degrees(rig.ir_yaw(0)))
        lat = rig.rows[-1]["t2_lateral_in_lead_frame_m"]
        verdict("hinge: +yaw command gives + IR relative yaw", yaw > 5.0, f"commanded +46 deg total (0.2+0.2 rad each way), IR yaw {yaw:+.1f} deg")
        verdict("hinge: +yaw swings the follower to the RIGHT (-y)", lat < -0.01, f"follower lateral offset in the lead frame {lat:+.3f} m (want < 0)")
        rows += rig.rows
        rig = _Rig(cfg)
        rig.step("2_hinge_settle", 0.5)
        rig.set_hinges(-0.2, -0.2)
        rig.step("2_hinge_minus", 3.0)
        yaw_m = float(np.degrees(rig.ir_yaw(0)))
        verdict("hinge: -yaw command gives - IR relative yaw", yaw_m < -5.0, f"IR yaw {yaw_m:+.1f} deg")
        rows += rig.rows
    else:
        verdicts.append("SKIP | hinge tests need bar_yaw_control (Variable build)")
        print(verdicts[-1])

    # 3. controller: kappa > 0 -> wheel commands -> lead turns left
    rig = _Rig(cfg)
    ctrl = create_controller(cfg)
    bars = [(cfg.rear_bar_range_m[1], cfg.front_bar_range_m[1])] * (rig.n - 1)
    ctrl.num_joints = rig.n - 1

    def drive(t):
        q = rig.quats()
        ctrl.last_ir_readings = [tuple(float(v) for v in relative_euler(q[j], q[j + 1])) for j in range(rig.n - 1)]
        wv = ctrl._ir_wheel_vels(1.0 * _ramp(t), 0.15, bars)
        for i in range(rig.n):
            rig.data.actuator(f"m_t{i + 1}_l").ctrl = wv[2 * i]
            rig.data.actuator(f"m_t{i + 1}_r").ctrl = wv[2 * i + 1]

    h0 = rig.lead_hdg()
    rig.step("3_controller_kappa_plus", 4.0, hook=drive)
    dh3 = float(np.degrees(np.unwrap([h0, rig.lead_hdg()])[1] - h0))
    verdict("controller: kappa > 0 (target on the left) turns the lead LEFT", dh3 > 3.0,
            f"heading change {dh3:+.1f} deg after 4 s at kappa 0.15 (ideal about +{np.degrees(0.15 * 1.0 * 3.0):.0f} deg incl. ramp)")
    rows += rig.rows

    # 4. controller hinge command sign for a left turn vs what the hinge physically does
    psi = cfg.yaw_gain * (np.arctan(cfg.rear_bar_range_m[1] * 0.15) + np.arctan(cfg.front_bar_range_m[1] * 0.15))
    cmd = cfg.joint_yaw_sign * psi
    verdict("consistency: controller hinge command for a left turn is NEGATIVE (follower yaw is negative in a left turn)", cmd < 0,
            f"yaw_gain {cfg.yaw_gain:+.2f} x joint_yaw_sign {cfg.joint_yaw_sign:+.0f} -> command {np.degrees(cmd):+.1f} deg for kappa +0.15")

    verdicts.append(f"RESULT | {'ALL CHECKS PASSED' if ok_all else 'SOME CHECKS FAILED'} | build_tag see header")
    print(verdicts[-1])

    fields = ["phase", "t", "lead_hdg_deg", "gyro_z_dps", "ir_yaw_deg", "t2_lateral_in_lead_frame_m"]
    with open(out_path, "w", newline="") as f:
        f.write("# selfcheck: sign conventions measured in MuJoCo\n")
        for v in verdicts:
            f.write(f"# {v}\n")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_path}")
    return ok_all
