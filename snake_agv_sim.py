from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

USE_IR_CAMERA_FEEDBACK = True

L_BAR = 0.35
W_Y = 0.24
H_Z = 0.165
L_SEG = 0.80
TRACK_W = 0.64
WHEEL_R = 0.18
PRELOAD_M = 0.002

SIGNS = {
    "tl": (+1.0, +1.0),
    "tr": (-1.0, +1.0),
    "bl": (+1.0, -1.0),
    "br": (-1.0, -1.0),
}


def calc_cable_len(yaw: float, pitch: float, roll: float, y_sign: float, z_sign: float) -> float:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    bx = L_BAR
    by = y_sign * W_Y
    bz = z_sign * H_Z

    wx = (cy * cp) * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    wy = (sy * cp) * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    wz = (-sp) * bx + (cp * sr) * by + (cp * cr) * bz

    return float(np.linalg.norm([wx + L_BAR, wy - by, wz - bz]))


def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw), float(pitch), float(roll)


def main() -> None:
    xml_path = Path(__file__).resolve().parent / "snake_agv_sim.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    neutral_len = 2.0 * L_BAR - PRELOAD_M
    for j_idx in (1, 2):
        for corner in SIGNS:
            data.actuator(f"cap{j_idx}_{corner}").ctrl = neutral_len

    path_history: list[tuple[float, float]] = []
    dist_traveled = 0.0
    last_print = time.time()
    interval_peak = 0.0
    peak_cable_name = "none"
    turn_complete = False

    def lookup_yaw(delay_m: float) -> float:
        target_d = dist_traveled - delay_m
        for d, y in reversed(path_history):
            if d <= target_d:
                return y
        return 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            _, lead_pitch, lead_hdg = quat_to_euler(data.joint("root_jnt").qpos[3:7])
            lead_hdg_deg = np.rad2deg(lead_hdg)

            v_center = 1.15
            if t < 5.0 or turn_complete:
                target_yaw = 0.0
            else:
                ramp = min(1.0, (t - 5.0) / 1.5)
                target_yaw = np.deg2rad(44.0) * ramp
                if abs(lead_hdg_deg) >= 176.0:
                    turn_complete = True
                    target_yaw = 0.0

            dist_traveled += v_center * model.opt.timestep
            path_history.append((dist_traveled, target_yaw))

            y1_cmd = lookup_yaw(0.40)
            y2_cmd = lookup_yaw(2.00)

            seg_yaws = [y1_cmd, 0.5 * (y1_cmd + y2_cmd), y2_cmd]
            for seg_num, psi in enumerate(seg_yaws, start=1):
                if abs(psi) < 1e-4:
                    wl = wr = v_center / WHEEL_R
                else:
                    radius = L_SEG / np.tan(psi / 2.0)
                    wl = (v_center * (1.0 - TRACK_W / (2.0 * radius))) / WHEEL_R
                    wr = (v_center * (1.0 + TRACK_W / (2.0 * radius))) / WHEEL_R
                data.actuator(f"m_t{seg_num}_l").ctrl = wl
                data.actuator(f"m_t{seg_num}_r").ctrl = wr

            y1_meas, p1_meas, r1_meas = quat_to_euler(data.joint("ball_joint_1").qpos)
            y2_meas, p2_meas, r2_meas = quat_to_euler(data.joint("ball_joint_2").qpos)

            p_feedback_1 = p1_meas if USE_IR_CAMERA_FEEDBACK else 0.0
            r_feedback_1 = r1_meas if USE_IR_CAMERA_FEEDBACK else 0.0
            p_feedback_2 = p2_meas if USE_IR_CAMERA_FEEDBACK else 0.0
            r_feedback_2 = r2_meas if USE_IR_CAMERA_FEEDBACK else 0.0

            pitch_level_assist = -0.35 * lead_pitch

            for corner, (ys, zs) in SIGNS.items():
                len_1 = calc_cable_len(-y1_cmd, p_feedback_1 + pitch_level_assist, r_feedback_1, ys, zs) - PRELOAD_M
                len_2 = calc_cable_len(-y2_cmd, p_feedback_2, r_feedback_2, ys, zs) - PRELOAD_M
                data.actuator(f"cap1_{corner}").ctrl = len_1
                data.actuator(f"cap2_{corner}").ctrl = len_2

            mujoco.mj_step(model, data)
            viewer.sync()

            for j_idx in (1, 2):
                for corner in SIGNS:
                    cap_name = f"cap{j_idx}_{corner}"
                    force_n = abs(float(data.actuator(cap_name).force[0]))
                    if force_n > interval_peak:
                        interval_peak = force_n
                        peak_cable_name = cap_name

            if time.time() - last_print > 0.6:
                print(
                    f"t={t:4.1f}s | Hdg={lead_hdg_deg:+6.1f}° | "
                    f"J1(cmd/meas)={np.rad2deg(y1_cmd):+5.1f}°/{np.rad2deg(-y1_meas):+5.1f}° | "
                    f"J2(cmd/meas)={np.rad2deg(y2_cmd):+5.1f}°/{np.rad2deg(-y2_meas):+5.1f}° | "
                    f"Peak={interval_peak:6.1f} N [{peak_cable_name}]"
                )
                interval_peak = 0.0
                peak_cable_name = "none"
                last_print = time.time()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
