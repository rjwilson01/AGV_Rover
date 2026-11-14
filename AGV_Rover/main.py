import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

from kinematics import SnakePathController, calc_cable_len, quat_to_euler
from telemetry import TelemetryManager

USE_IR_CAMERA_FEEDBACK = True
SAVE_CSV_LOG = False  # Toggle CSV logging on/off here

CAPSTAN_NAMES = [
    "cap_t1_t2_tl", "cap_t1_t2_tr", "cap_t1_t2_bl", "cap_t1_t2_br",
    "cap_t2_t3_tl", "cap_t2_t3_tr", "cap_t2_t3_bl", "cap_t2_t3_br",
]
TENDON_NAMES = [
    "t1_t2_tl", "t1_t2_tr", "t1_t2_bl", "t1_t2_br",
    "t2_t3_tl", "t2_t3_tr", "t2_t3_bl", "t2_t3_br",
]
WHEEL_NAMES = ["m_t1_l", "m_t1_r", "m_t2_l", "m_t2_r", "m_t3_l", "m_t3_r"]
SIGNS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

# Example paddock S-curve waypoints (x, y in meters)
WAYPOINTS = [
    [0.0, 0.0], [2.5, 0.0], [5.0, 1.5], [8.0, 1.5],
    [11.0, -1.5], [14.0, -1.5], [17.0, 0.0], [22.0, 0.0],
]


def main():
    base_dir = Path(__file__).resolve().parent
    model = mujoco.MjModel.from_xml_path(str(base_dir / "snake_agv_sim.xml"))
    data = mujoco.MjData(model)

    controller = SnakePathController(WAYPOINTS)
    telemetry = TelemetryManager(base_dir, udp_host="127.0.0.1", udp_port=9870, enable_csv=SAVE_CSV_LOG)

    for cname in CAPSTAN_NAMES:
        data.actuator(cname).ctrl = 0.698

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = float(data.time)

            root_qpos = data.joint("root_jnt").qpos
            pos_xy = root_qpos[:2]
            _, _, lead_hdg = quat_to_euler(root_qpos[3:7])

            wheel_vels, y1, y2 = controller.step(pos_xy, lead_hdg, v_center=1.15)
            for w_name, w_vel in zip(WHEEL_NAMES, wheel_vels):
                data.actuator(w_name).ctrl = w_vel

            j1_euler = quat_to_euler(data.joint("ball_joint_1").qpos)
            j2_euler = quat_to_euler(data.joint("ball_joint_2").qpos)
            p1_ctrl, r1_ctrl = (j1_euler[1], j1_euler[2]) if USE_IR_CAMERA_FEEDBACK else (0.0, 0.0)
            p2_ctrl, r2_ctrl = (j2_euler[1], j2_euler[2]) if USE_IR_CAMERA_FEEDBACK else (0.0, 0.0)

            for k, (ys, zs) in enumerate(SIGNS):
                data.actuator(CAPSTAN_NAMES[k]).ctrl = calc_cable_len(-y1, p1_ctrl, r1_ctrl, ys, zs) - 0.002
                data.actuator(CAPSTAN_NAMES[4 + k]).ctrl = calc_cable_len(-y2, p2_ctrl, r2_ctrl, ys, zs) - 0.002

            mujoco.mj_step(model, data)
            viewer.sync()

            forces = {
                t_name: abs(float(data.actuator(c_name).force[0]))
                for c_name, t_name in zip(CAPSTAN_NAMES, TENDON_NAMES)
            }
            telemetry.update(t, pos_xy, np.rad2deg(lead_hdg), y1, y2, j1_euler, j2_euler, forces)

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    telemetry.close()


if __name__ == "__main__":
    main()
