from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

from config import load_config
from kinematics import SnakePathController, calc_cable_len, quat_to_euler
from telemetry import TelemetryManager

CORNER_SIGNS = (
    ("tl",  1.0,  1.0),
    ("tr", -1.0,  1.0),
    ("bl",  1.0, -1.0),
    ("br", -1.0, -1.0),
)
STOP_WHEEL_VEL_EPS = 1e-6
WHEEL_NAMES = ["m_t1_l", "m_t1_r", "m_t2_l", "m_t2_r", "m_t3_l", "m_t3_r"]
SIGNS = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

# Example paddock S-curve waypoints (x, y in meters)
WAYPOINTS = [
    [0.0, 0.0], [2.5, 0.0], [5.0, 1.5], [8.0, 1.5],
    [11.0, -1.5], [14.0, -1.5], [17.0, 0.0], [22.0, 0.0],
]

def render_path_overlay(viewer, waypoints, lookahead_pt=None):
    scn = viewer.user_scn
    scn.ngeom = 0  # Clear previous frame's custom markers
    identity_mat = np.eye(3).flatten()

    # 1. Draw Waypoint Spheres (Yellow) & Connecting Path Capsules (Cyan)
    for i, wp in enumerate(waypoints):
        if scn.ngeom >= scn.maxgeom - 2:
            break
        # Waypoint sphere
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.08, 0, 0],
            pos=[wp[0], wp[1], 0.05],
            mat=identity_mat,
            rgba=[1.0, 0.85, 0.1, 0.9],
        )
        scn.ngeom += 1

        # Segment line to next waypoint
        if i < len(waypoints) - 1:
            nxt = waypoints[i + 1]
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[0.015, 0, 0],
                pos=[0, 0, 0],
                mat=identity_mat,
                rgba=[0.1, 0.8, 1.0, 0.7],
            )
            mujoco.mjv_connector(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=0.015,
                from_=[wp[0], wp[1], 0.04],
                to=[nxt[0], nxt[1], 0.04],
            )
            scn.ngeom += 1

    # 2. Draw Active Pure-Pursuit Lookahead Target (Bright Red/Orange Sphere)
    if lookahead_pt is not None and scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.11, 0, 0],
            pos=[lookahead_pt[0], lookahead_pt[1], 0.15],
            mat=identity_mat,
            rgba=[1.0, 0.2, 0.1, 0.95],
        )
        scn.ngeom += 1


def main():
    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir / "agv_config.toml")

    model = mujoco.MjModel.from_xml_path(str(base_dir / cfg.xml_filename))
    data = mujoco.MjData(model)

    lead_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trailer_1")
    controller = SnakePathController(cfg)
    telemetry = TelemetryManager(
        base_dir,
        udp_host=cfg.udp_host,
        udp_port=cfg.udp_port,
        enable_csv=cfg.save_csv_log,
    )

    # Build actuator & tendon name lists dynamically from num_trailers
    wheel_names = [
        f"m_t{i}_{side}"
        for i in range(1, cfg.num_trailers + 1)
        for side in ("l", "r")
    ]
    joint_names = [f"ball_joint_{j}" for j in range(1, cfg.num_trailers)]
    tendon_names = [
        f"t{j}_t{j + 1}_{code}"
        for j in range(1, cfg.num_trailers)
        for code, _, _ in CORNER_SIGNS
    ]
    capstan_names = [f"cap_{t_name}" for t_name in tendon_names]

    def apply_initial_capstan_preload():
        for cname in capstan_names:
            data.actuator(cname).ctrl = cfg.initial_capstan_ctrl_m

    apply_initial_capstan_preload()
    prev_sim_time = 0.0
    post_stop_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = lead_body_id
        viewer.cam.distance = cfg.camera_distance_m
        viewer.cam.azimuth = cfg.camera_azimuth_deg
        viewer.cam.elevation = cfg.camera_elevation_deg

        while viewer.is_running():
            # Handle MuJoCo viewer Reset button (data.time jumps back to 0)
            if data.time < prev_sim_time:
                post_stop_time = 0.0
                controller.reset()
                apply_initial_capstan_preload()
            prev_sim_time = data.time

            step_start = time.time()
            t = float(data.time)

            root_qpos = data.joint("root_jnt").qpos
            pos_xy = root_qpos[:2]
            lead_hdg, _, _ = quat_to_euler(root_qpos[3:7])

            # Single controller step per physics step
            wheel_vels, joint_yaws = controller.step(pos_xy, lead_hdg)
            for w_name, w_vel in zip(wheel_names, wheel_vels):
                data.actuator(w_name).ctrl = w_vel

            joint_eulers = [quat_to_euler(data.joint(j_name).qpos) for j_name in joint_names]

            for j_idx, (y_cmd, j_euler) in enumerate(zip(joint_yaws, joint_eulers)):
                p_ctrl, r_ctrl = (j_euler[1], j_euler[2]) if cfg.use_ir_camera_feedback else (0.0, 0.0)
                for k, (_, ys, zs) in enumerate(CORNER_SIGNS):
                    c_idx = 4 * j_idx + k
                    data.actuator(capstan_names[c_idx]).ctrl = (
                        calc_cable_len(-y_cmd, p_ctrl, r_ctrl, ys, zs, cfg) - cfg.cable_preload_offset_m
                    )

            mujoco.mj_step(model, data)

            with viewer.lock():
                render_path_overlay(viewer, cfg.waypoints, controller.last_lookahead)
            viewer.sync()

            forces = {
                t_name: abs(float(data.actuator(c_name).force[0]))
                for c_name, t_name in zip(capstan_names, tendon_names)
            }

            is_stopped = all(abs(v) < STOP_WHEEL_VEL_EPS for v in wheel_vels)
            if is_stopped:
                post_stop_time += model.opt.timestep
            else:
                post_stop_time = 0.0

            if post_stop_time <= cfg.post_stop_log_sec:
                # Passes joint1 and joint2 telemetry for 3-trailer setup (or first two joints)
                y1 = joint_yaws[0] if len(joint_yaws) > 0 else 0.0
                y2 = joint_yaws[1] if len(joint_yaws) > 1 else 0.0
                j1_euler = joint_eulers[0] if len(joint_eulers) > 0 else (0.0, 0.0, 0.0)
                j2_euler = joint_eulers[1] if len(joint_eulers) > 1 else (0.0, 0.0, 0.0)
                telemetry.update(t,pos_xy,lead_hdg,joint_yaws,joint_eulers,forces)

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

    telemetry.close()


if __name__ == "__main__":
    main()
