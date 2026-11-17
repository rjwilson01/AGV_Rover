import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

from bar_geometry import axle_distance, ball_ypr_from_trailer_relative, trailer_cable_len
from BaseSnakeController import calc_cable_len, create_controller, quat_to_euler
from config import config_diff, load_config
from ir_camera import IRCameraModel, relative_euler
from telemetry import TelemetryManager
from test_sequence import DEFAULT_DURATION_S, Segment, parse_sequence
from viewer_overlay import render_bar_yaw_overlay, render_path_overlay
from xml_builder import generate_mujoco_xml

CORNER_SIGNS = (
    ("tl",  1.0,  1.0),
    ("tr", -1.0,  1.0),
    ("bl",  1.0, -1.0),
    ("br", -1.0, -1.0),
)
STOP_WHEEL_VEL_EPS = 1e-6


class _NullViewer:
    """Stand-in used with --no-viewer so the loop runs headless (and as fast as the CPU allows)."""

    def is_running(self):
        return True

    def close(self):
        pass


def _set_status(viewer, line1: str, line2: str) -> None:
    """Show the current test step in the MuJoCo window (top-left). Silently skipped on old MuJoCo builds."""
    try:
        viewer.set_texts((mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, line1, line2))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Snake AGV simulation")
    parser.add_argument("preset", nargs="?", default=None, help="controller preset, e.g. VariableDrawbarController")
    parser.add_argument(
        "--tests", default=None, metavar="SPEC",
        help="test sequence, e.g. 200-a,b,c,100-d,e,100+f,g  (seconds, then - = reset to defaults / + = keep "
             "previous files, then files in configs/test/; each group runs once its files are loaded)",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one config value (TOML syntax); beats every test file. Repeatable.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="run headless, faster than real time")
    # argparse treats a value starting with '-' (e.g. "--tests -200-a,b") as a new option: glue it on as --tests=VALUE
    argv = sys.argv[1:]
    for i, tok in enumerate(argv[:-1]):
        if tok == "--tests" and argv[i + 1].startswith("-"):
            argv[i:i + 2] = [f"--tests={argv[i + 1]}"]
            break
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent

    # ---- plan every segment up front so a typo in any test file fails before anything runs
    if args.tests:
        segments = parse_sequence(args.tests)
    else:
        segments = [Segment(True, None, (), ())]
    plans = []
    for seg in segments:
        cfg = load_config(args.preset, test_files=seg.active_files, set_overrides=args.set)
        if args.tests:
            duration = seg.duration_s or cfg.test_duration_s or DEFAULT_DURATION_S
        elif args.no_viewer:
            duration = DEFAULT_DURATION_S                     # headless has no window to close
        else:
            duration = None                                   # interactive: run until the viewer is closed
        changes = config_diff(args.preset, test_files=seg.active_files, set_overrides=args.set)
        plans.append((seg, cfg, duration, changes))

    first_cfg = plans[0][1]
    meta = {"controller": first_cfg.controller_class}
    if args.tests:
        meta["TestName"] = "; ".join(p[1].test_name or "(defaults)" for p in plans)
        meta["test_sequence"] = args.tests
        for i, (seg, cfg, duration, changes) in enumerate(plans, start=1):
            meta[f"segment_{i}"] = (
                f"TestName={cfg.test_name or '(defaults)'} | {'reset' if seg.reset else 'add'} | "
                f"duration_s={duration:g} | files={','.join(seg.active_files) or '-'} | "
                f"changes={json.dumps(changes, default=str)}"
            )
    elif args.set:
        meta["set"] = " ".join(args.set)
    meta.update({
        "drawbar_mode": first_cfg.drawbar_mode,
        "bar_yaw_control": first_cfg.bar_yaw_control,
        "num_trailers": first_cfg.num_trailers,
        "ir_noise_std_deg": first_cfg.ir_noise_std_deg,
        "ir_latency_s": first_cfg.ir_latency_s,
        "use_ir_camera_feedback": first_cfg.use_ir_camera_feedback,
        "bar_yaw_kp": first_cfg.bar_yaw_kp,
        "bar_yaw_kv": first_cfg.bar_yaw_kv,
    })

    telemetry = TelemetryManager(
        base_dir,
        udp_host=first_cfg.udp_host,
        udp_port=first_cfg.udp_port,
        enable_csv=first_cfg.save_csv_log,
        bar_yaw=first_cfg.bar_yaw_control,
        meta=meta,
    )
    try:
        for i, (seg, cfg, duration, changes) in enumerate(plans, start=1):
            label = ""
            if args.tests:
                name = cfg.test_name or "(defaults)"
                label = f"Test {i}/{len(plans)}: {name}"
                print(f"\n{'=' * 70}\n[Test {i}/{len(plans)}] START  {name}  | {duration:g} s | "
                      f"{'reset' if seg.reset else 'add'} files={','.join(seg.active_files) or '-'}", flush=True)
                if changes:
                    print("    changes vs defaults: " + ", ".join(f"{k}={v}" for k, v in changes.items()), flush=True)
                telemetry.set_segment(i, name)
            finished = run_sim(cfg, telemetry, duration, show_viewer=not args.no_viewer, status_label=label)
            telemetry.flush_csv()
            if args.tests and finished:
                print(f"[Test {i}/{len(plans)}] DONE   {cfg.test_name or '(defaults)'}", flush=True)
            if not finished:
                print("[Test] viewer closed - stopping the sequence")
                break
    finally:
        telemetry.close()


def run_sim(cfg, telemetry, duration_s=None, show_viewer=True, status_label="") -> bool:
    """Run one simulation. Returns True if it ran to duration_s, False if the viewer was closed first."""
    print(f"[Sim] Loaded controller={cfg.controller_class} | mode={cfg.drawbar_mode} | path={cfg.active_path}")

    xml_str = generate_mujoco_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    lead_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trailer_1")
    controller = create_controller(cfg)
    ir_camera = IRCameraModel(cfg)

    wheel_names = [
        f"m_t{i}_{side}"
        for i in range(1, cfg.num_trailers + 1)
        for side in ("l", "r")
    ]
    joint_names = [f"ball_joint_{j}" for j in range(1, cfg.num_trailers)]
    bars = cfg.bar_yaw_control
    tendon_names = [
        f"t{j}_t{j + 1}_{code}"
        for j in range(1, cfg.num_trailers)
        for code, _, _ in CORNER_SIGNS
    ]
    capstan_names = [f"cap_{t_name}" for t_name in tendon_names]
    actuator_names = {model.actuator(i).name for i in range(model.nu)}
    joint_names_set = {model.joint(i).name for i in range(model.njnt)}

    def apply_neutral_controls():
        neutral_len = calc_cable_len(0.0, 0.0, 0.0, 1.0, 1.0, cfg) - cfg.cable_preload_offset_m
        for cname in capstan_names:
            data.actuator(cname).ctrl = neutral_len
        if bars:
            for j in range(1, cfg.num_trailers):
                data.actuator(f"act_yaw_rear_{j}").ctrl = 0.0
                data.actuator(f"act_yaw_front_{j + 1}").ctrl = 0.0

    apply_neutral_controls()
    prev_sim_time = 0.0
    post_stop_time = 0.0

    finished = False
    last_status = -1.0
    with (mujoco.viewer.launch_passive(model, data) if show_viewer else nullcontext(_NullViewer())) as viewer:
        if show_viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = lead_body_id
            viewer.cam.distance = cfg.camera_distance_m
            viewer.cam.azimuth = cfg.camera_azimuth_deg
            viewer.cam.elevation = cfg.camera_elevation_deg

        while viewer.is_running():
            if duration_s is not None and data.time >= duration_s:
                finished = True
                break
            if show_viewer and status_label and data.time - last_status >= 0.25:
                last_status = data.time
                _set_status(viewer, status_label, f"t = {data.time:5.1f} / {duration_s:g} s" if duration_s else "")
            if data.time < prev_sim_time:
                post_stop_time = 0.0
                controller.reset()
                ir_camera.reset()
                apply_neutral_controls()
            prev_sim_time = data.time

            step_start = time.time()
            t = float(data.time)

            root_qpos = data.joint("root_jnt").qpos
            pos_xy = root_qpos[:2]
            lead_hdg, _, _ = quat_to_euler(root_qpos[3:7])

            # What the IR cameras see: relative R/P/Y between neighbouring trailers.
            # Recompute body poses from current qpos (xquat lags one step after mj_step, and is unset at t=0).
            mujoco.mj_kinematics(model, data)
            trailer_quats = [data.body(f"trailer_{i}").xquat for i in range(1, cfg.num_trailers + 1)]
            ir_readings = ir_camera.measure(t, trailer_quats)

            wheel_vels, joint_yaws, bar_lengths = controller.step(pos_xy, lead_hdg, ir_readings=ir_readings)
            ramp = min(1.0, t / cfg.start_ramp_s) if cfg.start_ramp_s > 0.0 else 1.0
            for w_name, w_vel in zip(wheel_names, wheel_vels):
                data.actuator(w_name).ctrl = w_vel * ramp

            if bars:
                # With yaw tubes the ball joint is only part of the hitch: use the true trailer-to-trailer R/P/Y.
                joint_eulers = [
                    tuple(relative_euler(trailer_quats[j], trailer_quats[j + 1])) for j in range(cfg.num_joints)
                ]
            else:
                joint_eulers = [quat_to_euler(data.joint(j_name).qpos) for j_name in joint_names]
            _, r_max = cfg.rear_bar_range_m
            _, f_max = cfg.front_bar_range_m

            for j_idx, (y_cmd, (r_cmd, f_cmd)) in enumerate(zip(joint_yaws, bar_lengths)):
                rear_act = f"act_slide_rear_{j_idx + 1}"
                front_act = f"act_slide_front_{j_idx + 2}"
                if rear_act in actuator_names:
                    data.actuator(rear_act).ctrl = r_max - r_cmd
                if front_act in actuator_names:
                    data.actuator(front_act).ctrl = f_max - f_cmd

                ir = ir_readings[j_idx]
                if bars:
                    psi_r, psi_f = controller.last_bar_yaws[j_idx]
                    y_ball_cmd = y_cmd - psi_r - psi_f          # what the tubes cannot take goes to the ball
                    # Hold the ball's current pitch/roll: IR trailer R/P/Y minus the measured tube angles
                    if cfg.use_ir_camera_feedback:
                        psi_r_meas = float(data.joint(f"yaw_rear_{j_idx + 1}").qpos.item())
                        psi_f_meas = float(data.joint(f"yaw_front_{j_idx + 2}").qpos.item())
                        _, p_ball, r_ball = ball_ypr_from_trailer_relative(
                            (ir.yaw, ir.pitch, ir.roll), psi_r_meas, psi_f_meas
                        )
                    else:
                        p_ball = r_ball = 0.0
                    data.actuator(f"act_yaw_rear_{j_idx + 1}").ctrl = psi_r
                    data.actuator(f"act_yaw_front_{j_idx + 2}").ctrl = psi_f
                    for k, (_, ys, zs) in enumerate(CORNER_SIGNS):
                        data.actuator(capstan_names[4 * j_idx + k]).ctrl = (
                            trailer_cable_len(psi_r, psi_f, (y_ball_cmd, p_ball, r_ball), ys, zs, cfg, r_cmd, f_cmd)
                            - cfg.cable_preload_offset_m
                        )
                else:
                    p_ctrl, r_ctrl = (ir.pitch, ir.roll) if cfg.use_ir_camera_feedback else (0.0, 0.0)
                    for k, (_, ys, zs) in enumerate(CORNER_SIGNS):
                        c_idx = 4 * j_idx + k
                        data.actuator(capstan_names[c_idx]).ctrl = (
                            calc_cable_len(
                                # calc_cable_len uses a mirrored x axis: it matches the XML geometry for
                                # (yaw, pitch, roll) = (-in_yaw, -in_pitch, in_roll), so pitch must be negated too.
                                -y_cmd, -p_ctrl, r_ctrl, ys, zs, cfg,
                                rear_bar_m=r_cmd, front_bar_m=f_cmd,
                            )
                            - cfg.cable_preload_offset_m
                        )

            mujoco.mj_step(model, data)

            if show_viewer:
                with viewer.lock():
                    render_path_overlay(viewer, cfg.waypoints, controller.last_lookahead)
                    if bars:
                        render_bar_yaw_overlay(viewer, data, cfg, controller.last_bar_yaws, controller.last_bar_lengths)
                viewer.sync()

            forces = {
                t_name: abs(float(data.actuator(c_name).force.item()))
                for c_name, t_name in zip(capstan_names, tendon_names)
            }

            is_stopped = all(abs(v) < STOP_WHEEL_VEL_EPS for v in wheel_vels)
            if is_stopped:
                post_stop_time += model.opt.timestep
            else:
                post_stop_time = 0.0

            extra_metrics = controller.get_telemetry()
            # Absolute attitude, height and wheel torque of each trailer (diagnoses cart rocking / drive reaction)
            for i_t, q_t in enumerate(trailer_quats, start=1):
                _, p_abs, r_abs = quat_to_euler(q_t)
                extra_metrics[f"trailer{i_t}/pitch_abs_deg"] = round(float(np.rad2deg(p_abs)), 3)
                extra_metrics[f"trailer{i_t}/roll_abs_deg"] = round(float(np.rad2deg(r_abs)), 3)
                extra_metrics[f"trailer{i_t}/z_m"] = round(float(data.body(f"trailer_{i_t}").xpos[2]), 4)
                extra_metrics[f"trailer{i_t}/wheel_torque_l_Nm"] = round(float(data.actuator(f"m_t{i_t}_l").force.item()), 2)
                extra_metrics[f"trailer{i_t}/wheel_torque_r_Nm"] = round(float(data.actuator(f"m_t{i_t}_r").force.item()), 2)
            bar_meas_raw = [(r_max, f_max)] * cfg.num_joints    # unrounded, for the geometry below
            for j_idx in range(cfg.num_joints):
                r_jnt = f"slide_rear_{j_idx + 1}"
                f_jnt = f"slide_front_{j_idx + 2}"
                r_meas_raw = r_max - float(data.joint(r_jnt).qpos.item()) if r_jnt in joint_names_set else r_max
                f_meas_raw = f_max - float(data.joint(f_jnt).qpos.item()) if f_jnt in joint_names_set else f_max
                bar_meas_raw[j_idx] = (r_meas_raw, f_meas_raw)
                extra_metrics[f"joint{j_idx + 1}/rear_bar_meas_m"] = round(r_meas_raw, 4)
                extra_metrics[f"joint{j_idx + 1}/front_bar_meas_m"] = round(f_meas_raw, 4)
            if bars:
                # Per-bar tube yaw (commanded vs measured by the tube hinge), the ball joint, and the
                # axle distance implied by the measured geometry (compared with the real one in telemetry)
                for j_idx in range(cfg.num_joints):
                    n = j_idx + 1
                    psi_r_cmd, psi_f_cmd = controller.last_bar_yaws[j_idx]
                    psi_r_meas = float(data.joint(f"yaw_rear_{n}").qpos.item())
                    psi_f_meas = float(data.joint(f"yaw_front_{n + 1}").qpos.item())
                    b_yaw, b_pitch, b_roll = quat_to_euler(data.joint(f"ball_joint_{n}").qpos)
                    extra_metrics[f"joint{n}/rear_yaw_cmd_deg"] = round(float(np.rad2deg(psi_r_cmd)), 3)
                    extra_metrics[f"joint{n}/rear_yaw_meas_deg"] = round(float(np.rad2deg(psi_r_meas)), 3)
                    extra_metrics[f"joint{n}/front_yaw_cmd_deg"] = round(float(np.rad2deg(psi_f_cmd)), 3)
                    extra_metrics[f"joint{n}/front_yaw_meas_deg"] = round(float(np.rad2deg(psi_f_meas)), 3)
                    extra_metrics[f"joint{n}/ball_yaw_cmd_deg"] = round(
                        float(np.rad2deg(joint_yaws[j_idx] - psi_r_cmd - psi_f_cmd)), 3
                    )
                    extra_metrics[f"joint{n}/ball_yaw_meas_deg"] = round(float(np.rad2deg(b_yaw)), 3)
                    extra_metrics[f"joint{n}/ball_pitch_meas_deg"] = round(float(np.rad2deg(b_pitch)), 3)
                    extra_metrics[f"joint{n}/ball_roll_meas_deg"] = round(float(np.rad2deg(b_roll)), 3)
                    extra_metrics[f"joint{n}/rear_yaw_torque_Nm"] = round(
                        float(data.actuator(f"act_yaw_rear_{n}").force.item()), 3)
                    extra_metrics[f"joint{n}/front_yaw_torque_Nm"] = round(
                        float(data.actuator(f"act_yaw_front_{n + 1}").force.item()), 3)
                    extra_metrics[f"joint{n}/axle_dist_kin_m"] = round(
                        axle_distance(
                            psi_r_meas, psi_f_meas, (b_yaw, b_pitch, b_roll),
                            bar_meas_raw[j_idx][0], bar_meas_raw[j_idx][1], cfg,
                        ),
                        4,
                    )
            # Put the cables into a per joint list
            joint_cables = []
            for j_idx in range(cfg.num_joints):
                cables_j = {}
                for k, (code, _, _) in enumerate(CORNER_SIGNS):
                    c_idx = 4 * j_idx + k
                    t_name = tendon_names[c_idx]
                    c_name = capstan_names[c_idx]
                    cables_j[code] = (
                        float(data.actuator(c_name).ctrl.item()),
                        float(data.tendon(t_name).length.item()),
                        abs(float(data.actuator(c_name).force.item())),
                    )
                joint_cables.append(cables_j)

            trailer_positions = [
                tuple(float(v) for v in data.body(f"trailer_{i}").xpos)
                for i in range(1, cfg.num_trailers + 1)
            ]
            # Timed test runs log the whole duration; interactive runs stop logging shortly after the vehicle stops
            if duration_s is not None or post_stop_time <= cfg.post_stop_log_sec:
                telemetry.update(
                    t, pos_xy, lead_hdg, joint_yaws, joint_eulers, forces, extra_metrics,
                    joint_cables, trailer_positions, ir_readings=ir_readings,
                )


            if show_viewer:
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    return finished


if __name__ == "__main__":
    main()
