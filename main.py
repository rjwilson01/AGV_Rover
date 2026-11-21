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
from config import cg_above_axle_m, config_diff, load_config
from cli import build_parser
from drive_profile import parse_drive
from gnss import GNSSModel
from navfilter import LeadNav, wrap as wrap_angle
from imu import IMUModel
from loadcell import LoadCell
from safeguard import FollowerGuard, RetensionGuard
from ir_camera import IRCameraModel, relative_euler
from telemetry import TelemetryManager
from telemetry_pack import (CORNER_SIGNS, cables_and_positions, hitch_names, joint_eulers as read_joint_eulers,
                            loadcell_metrics, tension_forces, trailer_hitch_metrics)
from test_sequence import DEFAULT_DURATION_S, Segment, parse_sequence
from viewer_overlay import render_bar_yaw_overlay, render_path_overlay
from xml_builder import generate_mujoco_xml

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
    parser = build_parser()
    # argparse treats a value starting with '-' (e.g. "--tests -200-a,b") as a new option: glue it on as --tests=VALUE
    argv = sys.argv[1:]
    for i, tok in enumerate(argv[:-1]):
        if tok == "--tests" and argv[i + 1].startswith("-"):
            argv[i:i + 2] = [f"--tests={argv[i + 1]}"]
            break
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent

    if args.selfcheck:
        from selfcheck import run_selfcheck
        files = parse_sequence(args.tests)[-1].active_files if args.tests else ()
        cfg = load_config(args.preset, test_files=files, set_overrides=args.set)
        ok = run_selfcheck(cfg, base_dir / "sim_test_log.csv")
        return 0 if ok else 1

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
        mujoco.MjModel.from_xml_string(generate_mujoco_xml(cfg))   # compile now: a bad test file fails before anything runs
        changes = config_diff(args.preset, test_files=seg.active_files, set_overrides=args.set)
        plans.append((seg, cfg, duration, changes))

    if args.view:
        from viewmodel import view_model
        if len(plans) > 1:
            print("[view] only the first test group is shown (one configuration at a time)")
        from configsession import ConfigSession
        session = ConfigSession(args.preset, plans[0][0].active_files, args.set or ())
        return view_model(plans[0][1], args.dump_xml, args.base, args.no_friction, session=session, plotjuggler=args.plotjuggler)

    drive_fn = parse_drive(args.drive) if args.drive else None
    first_cfg = plans[0][1]
    meta = {"controller": first_cfg.controller_class}
    if args.drive:
        meta["drive"] = args.drive
    if args.tests:
        meta["TestName"] = "; ".join(p[1].test_name or "(defaults)" for p in plans)
        meta["test_sequence"] = args.tests
        for i, (seg, cfg, duration, changes) in enumerate(plans, start=1):
            meta[f"segment_{i}"] = (
                f"TestName={cfg.test_name or '(defaults)'} | {'reset' if seg.reset else 'add'} | "
                f"duration_s={duration:g} | files={','.join(seg.active_files) or '-'} | "
                f"cg_above_axle_m={cg_above_axle_m(cfg):.3f} | "
                f"clearance_m={cfg.wheel_radius_m - cfg.wheel_z_offset_m - cfg.chassis_half_height_m:.3f} | "
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
        csv_name="sim_test_log.csv" if (args.tests or args.set or args.drive) else "sim_log.csv",
        launch_pj=first_cfg.launch_plotjuggler or args.plotjuggler,
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
                print(f"    CG {cg_above_axle_m(cfg):+.3f} m above axle (negative = stable) | "
                      f"clearance {cfg.wheel_radius_m - cfg.wheel_z_offset_m - cfg.chassis_half_height_m:.3f} m", flush=True)
                if changes:
                    print("    changes vs defaults: " + ", ".join(f"{k}={v}" for k, v in changes.items()), flush=True)
                telemetry.set_segment(i, name)
            finished = run_sim(cfg, telemetry, duration, show_viewer=not args.no_viewer, status_label=label, drive_fn=drive_fn)
            telemetry.flush_csv()
            if args.tests and finished:
                print(f"[Test {i}/{len(plans)}] DONE   {cfg.test_name or '(defaults)'}", flush=True)
            if not finished:
                print("[Test] viewer closed - stopping the sequence")
                break
    finally:
        telemetry.close()


def run_sim(cfg, telemetry, duration_s=None, show_viewer=True, status_label="", drive_fn=None) -> bool:
    """Run one simulation. Returns True if it ran to duration_s, False if the viewer was closed first."""
    print(f"[Sim] Loaded controller={cfg.controller_class} | mode={cfg.drawbar_mode} | path={cfg.active_path}")

    xml_str = generate_mujoco_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    lead_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trailer_1")
    controller = create_controller(cfg)
    controller.drive = drive_fn
    ir_camera = IRCameraModel(cfg)
    imu_model = IMUModel(cfg) if cfg.imu_enabled else None
    guard = FollowerGuard(cfg)
    retension = RetensionGuard(cfg, cfg.num_trailers - 1)
    cable_cells = [LoadCell(cfg, ch) for ch in range(4 * (cfg.num_trailers - 1))]
    bar_cells = [LoadCell(cfg, 100 + ch) for ch in range(cfg.num_trailers - 1)]
    cable_sums_meas = [0.0] * (cfg.num_trailers - 1)
    gnss_model = GNSSModel(cfg) if cfg.gnss_enabled else None
    nav = LeadNav(cfg) if cfg.nav_gnss_in_loop else None
    if nav is not None and gnss_model is None:
        print("[Sim] nav.gnss_in_loop needs gnss.enabled = true: using the true pose instead")
        nav = None
    v_hint = 0.0                                          # signed lead speed from wheel odometry, last step

    wheel_names = [
        f"m_t{i}_{side}"
        for i in range(1, cfg.num_trailers + 1)
        for side in ("l", "r")
    ]
    joint_names = [f"ball_joint_{j}" for j in range(1, cfg.num_trailers)]
    bars = cfg.bar_yaw_control
    tendon_names, capstan_names = hitch_names(cfg)
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
                if imu_model is not None:
                    imu_model.reset()
                guard.reset()
                retension.reset()
                for _c in cable_cells + bar_cells:
                    _c.reset()
                cable_sums_meas = [0.0] * (cfg.num_trailers - 1)
                if gnss_model is not None:
                    gnss_model.reset()
                if nav is not None:
                    nav.reset()
                v_hint = 0.0
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
            imu_readings = None
            if imu_model is not None:
                imu_readings = imu_model.measure(
                    t,
                    [data.sensor(f"t{i}_acc").data for i in range(1, cfg.num_trailers + 1)],
                    [data.sensor(f"t{i}_gyro").data for i in range(1, cfg.num_trailers + 1)],
                )

            controller.sim_time = t
            true_vel = data.joint("root_jnt").qvel[:2]
            gnss = gnss_model.measure(t, pos_xy, true_vel) if gnss_model is not None else None
            nav_pos, nav_hdg = pos_xy, lead_hdg
            if nav is not None:
                gyro_z = imu_readings[0].gyro[2] if imu_readings else None
                nav_pos, nav_hdg = nav.update(t, gnss, gyro_z, v_hint, pos_xy, lead_hdg)
            wheel_vels, joint_yaws, bar_lengths = controller.step(nav_pos, nav_hdg, ir_readings=ir_readings)
            # Controller yaws already carry the physical sign (yaw_gain < 0: a follower's yaw relative to its leader is
            # negative in a left turn). joint_yaw_sign is kept only as a debugging switch (1.0 = no change).
            # Everything downstream (actuators, overlay, telemetry cmd-vs-meas) works in physical yaw.
            sgn = cfg.joint_yaw_sign
            joint_yaws = [sgn * y for y in joint_yaws]
            bar_yaws = [(sgn * a_, sgn * b_) for a_, b_ in controller.last_bar_yaws]
            ramp = min(1.0, t / cfg.start_ramp_s) if cfg.start_ramp_s > 0.0 else 1.0
            # Stall guard: lead commanded vs actual speed (stand-in for lead odometry / GNSS speed)
            v_lead_cmd = abs(0.5 * (wheel_vels[0] + wheel_vels[1])) * cfg.wheel_radius_m * ramp
            v_hint = 0.5 * (wheel_vels[0] + wheel_vels[1]) * cfg.wheel_radius_m * ramp
            if gnss is None:
                v_lead_meas = float(np.hypot(*true_vel))      # no GNSS modelled: ideal speed
            elif gnss.valid:
                v_lead_meas = gnss.speed
            else:
                v_lead_meas = v_lead_cmd                       # no fix: can't judge a stall, hold the guard
            guard_scale = guard.update(t, v_lead_cmd, v_lead_meas)
            for k, (w_name, w_vel) in enumerate(zip(wheel_names, wheel_vels)):
                data.actuator(w_name).ctrl = w_vel * ramp * (1.0 if k < 2 else guard_scale * retension.trailer_scale(k // 2))

            joint_eulers = read_joint_eulers(cfg, data, trailer_quats)
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
                    psi_r, psi_f = bar_yaws[j_idx]
                    psi_r_meas = float(data.joint(f"yaw_rear_{j_idx + 1}").qpos.item())
                    psi_f_meas = float(data.joint(f"yaw_front_{j_idx + 2}").qpos.item())
                    if cfg.cable_use_measured_tubes:
                        # Size the cables for where the tubes ARE, and let the ball take whatever the tubes have not yet
                        # delivered. Sizing them for the commanded tubes makes the cables fight a lagging tube and the
                        # passive ball swings far from its command (seen: -52 deg on the return leg).
                        psi_r_cab, psi_f_cab = psi_r_meas, psi_f_meas
                    else:
                        psi_r_cab, psi_f_cab = psi_r, psi_f
                    r_cab, f_cab = r_cmd, f_cmd
                    if cfg.cable_use_measured_tubes:         # same for the telescoping bars: cables follow where they ARE
                        if f"slide_rear_{j_idx + 1}" in joint_names_set:
                            r_cab = r_max - float(data.joint(f"slide_rear_{j_idx + 1}").qpos.item())
                        if f"slide_front_{j_idx + 2}" in joint_names_set:
                            f_cab = f_max - float(data.joint(f"slide_front_{j_idx + 2}").qpos.item())
                    y_ball_cmd = y_cmd - psi_r_cab - psi_f_cab  # what the tubes cannot take goes to the ball
                    # Hold the ball's current pitch/roll: IR trailer R/P/Y minus the measured tube angles
                    if cfg.use_ir_camera_feedback:
                        _, p_ball, r_ball = ball_ypr_from_trailer_relative(
                            (ir.yaw, ir.pitch, ir.roll), psi_r_meas, psi_f_meas
                        )
                    else:
                        p_ball = r_ball = 0.0
                    data.actuator(f"act_yaw_rear_{j_idx + 1}").ctrl = psi_r
                    data.actuator(f"act_yaw_front_{j_idx + 2}").ctrl = psi_f
                    for k, (_, ys, zs) in enumerate(CORNER_SIGNS):
                        data.actuator(capstan_names[4 * j_idx + k]).ctrl = (
                            trailer_cable_len(psi_r_cab, psi_f_cab, (y_ball_cmd, p_ball, r_ball), ys, zs, cfg, r_cab, f_cab)
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
                    if bars and cfg.show_bar_overlay:
                        render_bar_yaw_overlay(viewer, data, cfg, bar_yaws, controller.last_bar_lengths)
                viewer.sync()

            forces = tension_forces(data, capstan_names, tendon_names)

            # Simulated load cells on the four cables (and optionally the bar) of each hitch; compression guard input
            lc_metrics = loadcell_metrics(cfg, model, data, forces, tendon_names, cable_cells, bar_cells, cable_sums_meas)
            if cfg.loadcell_enabled:
                retension.update(float(data.time), cable_sums_meas)
                for j_h in range(cfg.num_trailers - 1):
                    lc_metrics[f"retension/hitch{j_h + 1}_scale"] = round(retension.scale[j_h], 3)

            is_stopped = all(abs(v) < STOP_WHEEL_VEL_EPS for v in wheel_vels)
            if is_stopped:
                post_stop_time += model.opt.timestep
            else:
                post_stop_time = 0.0

            extra_metrics = controller.get_telemetry()
            extra_metrics.update(lc_metrics)
            if nav is not None:
                extra_metrics["nav/pos_err_m"] = round(float(np.hypot(*(nav_pos - np.asarray(pos_xy)[:2]))), 3)
                extra_metrics["nav/hdg_err_deg"] = round(float(np.rad2deg(wrap_angle(nav_hdg - lead_hdg))), 3)
                extra_metrics["nav/fix_valid"] = int(nav.fix_valid)
            extra_metrics["guard/follower_scale"] = round(guard_scale, 3)
            extra_metrics["guard/lead_speed_cmd_mps"] = round(v_lead_cmd, 3)
            extra_metrics["guard/lead_speed_meas_mps"] = round(v_lead_meas, 3)
            extra_metrics.update(trailer_hitch_metrics(cfg, data, trailer_quats, joint_yaws, bar_yaws, joint_names_set))
            joint_cables, trailer_positions = cables_and_positions(cfg, data, capstan_names, tendon_names)
            # Timed test runs log the whole duration; interactive runs stop logging shortly after the vehicle stops
            if duration_s is not None or post_stop_time <= cfg.post_stop_log_sec:
                telemetry.update(
                    t, pos_xy, lead_hdg, joint_yaws, joint_eulers, forces, extra_metrics,
                    joint_cables, trailer_positions, ir_readings=ir_readings, imu_readings=imu_readings, gnss=gnss,
                )


            if show_viewer:
                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    return finished


if __name__ == "__main__":
    main()
