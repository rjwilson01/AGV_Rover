"""Per-step telemetry pieces shared by main.py (normal runs) and sandbox.py (--view Record): trailer attitude, wheel speeds and
torques, bar lengths, yaw tubes, the ball joint, per-cable lengths and tensions, trailer positions. Every function reads only
`data` (MuJoCo state) and the config, so both callers produce identical channels."""
from __future__ import annotations

import numpy as np

from BaseSnakeController import quat_to_euler
from bar_geometry import axle_distance, ball_ypr_from_joints
from ir_camera import relative_euler

CORNER_SIGNS = (
    ("tl", 1.0, 1.0),
    ("tr", -1.0, 1.0),
    ("bl", 1.0, -1.0),
    ("br", -1.0, -1.0),
)


def hitch_names(cfg) -> tuple[list[str], list[str]]:
    """(tendon names, capstan actuator names), four per hitch in CORNER_SIGNS order."""
    tendons = [f"t{j}_t{j + 1}_{code}" for j in range(1, cfg.num_trailers) for code, _, _ in CORNER_SIGNS]
    return tendons, [f"cap_{t}" for t in tendons]


def tension_forces(data, capstan_names, tendon_names) -> dict[str, float]:
    return {t: abs(float(data.actuator(c).force.item())) for c, t in zip(capstan_names, tendon_names)}


def joint_eulers(cfg, data, trailer_quats) -> list[tuple[float, float, float]]:
    if cfg.bar_yaw_control:
        # With yaw tubes the ball joint is only part of the hitch: use the true trailer-to-trailer R/P/Y.
        return [tuple(relative_euler(trailer_quats[j], trailer_quats[j + 1])) for j in range(cfg.num_joints)]
    return [quat_to_euler(data.joint(f"ball_joint_{j}").qpos) for j in range(1, cfg.num_trailers)]


def trailer_hitch_metrics(cfg, data, trailer_quats, joint_yaws, bar_yaws, joint_names_set) -> dict[str, float]:
    """Trailer attitude / height / wheel data, bar lengths and (with yaw tubes) tube yaw, ball angles and axle distance."""
    out: dict[str, float] = {}
    _, r_max = cfg.rear_bar_range_m
    _, f_max = cfg.front_bar_range_m
    # Absolute attitude, height and wheel torque of each trailer (diagnoses cart rocking / drive reaction)
    for i_t, q_t in enumerate(trailer_quats, start=1):
        _, p_abs, r_abs = quat_to_euler(q_t)
        out[f"trailer{i_t}/pitch_abs_deg"] = round(float(np.rad2deg(p_abs)), 3)
        out[f"trailer{i_t}/roll_abs_deg"] = round(float(np.rad2deg(r_abs)), 3)
        out[f"trailer{i_t}/z_m"] = round(float(data.body(f"trailer_{i_t}").xpos[2]), 4)
        out[f"trailer{i_t}/wheel_torque_l_Nm"] = round(float(data.actuator(f"m_t{i_t}_l").force.item()), 2)
        out[f"trailer{i_t}/wheel_torque_r_Nm"] = round(float(data.actuator(f"m_t{i_t}_r").force.item()), 2)
        # wheel speed: what was commanded (after the start ramp) vs what the wheel is actually doing
        for side in ("l", "r"):
            out[f"trailer{i_t}/wheel_speed_{side}_rad_s"] = round(float(data.joint(f"j_t{i_t}_{side}").qvel.item()), 3)
            out[f"trailer{i_t}/wheel_cmd_{side}_rad_s"] = round(float(data.actuator(f"m_t{i_t}_{side}").ctrl.item()), 3)
    bar_meas_raw = [(r_max, f_max)] * cfg.num_joints    # unrounded, for the geometry below
    for j_idx in range(cfg.num_joints):
        r_jnt = f"slide_rear_{j_idx + 1}"
        f_jnt = f"slide_front_{j_idx + 2}"
        r_meas_raw = r_max - float(data.joint(r_jnt).qpos.item()) if r_jnt in joint_names_set else r_max
        f_meas_raw = f_max - float(data.joint(f_jnt).qpos.item()) if f_jnt in joint_names_set else f_max
        bar_meas_raw[j_idx] = (r_meas_raw, f_meas_raw)
        out[f"joint{j_idx + 1}/rear_bar_meas_m"] = round(r_meas_raw, 4)
        out[f"joint{j_idx + 1}/front_bar_meas_m"] = round(f_meas_raw, 4)
    if cfg.bar_yaw_control:
        # Per-bar tube yaw (commanded vs measured by the tube hinge), the ball joint, and the
        # axle distance implied by the measured geometry (compared with the real one in telemetry)
        for j_idx in range(cfg.num_joints):
            n = j_idx + 1
            psi_r_cmd, psi_f_cmd = bar_yaws[j_idx]
            psi_r_meas = float(data.joint(f"yaw_rear_{n}").qpos.item())
            psi_f_meas = float(data.joint(f"yaw_front_{n + 1}").qpos.item())
            b_yaw, b_pitch, b_roll = ball_ypr_from_joints(cfg, data, n)
            out[f"joint{n}/rear_yaw_cmd_deg"] = round(float(np.rad2deg(psi_r_cmd)), 3)
            out[f"joint{n}/rear_yaw_meas_deg"] = round(float(np.rad2deg(psi_r_meas)), 3)
            out[f"joint{n}/front_yaw_cmd_deg"] = round(float(np.rad2deg(psi_f_cmd)), 3)
            out[f"joint{n}/front_yaw_meas_deg"] = round(float(np.rad2deg(psi_f_meas)), 3)
            out[f"joint{n}/ball_yaw_cmd_deg"] = round(float(np.rad2deg(joint_yaws[j_idx] - psi_r_cmd - psi_f_cmd)), 3)
            out[f"joint{n}/ball_yaw_meas_deg"] = round(float(np.rad2deg(b_yaw)), 3)
            out[f"joint{n}/ball_pitch_meas_deg"] = round(float(np.rad2deg(b_pitch)), 3)
            out[f"joint{n}/ball_roll_meas_deg"] = round(float(np.rad2deg(b_roll)), 3)
            out[f"joint{n}/rear_yaw_torque_Nm"] = round(float(data.actuator(f"act_yaw_rear_{n}").force.item()), 3)
            out[f"joint{n}/front_yaw_torque_Nm"] = round(float(data.actuator(f"act_yaw_front_{n + 1}").force.item()), 3)
            out[f"joint{n}/axle_dist_kin_m"] = round(
                axle_distance(psi_r_meas, psi_f_meas, (b_yaw, b_pitch, b_roll),
                              bar_meas_raw[j_idx][0], bar_meas_raw[j_idx][1], cfg), 4)
    return out


def cables_and_positions(cfg, data, capstan_names, tendon_names):
    """-> (per-hitch {corner: (length cmd, length meas, |tension|)}, [(x, y, z) of each trailer])."""
    joint_cables = []
    for j_idx in range(cfg.num_joints):
        cables_j = {}
        for k, (code, _, _) in enumerate(CORNER_SIGNS):
            c_idx = 4 * j_idx + k
            t_name, c_name = tendon_names[c_idx], capstan_names[c_idx]
            cables_j[code] = (
                float(data.actuator(c_name).ctrl.item()),
                float(data.tendon(t_name).length.item()),
                abs(float(data.actuator(c_name).force.item())),
            )
        joint_cables.append(cables_j)
    trailer_positions = [tuple(float(v) for v in data.body(f"trailer_{i}").xpos) for i in range(1, cfg.num_trailers + 1)]
    return joint_cables, trailer_positions


def loadcell_metrics(cfg, model, data, forces, tendon_names, cable_cells, bar_cells, cable_sums_meas) -> dict[str, float]:
    """Simulated load cells on the four cables (and optionally the bar) of each hitch. Fills `cable_sums_meas` in place (the
    compression guard's input) and returns the logged channels."""
    import mujoco
    lc: dict[str, float] = {}
    if cfg.loadcell_enabled:
        for j_h in range(cfg.num_trailers - 1):
            vals = [cable_cells[4 * j_h + k].measure(float(data.time), forces[tendon_names[4 * j_h + k]]) for k in range(4)]
            tl_, tr_, bl_, br_ = vals
            cable_sums_meas[j_h] = sum(vals)
            for (code, _, _), v_ in zip(CORNER_SIGNS, vals):
                lc[f"joint{j_h + 1}/loadcell_{code}_N"] = round(v_, 2)
            lc[f"joint{j_h + 1}/cable_sum_N"] = round(sum(vals), 2)
            lc[f"joint{j_h + 1}/cable_yaw_moment_N"] = round((tl_ + bl_) - (tr_ + br_), 2)
            lc[f"joint{j_h + 1}/cable_pitch_moment_N"] = round((tl_ + tr_) - (bl_ + br_), 2)
    if cfg.bar_loadcell_enabled and cfg.bar_yaw_control:
        mujoco.mj_rnePostConstraint(model, data)      # fills data.cfrc_int (no model element needed)
        for j_h in range(cfg.num_trailers - 1):
            # Force the parent transmits to the rear slider body through its joint, world frame, projected on
            # the bar axis (the slider body's x axis). Sign is set by bar_force_sign (see compare_loads.py).
            b_id = model.body(f"t{j_h + 1}_rear_slider").id
            axis = np.asarray(data.xmat[b_id]).reshape(3, 3)[:, 0]
            f_bar = cfg.bar_force_sign * float(np.dot(np.asarray(data.cfrc_int[b_id])[3:], axis))
            lc[f"joint{j_h + 1}/bar_axial_N"] = round(bar_cells[j_h].measure(float(data.time), f_bar), 2)
    return lc
