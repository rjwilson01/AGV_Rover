"""Static model review: build the model from the preset (+ one set of test files) and look at it. No physics steps,
no controller, no CSV, no PlotJuggler, no files written unless --dump-xml is given."""
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from bar_geometry import bar_packaging_warnings, pivot_x_m
from config import cg_above_axle_m
from xml_builder import generate_mujoco_xml


def describe(cfg, model) -> list[str]:
    mass = [float(model.body(i).mass[0]) if hasattr(model.body(i).mass, "__len__") else float(model.body(i).mass)
            for i in range(model.nbody)]
    lines = [
        f"controller {cfg.controller_class} | drawbar_mode {cfg.drawbar_mode} | bar_yaw_control {cfg.bar_yaw_control} | trailers {cfg.num_trailers}",
        f"test files: {cfg.test_name or '(defaults)'}",
        f"model: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, {model.ntendon} tendons, {model.nu} actuators, {model.nsensor} sensors",
        f"total mass {sum(mass):.1f} kg",
        f"wheel radius {cfg.wheel_radius_m:.3f} m | axle offset {cfg.wheel_z_offset_m:+.3f} m | track {cfg.track_width_m:.3f} m | "
        f"clearance {cfg.wheel_radius_m - cfg.wheel_z_offset_m - cfg.chassis_half_height_m:.3f} m",
        f"CG {cg_above_axle_m(cfg):+.3f} m above the axle (negative = pitch-stable)",
        f"chassis {2 * cfg.chassis_half_len_m:.2f} x {2 * cfg.chassis_half_width_m:.2f} m | bar pivot {pivot_x_m(cfg):.2f} m from the chassis centre",
        f"bars: rear {cfg.rear_bar_range_m} m, front {cfg.front_bar_range_m} m | joint yaw range {cfg.yaw_angle_range_deg} deg",
    ]
    lines += [f"WARNING: {w}" for w in bar_packaging_warnings(cfg)]
    return lines


def view_model(cfg, dump_xml: str | None = None) -> int:
    xml = generate_mujoco_xml(cfg)
    if dump_xml:
        Path(dump_xml).write_text(xml)
        print(f"[view] wrote {dump_xml}")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("[view] STATIC review: physics is NOT running; nothing is logged or written. Close the window to exit.")
    for line in describe(cfg, model):
        print("   " + line)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return 0
