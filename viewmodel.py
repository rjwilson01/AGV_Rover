"""Manual model review: build the model from the preset (+ one set of test files) and drive every actuator by hand
(see manual.py). No controller, no CSV, no PlotJuggler, no files written unless --dump-xml is given."""
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from bar_geometry import bar_packaging_warnings, pivot_x_m
from config import cg_above_axle_m
from sandbox import Sandbox
from xml_builder import generate_mujoco_xml


BASE_MODES = ("free", "fixed", "planar")


def apply_view_options(xml: str, base: str = "free", frictionless: bool = False) -> str:
    """Tweak the generated MJCF for hand testing (view mode only).

    base: 'free'   - the lead trailer is free (normal);
          'fixed'  - no root joint, so the lead is welded to the world and only the hitches/bars move;
          'planar' - two slide joints (x, y): the lead can translate over the ground but not rotate or lift.
    frictionless: floor friction 0 and condim=1 on every geom. (MuJoCo uses the larger friction of the two geoms in
          contact, so zeroing the floor alone is not enough: condim=1 removes all tangential contact force.)
    """
    if base not in BASE_MODES:
        raise ValueError(f"base must be one of {BASE_MODES}, got {base!r}")
    root = '<freejoint name="root_jnt"/>'
    if base != "free":
        if root not in xml:
            raise ValueError("no root freejoint found in the generated model")
        repl = "" if base == "fixed" else (
            '<joint name="root_x" type="slide" axis="1 0 0"/><joint name="root_y" type="slide" axis="0 1 0"/>')
        xml = xml.replace(root, repl, 1)
    if frictionless:
        xml = xml.replace('friction="0.85 0.005 0.0005"', 'friction="0 0 0"')
        i = xml.index("<option ")
        j = xml.index("/>", i) + 2
        xml = xml[:j] + '\n\t<default><geom condim="1"/></default>' + xml[j:]
    return xml


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


def view_model(cfg, dump_xml: str | None = None, base: str = "fixed", frictionless: bool = False) -> int:
    """Open the model with physics running, the REAL controller in the loop on manual inputs, and a pop-out Tk window
    with a switch per algorithm, the set-points, and live commanded-vs-measured values. Nothing is logged."""
    xml = apply_view_options(generate_mujoco_xml(cfg), base, frictionless)
    if dump_xml:
        Path(dump_xml).write_text(xml)
        print(f"[view] wrote {dump_xml}")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sb = Sandbox(cfg, model, data)
    ctl = sb.ctl
    print("[view] MANUAL sandbox: physics runs with the real controller on your inputs; switch each algorithm on/off and "
          "set the inputs in the pop-out window (or with the keys). Nothing is logged or written.")
    print(f"[view] base={base} | floor {'FRICTIONLESS' if frictionless else 'normal friction'}")
    for line in describe(cfg, model):
        print("   " + line)

    win = None
    try:
        from tkpanel import ControlWindow
        win = ControlWindow(ctl, sandbox=sb)
    except Exception as exc:                                    # no Tk (or no display): use the viewer overlay instead
        print(f"[view] pop-out window unavailable ({exc}); using the on-screen panel in the viewer")

    def on_key(code):
        ctl.key(code)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            sb.step()
            if ctl.paused:
                mujoco.mj_forward(model, data)
            else:
                mujoco.mj_step(model, data)
            if win is not None and not win.update():
                win = None                                       # closed by the user: carry on with the viewer panel
            try:
                if win is None:
                    left, right = ctl.panel()
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, ctl.readout, ""))
                else:
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      "controls are in the pop-out window" + ("  [PHYSICS PAUSED]" if ctl.paused else ""), ""))
            except Exception:
                pass
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))
    return 0
