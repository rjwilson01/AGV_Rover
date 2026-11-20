import math
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import jinja2
import mujoco
import mujoco.viewer
import numpy as np

# ==============================================================================
# JINJA2 TEMPLATE MANAGEMENT (./drawbar_and_balljoint/drawbar.xml.j2)
# ==============================================================================
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drawbar_and_balljoint")
TEMPLATE_FILE = "drawbar.xml.j2"

DEFAULT_MJCF_TEMPLATE = """<mujoco model="parametric_drawbar_ball_joint">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast">
    <flag warmstart="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <scale forcewidth="0.015" contactwidth="0.02" contactheight="0.02" jointlength="0.06" jointwidth="0.005"/>
    <rgba force="1 0.2 0.2 1"/>
  </visual>

  <worldbody>
    <light pos="0.4 -1.2 2.0" dir="0 0.5 -0.8" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 -0.35" size="3 3 0.1" rgba="0.90 0.90 0.90 1"/>

    <!-- =================================================================== -->
    <!-- REAR BASE ANCHOR (Holds Pivot 2 & Measures Transmitted Reactions)   -->
    <!-- =================================================================== -->
    <body name="rear_anchor" pos="0 0 0.2">
      <!-- Small semi-transparent node so force/torque arrows remain unobstructed -->
      <geom name="rear_housing" type="sphere" pos="0 0 0" size="0.022" rgba="0.20 0.65 0.35 0.45" mass="50.0"/>
      <site name="rear_reaction_site" pos="0 0 0" size="0.006" rgba="0 1 0 1"/>

      <!-- Pivot 2: True 1-DoF Hinge locked at deg2 -->
      <body name="pivot2_link" pos="0 0 0" euler="{{ rear_euler }}">
        <joint name="pivot2_hinge" type="hinge" axis="{{ hinge_axis }}" damping="0.5"/>
        <geom name="pivot2_pin" type="cylinder" fromto="{{ pin_fromto }}" size="0.007" rgba="0.85 0.30 0.20 0.9" mass="0.2"/>

        <!-- Bar 2 extends along local -X from Pivot 2 to the Ball Joint -->
        <geom name="bar2_geom" type="capsule" fromto="0 0 0  -{{ "%.6f"|format(l2) }} 0 0" size="0.011" rgba="0.18 0.72 0.55 1" mass="1.0"/>
        <site name="ball_end_bar2" pos="-{{ "%.6f"|format(l2) }} 0 0" size="0.006" rgba="0 1 1 1"/>
      </body>
    </body>

    <!-- =================================================================== -->
    <!-- FRONT INPUT NODE (Applies External Forces into Pivot 1)             -->
    <!-- =================================================================== -->
    <body name="front_node" pos="{{ "%.6f"|format(fx_pos) }} {{ "%.6f"|format(fy_pos) }} {{ "%.6f"|format(fz_pos) }}">
      <joint name="front_slide_x" type="slide" axis="1 0 0" damping="2.0"/>
      <joint name="front_slide_y" type="slide" axis="0 1 0" damping="2.0"/>
      <joint name="front_slide_z" type="slide" axis="0 0 1" damping="2.0"/>

      <geom name="front_housing" type="sphere" pos="0 0 0" size="0.022" rgba="0.25 0.35 0.50 0.45" mass="2.0"/>
      <site name="force_input_site" pos="0 0 0" size="0.006" rgba="1 0 0 1"/>

      <!-- Pivot 1: True 1-DoF Hinge locked at deg1 -->
      <body name="pivot1_link" pos="0 0 0" euler="{{ front_euler }}">
        <joint name="pivot1_hinge" type="hinge" axis="{{ hinge_axis }}" damping="0.5"/>
        <geom name="pivot1_pin" type="cylinder" fromto="{{ pin_fromto }}" size="0.007" rgba="0.85 0.30 0.20 0.9" mass="0.2"/>

        <!-- Bar 1 extends along local +X from Pivot 1 to the Ball Joint -->
        <geom name="bar1_geom" type="capsule" fromto="0 0 0  {{ "%.6f"|format(l1) }} 0 0" size="0.011" rgba="0.20 0.58 0.85 1" mass="1.0"/>

        <!-- Central 3-DoF Ball Joint -->
        <body name="ball_socket_body" pos="{{ "%.6f"|format(l1) }} 0 0">
          <joint name="center_ball_joint" type="ball" damping="0.05"/>
          <geom name="ball_visual" type="sphere" pos="0 0 0" size="0.020" rgba="0.95 0.75 0.10 0.85" mass="0.3"/>
          <site name="ball_end_bar1" pos="0 0 0" size="0.006" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- ===================================================================== -->
  <!-- CONSTRAINTS: Position-Only Spherical Ball Pin + Rigid End Angles      -->
  <!-- ===================================================================== -->
  <equality>
    <connect name="ball_spherical_pin" site1="ball_end_bar1" site2="ball_end_bar2" solref="0.002 1" solimp="0.95 0.99 0.001"/>
    <joint name="lock_pivot1_angle" joint1="pivot1_hinge" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="lock_pivot2_angle" joint1="pivot2_hinge" polycoef="0 0 0 0 0" solref="0.002 1"/>
    {% if lock_front %}
    <joint name="lock_front_x" joint1="front_slide_x" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="lock_front_y" joint1="front_slide_y" polycoef="0 0 0 0 0" solref="0.002 1"/>
    <joint name="lock_front_z" joint1="front_slide_z" polycoef="0 0 0 0 0" solref="0.002 1"/>
    {% endif %}
  </equality>

  <actuator>
    <general name="front_force_X" site="force_input_site" gear="1 0 0 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
    <general name="front_force_Y" site="force_input_site" gear="0 1 0 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
    <general name="front_force_Z" site="force_input_site" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
  </actuator>

  <sensor>
    <force  name="ball_joint_force"   site="ball_end_bar1"/>
    <force  name="rear_pivot2_force"  site="rear_reaction_site"/>
    <torque name="rear_pivot2_torque" site="rear_reaction_site"/>
  </sensor>
</mujoco>
"""


def ensure_template_exists() -> jinja2.Environment:
    """Creates ./drawbar_and_balljoint/drawbar.xml.j2 on first run and returns a Jinja2 Environment."""
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
    template_path = os.path.join(TEMPLATE_DIR, TEMPLATE_FILE)
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_MJCF_TEMPLATE)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def compute_geometry_context(l1: float, l2: float, deg1: float, deg2: float, axis_mode: str, lock_front: bool) -> dict:
    """Computes exact stress-free world coordinates and Euler angles for the Jinja2 template."""
    l1 = max(0.05, l1)
    l2 = max(0.05, l2)
    deg1 = max(-89.0, min(89.0, deg1))
    deg2 = max(-89.0, min(89.0, deg2))

    a1 = math.radians(deg1)
    a2 = math.radians(deg2)

    if axis_mode == "Yaw (Z)":
        hinge_axis = "0 0 1"
        pin_fromto = "0 0 -0.018 0 0 0.018"
        rear_euler = f"0 0 {deg2:.6f}"
        front_euler = f"0 0 {-deg1:.6f}"

        bx = -l2 * math.cos(a2)
        by = -l2 * math.sin(a2)
        bz = 0.2

        fx_pos = bx - l1 * math.cos(-a1)
        fy_pos = by - l1 * math.sin(-a1)
        fz_pos = 0.2
    else:
        hinge_axis = "0 1 0"
        pin_fromto = "0 -0.018 0 0 0.018 0"
        rear_euler = f"0 {-deg2:.6f} 0"
        front_euler = f"0 {deg1:.6f} 0"

        bx = -l2 * math.cos(a2)
        by = 0.0
        bz = 0.2 + l2 * math.sin(a2)

        fx_pos = bx - l1 * math.cos(a1)
        fy_pos = 0.0
        fz_pos = bz - l1 * math.sin(a1)

    span = math.hypot(math.hypot(fx_pos, fy_pos), fz_pos - 0.2)

    return {
        "l1": l1,
        "l2": l2,
        "deg1": deg1,
        "deg2": deg2,
        "hinge_axis": hinge_axis,
        "pin_fromto": pin_fromto,
        "rear_euler": rear_euler,
        "front_euler": front_euler,
        "fx_pos": fx_pos,
        "fy_pos": fy_pos,
        "fz_pos": fz_pos,
        "span": span,
        "lock_front": lock_front,
    }


# ==============================================================================
# 3D VIEWPORT VECTOR & ROTATIONAL RING RENDERING HELPERS
# ==============================================================================
def _orthonormal_basis(direction: np.ndarray):
    """Returns two unit vectors (u, v) perpendicular to unit vector `direction`."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


def add_vector_arrow(scn, origin: np.ndarray, vec: np.ndarray, scale: float, rgba: list, max_len: float = 0.45):
    """Draws a 3D shaft + tapered arrowhead (using mjGEOM_ARROW / mjv_connector) in scn.geoms."""
    mag = float(np.linalg.norm(vec))
    if mag < 1e-3 or scn.ngeom >= scn.maxgeom - 2:
        return

    direction = vec / mag
    length = min(mag * scale, max_len)
    if length < 0.015:
        length = 0.015

    # In MuJoCo, mjGEOM_ARROW natively draws a shaft + conical arrowhead when configured via mjv_connector
    g_arrow = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g_arrow,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g_arrow,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.006,
        origin,
        origin + direction * length,
    )
    scn.ngeom += 1


def add_torque_arrow_with_ring(scn, origin: np.ndarray, torque_vec: np.ndarray, scale: float, rgba: list):
    """Draws the torque normal arrow plus a 300-degree right-hand circular twist indicator around its shaft."""
    mag = float(np.linalg.norm(torque_vec))
    if mag < 1e-3:
        return

    add_vector_arrow(scn, origin, torque_vec, scale, rgba, max_len=0.40)

    direction = torque_vec / mag
    length = min(mag * scale, 0.40)
    if length < 0.025:
        length = 0.025

    u, v = _orthonormal_basis(direction)
    ring_center = origin + direction * (length * 0.52)
    ring_radius = 0.028

    segments = 16
    arc_angle = math.radians(300.0)
    prev_pt = None
    for i in range(segments + 1):
        theta = (i / segments) * arc_angle
        pt = ring_center + ring_radius * (math.cos(theta) * u + math.sin(theta) * v)
        if prev_pt is not None and scn.ngeom < scn.maxgeom - 2:
            g_seg = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g_seg,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.array(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(g_seg, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0022, prev_pt, pt)
            scn.ngeom += 1
        prev_pt = pt

    # Add a small tangential arrow tip at the end of the circular arc showing right-hand twist direction
    if scn.ngeom < scn.maxgeom - 1:
        end_theta = arc_angle
        tangent = -math.sin(end_theta) * u + math.cos(end_theta) * v
        tip_start = prev_pt - tangent * 0.004
        tip_end = prev_pt + tangent * 0.014
        g_tip = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g_tip,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(g_tip, mujoco.mjtGeom.mjGEOM_ARROW, 0.0035, tip_start, tip_end)
        scn.ngeom += 1


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
class DrawbarSimApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.rebuild_requested = False

        self.jinja_env = ensure_template_exists()

        # Default state
        self.l1 = 0.40
        self.l2 = 0.40
        self.deg1 = 20.0
        self.deg2 = 0.0
        self.axis_mode = "Yaw (Z)"
        self.lock_front = False
        self.forces = [200.0, 0.0, 0.0]

        # Arrow scales (meters per Newton / meters per Nm)
        self.force_scale = 0.0008
        self.torque_scale = 0.0030

        # Build initial model from Jinja2 template
        self.geo_ctx = compute_geometry_context(self.l1, self.l2, self.deg1, self.deg2, self.axis_mode, self.lock_front)
        xml_str = self.jinja_env.get_template(TEMPLATE_FILE).render(**self.geo_ctx)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self._apply_forces_locked()

        # Filtered world-frame sensor state
        self.filt_rear_force = np.zeros(3)
        self.filt_rear_torque = np.zeros(3)
        self.filt_ball_force = np.zeros(3)

        self.sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.sim_thread.start()

        self._build_gui()

    def _apply_forces_locked(self):
        self.data.actuator("front_force_X").ctrl = self.forces[0]
        self.data.actuator("front_force_Y").ctrl = self.forces[1]
        self.data.actuator("front_force_Z").ctrl = self.forces[2]

    def _sensor_to_world(self, site_name: str, local_vec: np.ndarray) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        xmat = self.data.site_xmat[site_id].reshape(3, 3)
        return xmat @ local_vec

    def _run_simulation(self):
        alpha = 0.12
        while self.running:
            with self.lock:
                model = self.model
                data = self.data
                self.rebuild_requested = False

            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

                while self.running and viewer.is_running() and not self.rebuild_requested:
                    step_start = time.time()
                    with self.lock:
                        mujoco.mj_step(model, data)

                        rf_w = self._sensor_to_world("rear_reaction_site", data.sensor("rear_pivot2_force").data)
                        rt_w = self._sensor_to_world("rear_reaction_site", data.sensor("rear_pivot2_torque").data)
                        bf_w = self._sensor_to_world("ball_end_bar1", data.sensor("ball_joint_force").data)

                        self.filt_rear_force = (1.0 - alpha) * self.filt_rear_force + alpha * rf_w
                        self.filt_rear_torque = (1.0 - alpha) * self.filt_rear_torque + alpha * rt_w
                        self.filt_ball_force = (1.0 - alpha) * self.filt_ball_force + alpha * bf_w

                        p_front = data.site("force_input_site").xpos.copy()
                        p_ball = data.site("ball_end_bar1").xpos.copy()
                        p_rear = data.site("rear_reaction_site").xpos.copy()

                        f_in_vec = np.array(self.forces, dtype=float)
                        f_scale = self.force_scale
                        t_scale = self.torque_scale

                        with viewer.lock():
                            scn = viewer.user_scn
                            scn.ngeom = 0
                            # Red: Applied front force
                            add_vector_arrow(scn, p_front, f_in_vec, f_scale, [0.95, 0.15, 0.15, 0.95])
                            # Yellow: Ball joint transmitted force
                            add_vector_arrow(scn, p_ball, self.filt_ball_force, f_scale, [0.98, 0.80, 0.10, 0.95])
                            # Green: Rear reaction force
                            add_vector_arrow(scn, p_rear, self.filt_rear_force, f_scale, [0.15, 0.85, 0.30, 0.95])
                            # Purple + Ring: Rear torque normal vector with right-hand twist indicator
                            add_torque_arrow_with_ring(scn, p_rear, self.filt_rear_torque, t_scale, [0.75, 0.25, 0.95, 0.95])

                        viewer.sync()

                    elapsed = time.time() - step_start
                    if elapsed < model.opt.timestep:
                        time.sleep(model.opt.timestep - elapsed)

                if not viewer.is_running() and not self.rebuild_requested:
                    self.running = False

    def rebuild_geometry(self, event=None):
        try:
            new_l1 = max(0.05, float(self.var_l1.get()))
            new_l2 = max(0.05, float(self.var_l2.get()))
            new_deg1 = max(-89.0, min(89.0, float(self.var_deg1.get())))
            new_deg2 = max(-89.0, min(89.0, float(self.var_deg2.get())))
            new_axis = self.var_axis.get()
            new_lock = bool(self.var_lock_front.get())

            self.var_l1.set(f"{new_l1:.2f}")
            self.var_l2.set(f"{new_l2:.2f}")
            self.var_deg1.set(f"{new_deg1:.1f}")
            self.var_deg2.set(f"{new_deg2:.1f}")

            ctx = compute_geometry_context(new_l1, new_l2, new_deg1, new_deg2, new_axis, new_lock)
            xml_str = self.jinja_env.get_template(TEMPLATE_FILE).render(**ctx)
            new_model = mujoco.MjModel.from_xml_string(xml_str)
            new_data = mujoco.MjData(new_model)

            with self.lock:
                self.l1 = new_l1
                self.l2 = new_l2
                self.deg1 = new_deg1
                self.deg2 = new_deg2
                self.axis_mode = new_axis
                self.lock_front = new_lock
                self.geo_ctx = ctx
                self.model = new_model
                self.data = new_data
                self.filt_rear_force = np.zeros(3)
                self.filt_rear_torque = np.zeros(3)
                self.filt_ball_force = np.zeros(3)
                self._apply_forces_locked()
                self.rebuild_requested = True

            self.var_info.set(
                f"Span: {ctx['span']:.3f} m  |  Front Offset: dX={ctx['fx_pos']:.3f}, dY={ctx['fy_pos']:.3f}, dZ={ctx['fz_pos'] - 0.2:.3f} m"
            )
        except Exception as exc:
            messagebox.showerror("Geometry Error", str(exc))

    def export_xml(self):
        ctx = compute_geometry_context(self.l1, self.l2, self.deg1, self.deg2, self.axis_mode, self.lock_front)
        xml_str = self.jinja_env.get_template(TEMPLATE_FILE).render(**ctx)
        path = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("MuJoCo MJCF XML", "*.xml")],
            initialfile="drawbar_ball_joint.xml",
            title="Save Rendered MJCF XML",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(xml_str)

    def update_forces(self, event=None):
        try:
            fx = float(self.var_fx.get())
            fy = float(self.var_fy.get())
            fz = float(self.var_fz.get())
            with self.lock:
                self.forces = [fx, fy, fz]
                self._apply_forces_locked()
        except ValueError:
            pass

    def _update_scales(self, _=None):
        with self.lock:
            self.force_scale = float(self.var_f_scale.get()) / 1000.0
            self.torque_scale = float(self.var_t_scale.get()) / 1000.0

    def _nudge_var(self, tk_var: tk.StringVar, delta: float, callback):
        try:
            val = float(tk_var.get()) + delta
            tk_var.set(f"{val:.2f}")
            callback()
        except ValueError:
            pass

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Drawbar Ball-Joint Vector Control")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        pad = {"padx": 6, "pady": 4}

        # 1. GEOMETRY FRAME
        geo_frame = ttk.LabelFrame(self.root, text="1. Drawbar Geometry (Jinja2 Template Driven)", padding=8)
        geo_frame.pack(fill="x", **pad)

        self.var_l1 = tk.StringVar(value=f"{self.l1:.2f}")
        self.var_l2 = tk.StringVar(value=f"{self.l2:.2f}")
        self.var_deg1 = tk.StringVar(value=f"{self.deg1:.1f}")
        self.var_deg2 = tk.StringVar(value=f"{self.deg2:.1f}")
        self.var_axis = tk.StringVar(value=self.axis_mode)
        self.var_lock_front = tk.BooleanVar(value=self.lock_front)
        self.var_info = tk.StringVar(value="")

        self._add_stepper_row(geo_frame, 0, "Front Bar 1 Length (m):", self.var_l1, 0.05, self.rebuild_geometry)
        self._add_stepper_row(geo_frame, 1, "Rear Bar 2 Length (m):", self.var_l2, 0.05, self.rebuild_geometry)
        self._add_stepper_row(geo_frame, 2, "Front Angle 1 (°):", self.var_deg1, 5.0, self.rebuild_geometry)
        self._add_stepper_row(geo_frame, 3, "Rear Angle 2 (°):", self.var_deg2, 5.0, self.rebuild_geometry)

        ttk.Label(geo_frame, text="Pivot Axis:").grid(row=4, column=0, sticky="w", pady=3)
        axis_box = ttk.Combobox(geo_frame, textvariable=self.var_axis, values=["Yaw (Z)", "Pitch (Y)"], state="readonly", width=12)
        axis_box.grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        axis_box.bind("<<ComboboxSelected>>", self.rebuild_geometry)

        ttk.Checkbutton(
            geo_frame,
            text="Lock Front Node Translation (Rigid Chassis Hitch vs. Floating)",
            variable=self.var_lock_front,
            command=self.rebuild_geometry,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 2))

        btn_geo = ttk.Frame(geo_frame)
        btn_geo.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        ttk.Button(btn_geo, text="Rebuild / Reset Pose", command=self.rebuild_geometry).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btn_geo, text="Export XML...", command=self.export_xml).pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Label(geo_frame, textvariable=self.var_info, foreground="#005a9e").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(2, 0)
        )

        # 2. INPUT FORCE FRAME
        frc_frame = ttk.LabelFrame(self.root, text="2. Front Input Forces (Newtons)", padding=8)
        frc_frame.pack(fill="x", **pad)

        self.var_fx = tk.StringVar(value=f"{self.forces[0]:.0f}")
        self.var_fy = tk.StringVar(value=f"{self.forces[1]:.0f}")
        self.var_fz = tk.StringVar(value=f"{self.forces[2]:.0f}")

        self._add_stepper_row(frc_frame, 0, "Fx (Axial Push/Pull):", self.var_fx, 50.0, self.update_forces)
        self._add_stepper_row(frc_frame, 1, "Fy (Lateral Side):", self.var_fy, 50.0, self.update_forces)
        self._add_stepper_row(frc_frame, 2, "Fz (Vertical Load):", self.var_fz, 50.0, self.update_forces)

        btn_row = ttk.Frame(frc_frame)
        btn_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Apply Forces", command=self.update_forces).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btn_row, text="Zero All Forces", command=self._zero_forces).pack(side="left", expand=True, fill="x", padx=(3, 0))

        # 3. 3D VECTORS & WORLD-FRAME READOUT
        out_frame = ttk.LabelFrame(self.root, text="3. 3D Vectors & World-Frame Readout", padding=8)
        out_frame.pack(fill="x", **pad)

        scale_frame = ttk.Frame(out_frame)
        scale_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(scale_frame, text="Force Arrow Scale:").grid(row=0, column=0, sticky="w")
        self.var_f_scale = tk.DoubleVar(value=0.8)
        ttk.Scale(scale_frame, from_=0.05, to=4.0, variable=self.var_f_scale, command=self._update_scales, length=160).grid(row=0, column=1, padx=6)

        ttk.Label(scale_frame, text="Torque Arrow Scale:").grid(row=1, column=0, sticky="w")
        self.var_t_scale = tk.DoubleVar(value=3.0)
        ttk.Scale(scale_frame, from_=0.2, to=15.0, variable=self.var_t_scale, command=self._update_scales, length=160).grid(row=1, column=1, padx=6)

        legend = (
            "Arrows: [RED] Applied Front Force   [YELLOW] Ball Joint Force\n"
            "        [GREEN] Rear Reaction Force [PURPLE + RING] Rear Torque Normal"
        )
        ttk.Label(out_frame, text=legend, foreground="#333333", font=("Consolas", 8)).pack(anchor="w", pady=(0, 4))

        self.var_out_force = tk.StringVar(value="Rear Force  [Fx,Fy,Fz]:   0.0,   0.0,   0.0 N")
        self.var_out_torque = tk.StringVar(value="Rear Torque [Mx,My,Mz]:  0.00,  0.00,  0.00 Nm")
        self.var_ball_force = tk.StringVar(value="Ball Joint  [Fx,Fy,Fz]:   0.0,   0.0,   0.0 N")

        ttk.Label(out_frame, textvariable=self.var_out_force, font=("Consolas", 9, "bold")).pack(anchor="w")
        ttk.Label(out_frame, textvariable=self.var_out_torque, font=("Consolas", 9, "bold")).pack(anchor="w", pady=(2, 0))
        ttk.Label(out_frame, textvariable=self.var_ball_force, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))

        self.var_info.set(
            f"Span: {self.geo_ctx['span']:.3f} m  |  Front Offset: dX={self.geo_ctx['fx_pos']:.3f}, dY={self.geo_ctx['fy_pos']:.3f}, dZ={self.geo_ctx['fz_pos'] - 0.2:.3f} m"
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_sensors()
        self.root.mainloop()

    def _add_stepper_row(self, parent, row: int, label: str, tk_var: tk.StringVar, step: float, callback):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Button(parent, text="-", width=3, command=lambda: self._nudge_var(tk_var, -step, callback)).grid(row=row, column=1, padx=2)
        entry = ttk.Entry(parent, textvariable=tk_var, width=9, justify="center")
        entry.grid(row=row, column=2, padx=2)
        entry.bind("<Return>", callback)
        ttk.Button(parent, text="+", width=3, command=lambda: self._nudge_var(tk_var, step, callback)).grid(row=row, column=3, padx=2)

    def _zero_forces(self):
        self.var_fx.set("0")
        self.var_fy.set("0")
        self.var_fz.set("0")
        self.update_forces()

    def _poll_sensors(self):
        if not self.running:
            self.root.destroy()
            return

        with self.lock:
            rf = self.filt_rear_force.copy()
            rt = self.filt_rear_torque.copy()
            bf = self.filt_ball_force.copy()

        self.var_out_force.set(f"Rear Force  [Fx,Fy,Fz]: {rf[0]:7.1f}, {rf[1]:7.1f}, {rf[2]:7.1f} N")
        self.var_out_torque.set(f"Rear Torque [Mx,My,Mz]: {rt[0]:6.2f}, {rt[1]:6.2f}, {rt[2]:6.2f} Nm")
        self.var_ball_force.set(f"Ball Joint  [Fx,Fy,Fz]: {bf[0]:7.1f}, {bf[1]:7.1f}, {bf[2]:7.1f} N")

        self.root.after(100, self._poll_sensors)

    def _on_close(self):
        self.running = False
        self.root.destroy()


if __name__ == "__main__":
    DrawbarSimApp()
