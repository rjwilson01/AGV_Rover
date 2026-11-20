import math
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
import mujoco
import mujoco.viewer
import numpy as np


def clamp_params(L1: float, L2: float, deg1: float, deg2: float):
    """Clamps bar lengths and end angles independently within safe physical bounds."""
    L1 = max(0.05, min(5.0, L1))
    L2 = max(0.05, min(5.0, L2))
    deg1 = max(-89.0, min(89.0, deg1))
    deg2 = max(-89.0, min(89.0, deg2))
    return L1, L2, deg1, deg2


def compute_kinematics(L1: float, L2: float, deg1: float, deg2: float, axis: str):
    """
    Computes exact 3D positions so Bar 1 (from Front Pivot 1 at angle deg1)
    and Bar 2 (from Rear Pivot 2 at angle deg2) meet cleanly at the Ball Joint
    with zero pre-stress, allowing the Front Node to shift laterally/vertically.
    Rear Pivot 2 sits at the origin [0, 0, 0.25].
    """
    a1 = math.radians(deg1)
    a2 = math.radians(deg2)

    if axis == "0 0 1":  # Yaw (X-Y plane)
        bx = -L2 * math.cos(a2)
        by = L2 * math.sin(a2)
        bz = 0.0

        fx = bx - L1 * math.cos(a1)
        fy = by - L1 * math.sin(a1)
        fz = 0.0

        bar2_tip = (bx, by, 0.0)
        bar1_tip = (L1 * math.cos(a1), L1 * math.sin(a1), 0.0)
    else:  # Pitch (X-Z plane)
        bx = -L2 * math.cos(a2)
        by = 0.0
        bz = L2 * math.sin(a2)

        fx = bx - L1 * math.cos(a1)
        fy = 0.0
        fz = bz - L1 * math.sin(a1)

        bar2_tip = (bx, 0.0, bz)
        bar1_tip = (L1 * math.cos(a1), 0.0, L1 * math.sin(a1))

    return (fx, fy, fz), (bx, by, bz), bar1_tip, bar2_tip


def build_drawbar_xml(L1: float, L2: float, deg1: float, deg2: float, axis: str = "0 0 1") -> str:
    """Generates the MuJoCo MJCF XML with compact spherical nodes and miniature hinges so 3D vectors stay unobstructed."""
    L1, L2, deg1, deg2 = clamp_params(L1, L2, deg1, deg2)
    (fx, fy, fz), _, bar1_tip, bar2_tip = compute_kinematics(L1, L2, deg1, deg2, axis)
    z_base = 0.25

    pin_fromto = "0 0 -0.018 0 0 0.018" if axis == "0 0 1" else "0 -0.018 0 0 0.018 0"

    return f"""<mujoco model="parametric_drawbar_ball_joint">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast">
    <flag warmstart="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <scale jointlength="0.06" jointwidth="0.005"/>
  </visual>

  <worldbody>
    <light pos="-0.2 -1.2 2.0" dir="0.1 0.5 -0.8" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="0.90 0.90 0.90 1"/>

    <!-- REAR ANCHOR NODE (Compact Sphere holding Miniature Pivot 2 at World [0, 0, {z_base}]) -->
    <body name="rear_block" pos="0 0 {z_base:.6f}">
      <geom name="rear_node" type="sphere" pos="0 0 0" size="0.022" rgba="0.25 0.30 0.35 0.85" mass="50"/>

      <!-- Pivot 2: Miniature 1-DoF Hinge locked at Rear Angle 2 -->
      <body name="pivot2_link" pos="0 0 0">
        <joint name="pivot2_hinge" type="hinge" axis="{axis}" damping="5.0"/>
        <site name="pivot2_sensor_site" pos="0 0 0" size="0.008" rgba="0 1 0 1"/>
        <geom name="pivot2_pin" type="cylinder" fromto="{pin_fromto}" size="0.007" rgba="0.85 0.30 0.20 1" mass="0.1"/>

        <!-- Bar 2 extends from Rear Pivot 2 to Central Ball Joint -->
        <geom name="bar2_geom" type="capsule" fromto="0 0 0  {bar2_tip[0]:.6f} {bar2_tip[1]:.6f} {bar2_tip[2]:.6f}" size="0.011" rgba="0.20 0.72 0.52 1" mass="0.8"/>
        <site name="ball_end_bar2" pos="{bar2_tip[0]:.6f} {bar2_tip[1]:.6f} {bar2_tip[2]:.6f}" size="0.008" rgba="0 1 1 1"/>
      </body>
    </body>

    <!-- FRONT INPUT NODE (Compact Sphere offset automatically for Angle 1 & Angle 2) -->
    <body name="front_block" pos="{fx:.6f} {fy:.6f} {z_base + fz:.6f}">
      <!-- Constrained to translation only (X, Y, Z) so chassis orientation stays fixed -->
      <joint name="front_slide_x" type="slide" axis="1 0 0" damping="15.0"/>
      <joint name="front_slide_y" type="slide" axis="0 1 0" damping="15.0"/>
      <joint name="front_slide_z" type="slide" axis="0 0 1" damping="15.0"/>

      <geom name="front_node" type="sphere" pos="0 0 0" size="0.022" rgba="0.35 0.35 0.45 0.85" mass="1.5"/>
      <site name="force_input_site" pos="0 0 0" size="0.008" rgba="1 0 0 1"/>

      <!-- Pivot 1: Miniature 1-DoF Hinge locked at Front Angle 1 -->
      <body name="pivot1_link" pos="0 0 0">
        <joint name="pivot1_hinge" type="hinge" axis="{axis}" damping="5.0"/>
        <site name="pivot1_sensor_site" pos="0 0 0" size="0.008" rgba="1 0.6 0 1"/>
        <geom name="pivot1_pin" type="cylinder" fromto="{pin_fromto}" size="0.007" rgba="0.85 0.30 0.20 1" mass="0.1"/>

        <!-- Bar 1 extends from Front Pivot 1 to Central Ball Joint -->
        <geom name="bar1_geom" type="capsule" fromto="0 0 0  {bar1_tip[0]:.6f} {bar1_tip[1]:.6f} {bar1_tip[2]:.6f}" size="0.011" rgba="0.20 0.58 0.85 1" mass="0.8"/>

        <!-- CENTRAL 3-DoF BALL JOINT -->
        <body name="ball_socket_body" pos="{bar1_tip[0]:.6f} {bar1_tip[1]:.6f} {bar1_tip[2]:.6f}">
          <joint name="center_ball_joint" type="ball" damping="0.2"/>
          <geom name="ball_visual" type="sphere" pos="0 0 0" size="0.020" rgba="0.95 0.75 0.10 0.90" mass="0.2"/>
          <site name="ball_end_bar1" pos="0 0 0" size="0.008" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- EQUALITY CONSTRAINTS: Lock both end hinge angles and close loop at the ball joint -->
  <equality>
    <joint name="lock_pivot1_angle" joint1="pivot1_hinge" polycoef="0 0 0 0 0" solref="0.002 1" solimp="0.98 0.999 0.0005"/>
    <joint name="lock_pivot2_angle" joint1="pivot2_hinge" polycoef="0 0 0 0 0" solref="0.002 1" solimp="0.98 0.999 0.0005"/>
    <weld name="ball_socket_weld" site1="ball_end_bar1" site2="ball_end_bar2" solref="0.002 1" solimp="0.98 0.999 0.0005"/>
  </equality>

  <actuator>
    <general name="front_force_X" site="force_input_site" gear="1 0 0 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
    <general name="front_force_Y" site="force_input_site" gear="0 1 0 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
    <general name="front_force_Z" site="force_input_site" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="-5000 5000"/>
  </actuator>

  <sensor>
    <force  name="front_pivot1_force"  site="pivot1_sensor_site"/>
    <torque name="front_pivot1_torque" site="pivot1_sensor_site"/>
    <force  name="ball_joint_force"    site="ball_end_bar1"/>
    <torque name="ball_joint_torque"   site="ball_end_bar1"/>
    <force  name="rear_pivot2_force"   site="pivot2_sensor_site"/>
    <torque name="rear_pivot2_torque"  site="pivot2_sensor_site"/>
  </sensor>
</mujoco>
"""


def add_vector_arrow(scn, origin: np.ndarray, vec: np.ndarray, scale: float, rgba: list, min_len: float = 0.025):
    """Draws a crisp 3D arrow in MuJoCo user_scn from origin in the direction of vec."""
    mag = float(np.linalg.norm(vec))
    if mag < 1e-3 or scn.ngeom >= scn.maxgeom:
        return

    length = min(0.65, max(min_len, mag * scale))
    end = origin + (vec / mag) * length

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.008,
        from_=origin,
        to=end,
    )
    scn.ngeom += 1


class DrawbarSimApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.rebuild_requested = False
        self.viewer = None

        # Default geometry & force state
        self.L1 = 0.40
        self.L2 = 0.40
        self.deg1 = 20.0
        self.deg2 = 0.0
        self.axis = "0 0 1"
        self.forces = [200.0, 0.0, 0.0]

        self.model = self._compile_model()
        self.data = mujoco.MjData(self.model)
        self._apply_forces_locked()

        # Smoothed world-frame sensor states
        self.rear_force_world = np.zeros(3)
        self.rear_torque_world = np.zeros(3)
        self.ball_force_world = np.zeros(3)

        self.sim_thread = threading.Thread(target=self._sim_supervisor, daemon=True)
        self.sim_thread.start()

        self._build_gui()

    def _compile_model(self) -> mujoco.MjModel:
        xml_str = build_drawbar_xml(self.L1, self.L2, self.deg1, self.deg2, self.axis)
        return mujoco.MjModel.from_xml_string(xml_str)

    def _apply_forces_locked(self):
        self.data.actuator("front_force_X").ctrl = self.forces[0]
        self.data.actuator("front_force_Y").ctrl = self.forces[1]
        self.data.actuator("front_force_Z").ctrl = self.forces[2]

    def _sim_supervisor(self):
        """Runs the viewer loop and cleanly relaunches it whenever geometry changes."""
        while self.running:
            with self.lock:
                self.rebuild_requested = False
                model = self.model
                data = self.data
                self.rear_force_world[:] = 0.0
                self.rear_torque_world[:] = 0.0
                self.ball_force_world[:] = 0.0

            with mujoco.viewer.launch_passive(model, data) as viewer:
                self.viewer = viewer
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

                # Configure clean camera framing
                viewer.cam.lookat[:] = [-0.30, 0.0, 0.25]
                viewer.cam.distance = 1.35
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -35.0

                alpha = 0.04
                while self.running and viewer.is_running() and not self.rebuild_requested:
                    step_start = time.time()
                    with self.lock:
                        mujoco.mj_step(model, data)

                        # Rotate site-local sensor readings into World Frame
                        R_p2 = data.site("pivot2_sensor_site").xmat.reshape(3, 3)
                        R_ball = data.site("ball_end_bar1").xmat.reshape(3, 3)

                        f_rear = R_p2 @ data.sensor("rear_pivot2_force").data
                        t_rear = R_p2 @ data.sensor("rear_pivot2_torque").data
                        f_ball = R_ball @ data.sensor("ball_joint_force").data

                        self.rear_force_world = (1.0 - alpha) * self.rear_force_world + alpha * f_rear
                        self.rear_torque_world = (1.0 - alpha) * self.rear_torque_world + alpha * t_rear
                        self.ball_force_world = (1.0 - alpha) * self.ball_force_world + alpha * f_ball

                        # Render 3D vector arrows directly in user_scn
                        scn = viewer.user_scn
                        scn.ngeom = 0

                        p_front = data.site("force_input_site").xpos.copy()
                        p_ball = data.site("ball_end_bar1").xpos.copy()
                        p_rear = data.site("pivot2_sensor_site").xpos.copy()

                        f_app = np.array(self.forces, dtype=float)

                        # 1. Red Arrow: Applied Force at Front Node
                        add_vector_arrow(scn, p_front, f_app, scale=0.0012, rgba=[0.95, 0.15, 0.15, 0.95])
                        # 2. Yellow Arrow: Transmitted Force at Central Ball Joint
                        add_vector_arrow(scn, p_ball, self.ball_force_world, scale=0.0012, rgba=[0.95, 0.75, 0.05, 0.95])
                        # 3. Green Arrow: Reaction Force at Rear Pivot 2
                        add_vector_arrow(scn, p_rear, self.rear_force_world, scale=0.0012, rgba=[0.10, 0.80, 0.25, 0.95])
                        # 4. Purple Arrow: Torque Normal Vector (Right-Hand Rule Axis) at Rear Pivot 2
                        add_vector_arrow(scn, p_rear, self.rear_torque_world, scale=0.0040, rgba=[0.75, 0.20, 0.90, 0.95])

                        viewer.sync()

                    elapsed = time.time() - step_start
                    if elapsed < model.opt.timestep:
                        time.sleep(model.opt.timestep - elapsed)

                if not viewer.is_running() and not self.rebuild_requested:
                    self.running = False

    def rebuild_geometry(self, event=None):
        try:
            raw_L1 = float(self.var_L1.get())
            raw_L2 = float(self.var_L2.get())
            raw_deg1 = float(self.var_deg1.get())
            raw_deg2 = float(self.var_deg2.get())
            new_axis = "0 0 1" if self.var_axis.get() == "Yaw (Z)" else "0 1 0"

            new_L1, new_L2, new_deg1, new_deg2 = clamp_params(raw_L1, raw_L2, raw_deg1, raw_deg2)

            self.var_L1.set(f"{new_L1:.2f}")
            self.var_L2.set(f"{new_L2:.2f}")
            self.var_deg1.set(f"{new_deg1:.1f}")
            self.var_deg2.set(f"{new_deg2:.1f}")

            new_model = mujoco.MjModel.from_xml_string(
                build_drawbar_xml(new_L1, new_L2, new_deg1, new_deg2, new_axis)
            )
            new_data = mujoco.MjData(new_model)

            with self.lock:
                self.L1 = new_L1
                self.L2 = new_L2
                self.deg1 = new_deg1
                self.deg2 = new_deg2
                self.axis = new_axis
                self.model = new_model
                self.data = new_data
                self._apply_forces_locked()
                self.rebuild_requested = True

            (fx, fy, fz), _, _, _ = compute_kinematics(new_L1, new_L2, new_deg1, new_deg2, new_axis)
            span = math.sqrt(fx * fx + fy * fy + fz * fz)
            self.var_span_info.set(
                f"Span: {span:.3f} m  |  Front Offset: dX={fx:.3f}, dY={fy:.3f}, dZ={fz:.3f} m"
            )

        except ValueError:
            pass

    def export_xml(self):
        path = filedialog.asksaveasfilename(
            title="Save MuJoCo Drawbar XML",
            defaultextension=".xml",
            initialfile="drawbar_ball_joint.xml",
            filetypes=[("MuJoCo XML", "*.xml"), ("All Files", "*.*")],
        )
        if path:
            xml_str = build_drawbar_xml(self.L1, self.L2, self.deg1, self.deg2, self.axis)
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

    def _nudge_var(self, tk_var: tk.StringVar, delta: float, callback, fmt: str = "{:.1f}"):
        try:
            val = float(tk_var.get()) + delta
            tk_var.set(fmt.format(val))
            callback()
        except ValueError:
            pass

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Drawbar Ball-Joint Vector Control")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        pad = {"padx": 6, "pady": 4}

        # --- 1. GEOMETRY FRAME ---
        geo_frame = ttk.LabelFrame(self.root, text="1. Drawbar Geometry (Locked Angles, -89° to +89°)", padding=8)
        geo_frame.pack(fill="x", **pad)

        self.var_L1 = tk.StringVar(value=f"{self.L1:.2f}")
        self.var_L2 = tk.StringVar(value=f"{self.L2:.2f}")
        self.var_deg1 = tk.StringVar(value=f"{self.deg1:.1f}")
        self.var_deg2 = tk.StringVar(value=f"{self.deg2:.1f}")
        self.var_axis = tk.StringVar(value="Yaw (Z)")
        self.var_span_info = tk.StringVar(value="")

        self._add_stepper_row(geo_frame, 0, "Front Bar 1 Length (m):", self.var_L1, 0.05, self.rebuild_geometry, "{:.2f}")
        self._add_stepper_row(geo_frame, 1, "Rear Bar 2 Length (m):", self.var_L2, 0.05, self.rebuild_geometry, "{:.2f}")
        self._add_stepper_row(geo_frame, 2, "Front Angle 1 (°):", self.var_deg1, 5.0, self.rebuild_geometry, "{:.1f}")
        self._add_stepper_row(geo_frame, 3, "Rear Angle 2 (°):", self.var_deg2, 5.0, self.rebuild_geometry, "{:.1f}")

        ttk.Label(geo_frame, text="Pivot Axis:").grid(row=4, column=0, sticky="w", pady=3)
        axis_box = ttk.Combobox(geo_frame, textvariable=self.var_axis, values=["Yaw (Z)", "Pitch (Y)"], state="readonly", width=12)
        axis_box.grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        axis_box.bind("<<ComboboxSelected>>", self.rebuild_geometry)

        geo_btns = ttk.Frame(geo_frame)
        geo_btns.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 2))
        ttk.Button(geo_btns, text="Rebuild / Reset Pose", command=self.rebuild_geometry).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(geo_btns, text="Export XML...", command=self.export_xml).pack(side="left", expand=True, fill="x", padx=(3, 0))

        ttk.Label(geo_frame, textvariable=self.var_span_info, foreground="#005a9e").grid(
            row=6, column=0, columnspan=4, sticky="w", pady=(2, 0)
        )

        # --- 2. INPUT FORCE FRAME ---
        frc_frame = ttk.LabelFrame(self.root, text="2. Front Input Forces (Newtons)", padding=8)
        frc_frame.pack(fill="x", **pad)

        self.var_fx = tk.StringVar(value=f"{self.forces[0]:.0f}")
        self.var_fy = tk.StringVar(value=f"{self.forces[1]:.0f}")
        self.var_fz = tk.StringVar(value=f"{self.forces[2]:.0f}")

        self._add_stepper_row(frc_frame, 0, "Fx (Axial Push/Pull):", self.var_fx, 50.0, self.update_forces, "{:.0f}")
        self._add_stepper_row(frc_frame, 1, "Fy (Lateral Side):", self.var_fy, 50.0, self.update_forces, "{:.0f}")
        self._add_stepper_row(frc_frame, 2, "Fz (Vertical Load):", self.var_fz, 50.0, self.update_forces, "{:.0f}")

        btn_row = ttk.Frame(frc_frame)
        btn_row.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Apply Forces", command=self.update_forces).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btn_row, text="Zero All Forces", command=self._zero_forces).pack(side="left", expand=True, fill="x", padx=(3, 0))

        # --- 3. 3D VECTOR LEGEND & READOUT FRAME ---
        out_frame = ttk.LabelFrame(self.root, text="3. 3D Vectors & World-Frame Readout", padding=8)
        out_frame.pack(fill="x", **pad)

        legend = (
            "Arrows: [RED] Applied Front Force   [YELLOW] Ball Joint Force"
            "        [GREEN] Rear Reaction Force  [PURPLE] Rear Torque Normal"
        )
        ttk.Label(out_frame, text=legend, foreground="#333333", font=("Consolas", 8)).pack(anchor="w", pady=(0, 4))

        self.var_out_force = tk.StringVar(value="Rear Force  [Fx,Fy,Fz]:    0.0,    0.0,    0.0 N")
        self.var_out_torque = tk.StringVar(value="Rear Torque [Mx,My,Mz]:   0.00,   0.00,   0.00 Nm")
        self.var_ball_force = tk.StringVar(value="Ball Joint  [Fx,Fy,Fz]:    0.0,    0.0,    0.0 N")

        ttk.Label(out_frame, textvariable=self.var_out_force, font=("Consolas", 9, "bold")).pack(anchor="w")
        ttk.Label(out_frame, textvariable=self.var_out_torque, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))
        ttk.Label(out_frame, textvariable=self.var_ball_force, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))

        (fx, fy, fz), _, _, _ = compute_kinematics(self.L1, self.L2, self.deg1, self.deg2, self.axis)
        span = math.sqrt(fx * fx + fy * fy + fz * fz)
        self.var_span_info.set(f"Span: {span:.3f} m  |  Front Offset: dX={fx:.3f}, dY={fy:.3f}, dZ={fz:.3f} m")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_sensors()
        self.root.mainloop()

    def _add_stepper_row(self, parent, row: int, label: str, tk_var: tk.StringVar, step: float, callback, fmt: str):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Button(parent, text="-", width=3, command=lambda: self._nudge_var(tk_var, -step, callback, fmt)).grid(row=row, column=1, padx=2)
        entry = ttk.Entry(parent, textvariable=tk_var, width=9, justify="center")
        entry.grid(row=row, column=2, padx=2)
        entry.bind("<Return>", callback)
        ttk.Button(parent, text="+", width=3, command=lambda: self._nudge_var(tk_var, step, callback, fmt)).grid(row=row, column=3, padx=2)

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
            rf = self.rear_force_world.copy()
            rt = self.rear_torque_world.copy()
            bf = self.ball_force_world.copy()

        self.var_out_force.set(f"Rear Force  [Fx,Fy,Fz]: {rf[0]:7.1f}, {rf[1]:7.1f}, {rf[2]:7.1f} N")
        self.var_out_torque.set(f"Rear Torque [Mx,My,Mz]: {rt[0]:7.2f}, {rt[1]:7.2f}, {rt[2]:7.2f} Nm")
        self.var_ball_force.set(f"Ball Joint  [Fx,Fy,Fz]: {bf[0]:7.1f}, {bf[1]:7.1f}, {bf[2]:7.1f} N")

        self.root.after(100, self._poll_sensors)

    def _on_close(self):
        self.running = False
        if self.viewer is not None:
            self.viewer.close()
        self.root.destroy()


if __name__ == "__main__":
    DrawbarSimApp()
