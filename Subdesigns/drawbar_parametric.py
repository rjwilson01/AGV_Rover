import math
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import mujoco
import mujoco.viewer
import numpy as np


def build_drawbar_xml(span: float, deg1: float, deg2: float, axis: str = "0 0 1") -> str:
    """Computes exact triangle geometry so Bar 1 and Bar 2 meet at the ball joint without pre-stress."""
    a1 = math.radians(deg1)
    a2 = math.radians(deg2)
    a3 = math.pi - (a1 + a2)

    if span <= 0.05:
        raise ValueError("DRAWBAR_SPAN must be greater than 0.05 m.")
    if deg1 <= 0.5 or deg2 <= 0.5:
        raise ValueError("ANGLE1_DEG and ANGLE2_DEG must each be greater than 0.5 deg to avoid collinear lockup.")
    if a3 <= 0.05:
        raise ValueError("ANGLE1_DEG + ANGLE2_DEG must be less than 175 deg so the triangle closes.")

    # Law of Sines: L1 / sin(a2) = L2 / sin(a1) = span / sin(a3)
    L1 = span * math.sin(a2) / math.sin(a3)
    L2 = span * math.sin(a1) / math.sin(a3)

    # Ball joint coordinates relative to Pivot 1 (at x=0, y=0)
    bx = L1 * math.cos(a1)
    by = L1 * math.sin(a1)

    return f"""
<mujoco model="parametric_drawbar_ball_joint">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast">
    <flag warmstart="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <scale forcewidth="0.025" contactwidth="0.04" contactheight="0.04" jointlength="0.12" jointwidth="0.012"/>
    <rgba force="1 0.2 0.2 1"/>
  </visual>

  <worldbody>
    <light pos="0.4 -1.2 2.0" dir="0 0.5 -0.8" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 -0.3" size="3 3 0.1" rgba="0.88 0.88 0.88 1"/>

    <!-- REAR BASE BLOCK (Fixed Reaction Anchor holding Pivot 2) -->
    <body name="rear_block" pos="{span:.6f} 0 0.2">
      <geom name="rear_housing" type="box" pos="0.06 0 0" size="0.06 0.08 0.06" rgba="0.25 0.3 0.35 1" mass="50"/>
      
      <!-- Pivot 2: True 1-DoF Hinge mounted on the Rear Block -->
      <body name="pivot2_link" pos="0 0 0">
        <joint name="pivot2_hinge" type="hinge" axis="{axis}" damping="2.0"/>
        <site name="pivot2_sensor_site" pos="0 0 0" size="0.014" rgba="0 1 0 1"/>
        <geom name="pivot2_pin" type="cylinder" fromto="0 0 -0.05 0 0 0.05" size="0.018" rgba="0.8 0.3 0.2 1" mass="0.2"/>

        <!-- Bar 2 extends from Pivot 2 to the Ball Joint -->
        <geom name="bar2_geom" type="capsule" fromto="0 0 0  {bx - span:.6f} {by:.6f} 0" size="0.018" rgba="0.2 0.75 0.55 1" mass="1.0"/>
        <site name="ball_end_bar2" pos="{bx - span:.6f} {by:.6f} 0" size="0.014" rgba="0 1 1 1"/>
      </body>
    </body>

    <!-- FRONT LOAD BLOCK (Guided along X-axis track with high damping to prevent free-flying whip) -->
    <body name="front_block" pos="0 0 0.2">
      <joint name="front_slide_x" type="slide" axis="1 0 0" damping="25.0"/>
      <geom name="front_housing" type="box" pos="-0.06 0 0" size="0.06 0.08 0.06" rgba="0.35 0.35 0.45 1" mass="5.0"/>
      <site name="force_input_site" pos="-0.06 0 0" size="0.016" rgba="1 0 0 1"/>

      <!-- Pivot 1: True 1-DoF Hinge mounted on the Front Block -->
      <body name="pivot1_link" pos="0 0 0">
        <joint name="pivot1_hinge" type="hinge" axis="{axis}" damping="2.0"/>
        <site name="pivot1_sensor_site" pos="0 0 0" size="0.014" rgba="1 0.6 0 1"/>
        <geom name="pivot1_pin" type="cylinder" fromto="0 0 -0.05 0 0 0.05" size="0.018" rgba="0.8 0.3 0.2 1" mass="0.2"/>

        <!-- Bar 1 extends from Pivot 1 to the Ball Joint -->
        <geom name="bar1_geom" type="capsule" fromto="0 0 0  {bx:.6f} {by:.6f} 0" size="0.018" rgba="0.2 0.6 0.85 1" mass="1.0"/>

        <!-- CENTRAL 3-DoF BALL JOINT -->
        <body name="ball_socket_body" pos="{bx:.6f} {by:.6f} 0">
          <joint name="center_ball_joint" type="ball" damping="0.5"/>
          <geom name="ball_visual" type="sphere" pos="0 0 0" size="0.032" rgba="0.95 0.75 0.1 1" mass="0.3"/>
          <site name="ball_end_bar1" pos="0 0 0" size="0.014" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <equality>
    <weld name="ball_socket_weld" site1="ball_end_bar1" site2="ball_end_bar2" solref="0.005 1" solimp="0.95 0.99 0.001"/>
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


class DrawbarSimApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.reload_requested = False
        self.viewer = None

        # Default geometry & force state
        self.span = 0.80
        self.deg1 = 20.0
        self.deg2 = 35.0
        self.axis = "0 0 1"
        self.forces = [200.0, 0.0, 0.0]

        # Build initial model
        self.model = mujoco.MjModel.from_xml_string(
            build_drawbar_xml(self.span, self.deg1, self.deg2, self.axis)
        )
        self.data = mujoco.MjData(self.model)
        self._apply_forces_locked()

        # Smoothed sensor snapshot for clean GUI readout
        self.rear_force = np.zeros(3)
        self.rear_torque = np.zeros(3)
        self.ball_force = np.zeros(3)

        # Start physics + MuJoCo viewer thread
        self.sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.sim_thread.start()

        self._build_gui()

    def _apply_forces_locked(self):
        self.data.actuator("front_force_X").ctrl = self.forces[0]
        self.data.actuator("front_force_Y").ctrl = self.forces[1]
        self.data.actuator("front_force_Z").ctrl = self.forces[2]

    def _run_simulation(self):
        while self.running:
            with self.lock:
                self.reload_requested = False
                current_model = self.model
                current_data = self.data

            with mujoco.viewer.launch_passive(current_model, current_data) as viewer:
                self.viewer = viewer
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

                while self.running and viewer.is_running() and not self.reload_requested:
                    step_start = time.time()
                    with self.lock:
                        mujoco.mj_step(self.model, self.data)
                        # Exponential moving average filter to suppress high-frequency constraint chatter
                        alpha = 0.05
                        self.rear_force = (1 - alpha) * self.rear_force + alpha * self.data.sensor("rear_pivot2_force").data
                        self.rear_torque = (1 - alpha) * self.rear_torque + alpha * self.data.sensor("rear_pivot2_torque").data
                        self.ball_force = (1 - alpha) * self.ball_force + alpha * self.data.sensor("ball_joint_force").data
                        viewer.sync()

                    elapsed = time.time() - step_start
                    if elapsed < self.model.opt.timestep:
                        time.sleep(self.model.opt.timestep - elapsed)

            # If viewer was closed by user X button rather than a geometry reload, exit app
            if not self.reload_requested:
                self.running = False

    def rebuild_geometry(self, event=None, trigger_reload=True):
        try:
            new_span = float(self.var_span.get())
            new_deg1 = float(self.var_deg1.get())
            new_deg2 = float(self.var_deg2.get())
            new_axis = "0 0 1" if self.var_axis.get() == "Yaw (Z)" else "0 1 0"

            xml_str = build_drawbar_xml(new_span, new_deg1, new_deg2, new_axis)
            new_model = mujoco.MjModel.from_xml_string(xml_str)
            new_data = mujoco.MjData(new_model)

            with self.lock:
                self.span = new_span
                self.deg1 = new_deg1
                self.deg2 = new_deg2
                self.axis = new_axis
                self.model = new_model
                self.data = new_data
                self.rear_force = np.zeros(3)
                self.rear_torque = np.zeros(3)
                self.ball_force = np.zeros(3)
                self._apply_forces_locked()
                if trigger_reload:
                    self.reload_requested = True

            # Update calculated bar lengths display
            a1, a2 = math.radians(new_deg1), math.radians(new_deg2)
            a3 = math.pi - (a1 + a2)
            L1 = new_span * math.sin(a2) / math.sin(a3)
            L2 = new_span * math.sin(a1) / math.sin(a3)
            self.var_lengths.set(f"Bar 1 (L1): {L1:.3f} m   |   Bar 2 (L2): {L2:.3f} m")

        except Exception as exc:
            messagebox.showerror("Invalid Geometry", str(exc))

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

    def _nudge_var(self, tk_var: tk.StringVar, delta: float, callback):
        try:
            val = float(tk_var.get()) + delta
            tk_var.set(f"{val:.3g}")
            callback()
        except ValueError:
            pass

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Drawbar Ball-Joint Control")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        pad = {"padx": 6, "pady": 4}

        # --- GEOMETRY FRAME ---
        geo_frame = ttk.LabelFrame(self.root, text="1. Drawbar Geometry (Press Enter or Rebuild)", padding=8)
        geo_frame.pack(fill="x", **pad)

        self.var_span = tk.StringVar(value=f"{self.span:.2f}")
        self.var_deg1 = tk.StringVar(value=f"{self.deg1:.1f}")
        self.var_deg2 = tk.StringVar(value=f"{self.deg2:.1f}")
        self.var_axis = tk.StringVar(value="Yaw (Z)")
        self.var_lengths = tk.StringVar(value="")

        self._add_stepper_row(geo_frame, 0, "Drawbar Span (m):", self.var_span, 0.05, self.rebuild_geometry)
        self._add_stepper_row(geo_frame, 1, "Front Angle 1 (deg):", self.var_deg1, 2.5, self.rebuild_geometry)
        self._add_stepper_row(geo_frame, 2, "Rear Angle 2 (deg):", self.var_deg2, 2.5, self.rebuild_geometry)

        ttk.Label(geo_frame, text="Pivot Axis:").grid(row=3, column=0, sticky="w", pady=3)
        axis_box = ttk.Combobox(geo_frame, textvariable=self.var_axis, values=["Yaw (Z)", "Pitch (Y)"], state="readonly", width=12)
        axis_box.grid(row=3, column=1, columnspan=3, sticky="ew", pady=3)
        axis_box.bind("<<ComboboxSelected>>", self.rebuild_geometry)

        ttk.Button(geo_frame, text="Rebuild Geometry / Reset Pose", command=self.rebuild_geometry).grid(
            row=4, column=0, columnspan=4, sticky="ew", pady=(6, 2)
        )
        ttk.Label(geo_frame, textvariable=self.var_lengths, foreground="#005a9e").grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(2, 0)
        )

        # --- INPUT FORCE FRAME ---
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

        # --- LIVE SENSOR READOUT FRAME ---
        out_frame = ttk.LabelFrame(self.root, text="3. Transmitted Rear Pivot 2 Readout (Filtered)", padding=8)
        out_frame.pack(fill="x", **pad)

        self.var_out_force = tk.StringVar(value="Rear Force  [Fx,Fy,Fz]:    0.0,    0.0,    0.0 N")
        self.var_out_torque = tk.StringVar(value="Rear Torque [Mx,My,Mz]:   0.00,   0.00,   0.00 Nm")
        self.var_ball_force = tk.StringVar(value="Ball Joint  [Fx,Fy,Fz]:    0.0,    0.0,    0.0 N")

        ttk.Label(out_frame, textvariable=self.var_out_force, font=("Consolas", 9, "bold")).pack(anchor="w")
        ttk.Label(out_frame, textvariable=self.var_out_torque, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))
        ttk.Label(out_frame, textvariable=self.var_ball_force, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))

        # Populate initial bar length display without triggering a redundant viewer restart
        self.rebuild_geometry(trigger_reload=False)
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
            rf = self.rear_force.copy()
            rt = self.rear_torque.copy()
            bf = self.ball_force.copy()

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
