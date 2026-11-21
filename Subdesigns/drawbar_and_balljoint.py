import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from jinja2 import Environment, FileSystemLoader
import mujoco
import mujoco.viewer
import numpy as np

# Template directory and file path relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "drawbar_and_balljoint"
TEMPLATE_NAME = "drawbar.xml.j2"
TEMPLATE_PATH = TEMPLATE_DIR / TEMPLATE_NAME

# Create Jinja2 environment reading directly from ./drawbar_and_balljoint/
jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def compute_geometry_context(l1: float, l2: float, deg1: float, deg2: float, axis_mode: str, lock_front: bool) -> dict:
    """Computes exact world-frame node positions and Euler angles for the Jinja2 template."""
    if l1 <= 0.05 or l2 <= 0.05:
        raise ValueError("Bar lengths must each be greater than 0.05 m.")
    if not (-89.0 <= deg1 <= 89.0 and -89.0 <= deg2 <= 89.0):
        raise ValueError("Angles must stay within -89.0° to +89.0°.")

    a1 = math.radians(deg1)
    a2 = math.radians(deg2)

    if axis_mode == "Yaw (Z)":
        hinge_axis = "0 0 1"
        pin_fromto = "0 0 -0.018 0 0 0.018"
        rear_euler = f"0 0 {deg2:.6f}"
        front_euler = f"0 0 {deg1:.6f}"

        bx = -l2 * math.cos(a2)
        by = -l2 * math.sin(a2)
        bz = 0.0

        fx_pos = bx - l1 * math.cos(a1)
        fy_pos = by - l1 * math.sin(a1)
        fz_pos = 0.2
    else:
        hinge_axis = "0 1 0"
        pin_fromto = "0 -0.018 0 0 0.018 0"
        rear_euler = f"0 {deg2:.6f} 0"
        front_euler = f"0 {deg1:.6f} 0"

        bx = -l2 * math.cos(a2)
        by = 0.0
        bz = l2 * math.sin(a2)

        fx_pos = bx - l1 * math.cos(a1)
        fy_pos = 0.0
        fz_pos = 0.2 + bz + l1 * math.sin(a1)

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
        "lock_front": lock_front,
    }


def render_drawbar_xml(l1: float, l2: float, deg1: float, deg2: float, axis_mode: str, lock_front: bool) -> str:
    """Loads ./drawbar_and_balljoint/drawbar.xml.j2 from disk and renders the MJCF XML in memory."""
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            f"Template file not found:\n{TEMPLATE_PATH}\n\n"
            f"Ensure 'drawbar.xml.j2' is placed inside './drawbar_and_balljoint/'."
        )
    ctx = compute_geometry_context(l1, l2, deg1, deg2, axis_mode, lock_front)
    template = jinja_env.get_template(TEMPLATE_NAME)
    return template.render(**ctx)


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix whose local Z-axis aligns with unit vector `vec`."""
    z = vec / np.linalg.norm(vec)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def add_vector_arrow(scn, origin: np.ndarray, vec: np.ndarray, scale: float, rgba: list, shaft_radius: float = 0.0045):
    """Draws a crisp 3D arrow (shaft + conical arrowhead via mjGEOM_ARROW) inside the MuJoCo viewer."""
    mag = float(np.linalg.norm(vec))
    length = mag * scale
    if length < 1e-4 or scn.ngeom >= scn.maxgeom - 2:
        return

    length = min(length, 1.5)
    end = origin + (vec / mag) * length

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        shaft_radius,
        origin.astype(np.float64),
        end.astype(np.float64),
    )
    scn.ngeom += 1


def add_torque_ring(scn, origin: np.ndarray, torque_vec: np.ndarray, scale: float, rgba: list):
    """Draws a right-hand rotational ring and tangential arrowhead orbiting the torque vector shaft."""
    mag = float(np.linalg.norm(torque_vec))
    length = mag * scale
    if length < 1e-4:
        return

    length = min(length, 1.5)
    R = _rotation_matrix_from_z(torque_vec)
    axis_dir = R[:, 2]
    ring_center = origin + axis_dir * min(0.055, length * 0.45)
    ring_radius = 0.026
    segments = 16
    arc_angle = math.radians(300.0)

    pts = []
    for i in range(segments + 1):
        theta = (i / segments) * arc_angle
        local_pt = np.array([ring_radius * math.cos(theta), ring_radius * math.sin(theta), 0.0])
        pts.append(ring_center + R @ local_pt)

    for i in range(segments - 1):
        if scn.ngeom >= scn.maxgeom - 2:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.0025,
            pts[i].astype(np.float64),
            pts[i + 1].astype(np.float64),
        )
        scn.ngeom += 1

    # Final segment as an arrow showing right-hand rotational twist direction
    if scn.ngeom < scn.maxgeom - 2:
        g_tip = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g_tip,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            g_tip,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.0035,
            pts[-2].astype(np.float64),
            pts[-1].astype(np.float64),
        )
        scn.ngeom += 1


class DrawbarSimApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.rebuild_requested = False
        self.viewer = None

        # Default geometry & force parameters
        self.l1 = 0.40
        self.l2 = 0.40
        self.deg1 = 20.0
        self.deg2 = 0.0
        self.axis_mode = "Yaw (Z)"
        self.lock_front = False
        self.forces = [200.0, 0.0, 0.0]

        # Display scaling for 3D vectors
        self.force_scale = 0.0012
        self.torque_scale = 0.006

        # Build initial model from Jinja2 template
        self.current_xml = render_drawbar_xml(
            self.l1, self.l2, self.deg1, self.deg2, self.axis_mode, self.lock_front
        )
        self.model = mujoco.MjModel.from_xml_string(self.current_xml)
        self.data = mujoco.MjData(self.model)
        self._apply_forces_locked()

        # Filtered world-frame sensor telemetry
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

    def _read_world_sensors_locked(self):
        """Transforms local site sensor outputs into world coordinates."""
        id_rear = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rear_reaction_site")
        id_ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball_end_bar1")

        R_rear = self.data.site_xmat[id_rear].reshape(3, 3)
        R_ball = self.data.site_xmat[id_ball].reshape(3, 3)

        # MuJoCo force/torque sensors report force exerted ON parent body by child; negate for reaction
        f_rear_world = -(R_rear @ self.data.sensor("rear_pivot2_force").data)
        t_rear_world = -(R_rear @ self.data.sensor("rear_pivot2_torque").data)
        f_ball_world = -(R_ball @ self.data.sensor("ball_joint_force").data)

        alpha = 0.08
        self.filt_rear_force = (1.0 - alpha) * self.filt_rear_force + alpha * f_rear_world
        self.filt_rear_torque = (1.0 - alpha) * self.filt_rear_torque + alpha * t_rear_world
        self.filt_ball_force = (1.0 - alpha) * self.filt_ball_force + alpha * f_ball_world

    def _run_simulation(self):
        while self.running:
            with self.lock:
                self.rebuild_requested = False
                m = self.model
                d = self.data

            with mujoco.viewer.launch_passive(m, d) as viewer:
                self.viewer = viewer
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

                while self.running and viewer.is_running() and not self.rebuild_requested:
                    step_start = time.time()
                    with self.lock:
                        if self.rebuild_requested:
                            break
                        mujoco.mj_step(self.model, self.data)
                        self._read_world_sensors_locked()

                        # Update custom 3D vector overlay arrows in the user scene
                        scn = viewer.user_scn
                        scn.ngeom = 0

                        id_front = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "force_input_site")
                        id_ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball_end_bar1")
                        id_rear = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "rear_reaction_site")

                        p_front = self.data.site_xpos[id_front].copy()
                        p_ball = self.data.site_xpos[id_ball].copy()
                        p_rear = self.data.site_xpos[id_rear].copy()

                        f_in_vec = np.array(self.forces, dtype=float)
                        f_scale = self.force_scale
                        t_scale = self.torque_scale

                        # Red: Applied front force
                        add_vector_arrow(scn, p_front, f_in_vec, f_scale, [0.95, 0.15, 0.15, 0.95])
                        # Yellow: Ball joint transmitted force
                        add_vector_arrow(scn, p_ball, self.filt_ball_force, f_scale, [0.95, 0.78, 0.10, 0.95])
                        # Green: Rear reaction force
                        add_vector_arrow(scn, p_rear, self.filt_rear_force, f_scale, [0.15, 0.85, 0.25, 0.95])
                        # Purple: Rear torque normal vector + twist ring
                        add_vector_arrow(scn, p_rear, self.filt_rear_torque, t_scale, [0.72, 0.22, 0.95, 0.95])
                        add_torque_ring(scn, p_rear, self.filt_rear_torque, t_scale, [0.72, 0.22, 0.95, 0.95])

                        viewer.sync()

                    elapsed = time.time() - step_start
                    if elapsed < m.opt.timestep:
                        time.sleep(m.opt.timestep - elapsed)

            self.viewer = None
            if not self.rebuild_requested:
                self.running = False

    def rebuild_geometry(self, event=None):
        try:
            new_l1 = float(self.var_l1.get())
            new_l2 = float(self.var_l2.get())
            new_deg1 = max(-89.0, min(89.0, float(self.var_deg1.get())))
            new_deg2 = max(-89.0, min(89.0, float(self.var_deg2.get())))
            new_axis = self.var_axis.get()
            new_lock = bool(self.var_lock_front.get())

            self.var_deg1.set(f"{new_deg1:.1f}")
            self.var_deg2.set(f"{new_deg2:.1f}")

            xml_str = render_drawbar_xml(new_l1, new_l2, new_deg1, new_deg2, new_axis, new_lock)
            new_model = mujoco.MjModel.from_xml_string(xml_str)
            new_data = mujoco.MjData(new_model)

            with self.lock:
                self.l1 = new_l1
                self.l2 = new_l2
                self.deg1 = new_deg1
                self.deg2 = new_deg2
                self.axis_mode = new_axis
                self.lock_front = new_lock
                self.current_xml = xml_str
                self.model = new_model
                self.data = new_data
                self.filt_rear_force = np.zeros(3)
                self.filt_rear_torque = np.zeros(3)
                self.filt_ball_force = np.zeros(3)
                self._apply_forces_locked()
                self.rebuild_requested = True

            ctx = compute_geometry_context(new_l1, new_l2, new_deg1, new_deg2, new_axis, new_lock)
            span = math.hypot(ctx["fx_pos"], ctx["fy_pos"], ctx["fz_pos"] - 0.2)
            self.var_info.set(
                f"Span: {span:.3f} m | Front Offset: dX={ctx['fx_pos']:.3f}, dY={ctx['fy_pos']:.3f}, dZ={ctx['fz_pos'] - 0.2:.3f} m"
            )

        except Exception as exc:
            messagebox.showerror("Geometry / Template Error", str(exc))

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
            self.force_scale = float(self.var_fscale.get())
            self.torque_scale = float(self.var_tscale.get())

    def _export_xml(self):
        path = filedialog.asksaveasfilename(
            title="Save Rendered MJCF XML",
            defaultextension=".xml",
            initialfile="drawbar_ball_joint.xml",
            filetypes=[("MuJoCo XML", "*.xml"), ("All Files", "*.*")],
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.current_xml)
            messagebox.showinfo("Exported", f"Saved rendered MJCF XML to:\n{path}")

    def _nudge_var(self, tk_var: tk.StringVar, delta: float, callback):
        try:
            val = float(tk_var.get()) + delta
            tk_var.set(f"{val:.2f}" if abs(delta) < 1.0 else f"{val:.1f}")
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
        geo_frame = ttk.LabelFrame(self.root, text="1. Drawbar Geometry (Locked End Angles, -89° to +89°)", padding=8)
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
        axis_box = ttk.Combobox(
            geo_frame, textvariable=self.var_axis, values=["Yaw (Z)", "Pitch (Y)"], state="readonly", width=12
        )
        axis_box.grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)
        axis_box.bind("<<ComboboxSelected>>", self.rebuild_geometry)

        chk_lock = ttk.Checkbutton(
            geo_frame,
            text="Lock Front Node Translation (Rigid Chassis Hitch vs. Floating)",
            variable=self.var_lock_front,
            command=self.rebuild_geometry,
        )
        chk_lock.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 2))

        btn_geo = ttk.Frame(geo_frame)
        btn_geo.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(4, 2))
        ttk.Button(btn_geo, text="Rebuild / Reset Pose", command=self.rebuild_geometry).pack(
            side="left", expand=True, fill="x", padx=(0, 3)
        )
        ttk.Button(btn_geo, text="Export XML...", command=self._export_xml).pack(
            side="left", expand=True, fill="x", padx=(3, 0)
        )

        ttk.Label(geo_frame, textvariable=self.var_info, foreground="#005a9e").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(2, 0)
        )

        # --- 2. INPUT FORCE FRAME ---
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
        ttk.Button(btn_row, text="Apply Forces", command=self.update_forces).pack(
            side="left", expand=True, fill="x", padx=(0, 3)
        )
        ttk.Button(btn_row, text="Zero All Forces", command=self._zero_forces).pack(
            side="left", expand=True, fill="x", padx=(3, 0)
        )

        # --- 3. 3D VECTORS & SENSOR READOUT FRAME ---
        out_frame = ttk.LabelFrame(self.root, text="3. 3D Vectors & World-Frame Readout", padding=8)
        out_frame.pack(fill="x", **pad)

        scale_grid = ttk.Frame(out_frame)
        scale_grid.pack(fill="x", pady=(0, 4))
        self.var_fscale = tk.DoubleVar(value=self.force_scale)
        self.var_tscale = tk.DoubleVar(value=self.torque_scale)

        ttk.Label(scale_grid, text="Force Arrow Scale:").grid(row=0, column=0, sticky="w")
        ttk.Scale(
            scale_grid, from_=0.0001, to=0.005, variable=self.var_fscale, command=self._update_scales, length=170
        ).grid(row=0, column=1, padx=6)

        ttk.Label(scale_grid, text="Torque Arrow Scale:").grid(row=1, column=0, sticky="w")
        ttk.Scale(
            scale_grid, from_=0.0005, to=0.03, variable=self.var_tscale, command=self._update_scales, length=170
        ).grid(row=1, column=1, padx=6)

        legend = (
            "Arrows: [RED] Applied Front Force   [YELLOW] Ball Joint Force\n"
            "        [GREEN] Rear Reaction Force [PURPLE + RING] Rear Torque Normal"
        )
        ttk.Label(out_frame, text=legend, foreground="#333333", font=("Consolas", 8)).pack(anchor="w", pady=(0, 4))

        self.var_out_force = tk.StringVar(value="")
        self.var_out_torque = tk.StringVar(value="")
        self.var_ball_force = tk.StringVar(value="")

        ttk.Label(out_frame, textvariable=self.var_out_force, font=("Consolas", 9, "bold")).pack(anchor="w")
        ttk.Label(out_frame, textvariable=self.var_out_torque, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))
        ttk.Label(out_frame, textvariable=self.var_ball_force, font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))

        ctx = compute_geometry_context(self.l1, self.l2, self.deg1, self.deg2, self.axis_mode, self.lock_front)
        span = math.hypot(ctx["fx_pos"], ctx["fy_pos"], ctx["fz_pos"] - 0.2)
        self.var_info.set(
            f"Span: {span:.3f} m | Front Offset: dX={ctx['fx_pos']:.3f}, dY={ctx['fy_pos']:.3f}, dZ={ctx['fz_pos'] - 0.2:.3f} m"
        )

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_sensors()
        self.root.mainloop()

    def _add_stepper_row(self, parent, row: int, label: str, tk_var: tk.StringVar, step: float, callback):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Button(parent, text="-", width=3, command=lambda: self._nudge_var(tk_var, -step, callback)).grid(
            row=row, column=1, padx=2
        )
        entry = ttk.Entry(parent, textvariable=tk_var, width=9, justify="center")
        entry.grid(row=row, column=2, padx=2)
        entry.bind("<Return>", callback)
        ttk.Button(parent, text="+", width=3, command=lambda: self._nudge_var(tk_var, step, callback)).grid(
            row=row, column=3, padx=2
        )

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
