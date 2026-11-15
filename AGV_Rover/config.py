from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AGVConfig:
    # Geometry
    num_trailers: int
    half_drawbar_len_m: float
    chassis_half_len_m: float
    chassis_half_width_m: float
    chassis_half_height_m: float
    wheel_y_offset_m: float
    wheel_z_offset_m: float
    wheel_radius_m: float
    wheel_half_width_m: float
    wheel_mass_kg: float
    lead_dry_mass_kg: float
    trailer_dry_mass_kg: float
    cable_anchor_y_m: float
    cable_anchor_z_m: float
    cable_preload_offset_m: float
    initial_capstan_ctrl_m: float
    ball_joint_limit_deg: float
    ball_joint_damping: float

    # Tank Slosh
    water_liters_per_trailer: float
    slosh_mass_fraction: float
    slosh_pendulum_len_m: float
    slosh_roll_stiffness: float
    slosh_pitch_stiffness: float
    slosh_damping: float
    slosh_roll_limit_deg: float
    slosh_pitch_limit_deg: float

    # Actuators
    lead_wheel_kv: float
    trailer_wheel_kv: float
    capstan_kp: float
    capstan_kv: float
    capstan_max_pull_n: float

    # Controller
    v_center_mps: float
    lookahead_dist_m: float
    goal_tol_m: float
    max_curvature: float
    yaw_gain: float
    joint_yaw_margin_deg: float
    actuator_lead_offset_m: float
    use_ir_camera_feedback: bool
    waypoints: tuple[tuple[float, float], ...]
    active_name: str

    # Simulation & Viewer
    xml_filename: str
    udp_host: str
    udp_port: int
    save_csv_log: bool
    post_stop_log_sec: float
    camera_distance_m: float
    camera_azimuth_deg: float
    camera_elevation_deg: float

    # --- Derived Properties ---
    @property
    def cable_joint_to_plate_x_m(self) -> float:
        return self.half_drawbar_len_m - self.chassis_half_len_m  # 0.35 m

    @property
    def track_width_m(self) -> float:
        return 2.0 * self.wheel_y_offset_m  # 0.64 m

    @property
    def max_joint_yaw_deg(self) -> float:
        return self.ball_joint_limit_deg - self.joint_yaw_margin_deg  # 38.0 deg

    def joint_delays_m(self) -> list[float]:
        """Spatial delay from Trailer 1 axle to each ball joint (minus lead compensation)."""
        return [
            (2 * i + 1) * self.half_drawbar_len_m - self.actuator_lead_offset_m
            for i in range(self.num_trailers - 1)
        ]


def load_config(toml_path: Path | str | None = None) -> AGVConfig:
    path = Path(toml_path) if toml_path else Path(__file__).resolve().parent / "agv_config.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    g = raw["geometry"]
    s = raw["tank_slosh"]
    a = raw["actuators"]
    c = raw["controller"]
    sim = raw["simulation"]
    v = raw["viewer"]

    # Compute local variables BEFORE calling return AGVConfig(...)
    active_name = str(c["active_path"])
    raw_pts = raw["paths"][active_name]
    waypoints = tuple((float(pt[0]), float(pt[1])) for pt in raw_pts)

    return AGVConfig(
        # Geometry
        num_trailers=int(g["num_trailers"]),
        half_drawbar_len_m=float(g["half_drawbar_len_m"]),
        chassis_half_len_m=float(g["chassis_half_len_m"]),
        chassis_half_width_m=float(g["chassis_half_width_m"]),
        chassis_half_height_m=float(g["chassis_half_height_m"]),
        wheel_y_offset_m=float(g["wheel_y_offset_m"]),
        wheel_z_offset_m=float(g["wheel_z_offset_m"]),
        wheel_radius_m=float(g["wheel_radius_m"]),
        wheel_half_width_m=float(g["wheel_half_width_m"]),
        wheel_mass_kg=float(g["wheel_mass_kg"]),
        lead_dry_mass_kg=float(g["lead_dry_mass_kg"]),
        trailer_dry_mass_kg=float(g["trailer_dry_mass_kg"]),
        cable_anchor_y_m=float(g["cable_anchor_y_m"]),
        cable_anchor_z_m=float(g["cable_anchor_z_m"]),
        cable_preload_offset_m=float(g["cable_preload_offset_m"]),
        initial_capstan_ctrl_m=float(g["initial_capstan_ctrl_m"]),
        ball_joint_limit_deg=float(g["ball_joint_limit_deg"]),
        ball_joint_damping=float(g["ball_joint_damping"]),
        # Tank slosh
        water_liters_per_trailer=float(s["water_liters_per_trailer"]),
        slosh_mass_fraction=float(s["slosh_mass_fraction"]),
        slosh_pendulum_len_m=float(s["slosh_pendulum_len_m"]),
        slosh_roll_stiffness=float(s["slosh_roll_stiffness"]),
        slosh_pitch_stiffness=float(s["slosh_pitch_stiffness"]),
        slosh_damping=float(s["slosh_damping"]),
        slosh_roll_limit_deg=float(s["slosh_roll_limit_deg"]),
        slosh_pitch_limit_deg=float(s["slosh_pitch_limit_deg"]),
        # Actuators
        lead_wheel_kv=float(a["lead_wheel_kv"]),
        trailer_wheel_kv=float(a["trailer_wheel_kv"]),
        capstan_kp=float(a["capstan_kp"]),
        capstan_kv=float(a["capstan_kv"]),
        capstan_max_pull_n=float(a["capstan_max_pull_n"]),
        # Controller
        v_center_mps=float(c["v_center_mps"]),
        lookahead_dist_m=float(c["lookahead_dist_m"]),
        goal_tol_m=float(c["goal_tol_m"]),
        max_curvature=float(c["max_curvature"]),
        yaw_gain=float(c["yaw_gain"]),
        joint_yaw_margin_deg=float(c["joint_yaw_margin_deg"]),
        actuator_lead_offset_m=float(c["actuator_lead_offset_m"]),
        use_ir_camera_feedback=bool(c["use_ir_camera_feedback"]),
        waypoints=waypoints,
        active_name=active_name,
        # Simulation & viewer
        xml_filename=str(sim["xml_filename"]),
        udp_host=str(sim["udp_host"]),
        udp_port=int(sim["udp_port"]),
        save_csv_log=bool(sim["save_csv_log"]),
        post_stop_log_sec=float(sim["post_stop_log_sec"]),
        camera_distance_m=float(v["camera_distance_m"]),
        camera_azimuth_deg=float(v["camera_azimuth_deg"]),
        camera_elevation_deg=float(v["camera_elevation_deg"]),
    )
