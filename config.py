from dataclasses import dataclass
from pathlib import Path
import tomllib


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dictionary into base dictionary."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


@dataclass(frozen=True)
class AGVConfig:
    # Drawbar mode & ranges
    controller_class: str
    drawbar_mode: str
    rear_bar_range_m: tuple[float, float]
    front_bar_range_m: tuple[float, float]
    yaw_angle_range_deg: tuple[float, float]
    telescoping_kp: float
    telescoping_kv: float
    telescoping_max_force_n: float

    # Geometry
    num_trailers: int
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
    active_path: str
    v_center_mps: float
    lookahead_dist_m: float
    goal_tol_m: float
    max_curvature: float
    yaw_gain: float
    actuator_lead_offset_m: float
    use_ir_camera_feedback: bool
    waypoints: tuple[tuple[float, float], ...]

    # Simulation & Viewer
    udp_host: str
    udp_port: int
    stream_hz: float
    save_csv_log: bool
    post_stop_log_sec: float
    camera_distance_m: float
    camera_azimuth_deg: float
    camera_elevation_deg: float

    # --- Derived Properties ---
    @property
    def num_joints(self) -> int:
        return self.num_trailers - 1

    @property
    def nominal_rear_bar_m(self) -> float:
        _, r_max = self.rear_bar_range_m
        return r_max

    @property
    def nominal_front_bar_m(self) -> float:
        _, f_max = self.front_bar_range_m
        return f_max

    @property
    def track_width_m(self) -> float:
        return 2.0 * self.wheel_y_offset_m


def load_config(
    impl_toml_path: Path | str | None = None,
    common_toml_path: Path | str | None = None,
) -> AGVConfig:
    base_dir = Path(__file__).resolve().parent

    common_path = Path(common_toml_path) if common_toml_path else base_dir / "configs" / "common.toml"
    if not common_path.is_absolute():
        common_path = base_dir / common_path

    if impl_toml_path is None:
        impl_path = base_dir / "configs" / "FixedDrawbarController.toml"
    else:
        impl_path = Path(impl_toml_path)
        if not impl_path.suffix:
            impl_path = base_dir / "configs" / f"{impl_path.name}.toml"
        elif not impl_path.is_absolute():
            impl_path = base_dir / impl_path

    with open(common_path, "rb") as f:
        common_raw = tomllib.load(f)
    with open(impl_path, "rb") as f:
        impl_raw = tomllib.load(f)

    raw = _deep_merge(common_raw, impl_raw)

    d = raw["drawbar"]
    g = raw["geometry"]
    s = raw["tank_slosh"]
    a = raw["actuators"]
    c = raw["controller"]
    sim = raw["simulation"]
    v = raw["viewer"]

    r_min, r_max = d["rear_bar_range_m"]
    f_min, f_max = d["front_bar_range_m"]
    yaw_min, yaw_max = d["yaw_angle_range_deg"]

    active_name = str(c["active_path"])
    raw_pts = raw["paths"][active_name]
    waypoints = tuple((float(x), float(y)) for x, y in raw_pts)

    return AGVConfig(
        controller_class=str(d["controller_class"]),
        drawbar_mode=str(d["mode"]).lower(),
        rear_bar_range_m=(float(r_min), float(r_max)),
        front_bar_range_m=(float(f_min), float(f_max)),
        yaw_angle_range_deg=(float(yaw_min), float(yaw_max)),
        telescoping_kp=float(d["telescoping_kp"]),
        telescoping_kv=float(d["telescoping_kv"]),
        telescoping_max_force_n=float(d["telescoping_max_force_n"]),
        num_trailers=int(g["num_trailers"]),
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
        ball_joint_limit_deg=float(g["ball_joint_limit_deg"]),
        ball_joint_damping=float(g["ball_joint_damping"]),
        water_liters_per_trailer=float(s["water_liters_per_trailer"]),
        slosh_mass_fraction=float(s["slosh_mass_fraction"]),
        slosh_pendulum_len_m=float(s["slosh_pendulum_len_m"]),
        slosh_roll_stiffness=float(s["slosh_roll_stiffness"]),
        slosh_pitch_stiffness=float(s["slosh_pitch_stiffness"]),
        slosh_damping=float(s["slosh_damping"]),
        slosh_roll_limit_deg=float(s["slosh_roll_limit_deg"]),
        slosh_pitch_limit_deg=float(s["slosh_pitch_limit_deg"]),
        lead_wheel_kv=float(a["lead_wheel_kv"]),
        trailer_wheel_kv=float(a["trailer_wheel_kv"]),
        capstan_kp=float(a["capstan_kp"]),
        capstan_kv=float(a["capstan_kv"]),
        capstan_max_pull_n=float(a["capstan_max_pull_n"]),
        active_path=active_name,
        v_center_mps=float(c["v_center_mps"]),
        lookahead_dist_m=float(c["lookahead_dist_m"]),
        goal_tol_m=float(c["goal_tol_m"]),
        max_curvature=float(c["max_curvature"]),
        yaw_gain=float(c["yaw_gain"]),
        actuator_lead_offset_m=float(c["actuator_lead_offset_m"]),
        use_ir_camera_feedback=bool(c["use_ir_camera_feedback"]),
        waypoints=waypoints,
        udp_host=str(sim["udp_host"]),
        udp_port=int(sim["udp_port"]),
        stream_hz=float(sim["stream_hz"]),
        save_csv_log=bool(sim["save_csv_log"]),
        post_stop_log_sec=float(sim["post_stop_log_sec"]),
        camera_distance_m=float(v["camera_distance_m"]),
        camera_azimuth_deg=float(v["camera_azimuth_deg"]),
        camera_elevation_deg=float(v["camera_elevation_deg"]),
    )
