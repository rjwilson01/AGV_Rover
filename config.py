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
    start_ramp_s: float
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

    # Simulated IR camera (relative R/P/Y between adjacent trailers). Defaults = ideal sensor.
    ir_noise_std_deg: float = 0.0
    ir_latency_s: float = 0.0
    ir_quantization_deg: float = 0.0
    ir_seed: int = 0

    # Yaw-controlled drawbar tubes (see bar_geometry.py). Off = original rigid bars.
    bar_yaw_control: bool = False
    rear_bar_yaw_range_deg: tuple[float, float] = (-25.0, 25.0)
    front_bar_yaw_range_deg: tuple[float, float] = (-25.0, 25.0)
    bar_pivot_inset_m: float = 0.05             # tube pivot sits this far inside the chassis end
    bar_yaw_kp: float = 400.0                   # tube yaw position actuator gain (N*m per rad)
    bar_yaw_kv: float = 20.0                    # tube yaw actuator damping (N*m per rad/s)
    bar_yaw_max_torque_nm: float = 200.0
    bar_yaw_damping: float = 2.0                # passive hinge damping
    bar_yaw_hinge_margin_deg: float = 5.0       # hinge stop sits this far beyond the commanded range

    # Follower wheel speeds from the MEASURED joint yaw (IR camera) via the hitch kinematics, not the commanded yaw
    wheel_speed_from_ir: bool = False
    joint_yaw_sign: float = 1.0                 # -1 flips controller yaws to physical hinge/ball commands

    # --- Tooling / display ---
    launch_plotjuggler: bool = False
    view_lead_speed_mps: float = 0.0           # --view starting lead speed
    sim_timestep_s: float = 0.002              # MuJoCo step (XML)
    sim_integrator: str = "RK4"
    sim_gravity_mps2: float = 9.81
    ground_friction: tuple = (0.85, 0.005, 0.0005)   # sliding, torsional, rolling
    bar_slide_damping: float = 300.0           # telescoping-bar slide joints (XML)
    cable_damping: float = 200.0
    cable_width_m: float = 0.005
    follower_max_speed_frac: float = 2.0       # follower speed limit, times the lead speed
    follower_max_yaw_rate: float = 3.0         # rad/s limit on the follower yaw rate in the wheel law
    view_pause_on_compression: bool = False    # --view: freeze the physics when a hitch goes into compression
    view_show_force_window: bool = True        # --view sub-windows (can also be toggled in the control window)
    view_show_config_window: bool = True
    view_show_param_window: bool = True
    show_bar_overlay: bool = False             # orange (commanded) / green (measured) bar lines drawn over the model
    bar_min_engagement_m: float = 0.10         # minimum rod-in-tube overlap at full extension (build check)

    # --- Idealised reference path for the followers ---
    follow_ideal_path: bool = True             # followers use the smooth planned path's curvature, not the lead's history
    ref_smooth_m: float = 1.0                  # moving-average length applied to the reference curvature
    ref_progress_gain: float = 1.5             # max reference progress per metre the lead actually travels

    # --- Steering smoothness ---
    lookahead_gain_s: float = 2.0              # lookahead = max(lookahead_dist_m, gain * speed)
    curv_filter_m: float = 0.5                 # curvature low-pass length (distance domain); 0 = off
    curv_rate_limit_per_m: float = 0.6         # max change of curvature per metre travelled (1/m^2); 0 = off
    follower_min_speed_frac: float = 0.3       # follower speed floor as a fraction of the lead's commanded speed

    # --- Path planner ---
    planner_enabled: bool = True
    planner_margin: float = 0.85               # fraction of the hitch yaw range the planner may use
    planner_spacing_m: float = 0.05
    planner_radius_cap_m: float = 8.0
    ref_dev_gain: float = 1.0                  # share of the lead's deviation from the ideal path passed on to followers (0 = pure ideal path)
    ref_dev_filter_m: float = 2.0              # distance low-pass on that deviation
    ref_dev_max: float = 0.35                  # clip on the deviation curvature, 1/m
    wheel_use_tube_geometry: bool = True        # follower wheel speeds use the ball's off-axis position from the tube yaws
    cable_use_measured_tubes: bool = True       # size the hitch cables for the measured tube angles (ball takes the lag)
    bar_pitch_limit: bool = True                # shorten the bars when hitch pitch would put a bar end on the ground
    bar_ground_margin_m: float = 0.06           # clearance kept under the bars (tube radius plus a little)
    planner_runout_m: float = 3.0              # straight extension past the goal; the train still stops at the real goal

    # --- Follower stall guard ---
    guard_enabled: bool = True
    guard_stall_ratio: float = 0.3
    guard_release_ratio: float = 0.6
    guard_stall_time_s: float = 0.5
    retension_guard: bool = False              # compression at a hitch (cable sum below its resting value) slows the trailer behind
    retension_baseline_s: float = 0.5          # resting cable sum is measured over the first this-many seconds
    retension_thresh_n: float = 30.0           # compression declared when the sum is this far below the baseline
    retension_gain_per_n: float = 0.005        # follower speed scale falls by this per N beyond the threshold
    retension_min_scale: float = 0.5
    retension_release_s: float = 1.0           # time to recover from min scale back to 1
    guard_min_cmd_mps: float = 0.15
    guard_ramp_s: float = 0.5

    # --- Lead GNSS (u-blox ZED-F9P class, RTK fixed) ---
    gnss_enabled: bool = True
    gnss_rate_hz: float = 10.0
    gnss_latency_s: float = 0.1
    gnss_pos_noise_m: float = 0.014
    gnss_bias_m: float = 0.01
    gnss_bias_tau_s: float = 10.0
    gnss_vel_noise_mps: float = 0.05
    gnss_outages_s: tuple = ()
    loadcell_enabled: bool = True              # simulated cable load cells (fixed end of each cable)
    loadcell_noise_n: float = 1.0
    loadcell_resolution_n: float = 0.5
    loadcell_range_n: float = 1000.0
    loadcell_bw_hz: float = 50.0
    loadcell_seed: int = 3
    bar_loadcell_enabled: bool = False         # optional axial force sensor on the bar (yaw-tube build)
    bar_force_sign: float = 1.0                # flip to -1 if a pull on the hitch reads negative
    gnss_seed: int = 2

    # --- IMU per trailer (defaults = Bosch BMI088 datasheet) ---
    imu_enabled: bool = True
    imu_rate_hz: float = 400.0
    imu_acc_noise_ug_rthz: tuple[float, float, float] = (160.0, 160.0, 190.0)
    imu_acc_bw_hz: float = 145.0
    imu_acc_offset_mg: float = 20.0
    imu_acc_range_g: float = 6.0
    imu_gyro_noise_dps_rthz: float = 0.014
    imu_gyro_bw_hz: float = 116.0
    imu_gyro_offset_dps: float = 1.0
    imu_gyro_g_sens_dps_per_g: float = 0.1
    imu_gyro_range_dps: float = 500.0
    imu_vibration_g_rms: float = 0.0            # extra broadband vibration on the accelerometer (engine/blade)
    imu_seed: int = 1

    # --- Mass layout (defaults reproduce the original model: all dry mass at the chassis centre) ---
    battery_mass_kg: float = 0.0                 # part of the dry mass, placed as its own block
    battery_z_m: float | None = None             # height vs chassis centre; None = axle height
    water_rigid_z_m: float = 0.06                # the non-sloshing water block
    slosh_pivot_z_m: float | None = None         # slosh pendulum pivot; None = chassis top + pendulum length

    # --- Test sequence info (set by load_config when test files are applied) ---
    test_name: str = ""
    test_duration_s: float | None = None

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


TEST_DIR = Path("configs") / "test"


def resolve_test_path(name: str) -> Path:
    """A bare name means configs/test/<name>.toml; anything with a folder or a .toml suffix is used as given."""
    base_dir = Path(__file__).resolve().parent
    p = Path(name)
    if not p.suffix:
        p = p.with_suffix(".toml")
    if len(p.parts) == 1:
        p = TEST_DIR / p
    return p if p.is_absolute() else base_dir / p


def _parse_set(item: str) -> tuple[list[str], object]:
    """'section.key=value' -> (['section','key'], parsed value). Value is TOML (1.05, true, [1,2], "x")."""
    if "=" not in item:
        raise ValueError(f"--set needs section.key=value, got {item!r}")
    key, val = item.split("=", 1)
    try:
        parsed = tomllib.loads(f"v = {val.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = val.strip()
    return key.strip().split("."), parsed


def read_raw_config(
    impl_toml_path: Path | str | None = None,
    common_toml_path: Path | str | None = None,
    test_files=(),
    set_overrides=(),
) -> tuple[dict, list[dict]]:
    """Merged raw config: common < controller < each test file in order < --set overrides.

    Returns (raw, test_infos) where test_infos is one {file, TestName, duration_s} per test file."""
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

    infos = []
    for name in test_files:
        tpath = resolve_test_path(name)
        if not tpath.exists():
            raise FileNotFoundError(f"test file not found: {tpath} (from {name!r})")
        with open(tpath, "rb") as f:
            t_raw = tomllib.load(f)
        meta = t_raw.pop("test", {})      # [test] holds TestName / duration_s, not simulation parameters
        infos.append({
            "file": Path(name).stem,
            "TestName": str(meta.get("TestName", Path(name).stem)),
            "duration_s": float(meta["duration_s"]) if "duration_s" in meta else None,
        })
        raw = _deep_merge(raw, t_raw)

    for item in set_overrides:
        path, val = _parse_set(item)
        node = raw
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = val
    return raw, infos


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(impl_toml_path=None, common_toml_path=None, test_files=(), set_overrides=()) -> dict:
    """Flat {'section.key': value} of everything that differs from the plain controller + common defaults."""
    base, _ = read_raw_config(impl_toml_path, common_toml_path)
    cur, _ = read_raw_config(impl_toml_path, common_toml_path, test_files, set_overrides)
    fb, fc = _flatten(base), _flatten(cur)
    return {k: v for k, v in fc.items() if fb.get(k, object()) != v}


def load_config(
    impl_toml_path: Path | str | None = None,
    common_toml_path: Path | str | None = None,
    test_files=(),
    set_overrides=(),
) -> AGVConfig:
    raw, infos = read_raw_config(impl_toml_path, common_toml_path, test_files, set_overrides)
    return config_from_raw(raw, infos)


def config_from_raw(raw: dict, infos: list[dict] | None = None) -> AGVConfig:
    """Build the config from an already merged raw dict (the --view parameter window edits that dict directly)."""
    infos = infos or []
    test_name = " + ".join(i["TestName"] for i in infos)
    test_duration = next((i["duration_s"] for i in reversed(infos) if i["duration_s"] is not None), None)

    d = raw["drawbar"]
    g = raw["geometry"]
    s = raw["tank_slosh"]
    a = raw["actuators"]
    c = raw["controller"]
    sim = raw["simulation"]
    v = raw["viewer"]
    ir = raw.get("ir_camera", {})
    imu = raw.get("imu", {})
    bars = raw.get("bars", {})

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
        start_ramp_s=float(c.get("start_ramp_s", 0.0)),
        launch_plotjuggler=bool(sim.get("launch_plotjuggler", False)),
        show_bar_overlay=bool(v.get("show_bar_overlay", False)),
        view_lead_speed_mps=float(v.get("lead_speed_mps", 0.0)),
        sim_timestep_s=float(sim.get("timestep_s", 0.002)),
        sim_integrator=str(sim.get("integrator", "RK4")),
        sim_gravity_mps2=float(sim.get("gravity_mps2", 9.81)),
        ground_friction=tuple(float(x) for x in raw.get("terrain", {}).get("ground_friction", [0.85, 0.005, 0.0005])),
        bar_slide_damping=float(a.get("bar_slide_damping", 300.0)),
        cable_damping=float(a.get("cable_damping", 200.0)),
        cable_width_m=float(a.get("cable_width_m", 0.005)),
        follower_max_speed_frac=float(raw["controller"].get("follower_max_speed_frac", 2.0)),
        follower_max_yaw_rate=float(raw["controller"].get("follower_max_yaw_rate", 3.0)),
        view_pause_on_compression=bool(v.get("pause_on_compression", False)),
        view_show_force_window=bool(v.get("show_force_window", True)),
        view_show_config_window=bool(v.get("show_config_window", True)),
        view_show_param_window=bool(v.get("show_param_window", True)),
        bar_min_engagement_m=float(bars.get("min_engagement_m", 0.10)),
        follow_ideal_path=bool(raw["controller"].get("follow_ideal_path", True)),
        ref_smooth_m=float(raw["controller"].get("ref_smooth_m", 1.0)),
        ref_progress_gain=float(raw["controller"].get("ref_progress_gain", 1.5)),
        lookahead_gain_s=float(raw["controller"].get("lookahead_gain_s", 2.0)),
        curv_filter_m=float(raw["controller"].get("curv_filter_m", 0.5)),
        curv_rate_limit_per_m=float(raw["controller"].get("curv_rate_limit_per_m", 0.6)),
        follower_min_speed_frac=float(raw["controller"].get("follower_min_speed_frac", 0.3)),
        planner_enabled=bool(c.get("planner_enabled", True)),
        planner_margin=float(c.get("planner_margin", 0.85)),
        planner_spacing_m=float(c.get("planner_spacing_m", 0.05)),
        planner_radius_cap_m=float(c.get("planner_radius_cap_m", 8.0)),
        planner_runout_m=float(c.get("planner_runout_m", 3.0)),
        bar_pitch_limit=bool(c.get("bar_pitch_limit", True)),
        wheel_use_tube_geometry=bool(c.get("wheel_use_tube_geometry", True)),
        cable_use_measured_tubes=bool(c.get("cable_use_measured_tubes", True)),
        bar_ground_margin_m=float(c.get("bar_ground_margin_m", 0.06)),
        ref_dev_gain=float(c.get("ref_dev_gain", 1.0)),
        ref_dev_filter_m=float(c.get("ref_dev_filter_m", 2.0)),
        ref_dev_max=float(c.get("ref_dev_max", 0.35)),
        guard_enabled=bool(c.get("guard_enabled", True)),
        guard_stall_ratio=float(c.get("guard_stall_ratio", 0.3)),
        guard_release_ratio=float(c.get("guard_release_ratio", 0.6)),
        guard_stall_time_s=float(c.get("guard_stall_time_s", 0.5)),
        retension_guard=bool(c.get("retension_guard", False)),
        retension_baseline_s=float(c.get("retension_baseline_s", 0.5)),
        retension_thresh_n=float(c.get("retension_thresh_n", 30.0)),
        retension_gain_per_n=float(c.get("retension_gain_per_n", 0.005)),
        retension_min_scale=float(c.get("retension_min_scale", 0.5)),
        retension_release_s=float(c.get("retension_release_s", 1.0)),
        guard_min_cmd_mps=float(c.get("guard_min_cmd_mps", 0.15)),
        guard_ramp_s=float(c.get("guard_ramp_s", 0.5)),
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
        ir_noise_std_deg=float(ir.get("noise_std_deg", 0.0)),
        ir_latency_s=float(ir.get("latency_s", 0.0)),
        ir_quantization_deg=float(ir.get("quantization_deg", 0.0)),
        ir_seed=int(ir.get("seed", 0)),
        bar_yaw_control=bool(bars.get("yaw_control", False)),
        rear_bar_yaw_range_deg=tuple(float(x) for x in bars.get("rear_yaw_range_deg", (-25.0, 25.0))),
        front_bar_yaw_range_deg=tuple(float(x) for x in bars.get("front_yaw_range_deg", (-25.0, 25.0))),
        bar_pivot_inset_m=float(bars.get("pivot_inset_m", 0.05)),
        bar_yaw_kp=float(bars.get("yaw_kp", 400.0)),
        bar_yaw_kv=float(bars.get("yaw_kv", 20.0)),
        bar_yaw_max_torque_nm=float(bars.get("yaw_max_torque_nm", 200.0)),
        bar_yaw_damping=float(bars.get("yaw_damping", 2.0)),
        bar_yaw_hinge_margin_deg=float(bars.get("yaw_hinge_margin_deg", 5.0)),
        joint_yaw_sign=float(raw["controller"].get("joint_yaw_sign", 1.0)),
        gnss_enabled=bool(raw.get("gnss", {}).get("enabled", True)),
        gnss_rate_hz=float(raw.get("gnss", {}).get("rate_hz", 10.0)),
        gnss_latency_s=float(raw.get("gnss", {}).get("latency_s", 0.1)),
        gnss_pos_noise_m=float(raw.get("gnss", {}).get("pos_noise_m", 0.014)),
        gnss_bias_m=float(raw.get("gnss", {}).get("bias_m", 0.01)),
        gnss_bias_tau_s=float(raw.get("gnss", {}).get("bias_tau_s", 10.0)),
        gnss_vel_noise_mps=float(raw.get("gnss", {}).get("vel_noise_mps", 0.05)),
        gnss_outages_s=tuple((float(a), float(b)) for a, b in raw.get("gnss", {}).get("outages_s", [])),
        gnss_seed=int(raw.get("gnss", {}).get("seed", 2)),
        loadcell_enabled=bool(raw.get("loadcell", {}).get("enabled", True)),
        loadcell_noise_n=float(raw.get("loadcell", {}).get("noise_n", 1.0)),
        loadcell_resolution_n=float(raw.get("loadcell", {}).get("resolution_n", 0.5)),
        loadcell_range_n=float(raw.get("loadcell", {}).get("range_n", 1000.0)),
        loadcell_bw_hz=float(raw.get("loadcell", {}).get("bw_hz", 50.0)),
        loadcell_seed=int(raw.get("loadcell", {}).get("seed", 3)),
        bar_loadcell_enabled=bool(raw.get("loadcell", {}).get("bar_enabled", False)),
        bar_force_sign=float(raw.get("loadcell", {}).get("bar_force_sign", 1.0)),
        imu_enabled=bool(imu.get("enabled", True)),
        imu_rate_hz=float(imu.get("rate_hz", 400.0)),
        imu_acc_noise_ug_rthz=tuple(float(v) for v in imu.get("acc_noise_ug_rthz", (160.0, 160.0, 190.0))),
        imu_acc_bw_hz=float(imu.get("acc_bw_hz", 145.0)),
        imu_acc_offset_mg=float(imu.get("acc_offset_mg", 20.0)),
        imu_acc_range_g=float(imu.get("acc_range_g", 6.0)),
        imu_gyro_noise_dps_rthz=float(imu.get("gyro_noise_dps_rthz", 0.014)),
        imu_gyro_bw_hz=float(imu.get("gyro_bw_hz", 116.0)),
        imu_gyro_offset_dps=float(imu.get("gyro_offset_dps", 1.0)),
        imu_gyro_g_sens_dps_per_g=float(imu.get("gyro_g_sens_dps_per_g", 0.1)),
        imu_gyro_range_dps=float(imu.get("gyro_range_dps", 500.0)),
        imu_vibration_g_rms=float(imu.get("vibration_g_rms", 0.0)),
        imu_seed=int(imu.get("seed", 1)),
        wheel_speed_from_ir=bool(raw["controller"].get("wheel_speed_from_ir", False)),
        battery_mass_kg=float(raw.get("mass_layout", {}).get("battery_mass_kg", 0.0)),
        battery_z_m=raw.get("mass_layout", {}).get("battery_z_m"),
        water_rigid_z_m=float(raw.get("mass_layout", {}).get("water_rigid_z_m", 0.06)),
        slosh_pivot_z_m=raw.get("mass_layout", {}).get("slosh_pivot_z_m"),
        test_name=test_name,
        test_duration_s=test_duration,
    )


def cg_above_axle_m(cfg: AGVConfig, lead: bool = False) -> float:
    """Height of one cart's centre of gravity above its axle (negative = CG below the axle, pendulum-stable).

    Cart = chassis (dry mass less battery) at the chassis centre + battery block + rigid and slosh water + wheels."""
    dry = cfg.lead_dry_mass_kg if lead else cfg.trailer_dry_mass_kg
    water = cfg.water_liters_per_trailer
    m_slosh = water * cfg.slosh_mass_fraction
    pivot = cfg.slosh_pivot_z_m if cfg.slosh_pivot_z_m is not None else cfg.chassis_half_height_m + cfg.slosh_pendulum_len_m
    batt_z = cfg.battery_z_m if cfg.battery_z_m is not None else cfg.wheel_z_offset_m
    parts = [
        (dry - cfg.battery_mass_kg, 0.0),
        (cfg.battery_mass_kg, batt_z),
        (water - m_slosh, cfg.water_rigid_z_m),
        (m_slosh, pivot - cfg.slosh_pendulum_len_m),
        (2.0 * cfg.wheel_mass_kg, cfg.wheel_z_offset_m),
    ]
    total = sum(m for m, _ in parts)
    return sum(m * z for m, z in parts) / total - cfg.wheel_z_offset_m
