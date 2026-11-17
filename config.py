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
        test_name=test_name,
        test_duration_s=test_duration,
    )
