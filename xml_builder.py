from config import AGVConfig

SITE_CORNERS = (
    ("tl",  1,  1),
    ("tr", -1,  1),
    ("bl",  1, -1),
    ("br", -1, -1),
)


def _build_slosh_tank_xml(idx: int, cfg: AGVConfig, indent: str) -> str:
    if cfg.water_liters_per_trailer <= 0.0:
        return ""
    total_water_kg = cfg.water_liters_per_trailer
    m_slosh = total_water_kg * cfg.slosh_mass_fraction
    m_rigid = total_water_kg - m_slosh
    pivot_z = cfg.chassis_half_height_m + cfg.slosh_pendulum_len_m

    return (
        f'{indent}<geom name="t{idx}_water_rigid" type="box" pos="0 0 0.06" '
        f'size="{cfg.chassis_half_len_m * 0.75:.3f} {cfg.chassis_half_width_m * 0.80:.3f} 0.05" '
        f'mass="{m_rigid:.1f}" rgba="0.1 0.5 0.9 0.35" contype="0" conaffinity="0"/>\n'
        f'{indent}<body name="t{idx}_slosh_pivot" pos="0 0 {pivot_z:.3f}">\n'
        f'{indent}\t<joint name="t{idx}_slosh_roll" type="hinge" axis="1 0 0" damping="{cfg.slosh_damping}" '
        f'stiffness="{cfg.slosh_roll_stiffness}" limited="true" range="-{cfg.slosh_roll_limit_deg} {cfg.slosh_roll_limit_deg}"/>\n'
        f'{indent}\t<joint name="t{idx}_slosh_pitch" type="hinge" axis="0 1 0" damping="{cfg.slosh_damping}" '
        f'stiffness="{cfg.slosh_pitch_stiffness}" limited="true" range="-{cfg.slosh_pitch_limit_deg} {cfg.slosh_pitch_limit_deg}"/>\n'
        f'{indent}\t<geom name="t{idx}_water_slosh" type="sphere" pos="0 0 -{cfg.slosh_pendulum_len_m:.3f}" '
        f'size="0.13" mass="{m_slosh:.1f}" rgba="0.0 0.7 1.0 0.6" contype="0" conaffinity="0"/>\n'
        f'{indent}</body>'
    )


def _build_trailer_body(idx: int, cfg: AGVConfig, depth: int = 2) -> str:
    ind = "\t" * depth
    is_lead = (idx == 1)
    is_last = (idx == cfg.num_trailers)
    is_var = (cfg.drawbar_mode == "variable")

    cx = cfg.chassis_half_len_m
    r_min, r_max = cfg.rear_bar_range_m
    f_min, f_max = cfg.front_bar_range_m
    has_rear_slider = is_var and (r_max > r_min)
    has_front_slider = is_var and (f_max > f_min)

    lines = []
    if is_lead:
        body_pos = f"0 0 {cfg.wheel_radius_m - cfg.wheel_z_offset_m:.3f}"
        lines.append(f'{ind}<body name="trailer_{idx}" pos="{body_pos}">')
        lines.append(f'{ind}\t<freejoint name="root_jnt"/>')
    else:
        j_idx = idx - 1
        socket_pos = "0 0 0" if has_rear_slider else f"{-r_max:.3f} 0 0"
        lines.append(f'{ind}<body name="t{idx}_front_socket" pos="{socket_pos}">')
        lines.append(
            f'{ind}\t<joint name="ball_joint_{j_idx}" type="ball" pos="0 0 0" '
            f'damping="{cfg.ball_joint_damping}" limited="true" range="0 {cfg.ball_joint_limit_deg}"/>'
        )
        lines.append(f'{ind}\t<geom name="hitch_{j_idx}_ball" type="sphere" pos="0 0 0" size="0.036" material="cable_mat"/>')

        if has_front_slider:
            # Inner telescoping rod attached to the ball-joint socket (slides inside front outer sleeve)
            rod_len_f = max(0.05, f_max - cx)
            lines.append(
                f'{ind}\t<geom name="t{idx}_drawbar_front_rod" type="capsule" '
                f'fromto="0 0 0 -{rod_len_f:.3f} 0 0" size="0.022" mass="2.0" '
                f'material="drawbar" contype="0" conaffinity="0"/>'
            )

        lines.append(f'{ind}\t<body name="trailer_{idx}" pos="{-f_max:.3f} 0 0">')
        if has_front_slider:
            stroke_f = f_max - f_min
            lines.append(
                f'{ind}\t\t<joint name="slide_front_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_f:.4f}" damping="300"/>'
            )
            # Outer sleeve fixed to chassis from +cx to +f_min
            sleeve_end_f = max(cx + 0.05, f_min)
            lines.append(
                f'{ind}\t\t<geom name="t{idx}_drawbar_front_sleeve" type="capsule" '
                f'fromto="{sleeve_end_f:.3f} 0 0 {cx:.3f} 0 0" size="0.032" mass="3.0" '
                f'material="drawbar" contype="0" conaffinity="0"/>'
            )
        else:
            lines.append(
                f'{ind}\t\t<geom name="t{idx}_drawbar_front" type="capsule" '
                f'fromto="{f_max:.3f} 0 0 {cx:.3f} 0 0" size="0.025" material="drawbar"/>'
            )

        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{ind}\t\t<site name="j_t{idx}_b{code}" '
                f'pos="{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

    b_ind = ind if is_lead else ind + "\t"
    mass = cfg.lead_dry_mass_kg if is_lead else cfg.trailer_dry_mass_kg
    mat = "chassis_lead" if is_lead else "chassis_trl"

    lines.append(
        f'{b_ind}\t<geom name="t{idx}_chassis" type="box" '
        f'size="{cx:.3f} {cfg.chassis_half_width_m:.3f} {cfg.chassis_half_height_m:.3f}" mass="{mass:.1f}" material="{mat}"/>'
    )
    slosh_xml = _build_slosh_tank_xml(idx, cfg, b_ind + "\t")
    if slosh_xml:
        lines.append(slosh_xml)

    # Wheels
    for side, y_sign in (("l", 1), ("r", -1)):
        lines.append(f'{b_ind}\t<body name="t{idx}_{side}" pos="0 {y_sign * cfg.wheel_y_offset_m:.3f} {cfg.wheel_z_offset_m:.3f}">')
        lines.append(f'{b_ind}\t\t<joint name="j_t{idx}_{side}" type="hinge" axis="0 1 0"/>')
        lines.append(
            f'{b_ind}\t\t<geom type="cylinder" size="{cfg.wheel_radius_m:.3f} {cfg.wheel_half_width_m:.3f}" '
            f'euler="90 0 0" mass="{cfg.wheel_mass_kg:.1f}" material="tire"/>'
        )
        lines.append(f'{b_ind}\t</body>')

    # Rear drawbar + sites + nested next trailer
    if not is_last:
        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{b_ind}\t<site name="j_t{idx}_f{code}" '
                f'pos="-{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

        if has_rear_slider:
            stroke_r = r_max - r_min
            # Outer sleeve fixed to chassis from -cx to -r_min
            sleeve_end_r = max(cx + 0.05, r_min)
            lines.append(
                f'{b_ind}\t<geom name="t{idx}_drawbar_rear_sleeve" type="capsule" '
                f'fromto="-{cx:.3f} 0 0 -{sleeve_end_r:.3f} 0 0" size="0.032" mass="3.0" '
                f'material="drawbar" contype="0" conaffinity="0"/>'
            )
            lines.append(f'{b_ind}\t<body name="t{idx}_rear_slider" pos="-{r_max:.3f} 0 0">')
            lines.append(
                f'{b_ind}\t\t<joint name="slide_rear_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_r:.4f}" damping="300"/>'
            )
            # Inner telescoping rod extending from hitch tip (0 0 0) back into outer sleeve
            rod_len_r = max(0.05, r_max - cx)
            lines.append(
                f'{b_ind}\t\t<geom name="t{idx}_drawbar_rear_rod" type="capsule" '
                f'fromto="{rod_len_r:.3f} 0 0 0 0 0" size="0.022" mass="2.0" '
                f'material="drawbar" contype="0" conaffinity="0"/>'
            )
            lines.append(_build_trailer_body(idx + 1, cfg, depth + (2 if is_lead else 3)))
            lines.append(f'{b_ind}\t</body>')
        else:
            lines.append(
                f'{b_ind}\t<geom name="t{idx}_drawbar_rear" type="capsule" '
                f'fromto="-{cx:.3f} 0 0 -{r_max:.3f} 0 0" size="0.025" material="drawbar"/>'
            )
            lines.append(_build_trailer_body(idx + 1, cfg, depth + (1 if is_lead else 2)))

    lines.append(f'{b_ind}</body>')
    if not is_lead:
        lines.append(f'{ind}</body>')
    return "\n".join(lines)

    ind = "\t" * depth
    is_lead = (idx == 1)
    is_last = (idx == cfg.num_trailers)
    is_var = (cfg.drawbar_mode == "variable")

    cx = cfg.chassis_half_len_m
    r_min, r_max = cfg.rear_bar_range_m
    f_min, f_max = cfg.front_bar_range_m
    has_rear_slider = is_var and (r_max > r_min)
    has_front_slider = is_var and (f_max > f_min)

    mass = cfg.lead_dry_mass_kg if is_lead else cfg.trailer_dry_mass_kg
    mat = "chassis_lead" if is_lead else "chassis_trl"

    lines = []
    if is_lead:
        body_pos = f"0 0 {cfg.wheel_radius_m - cfg.wheel_z_offset_m:.3f}"
        lines.append(f'{ind}<body name="trailer_{idx}" pos="{body_pos}">')
        lines.append(f'{ind}\t<freejoint name="root_jnt"/>')
    else:
        j_idx = idx - 1
        # Socket sits at the exact tip of parent's rear drawbar (0 0 0 if parent has slider, else -r_max)
        socket_pos = "0 0 0" if has_rear_slider else f"{-r_max:.3f} 0 0"
        lines.append(f'{ind}<body name="t{idx}_front_socket" pos="{socket_pos}">')
        lines.append(
            f'{ind}\t<joint name="ball_joint_{j_idx}" type="ball" pos="0 0 0" '
            f'damping="{cfg.ball_joint_damping}" limited="true" range="0 {cfg.ball_joint_limit_deg}"/>'
        )
        lines.append(f'{ind}\t<geom name="hitch_{j_idx}_ball" type="sphere" pos="0 0 0" size="0.035" material="cable_mat"/>')

        # Trailer body hangs behind the socket at -f_max and slides forward (+X) to shorten front bar
        lines.append(f'{ind}\t<body name="trailer_{idx}" pos="{-f_max:.3f} 0 0">')
        if has_front_slider:
            stroke_f = f_max - f_min
            lines.append(
                f'{ind}\t\t<joint name="slide_front_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_f:.4f}" damping="300"/>'
            )
        lines.append(f'{ind}\t\t<geom name="t{idx}_drawbar_front" type="capsule" fromto="{f_max:.3f} 0 0 {cx:.3f} 0 0" size="0.025" material="drawbar"/>')
        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{ind}\t\t<site name="j_t{idx}_b{code}" '
                f'pos="{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

    b_ind = ind if is_lead else ind + "\t"
    lines.append(
        f'{b_ind}\t<geom name="t{idx}_chassis" type="box" '
        f'size="{cx:.3f} {cfg.chassis_half_width_m:.3f} {cfg.chassis_half_height_m:.3f}" mass="{mass:.1f}" material="{mat}"/>'
    )
    slosh_xml = _build_slosh_tank_xml(idx, cfg, b_ind + "\t")
    if slosh_xml:
        lines.append(slosh_xml)

    # Wheels
    for side, y_sign in (("l", 1), ("r", -1)):
        lines.append(f'{b_ind}\t<body name="t{idx}_{side}" pos="0 {y_sign * cfg.wheel_y_offset_m:.3f} {cfg.wheel_z_offset_m:.3f}">')
        lines.append(f'{b_ind}\t\t<joint name="j_t{idx}_{side}" type="hinge" axis="0 1 0"/>')
        lines.append(
            f'{b_ind}\t\t<geom type="cylinder" size="{cfg.wheel_radius_m:.3f} {cfg.wheel_half_width_m:.3f}" '
            f'euler="90 0 0" mass="{cfg.wheel_mass_kg:.1f}" material="tire"/>'
        )
        lines.append(f'{b_ind}\t</body>')

    # Rear drawbar + sites + nested next trailer
    if not is_last:
        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{b_ind}\t<site name="j_t{idx}_f{code}" '
                f'pos="-{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

        if has_rear_slider:
            stroke_r = r_max - r_min
            lines.append(f'{b_ind}\t<body name="t{idx}_rear_slider" pos="-{r_max:.3f} 0 0">')
            lines.append(
                f'{b_ind}\t\t<joint name="slide_rear_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_r:.4f}" damping="300"/>'
            )
            lines.append(f'{b_ind}\t\t<geom name="t{idx}_drawbar_rear" type="capsule" fromto="{r_max - cx:.3f} 0 0 0 0 0" size="0.025" mass="4.0" material="drawbar"/>')
            lines.append(_build_trailer_body(idx + 1, cfg, depth + (2 if is_lead else 3)))
            lines.append(f'{b_ind}\t</body>')
        else:
            lines.append(f'{b_ind}\t<geom name="t{idx}_drawbar_rear" type="capsule" fromto="-{cx:.3f} 0 0 -{r_max:.3f} 0 0" size="0.025" material="drawbar"/>')
            lines.append(_build_trailer_body(idx + 1, cfg, depth + (1 if is_lead else 2)))

    lines.append(f'{b_ind}</body>')
    if not is_lead:
        lines.append(f'{ind}</body>')
    return "\n".join(lines)

    ind = "\t" * depth
    is_lead = (idx == 1)
    is_last = (idx == cfg.num_trailers)
    is_var = (cfg.drawbar_mode == "variable")

    cx = cfg.chassis_half_len_m
    r_min, r_max = cfg.rear_bar_range_m
    f_min, f_max = cfg.front_bar_range_m

    mass = cfg.lead_dry_mass_kg if is_lead else cfg.trailer_dry_mass_kg
    mat = "chassis_lead" if is_lead else "chassis_trl"
    has_rear_slider = is_var and (r_max > r_min)
    if is_lead:
        body_pos = f"0 0 {cfg.wheel_radius_m - cfg.wheel_z_offset_m:.3f}"
    elif has_rear_slider:
        body_pos = f"{-f_max:.3f} 0 0"
    else:
        body_pos = f"{-(r_max + f_max):.3f} 0 0"

    
    lines = [f'{ind}<body name="trailer_{idx}" pos="{body_pos}">']

    if is_lead:
        lines.append(f'{ind}\t<freejoint name="root_jnt"/>')
    else:
        j_idx = idx - 1
        lines.append(
            f'{ind}\t<joint name="ball_joint_{j_idx}" type="ball" pos="{f_max:.3f} 0 0" '
            f'damping="{cfg.ball_joint_damping}" limited="true" range="0 {cfg.ball_joint_limit_deg}"/>'
        )
        if is_var and (f_max > f_min):
            stroke_f = f_max - f_min
            lines.append(
                f'{ind}\t<joint name="slide_front_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_f:.4f}" damping="300"/>'
            )
        lines.append(f'{ind}\t<geom name="hitch_{j_idx}_ball" type="sphere" pos="{f_max:.3f} 0 0" size="0.032" material="cable_mat"/>')
        lines.append(f'{ind}\t<geom name="t{idx}_drawbar_front" type="capsule" fromto="{f_max:.3f} 0 0 {cx:.3f} 0 0" size="0.025" material="drawbar"/>')
        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{ind}\t<site name="j_t{idx}_b{code}" '
                f'pos="{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

    lines.append(
        f'{ind}\t<geom name="t{idx}_chassis" type="box" '
        f'size="{cx:.3f} {cfg.chassis_half_width_m:.3f} {cfg.chassis_half_height_m:.3f}" mass="{mass:.1f}" material="{mat}"/>'
    )
    slosh_xml = _build_slosh_tank_xml(idx, cfg, ind + "\t")
    if slosh_xml:
        lines.append(slosh_xml)

    # Wheels
    for side, y_sign in (("l", 1), ("r", -1)):
        lines.append(f'{ind}\t<body name="t{idx}_{side}" pos="0 {y_sign * cfg.wheel_y_offset_m:.3f} {cfg.wheel_z_offset_m:.3f}">')
        lines.append(f'{ind}\t\t<joint name="j_t{idx}_{side}" type="hinge" axis="0 1 0"/>')
        lines.append(
            f'{ind}\t\t<geom type="cylinder" size="{cfg.wheel_radius_m:.3f} {cfg.wheel_half_width_m:.3f}" '
            f'euler="90 0 0" mass="{cfg.wheel_mass_kg:.1f}" material="tire"/>'
        )
        lines.append(f'{ind}\t</body>')

    # Rear drawbar + sites + nested next trailer
    if not is_last:
        for code, ys, zs in SITE_CORNERS:
            lines.append(
                f'{ind}\t<site name="j_t{idx}_f{code}" '
                f'pos="-{cx:.3f} {ys * cfg.cable_anchor_y_m:.3f} {zs * cfg.cable_anchor_z_m:.3f}" size="0.015"/>'
            )

        if is_var and (r_max > r_min):
            stroke_r = r_max - r_min
            lines.append(f'{ind}\t<body name="t{idx}_rear_slider" pos="-{r_max:.3f} 0 0">')
            lines.append(
                f'{ind}\t\t<joint name="slide_rear_{idx}" type="slide" axis="1 0 0" '
                f'limited="true" range="0 {stroke_r:.4f}" damping="300"/>'
            )
            lines.append(f'{ind}\t\t<geom name="t{idx}_drawbar_rear" type="capsule" fromto="{r_max - cx:.3f} 0 0 0 0 0" size="0.025" mass="4.0" material="drawbar"/>')
            lines.append(_build_trailer_body(idx + 1, cfg, depth + 2))
            lines.append(f'{ind}\t</body>')
        else:
            lines.append(f'{ind}\t<geom name="t{idx}_drawbar_rear" type="capsule" fromto="-{cx:.3f} 0 0 -{r_max:.3f} 0 0" size="0.025" material="drawbar"/>')
            lines.append(_build_trailer_body(idx + 1, cfg, depth + 1))

    lines.append(f'{ind}</body>')
    return "\n".join(lines)


def generate_mujoco_xml(cfg: AGVConfig) -> str:
    trailers_xml = _build_trailer_body(1, cfg)
    tendons, wheel_acts, capstan_acts, slide_acts = [], [], [], []
    is_var = (cfg.drawbar_mode == "variable")
    r_min, r_max = cfg.rear_bar_range_m
    f_min, f_max = cfg.front_bar_range_m

    for i in range(1, cfg.num_trailers + 1):
        kv = cfg.lead_wheel_kv if i == 1 else cfg.trailer_wheel_kv
        for side in ("l", "r"):
            wheel_acts.append(f'\t\t<velocity name="m_t{i}_{side}" joint="j_t{i}_{side}" kv="{kv}"/>')

    for i in range(1, cfg.num_trailers):
        nxt = i + 1
        for code, _, _ in SITE_CORNERS:
            t_name = f"t{i}_t{nxt}_{code}"
            tendons.append(
                f'\t\t<spatial name="{t_name}" width="0.005" damping="200" material="cable_mat">\n'
                f'\t\t\t<site site="j_t{i}_f{code}"/>\n'
                f'\t\t\t<site site="j_t{nxt}_b{code}"/>\n'
                f'\t\t</spatial>'
            )
            capstan_acts.append(
                f'\t\t<position name="cap_{t_name}" tendon="{t_name}" '
                f'kp="{cfg.capstan_kp}" kv="{cfg.capstan_kv}" forcerange="-{cfg.capstan_max_pull_n} 0"/>'
            )

        if is_var and (r_max > r_min):
            slide_acts.append(
                f'\t\t<position name="act_slide_rear_{i}" joint="slide_rear_{i}" '
                f'kp="{cfg.telescoping_kp}" kv="{cfg.telescoping_kv}" '
                f'forcerange="-{cfg.telescoping_max_force_n} {cfg.telescoping_max_force_n}"/>'
            )
        if is_var and (f_max > f_min):
            slide_acts.append(
                f'\t\t<position name="act_slide_front_{nxt}" joint="slide_front_{nxt}" '
                f'kp="{cfg.telescoping_kp}" kv="{cfg.telescoping_kv}" '
                f'forcerange="-{cfg.telescoping_max_force_n} {cfg.telescoping_max_force_n}"/>'
            )

    return f"""<mujoco model="agv_{cfg.num_trailers}trailer_{cfg.drawbar_mode}">
\t<option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
\t<visual>
\t\t<global offwidth="1280" offheight="720"/>
\t\t<rgba haze="0.15 0.25 0.35 1"/>
\t</visual>
\t<asset>
\t\t<texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.15 0.2" width="512" height="512"/>
\t\t<texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.36 0.22" rgb2="0.33 0.42 0.26" width="512" height="512"/>
\t\t<material name="paddock" texture="grid" texrepeat="20 20" reflectance="0.1"/>
\t\t<material name="chassis_lead" rgba="0.20 0.40 0.65 1"/>
\t\t<material name="chassis_trl" rgba="0.25 0.55 0.35 1"/>
\t\t<material name="drawbar" rgba="0.35 0.35 0.38 1"/>
\t\t<material name="tire" rgba="0.12 0.12 0.14 1"/>
\t\t<material name="cable_mat" rgba="0.90 0.25 0.15 1"/>
\t</asset>
\t<worldbody>
\t\t<light pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
\t\t<geom name="ground" type="plane" size="50 50 0.1" material="paddock" friction="0.85 0.005 0.0005"/>
\t\t<geom name="rut_1" type="box" pos="2.2 0.30 0.025" size="0.35 0.25 0.025" euler="0 5 0" material="paddock"/>
\t\t<geom name="rut_2" type="box" pos="3.6 -0.30 0.030" size="0.40 0.25 0.030" euler="0 -6 0" material="paddock"/>
\t\t<geom name="rut_3" type="box" pos="5.0 0.28 0.028" size="0.35 0.25 0.028" euler="4 0 0" material="paddock"/>
{trailers_xml}
\t</worldbody>
\t<tendon>
{chr(10).join(tendons)}
\t</tendon>
\t<actuator>
{chr(10).join(wheel_acts)}
{chr(10).join(capstan_acts)}
{chr(10).join(slide_acts)}
\t</actuator>
</mujoco>"""
