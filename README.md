# AGV_Rover: snake AGV simulation

MuJoCo simulation of a "snake" farm AGV: several two-wheeled trailers (water tank + battery on each) joined by telescoping
drawbars and cables, driven by a leader / follow-the-leader controller. Two controller presets:

* **FixedDrawbarController**: fixed-length bars.
* **VariableDrawbarController**: telescoping bars with commanded length and hinge angle. The variable bar was introduced
  for two reasons: **(1) tighter, faster turns** (shorten the bars in a sharp curve) and **(2) less tyre skid**.
  Tyre skid has not been studied yet; stabilise first.

> This file is kept up to date as the project changes. The sections between `<!-- BEGIN:x -->` and `<!-- END:x -->` are
> generated: run `python make_readme.py` after changing any TOML, test file or command-line option.

## 1. Running

```
python main.py <Preset> [options]
```

On your machine the usual form is the PowerShell wrapper (pulls, tags the build, runs, then commits `sim_log.csv` with the
command as the commit message and pushes):

```
runtest VariableDrawbarController --tests 30-t_real_layout,t_w35_p03,t_wide_uturn,t_ideal_sensors
```

| Mode | Command | Writes |
|---|---|---|
| Normal run (viewer, runs until closed) | `python main.py VariableDrawbarController` | `sim_log.csv` (if `save_csv_log`) |
| Timed test sequence | `python main.py VariableDrawbarController --tests 30-t_real_layout,t_w35_p03` | `sim_test_log.csv` |
| **Manual model review** (physics on, controller off, keyboard set-points) | `python main.py VariableDrawbarController --view --tests 1-t_real_layout,t_w35_p03` | nothing (`--dump-xml FILE` writes the MJCF) |
| **Impulse / step response** (open-loop lead) | `python main.py VariableDrawbarController --tests 15-t_real_layout,t_w35_p03,t_two_trailers,t_ideal_sensors --drive 0:3:1.0:0,3:3.5:1.0:0.25,3.5:15:1.0:0` | `sim_test_log.csv` |
| **Sign-convention self-check** | `python main.py VariableDrawbarController --selfcheck --tests 1-t_real_layout,t_w35_p03` | `sim_test_log.csv` (verdicts as `# PASS/FAIL` lines) |
| Headless | add `--no-viewer` | as above |
| PlotJuggler | add `--plotjuggler` (off by default; UDP streaming is always on) | `plotjuggler.log` |

There are two log files, each overwritten by every run: runs given test parameters (`--tests`, `--set`, `--drive`, `--selfcheck`) write `sim_test_log.csv`; a plain run writes `sim_log.csv`; nothing is written if `save_csv_log` is off. Its header records `git_commit`, `build_tag` (from
`$env:BUILD_TAG`), the controller, test names and the sequence.

### Manual sandbox (`--view`)
Builds the model from the preset plus **one** set of test files (only the first group of `--tests` is used) and runs physics
with the **real controller in the loop on manual inputs**. A pop-out Tk window (falls back to an on-screen panel if Tk is not
available) has:

* a **checkbox per algorithm** (wheel law, bar-length law, tube-yaw law, pitch limit, tube geometry in the wheel law, measured vs
  commanded cables, ball pitch/roll hold): checked = the controller's output is used, unchecked = your set-point;
* **set-points** with -/+ buttons and keys (held keys and held buttons repeat): lead speed, lead curvature, the curvature the followers
  see, an all-wheel speed (for wheel law off), and per hitch the yaw, ball pitch/roll, rear and front bar length;
* a **live readout**: IR hitch angle, bar and tube yaw commanded vs measured, cable commanded vs actual length, wheel commands and the
  follower law's (v, w).

Keys in the window: Left/Right select, Up/Down (or +/-) change, PgUp/PgDn step size, Backspace zero, R reset, Space pause physics.
Default is `--base fixed` (the lead is welded to the world; wheels spin in the air or drive against the hitches).
The wheel law only gives different left/right speeds when the lead has a speed: its correction scales with `v`.
See `docs/ALGORITHMS.md` and `docs/architecture.drawio` for what each stage does.

| Option | Effect |
|---|---|
| `--base fixed` (default) | no root joint: the lead trailer is welded to the world, so only the hitches, bars, cables and wheels move |
| `--base planar` | the lead can slide in x and y only (no rotation or lifting) |
| `--base free` | normal free vehicle |
| `--no-friction` | floor friction 0 **and** `condim=1` on every geom. MuJoCo uses the larger friction of the two geoms in contact, so zeroing the floor alone would not be enough |

Example: `python main.py VariableDrawbarController --view --no-friction --tests 1-t_real_layout,t_w35_p03`

### Impulse / step response (`--drive`)
Replaces the lead's path following with an open-loop profile; the follower algorithm runs unchanged. Spec:
`start_s:end_s:speed_mps:curvature_per_m,...` (curvature > 0 = left turn; outside every window the lead stands still).
Handy two-trailer set-up: test file `t_two_trailers`. Examples:

| Experiment | `--drive` |
|---|---|
| Straight, 0.5 s curvature impulse, straight | `0:3:1.0:0,3:3.5:1.0:0.25,3.5:15:1.0:0` |
| Speed step 0.5 → 1.0 m/s | `0:4:0.5:0,4:15:1.0:0` |
| Constant left turn | `0:15:1.0:0.15` |

Watch `joint1/yaw_cmd_deg` vs `joint1/yaw_meas_deg`, `trailer2/*`, `tension/*`. In drive mode the followers use the lead's
steering history (there is no path to take a reference from). Add `--set controller.start_ramp_s=0` for a sharp start.

### Sign-convention self-check (`--selfcheck`)
Short open-loop moves in the real model with PASS/FAIL verdicts: hinge command sign, follower displacement, and a whole-chain
left and right turn. Conventions: x forward, y left, z up, yaw counter-clockwise positive; a follower's yaw relative to its
leader is **negative in a left turn**; in a left turn the **right** wheel is faster.

## 2. Configuration layers

1. `configs/common.toml`: shared hardware, sensors, test courses.
2. `configs/<Preset>.toml`: controller-specific values (deep-merged over common).
3. Test files (`configs/test/*.toml`), in the order named; later files win. A `[test]` table may hold `TestName` (written to the CSV header) and `duration_s`.
4. `--set section.key=value` (TOML syntax, repeatable): beats everything.

### Test sequence grammar (`--tests`)
`<seconds>(-|+)<file>[,<file>...]` groups, comma-joined. `-` resets to defaults before loading the files, `+` keeps the files
from the previous group. Duration = the number, else the file's `[test] duration_s`, else 30 s. Each group runs a fresh model.
Example: `20-t_real_layout,t_w35_p03,20+t_wide_uturn`.

## 3. How the control works

* **Lead steering:** pure pursuit on the planned path. Lookahead = max(`lookahead_dist_m`, `lookahead_gain_s` × speed).
  The curvature is low-passed and slew-limited in the distance domain (`curv_filter_m`, `curv_rate_limit_per_m`).
* **Path planner:** rounds waypoint corners with arcs and caps curvature at what the hitches can follow
  (`planner_margin` × the hitch yaw range). It warns when waypoints are too tight (e.g. `u_turn_180`; use `u_turn_wide`).
* **Followers follow an idealised path** (`follow_ideal_path`): their hinge commands and wheel law use the curvature of the
  smooth planned path at the lead's smoothed progress along it (monotone, rate-limited, `ref_progress_gain`), not the lead's
  own steering history. The lead's position matters only through that progress, so a wiggling lead does not wiggle the
  train. Before the start and after the end the reference is straight, so a stationary train holds a stable, defined
  posture. `t_follow_lead_history` restores the old behaviour.
* **Follower wheel speeds** (`wheel_speed_from_ir`): from the hitch kinematics and the measured IR yaw; followers never drop
  below `follower_min_speed_frac` of the lead's speed.
* **Follower stall guard:** followers stop if the lead is commanded to move but (by GNSS speed) is not.
* **Sensors (simulated):** IR camera (relative RPY of each trailer; perfect by default), BMI088 IMU per trailer, ZED-F9P-class
  RTK GNSS on the lead. IMU and GNSS are logged; only the GNSS speed feeds the stall guard. Test files
  `t_ideal_sensors` → `t_real_sensors` → `t_rough_field` stage the noise.
* **Mechanical checks at build time:** the build prints `[bars] WARNING` lines for rod/tube overlap and tube packaging.

## 4. Command-line options
<!-- BEGIN:cli -->
| Option | Description |
|---|---|
| `preset` | controller preset, e.g. VariableDrawbarController |
| `--tests SPEC` | test sequence, e.g. 200-a,b,c,100-d,e,100+f,g  (seconds, then - = reset to defaults / + = keep previous files, then files in configs/test/; each group runs once its files are loaded) |
| `--set SECTION.KEY=VALUE` | override one config value (TOML syntax); beats every test file. Repeatable. |
| `--plotjuggler` | launch PlotJuggler (off by default) |
| `--selfcheck` | measure the sign conventions with short open-loop moves, then exit |
| `--view` | MANUAL review: show the model (preset + the first --tests group) with physics running and every actuator on keyboard hot keys (listed on screen); no controller, CSV or PlotJuggler |
| `--base` | with --view (default fixed): 'fixed' welds the lead trailer to the world (only hitches/bars/wheels move), 'planar' lets it slide in x and y only, 'free' is normal |
| `--no-friction` | with --view: frictionless floor (floor friction 0 and condim=1 on all geoms) |
| `--dump-xml FILE` | with --view: also write the generated MJCF to FILE |
| `--drive SPEC` | open-loop lead profile for impulse/step-response tests: start:end:speed:curvature,... (replaces path following; followers run their normal algorithm) |
| `--no-viewer` | run headless, faster than real time |
<!-- END:cli -->

## 5. Parameter reference
<!-- BEGIN:params -->
Value shown = `common.toml`; the Variable / Fixed columns show a controller-specific override (blank = inherits).

### `[geometry]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `num_trailers` | 3 |  |  |  |
| `chassis_half_len_m` | 0.45 |  |  |  |
| `chassis_half_width_m` | 0.25 |  |  |  |
| `chassis_half_height_m` | 0.12 |  |  |  |
| `wheel_y_offset_m` | 0.32 |  |  |  |
| `wheel_z_offset_m` | -0.14 |  |  |  |
| `wheel_radius_m` | 0.18 |  |  |  |
| `wheel_half_width_m` | 0.05 |  |  |  |
| `wheel_mass_kg` | 8.0 |  |  |  |
| `lead_dry_mass_kg` | 110.0 |  |  |  |
| `trailer_dry_mass_kg` | 100.0 |  |  |  |
| `cable_anchor_y_m` | 0.24 |  |  |  |
| `cable_anchor_z_m` | 0.165 |  |  |  |
| `cable_preload_offset_m` | 0.002 |  |  |  |
| `ball_joint_limit_deg` | 45.0 | 55.0 |  |  |
| `ball_joint_damping` | 12.0 |  |  |  |

### `[tank_slosh]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `water_liters_per_trailer` | 120.0 |  |  |  |
| `slosh_mass_fraction` | 0.60 |  |  |  |
| `slosh_pendulum_len_m` | 0.18 |  |  |  |
| `slosh_roll_stiffness` | 65.0 |  |  |  |
| `slosh_pitch_stiffness` | 35.0 |  |  |  |
| `slosh_damping` | 18.0 |  |  |  |
| `slosh_roll_limit_deg` | 35.0 |  |  |  |
| `slosh_pitch_limit_deg` | 30.0 |  |  |  |

### `[actuators]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `lead_wheel_kv` | 45.0 |  |  |  |
| `trailer_wheel_kv` | 35.0 |  |  |  |
| `capstan_kp` | 20000.0 |  |  |  |
| `capstan_kv` | 500.0 |  |  |  |
| `capstan_max_pull_n` | 6000.0 |  |  |  |

### `[controller]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `active_path` | "paddock_turn" | "u_turn_180" |  |  |
| `goal_tol_m` | 0.25 |  |  |  |
| `follow_ideal_path` | true |  |  | followers take curvature from the smooth planned path (at the lead's progress), not the lead's own steering history |
| `ref_smooth_m` | 1.0 |  |  | moving-average length of the reference curvature (m) |
| `ref_progress_gain` | 1.5 |  |  | reference progress is limited to this x the distance the lead really travels |
| `lookahead_gain_s` | 2.0 |  |  | lookahead = max(lookahead_dist_m, gain x speed): keeps the steering loop stable |
| `curv_filter_m` | 0.5 |  |  | low-pass on the curvature command, in metres travelled (0 = off) |
| `curv_rate_limit_per_m` | 0.6 |  |  | max curvature change per metre travelled (0 = off) |
| `follower_min_speed_frac` | 0.3 |  |  | followers never stall below this fraction of the lead's speed |
| `planner_enabled` | true |  |  | Round waypoint corners with arcs and cap curvature at what the hitches can follow |
| `planner_margin` | 0.85 |  |  | fraction of the hitch yaw range the planner may use |
| `planner_spacing_m` | 0.05 |  |  |  |
| `planner_radius_cap_m` | 8.0 |  |  |  |
| `ref_dev_gain` | 1.0 |  |  | share of the lead's deviation from the ideal path passed to followers (0 = pure ideal path) |
| `ref_dev_filter_m` | 2.0 |  |  | distance low-pass on that deviation, m |
| `ref_dev_max` | 0.35 |  |  | clip on the deviation curvature, 1/m |
| `wheel_use_tube_geometry` | true |  |  | yaw-tube build: follower wheel speeds use the ball's true (off-axis) position from the tube yaws and bar lengths |
| `cable_use_measured_tubes` | true |  |  | yaw-tube build: size the hitch cables for the MEASURED tube angles (false = commanded, the old behaviour) |
| `bar_pitch_limit` | true |  |  | Variable build: shorten the bars when hitch pitch would put a bar end on the ground |
| `bar_ground_margin_m` | 0.06 |  |  | clearance kept under the bars, m |
| `planner_runout_m` | 3.0 |  |  | straight extension past the goal so the lookahead never collapses onto it; the train still stops at the real goal |
| `guard_enabled` | true |  |  | Followers stop if the lead is commanded to move but isn't (blocked/collided) |
| `guard_stall_ratio` | 0.3 |  |  | stalled when measured lead speed < this x commanded ... |
| `guard_stall_time_s` | 0.5 |  |  | ... for this long |
| `guard_release_ratio` | 0.6 |  |  | resume when lead speed recovers above this x commanded |
| `guard_min_cmd_mps` | 0.15 |  |  |  |
| `retension_guard` | false |  |  | COMPRESSION = FAILURE: when a hitch's cable-tension sum drops below its resting value the trailer behind is slowed at once |
| `retension_baseline_s` | 0.5 |  |  | resting cable sum is learned over the first this-many seconds |
| `retension_thresh_n` | 30.0 |  |  | compression declared this many N below the baseline |
| `retension_gain_per_n` | 0.005 |  |  | follower speed scale falls by this per N beyond the threshold |
| `retension_min_scale` | 0.5 |  |  |  |
| `retension_release_s` | 1.0 |  |  | recovery time from the minimum scale back to 1 |
| `guard_ramp_s` | 0.5 |  |  | follower speed ramps to/from zero over this long |
| `start_ramp_s` | 2.0 |  |  | Wheel speed ramps 0 -> full over this long (0 = step). Avoids a launch torque kick. |
| `use_ir_camera_feedback` | false |  |  |  |
| `cross_track_gain` | 0.55 |  |  | Pulls trailing axles onto the path (rad per meter of lateral error) |
| `joint_angle_kp` | 0.35 |  |  | IR-camera joint angle P-trim against cable/tire scrub |
| `joint_yaw_sign` | – | 1.0 |  | follower yaw relative to its leader is negative in a left turn |
| `wheel_speed_from_ir` | – | true |  | follower wheel speeds from the measured (IR) joint yaw and hitch kinematics |
| `v_center_mps` | – | 1.05 | 1.15 |  |
| `lookahead_dist_m` | – | 1.60 | 1.65 |  |
| `max_curvature` | – | 0.55 | 0.42 |  |
| `yaw_gain` | – | -1.00 | -1.35 |  |
| `actuator_lead_offset_m` | – | 0.35 | 0.40 |  |

### `[simulation]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `udp_host` | "127.0.0.1" |  |  |  |
| `udp_port` | 9870 |  |  |  |
| `stream_hz` | 50.0 |  |  |  |
| `save_csv_log` | true |  |  |  |
| `launch_plotjuggler` | false |  |  | true (or --plotjuggler) starts PlotJuggler automatically; UDP streaming stays on |
| `post_stop_log_sec` | 2.0 |  |  |  |

### `[viewer]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `camera_distance_m` | 6.0 |  |  |  |
| `camera_azimuth_deg` | 90.0 |  |  |  |
| `camera_elevation_deg` | -45.0 |  |  |  |
| `lead_speed_mps` | 0.0 |  |  | --view starts with the lead at this speed (the wheel law's straightening needs v > 0; raise it in the window) |
| `show_bar_overlay` | false |  |  | orange/green debug lines for commanded/measured bar direction |

### `[paths]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `paddock_turn` | [ |  |  |  |
| `u_turn_wide` | [ |  |  |  |
| `u_turn_180` | [ |  |  |  |

### `[imu]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `enabled` | true |  |  | One 6-axis IMU per trailer (Bosch BMI088). Values are datasheet figures; sample-and-hold at rate_hz. |
| `rate_hz` | 400 |  |  |  |
| `acc_noise_ug_rthz` | [160.0, 160.0, 190.0] |  |  | X, Y, Z (+-3 g range) |
| `acc_bw_hz` | 145.0 |  |  |  |
| `acc_offset_mg` | 20.0 |  |  |  |
| `acc_range_g` | 6.0 |  |  |  |
| `gyro_noise_dps_rthz` | 0.014 |  |  |  |
| `gyro_bw_hz` | 116.0 |  |  |  |
| `gyro_offset_dps` | 1.0 |  |  |  |
| `gyro_g_sens_dps_per_g` | 0.1 |  |  |  |
| `gyro_range_dps` | 500.0 |  |  |  |
| `vibration_g_rms` | 0.0 |  |  | add broadband accelerometer vibration (engine / cutter blade) |
| `seed` | 1 |  |  |  |

### `[loadcell]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `enabled` | true |  |  | Cable load cells at the fixed end of each cable (signed, N). The optional bar cell measures axial force through the bar. |
| `noise_n` | 1.0 |  |  | rms noise |
| `resolution_n` | 0.5 |  |  | quantisation step |
| `range_n` | 1000.0 |  |  | saturates beyond +-this |
| `bw_hz` | 50.0 |  |  | one-pole low-pass |
| `seed` | 3 |  |  |  |
| `bar_enabled` | false |  |  | optional bar axial-force reading, computed from the simulator's interaction force (nothing is added to the model); yaw-tube build only |
| `bar_force_sign` | 1.0 |  |  | -1 if a pull on the hitch reads negative (confirm with a tow test; compare_loads.py reports the slope sign) |

### `[gnss]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `enabled` | true |  |  | Lead-trailer RTK GNSS (u-blox ZED-F9P class). Used for the lead speed seen by the follower stall guard. |
| `rate_hz` | 10 |  |  |  |
| `latency_s` | 0.1 |  |  |  |
| `pos_noise_m` | 0.014 |  |  |  |
| `bias_m` | 0.01 |  |  | slowly varying position error (std) |
| `bias_tau_s` | 10.0 |  |  |  |
| `vel_noise_mps` | 0.05 |  |  |  |
| `outages_s` | [] |  |  | e.g. [[10.0, 14.0]] = no fix from t=10 to 14 s |
| `seed` | 2 |  |  |  |

### `[ir_camera]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `noise_std_deg` | 0.0 |  |  | Gaussian noise per axis (deg) |
| `latency_s` | 0.0 |  |  | measurement delay (s) |
| `quantization_deg` | 0.0 |  |  | output resolution (deg); 0 = none |
| `seed` | 0 |  |  |  |

### `[mass_layout]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `battery_mass_kg` | 0.0 |  |  | part of trailer_dry_mass_kg, modelled as its own block (kg) |
| `water_rigid_z_m` | 0.06 |  |  | height of the non-sloshing water block above the chassis centre (m) |

### `[bars]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `min_engagement_m` | 0.10 |  |  | build check: minimum rod-in-tube overlap at full extension (m) |
| `yaw_control` | – | true |  | Each drawbar sits in a tube that pivots (yaw) near the trailer edge. The simulation commands the tube yaw angle directly (the real build drives it with pull/pull cables); the bar length is the separate slider actuator. The ball joint between the two bar tips is passive, and the 4 trailer-to-trailer cables are kept as on the Fixed build. Values below are PLACEHOLDERS - replace with the real build. |
| `rear_yaw_range_deg` | – | [-25.0, 25.0] |  | commanded yaw range of each rear tube (rear bar of trailer n) |
| `front_yaw_range_deg` | – | [-25.0, 25.0] |  | commanded yaw range of each front tube (front bar of trailer n+1) |
| `pivot_inset_m` | – | 0.05 |  | tube pivot sits this far inside the chassis end |
| `yaw_kp` | – | 400.0 |  | tube yaw position actuator gain (N*m per rad) |
| `yaw_kv` | – | 20.0 |  | tube yaw actuator damping (N*m per rad/s) |
| `yaw_max_torque_nm` | – | 200.0 |  |  |
| `yaw_damping` | – | 2.0 |  | passive hinge damping |
| `yaw_hinge_margin_deg` | – | 5.0 |  | hinge stop sits this far beyond the commanded range |

### `[drawbar]`

| Key | Default | Variable | Fixed | Description |
|---|---|---|---|---|
| `controller_class` | – | "VariableDrawbarController" | "FixedDrawbarController" |  |
| `mode` | – | "variable" | "fixed" |  |
| `rear_bar_range_m` | – | [1.00, 1.50] | [0.80, 0.80] |  |
| `front_bar_range_m` | – | [1.00, 1.50] | [0.80, 0.80] |  |
| `yaw_angle_range_deg` | – | [-40.0, 50.0] | [-38.0, 38.0] | Limit on the commanded trailer-to-trailer yaw per hitch (rear bar yaw + front bar yaw + ball yaw). The per-bar tube limits are in [bars] below; ball_joint_limit_deg is the physical stop on the ball. |
| `telescoping_kp` | – | 18000.0 | 15000.0 |  |
| `telescoping_kv` | – | 900.0 | 800.0 |  |
| `telescoping_max_force_n` | – | 8000.0 | 8000.0 |  |

<!-- END:params -->

Parameters that exist only in `[mass_layout]` when set: `battery_z_m` (battery height; unset = axle height) and
`slosh_pivot_z_m` (slosh pendulum pivot; unset = chassis top + pendulum length).

## 6. Test files
<!-- BEGIN:tests -->
| File | What it does | Overrides |
|---|---|---|
| `example_softstart_off` | Hard launch (no soft start).  | controller.start_ramp_s=0.0 |
| `example_stationary` | Stationary, level cables. Example test file: any section/key from common.toml or the controller toml can be overridden here. [test] is special: TestName is written to the CSV header; duration_s is how long to run (an explicit seconds value in --tests, e.g. 200-example_stationary, takes precedence). | controller.v_center_mps=0.0, controller.use_ir_camera_feedback=false |
| `soft_start_on` | Soft start on (2 s ramp). Wheel speed ramps from 0 to full over 2 s instead of stepping (avoids the launch torque kick). Compare against example_softstart_off. If it works it is already the default in common.toml. | controller.start_ramp_s=2.0 |
| `t_axle_above_cg` | Axle above CG (+0.06).  | geometry.wheel_z_offset_m=0.06 |
| `t_axle_at_cg` | Axle at chassis centre. Axle at chassis-centre height instead of 0.14 m below it (CG above the axle is pitch-unstable per cart). The trailer body height follows automatically. Try +0.06 for CG below the axle (pendulum-stable). | geometry.wheel_z_offset_m=0.0 |
| `t_ball_damping` | Ball damping 200. Heavy damping on the ball joint (default 12) to damp relative pitch/roll between carts. | geometry.ball_joint_damping=200.0 |
| `t_batt_low` | Batteries on the chassis floor (-0.10). Batteries mounted low on the chassis floor (-0.10 m) instead of following the axle height. Lets the axle stay higher without raising the CG. Use together with t_real_layout and a wheel file. | mass_layout.battery_z_m=-0.10 |
| `t_batt_under` | Batteries slung low (-0.15). Batteries slung low, under the axle line and between the wheels (-0.15 m from the chassis centre) so the 65 kg sits as low as the ground clearance allows. Use with t_real_layout and a wheel file. | mass_layout.battery_z_m=-0.15 |
| `t_big_wheels_marginal` | Wheels 0.25 m, axle +0.04 (axle at CG). Axle just at the CG height (neutral pitch stability): body height 0.21 m. | geometry.wheel_radius_m=0.25, geometry.wheel_z_offset_m=0.04 |
| `t_big_wheels_r30` | Wheels 0.30 m, axle +0.08 (CG below axle, 0.10 m clearance). Same idea as t_big_wheels_stable but with ground clearance for the 5 cm ruts (chassis bottom was only 0.05 m up and snagged on them). Radius 0.30, axle +0.08 above the chassis centre: body height 0.22 m, chassis bottom 0.10 m off the ground, CG ~0.04 m below the axle. | geometry.wheel_radius_m=0.30, geometry.wheel_z_offset_m=0.08 |
| `t_big_wheels_stable` | Wheels 0.25 m, axle +0.08 (CG below axle). Each cart's CG is ~0.04 m above the chassis centre; the default axle is 0.14 m BELOW it, so a cart is an inverted pendulum. Bigger wheels let the axle sit ABOVE the CG without the chassis touching the ground: body height = radius - offset = 0.17 m, chassis bottom 0.05 m off the ground, CG ~0.04 m below the axle. | geometry.wheel_radius_m=0.25, geometry.wheel_z_offset_m=0.08 |
| `t_follow_lead_history` | Old follower reference: replay the lead's own steering history instead of the smooth planned path (A/B). | controller.follow_ideal_path=false |
| `t_gnss_outage` | Lead GNSS fix lost 8-12 s: guard must hold (not trip) and the run should carry on. | gnss.outages_s=[[8.0, 12.0]] |
| `t_ideal_hitch` | ideal light hitch rig. "Ideal" hitch-mechanics rig: nearly massless trailers, no water, light wheels. Use with  --view --base fixed --no-friction. Mass cannot be exactly zero: MuJoCo rejects zero-mass moving bodies, and a near-zero mass against stiff actuators makes the integration stiff. These are the lightest values that stay sensible with the default actuator gains (untested in MuJoCo). | geometry.lead_dry_mass_kg=10.0, geometry.trailer_dry_mass_kg=10.0, geometry.wheel_mass_kg=0.5, tank_slosh.water_liters_per_trailer=0.0 |
| `t_ideal_sensors` | Stage 1: perfect instruments. Any steering/path-following problem seen with this file is a control or physics problem, not a sensor problem. | ir_camera.noise_std_deg=0.0, ir_camera.latency_s=0.0, ir_camera.quantization_deg=0.0, imu.acc_noise_ug_rthz=[0.0, 0.0, 0.0], imu.acc_offset… |
| `t_no_dev` | Return-leg experiment: followers follow the pure ideal path (no share of the lead's deviation from it). | controller.ref_dev_gain=0.0 |
| `t_no_slosh` | Slosh cut 60x. Is the sloshing water driving the pitch mode? The moving mass is cut 60x (72 kg -> 1.2 kg) and its stiffness/damping are cut by the same factor, so the pendulum keeps its time constants (a tiny mass with the default damping is numerically unstable at a 2 ms step) but barely pushes on the carts. The water is not removed: the other 98% becomes fixed mass. | tank_slosh.slosh_mass_fraction=0.01, tank_slosh.slosh_roll_stiffness=1.08, tank_slosh.slosh_pitch_stiffness=0.58, tank_slosh.slosh_damping=… |
| `t_old_cables` | Return-leg experiment: size the hitch cables for the COMMANDED tube angles (behaviour before 24bcc2e). | controller.cable_use_measured_tubes=false |
| `t_planner_off` | Old behaviour: raw waypoint polyline, curvature clipped only by max_curvature. | controller.planner_enabled=false |
| `t_real_layout` | Real mass layout (batt at axle, water mid). Real mass layout, per cart: 110 kg dry INCLUDING a ~65 kg battery (10 kWh Li, 54 V) in a layer at axle height; 100 L of water spread over the full chassis footprint, so its centre is about the chassis centre. Water is nearly a full tank, so only 30% is modelled as slosh, hanging from a pivot at +0.12 m (bob at the centre). stiffness/damping scaled with the smaller slosh mass (30 kg vs 72 kg) so the pendulum stays well-conditioned | geometry.trailer_dry_mass_kg=110.0, mass_layout.battery_mass_kg=65.0, mass_layout.water_rigid_z_m=0.0, mass_layout.slosh_pivot_z_m=0.12, ta… |
| `t_real_sensors` | Stage 2: realistic instruments (IR camera noise, delay and quantisation; BMI088 and ZED-F9P figures are the defaults). Compare against t_ideal_sensors: the difference is what the sensors cost you. | ir_camera.noise_std_deg=0.3, ir_camera.latency_s=0.05, ir_camera.quantization_deg=0.1 |
| `t_rough_field` | Stage 3: add engine/cutter-blade vibration on the IMU on top of t_real_sensors. | imu.vibration_g_rms=0.3 |
| `t_short_bars` | short bars 0.80-1.00 m. Shorter telescoping bars (current 1.00-1.50 m look long against a 0.9 m chassis). Bar length = trailer centre to ball. Pivot is 0.40 m from the centre, so 0.80-1.00 m leaves 0.40-0.60 m of bar beyond the pivot. Needs a rebuild: set at startup only. | drawbar.rear_bar_range_m=[0.80, 1.00], drawbar.front_bar_range_m=[0.80, 1.00] |
| `t_slow_unwind` | Return-leg experiment: slower and with gentler curvature changes, so the chain has time to unwind after the U. | controller.v_center_mps=0.7, controller.curv_rate_limit_per_m=0.3 |
| `t_stiff_cables` | Stiff + preloaded cables. Stiffer cables and ~900 N preload so top AND bottom cables stay taut (currently ~4 N, top cables go slack). | geometry.cable_preload_offset_m=0.015, actuators.capstan_kp=60000.0 |
| `t_twitchy_old` | Old steering: fixed 1.6 m lookahead, raw curvature, followers may stall. For A/B against the smoothed default. | controller.lookahead_gain_s=0.0, controller.curv_filter_m=0.0, controller.curv_rate_limit_per_m=0.0, controller.follower_min_speed_frac=0.0 |
| `t_two_trailers` | Two-trailer train: the simplest set-up for impulse / step-response studies (use with --drive). | geometry.num_trailers=2 |
| `t_w30_p03` | Wheels 0.30 m, axle +0.03 (clearance 0.15). Wheels 0.30 m, axle 0.03 m ABOVE the chassis centre (batteries ride with the axle): body height 0.27 m, chassis clearance 0.15 m. Use together with t_real_layout. | geometry.wheel_radius_m=0.30, geometry.wheel_z_offset_m=0.03 |
| `t_w35_p03` | Wheels 0.35 m, axle +0.03 (clearance 0.20). Wheels 0.35 m, axle 0.03 m ABOVE the chassis centre: body height 0.32 m, chassis clearance 0.20 m (= original). | geometry.wheel_radius_m=0.35, geometry.wheel_z_offset_m=0.03 |
| `t_wheel_ir` | Wheel speeds from IR yaw. Follower wheel speeds from the MEASURED joint yaw (IR camera) via the hitch kinematics, instead of from the commanded yaw. Lead follows the pure-pursuit curvature. | controller.wheel_speed_from_ir=true |
| `t_wide_uturn` | Wider U-turn (radius >= ~3.4 m) that the hitch-limited planner can follow without saturating. | controller.active_path="u_turn_wide" |
| `t_yaw_flip` | Yaw command sign flipped. Flip the sign of the commanded hinge/ball yaw. The planar kinematics say a follower's yaw relative to its leader is NEGATIVE in a left turn, but the controller yaw is positive for a left turn and was sent to the hinges unchanged, bending the hitch the wrong way. | controller.joint_yaw_sign=-1.0 |
<!-- END:tests -->

## 7. The CSV log

Header lines start with `# ` (`log_version`, `git_commit`, `build_tag`, `started`, `controller`, `TestName`,
`test_sequence`, `segment_N`, then the configuration). Columns (a row per sample):

| Prefix | Content |
|---|---|
| `timestamp`, `test/segment`, `test/name` | time and which test segment |
| `pose/*` | lead position and heading |
| `joint<N>/*` | commanded vs measured hinge yaw/pitch/roll, bar lengths, tube yaw and torque |
| `trailer<N>/*` | commanded speed and yaw rate, wheel commands/speeds/torques, pitch, roll, height |
| `lead/*`, `ref/*` | raw and shaped curvature, lookahead; reference-path progress and curvature |
| `imu<N>/*` | simulated accelerometer/gyro and accelerometer tilt |
| `gnss/*` | simulated lead GNSS position, speed, validity, error |
| `guard/*` | follower stall guard scale and lead speed command/measured |
| `tension/*` | cable tension |

## 8. Known issues / open items

* The hinge actuators reach only part of a commanded angle with wheels held (about 10° of a 46° command in `--selfcheck`);
  the chain does not yet turn in the whole-chain self-check. Under investigation with `--view` and `--drive`.
* Telescoping bars keep only ~8 cm of rod in the tube at full extension (build warning); tubes cannot be housed inside the
  chassis (they would meet).
* The IMU is logged only; it does not yet feed control.
* Tyre-skid reduction (a purpose of the variable bar) is not yet studied.

## 9. Change log (newest first)

* Force window: slack cables are labelled SLACK, and an INTERNAL FIGHT banner appears when the cables carry far more than the resting sum while the bar pushes back (the two cancel in N). `--view` readout gains a 'formula check' per hitch: the cable-length formula evaluated at the measured pose minus the real tendon length (mm); ~0 means the formula includes the angles correctly.
* `--view` opens a second window with a LIVE force diagram of one hitch (`forceview.py`): cable tensions, bar axial force, wheel thrust, net force on the trailer behind and a COMPRESSION banner; buttons pick the hitch, tare the resting tension and scale the arrows. `--view` now starts with the lead at `viewer.lead_speed_mps` (default 0).
* Test files `t_short_bars` (bars 0.80-1.00 m instead of 1.00-1.50; tightest turn radius 2.62 m vs 3.27 m, no packaging warnings) and `t_ideal_hitch` (light trailers, no water). Bar ranges are read at startup only, so they are set in TOML.
* Simulated cable load cells (`loadcell.py`, `[loadcell]`; channels `jointJ/loadcell_*_N`, `cable_sum_N`, yaw/pitch moments), an optional bar axial-force reading computed from the simulator (no model element), `compare_loads.py` (is a bar sensor worthwhile?), and an optional compression guard `retension_guard` (default off) that slows the trailer behind a hitch that is being pushed. `docs/hitch_forces.drawio` shows the forces on one hitch; `docs/ALGORITHMS.md` covers the spring-oscillation question.
* `--view` is now a sandbox: the real controller runs on manual inputs, each algorithm has an on/off switch, pop-out Tk window with key/button repeat and a commanded-vs-measured readout (`sandbox.py`, `tkpanel.py`, `manual.py`); default `--base fixed`.
* Wheels carry a white cross so rotation is visible. `docs/ALGORITHMS.md` and `docs/architecture.drawio` (`make_diagram.py`) describe the algorithms.
* Follower wheel speeds now use the true hitch geometry of the yaw-tube build (`wheel_use_tube_geometry`): the ball is off the trailer axis when a tube is yawed, so the velocity-matching recursion uses its position from the tube yaws and bar lengths (reduces to the old law at zero tube yaw; verified against the rigid-body geometry).
* Hitch cables follow the MEASURED telescoping-bar lengths as well as the measured tube yaws (`cable_use_measured_tubes`), in the controller run and in `--view`.
* Return-leg experiment test files: `t_slow_unwind`, `t_old_cables`, `t_no_dev`.
* `--view --base free|fixed|planar` and `--view --no-friction` for hand-testing the hitches with the lead pinned and/or a frictionless floor.
* Hitch cables are sized for the MEASURED tube angles (`cable_use_measured_tubes`, default on) so a lagging tube no longer makes the cables fight it and swing the passive ball (ball yaw reached -52 deg on the return leg). `false` restores the old behaviour.
* `--view` is now a manual review with keyboard hot keys and an on-screen panel (`manual.py`); test-parameter runs write `sim_test_log.csv`, plain runs `sim_log.csv`.
* Variable build shortens the bars when hitch pitch would put a bar end on the ground (`bar_pitch_limit`, `bar_ground_margin_m`): max length = (axle height - margin) / sin|IR pitch|, never below the minimum.
* Followers now follow the ideal path plus the lead's smoothed deviation from it (`ref_dev_gain`, `ref_dev_filter_m`, `ref_dev_max`), so a lead that cuts a corner and corrects back no longer drags the chain the wrong way (return-leg jackknife). `ref_dev_gain=0` restores the pure ideal path.
* Return-leg fix: planned path gets a straight run-out (`planner_runout_m`, 3 m) past the goal so the lookahead and reference curvature don't collapse at the end; the train stops on arc-length progress to the real goal (not a geometric test), so the followers keep their reference until the lead actually arrives.
* Idealised reference path for followers (`follow_ideal_path`); `--view`, `--dump-xml`, `--drive`; this README and `make_readme.py`.
* `--selfcheck`; passed-end stop; speed-scaled lookahead and curvature shaping; follower speed floor.
* Steering-sign fix in the IR wheel mode; `joint_yaw_sign` back to 1.
* Simulated lead GNSS; follower stall guard; BMI088 IMU; curvature-limited path planner; staged sensor test files.
* Build tag (`BUILD_TAG`) in the CSV header; bar packaging checks; overlay and PlotJuggler off by default.
