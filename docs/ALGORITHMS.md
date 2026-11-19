# What each algorithm does (and what it does not)

Diagram: `docs/architecture.drawio` (open in https://app.diagrams.net; regenerate with `python make_diagram.py`).
Conventions: x forward, y left, z up, yaw counter-clockwise positive. In a left turn a follower's yaw relative to its leader
is negative and the RIGHT wheel is faster. `kappa` is path curvature, 1/m.

## Lead (steering)
1. **Planner** (`planner.py`): rounds the waypoint corners so the curvature never exceeds what the hitches can follow
   (`planner_margin` x the hitch yaw range, shortest bars). Adds a 3 m straight run-out past the goal.
2. **Pure pursuit**: target = point on the path one lookahead ahead, `L = max(lookahead_dist_m, lookahead_gain_s * v)`;
   `kappa_raw = 2 y_local / L^2`.
3. **Shaping**: distance-domain low-pass (`curv_filter_m`) and slew limit (`curv_rate_limit_per_m`) -> `kappa_lead`.
4. Lead wheels: `v = v_center` (start ramp), `w = v * kappa_lead`.

## What the followers are told
`kappa_follow(j) = kappa_ref(s_ref - delay_j) + ref_dev_gain * deviation(delay_j)`
* `kappa_ref`: curvature of the smoothed planned path at the lead's progress `s_ref` (monotone, rate-limited), `delay_j` = distance
  back to hitch j.
* `deviation`: `kappa_lead - kappa_ref` low-passed over `ref_dev_filter_m`, clipped to `ref_dev_max`.
* **Stall guard**: followers are scaled down if the lead's GNSS speed lags its command (stops the followers shoving a stalled lead).

## Hitch laws (Variable build, per hitch j) - all FEED-FORWARD on `kappa_follow`
* **Bar length** (tightness law): `psi_full = g (atan(r_max k) + atan(f_max k))`; `tightness = |psi_full| / yaw_limit` (0..1);
  `r = r_max - tightness (r_max - r_min)`, same for `f`. Straight path -> bars fully extended; tight turn -> bars shortened so the
  hitch can reach the angle. Then shortened further if the measured hitch **pitch** would put a bar end on the ground.
* **Tube yaw**: `psi_r = g atan(r k)`, `psi_f = g atan(f k)` (g = `yaw_gain`, negative), each clipped to its tube range; the
  total is clipped to the hitch yaw range; the passive ball takes whatever the tubes do not.
* **Cables**: four per hitch; each commanded to the distance between its two anchors for the current geometry, minus a preload offset.
  With `cable_use_measured_tubes` (default) the geometry is the MEASURED tube yaws and bar lengths, so a lagging tube or bar does
  not make the cables fight it. Longer bar -> longer cables; the computation includes the bar lengths and the trailer spacing
  (`bar_geometry.trailer_cable_len`).

## Wheel law (the part that uses the measured hitch angle)
For trailer j with speed `v_j` and yaw rate `w_j`, and the MEASURED hitch yaw `psi` (IR camera): the ball is the hitch point.
Its position in trailer j is `b = (-p - s_r cos psi_r, -s_r sin psi_r)` and in trailer j+1 `a = (p + s_f cos psi_f, -s_f sin psi_f)`
(`p` = tube pivot distance, `s` = pivot-to-ball reach). Match the ball's velocity on both sides and forbid sideways slip of trailer
j+1's axle:
```
v_ball = (v_j - w_j b_y, w_j b_x)  rotated into trailer j+1's frame by -psi
w_{j+1} = v_ball_y / a_x ;   v_{j+1} = v_ball_x + w_{j+1} a_y
```
With `psi_r = psi_f = 0` this is `v' = v cos psi - w L_r sin psi`, `w' = -(v sin psi + w L_r cos psi)/L_f`.
So if the hitch is bent and the lead drives straight, `w' = -v sin(psi)/L_f`: the follower steers to **straighten the hitch** - its
left/right wheel speeds differ. This needs the lead to be moving (`v > 0`): the correction scales with speed. Then
`left = (v - w half_track)/r`, `right = (v + w half_track)/r`.

## What is NOT there
* Bar lengths and tube yaws do **not** respond to the measured hitch angle; they only follow the path curvature. The only feedback on
  the measured angle is the wheel law above (and the bar pitch limit, and ball pitch/roll hold).
* The relationship between bar length, the two tube yaws and non-skidding wheels is only in the wheel law (it uses the commanded
  tube yaws and bar lengths, since the controller sees only the total hitch yaw). Tyre skid is not measured or minimised yet.
* IMU data is logged only.

## `--view` sandbox
The pop-out window runs this exact controller on manual inputs. Each checkbox picks, for one stage, the controller's output (checked)
or your set-point (unchecked): wheel law, bar-length law, tube-yaw law, pitch limit, tube geometry in the wheel law, measured-vs-commanded
cables, ball pitch/roll hold. The readout shows commanded vs measured for bars, tubes, cables and wheels.
