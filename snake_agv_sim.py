import mujoco
import mujoco.viewer
import numpy as np
import time

# 3-Trailer Snake AGV: Single-Axle Articulated Segments + 2 Active 4-Tendon Ball Joints
XML = """
<mujoco model="agv_3trailer_snake_v4">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.36 0.22" rgb2="0.33 0.42 0.26" width="512" height="512"/>
    <material name="paddock" texture="grid" texrepeat="20 20" reflectance="0.1"/>
    <material name="chassis_lead" rgba="0.20 0.40 0.65 1"/>
    <material name="chassis_trl" rgba="0.25 0.55 0.35 1"/>
    <material name="drawbar" rgba="0.35 0.35 0.38 1"/>
    <material name="tire" rgba="0.12 0.12 0.14 1"/>
    <material name="cable_mat" rgba="0.90 0.25 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" size="50 50 0.1" material="paddock" friction="0.85 0.005 0.0005"/>

    <!-- Simulated Paddock Ruts / Bumps -->
    <geom name="rut_1" type="box" pos="2.2  0.30 0.025" size="0.35 0.25 0.025" euler="0 5 0" material="paddock"/>
    <geom name="rut_2" type="box" pos="3.6 -0.30 0.030" size="0.40 0.25 0.030" euler="0 -6 0" material="paddock"/>
    <geom name="rut_3" type="box" pos="5.0  0.28 0.028" size="0.35 0.25 0.028" euler="4 0 0" material="paddock"/>

    <!-- TRAILER 1 (LEAD POWERED AGV - Single Mid-Axle so it pivots cleanly as a snake) -->
    <body name="trailer_1" pos="0 0 0.32">
      <freejoint name="root_jnt"/>
      <geom type="box" size="0.45 0.25 0.12" mass="110" material="chassis_lead"/>
      <body name="t1_l" pos="0  0.32 -0.14"><joint name="j_t1_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
      <body name="t1_r" pos="0 -0.32 -0.14"><joint name="j_t1_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

      <!-- Rear Drawbar T1 (350mm: -0.45 to -0.80) + 4 Capstan Fairlead Sites -->
      <geom type="capsule" fromto="-0.45 0 0 -0.80 0 0" size="0.025" material="drawbar"/>
      <site name="j1_a_tl" pos="-0.45  0.24  0.165" size="0.015"/>
      <site name="j1_a_tr" pos="-0.45 -0.24  0.165" size="0.015"/>
      <site name="j1_a_bl" pos="-0.45  0.24 -0.165" size="0.015"/>
      <site name="j1_a_br" pos="-0.45 -0.24 -0.165" size="0.015"/>

      <!-- BALL JOINT 1 at (-0.80, 0, 0) -> TRAILER 2 (CENTER TRAILER at -1.60) -->
      <body name="trailer_2" pos="-1.60 0 0">
        <joint name="ball_joint_1" type="ball" pos="0.80 0 0" damping="12"/>
        <geom type="sphere" pos="0.80 0 0" size="0.032" material="cable_mat"/>
        <geom type="capsule" fromto="0.80 0 0 0.45 0 0" size="0.025" material="drawbar"/>
        <geom type="box" size="0.45 0.25 0.12" mass="100" material="chassis_trl"/>

        <!-- Front Anchor Sites J1 (on Trailer 2) -->
        <site name="j1_b_tl" pos="0.45  0.24  0.165" size="0.015"/>
        <site name="j1_b_tr" pos="0.45 -0.24  0.165" size="0.015"/>
        <site name="j1_b_bl" pos="0.45  0.24 -0.165" size="0.015"/>
        <site name="j1_b_br" pos="0.45 -0.24 -0.165" size="0.015"/>

        <!-- Wheels T2 -->
        <body name="t2_l" pos="0  0.32 -0.14"><joint name="j_t2_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        <body name="t2_r" pos="0 -0.32 -0.14"><joint name="j_t2_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

        <!-- Rear Drawbar T2 (350mm) + 4 Capstan Fairlead Sites J2 -->
        <geom type="capsule" fromto="-0.45 0 0 -0.80 0 0" size="0.025" material="drawbar"/>
        <site name="j2_a_tl" pos="-0.45  0.24  0.165" size="0.015"/>
        <site name="j2_a_tr" pos="-0.45 -0.24  0.165" size="0.015"/>
        <site name="j2_a_bl" pos="-0.45  0.24 -0.165" size="0.015"/>
        <site name="j2_a_br" pos="-0.45 -0.24 -0.165" size="0.015"/>

        <!-- BALL JOINT 2 at (-0.80, 0, 0) -> TRAILER 3 (REAR TRAILER at -1.60) -->
        <body name="trailer_3" pos="-1.60 0 0">
          <joint name="ball_joint_2" type="ball" pos="0.80 0 0" damping="12"/>
          <geom type="sphere" pos="0.80 0 0" size="0.032" material="cable_mat"/>
          <geom type="capsule" fromto="0.80 0 0 0.45 0 0" size="0.025" material="drawbar"/>
          <geom type="box" size="0.45 0.25 0.12" mass="100" material="chassis_lead"/>

          <!-- Front Anchor Sites J2 (on Trailer 3) -->
          <site name="j2_b_tl" pos="0.45  0.24  0.165" size="0.015"/>
          <site name="j2_b_tr" pos="0.45 -0.24  0.165" size="0.015"/>
          <site name="j2_b_bl" pos="0.45  0.24 -0.165" size="0.015"/>
          <site name="j2_b_br" pos="0.45 -0.24 -0.165" size="0.015"/>

          <!-- Wheels T3 -->
          <body name="t3_l" pos="0  0.32 -0.14"><joint name="j_t3_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
          <body name="t3_r" pos="0 -0.32 -0.14"><joint name="j_t3_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 8 Spatial Tendons (4 per Ball Joint) with Inline Die-Spring Stiffness (150 N/mm) -->
  <tendon>
    <spatial name="t1_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="180"><site site="j1_a_tl"/><site site="j1_b_tl"/></spatial>
    <spatial name="t1_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="180"><site site="j1_a_tr"/><site site="j1_b_tr"/></spatial>
    <spatial name="t1_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="180"><site site="j1_a_bl"/><site site="j1_b_bl"/></spatial>
    <spatial name="t1_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="180"><site site="j1_a_br"/><site site="j1_b_br"/></spatial>

    <spatial name="t2_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="180"><site site="j2_a_tl"/><site site="j2_b_tl"/></spatial>
    <spatial name="t2_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="180"><site site="j2_a_tr"/><site site="j2_b_tr"/></spatial>
    <spatial name="t2_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="180"><site site="j2_a_bl"/><site site="j2_b_bl"/></spatial>
    <spatial name="t2_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="180"><site site="j2_a_br"/><site site="j2_b_br"/></spatial>
  </tendon>

  <actuator>
    <!-- Standard Wheel Velocity Drives (6 wheels across 3 single-axle segments) -->
    <velocity name="m_t1_l" joint="j_t1_l" kv="45"/>
    <velocity name="m_t1_r" joint="j_t1_r" kv="45"/>
    <velocity name="m_t2_l" joint="j_t2_l" kv="40"/>
    <velocity name="m_t2_r" joint="j_t2_r" kv="40"/>
    <velocity name="m_t3_l" joint="j_t3_l" kv="40"/>
    <velocity name="m_t3_r" joint="j_t3_r" kv="40"/>

    <!-- Capstan Position Actuators (Pull-only via force range [-6000, 0] N) -->
    <position name="cap1_tl" tendon="t1_tl" kp="150000" forcerange="-6000 0"/>
    <position name="cap1_tr" tendon="t1_tr" kp="150000" forcerange="-6000 0"/>
    <position name="cap1_bl" tendon="t1_bl" kp="150000" forcerange="-6000 0"/>
    <position name="cap1_br" tendon="t1_br" kp="150000" forcerange="-6000 0"/>

    <position name="cap2_tl" tendon="t2_tl" kp="150000" forcerange="-6000 0"/>
    <position name="cap2_tr" tendon="t2_tr" kp="150000" forcerange="-6000 0"/>
    <position name="cap2_bl" tendon="t2_bl" kp="150000" forcerange="-6000 0"/>
    <position name="cap2_br" tendon="t2_br" kp="150000" forcerange="-6000 0"/>
  </actuator>
</mujoco>
"""

USE_IR_CAMERA_FEEDBACK = True

def calc_cable_len(yaw, pitch, roll, y_sign, z_sign, L_bar=0.35, W_y=0.24, H_z=0.165):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy*cp)*bx + (cy*sp*sr - sy*cr)*by + (cy*sp*cr + sy*sr)*bz
    wy = (sy*cp)*bx + (sy*sp*sr + cy*cr)*by + (sy*sp*cr - cy*sr)*bz
    wz = (-sp)*bx   + (cp*sr)*by            + (cp*cr)*bz
    return np.linalg.norm([wx + L_bar, wy - by, wz - bz])

def quat_to_euler(q):
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return yaw, pitch, roll

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# Neutral cable length = 0.700m minus 2mm preload (300 N baseline)
for i in range(6, 14):
    data.ctrl[i] = 0.698

signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
path_history = []
L_SEG = 0.80      # Half-segment distance from mid-axle to ball joint (m)
TRACK_W = 0.64    # Wheel track width (m)
WHEEL_R = 0.18    # Wheel radius (m)

with mujoco.viewer.launch_passive(model, data) as viewer:
    dist_traveled = 0.0
    last_print = time.time()
    interval_peak = 0.0
    turn_complete = False

    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # Measure lead AGV heading
        _, _, lead_hdg = quat_to_euler(data.qpos[3:7])
        lead_hdg_deg = np.rad2deg(lead_hdg)

        # Drive straight over ruts (t < 5.0s), then execute full 180 deg U-turn
        v_center = 1.15  # m/s
        if t < 5.0 or turn_complete:
            target_yaw = 0.0
        else:
            ramp = min(1.0, (t - 5.0) / 1.5)
            target_yaw = np.deg2rad(44.0) * ramp
            if abs(lead_hdg_deg) >= 176.0:
                turn_complete = True
                target_yaw = 0.0

        dist_traveled += v_center * model.opt.timestep
        path_history.append((dist_traveled, target_yaw))

        def lookup_yaw(delay_m):
            target_d = dist_traveled - delay_m
            for d, y in reversed(path_history):
                if d <= target_d:
                    return y
            return 0.0

        y1 = lookup_yaw(0.40)
        y2 = lookup_yaw(2.00)

        # Matched Ackermann differential wheel speeds for each segment
        seg_yaws = [y1, 0.5 * (y1 + y2), y2]
        for s_idx, psi in enumerate(seg_yaws):
            if abs(psi) < 1e-4:
                wl = wr = v_center / WHEEL_R
            else:
                R = L_SEG / np.tan(psi / 2.0)
                wl = (v_center * (1.0 - TRACK_W / (2.0 * R))) / WHEEL_R
                wr = (v_center * (1.0 + TRACK_W / (2.0 * R))) / WHEEL_R
            data.ctrl[2 * s_idx]     = wl
            data.ctrl[2 * s_idx + 1] = wr

        # Simulated IR camera measured pitch & roll across Joint 1 and Joint 2
        if USE_IR_CAMERA_FEEDBACK:
            _, p1_m, r1_m = quat_to_euler(data.qpos[9:13])
            _, p2_m, r2_m = quat_to_euler(data.qpos[15:19])
        else:
            p1_m = r1_m = p2_m = r2_m = 0.0

        # Note -y1, -y2 sign match for rear-mounted trailers along -X
        for k, (ys, zs) in enumerate(signs):
            data.ctrl[6 + k]  = calc_cable_len(-y1, p1_m, r1_m, ys, zs) - 0.002
            data.ctrl[10 + k] = calc_cable_len(-y2, p2_m, r2_m, ys, zs) - 0.002

        mujoco.mj_step(model, data)
        viewer.sync()

        live_peak = float(np.max(np.abs(data.actuator_force[6:14])))
        if live_peak > interval_peak:
            interval_peak = live_peak

        if time.time() - last_print > 0.6:
            print(
                f"t={t:4.1f}s | Hdg={lead_hdg_deg:+6.1f}° | J1={np.rad2deg(y1):+5.1f}° | J2={np.rad2deg(y2):+5.1f}° | "
                f"Interval Peak={interval_peak:6.1f} N (Live={live_peak:6.1f} N)"
            )
            interval_peak = 0.0
            last_print = time.time()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
