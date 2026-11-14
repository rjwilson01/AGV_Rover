import mujoco
import mujoco.viewer
import numpy as np
import time

# Toggle this between True and False to compare IR-camera roll/pitch compliance vs open-loop!
USE_IR_CAMERA_FEEDBACK = True

# 3-Trailer Snake AGV with 2 Active 4-Tendon Ball Joints (True 350mm Drawbars) & Paddock Ruts
XML = """
<mujoco model="agv_3trailer_snake_v2">
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
    <geom name="ground" type="plane" size="50 50 0.1" material="paddock" friction="0.85 0.01 0.001"/>

    <!-- Simulated Paddock Ruts / Bumps -->
    <geom name="rut_1" type="box" pos="3.5  0.32 0.03" size="0.4 0.3 0.03" euler="0 5 0" material="paddock"/>
    <geom name="rut_2" type="box" pos="5.2 -0.32 0.04" size="0.5 0.3 0.04" euler="0 -6 0" material="paddock"/>
    <geom name="rut_3" type="box" pos="7.0  0.30 0.035" size="0.4 0.3 0.035" euler="4 0 0" material="paddock"/>

    <!-- TRAILER 1 (LEAD POWERED AGV: body half-len=0.55m, rear drawbar=0.35m -> ball at -0.90m) -->
    <body name="trailer_1" pos="0 0 0.32">
      <freejoint name="root_jnt"/>
      <geom type="box" size="0.55 0.25 0.12" mass="120" material="chassis_lead"/>
      <!-- Wheels T1 (track width = 0.64m, wheel radius = 0.18m) -->
      <body name="t1_fl" pos="0.35 0.32 -0.14"><joint name="j_t1_fl" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
      <body name="t1_fr" pos="0.35 -0.32 -0.14"><joint name="j_t1_fr" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
      <body name="t1_rl" pos="-0.35 0.32 -0.14"><joint name="j_t1_rl" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
      <body name="t1_rr" pos="-0.35 -0.32 -0.14"><joint name="j_t1_rr" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

      <!-- Rear Drawbar T1 (350mm) + 4 Capstan Fairlead Sites -->
      <geom type="capsule" fromto="-0.55 0 0 -0.90 0 0" size="0.025" material="drawbar"/>
      <site name="j1_a_tl" pos="-0.55  0.24  0.165" size="0.015"/>
      <site name="j1_a_tr" pos="-0.55 -0.24  0.165" size="0.015"/>
      <site name="j1_a_bl" pos="-0.55  0.24 -0.165" size="0.015"/>
      <site name="j1_a_br" pos="-0.55 -0.24 -0.165" size="0.015"/>

      <!-- BALL JOINT 1 -> TRAILER 2 (front drawbar=0.35m, body half-len=0.55m -> center at -1.80m, ball at +0.90m local = 0.35m from front plate!) -->
      <body name="trailer_2" pos="-1.80 0 0">
        <joint name="ball_joint_1" type="ball" pos="0.90 0 0" damping="12"/>
        <geom type="sphere" pos="0.90 0 0" size="0.032" material="cable_mat"/>
        <geom type="capsule" fromto="0.90 0 0 0.55 0 0" size="0.025" material="drawbar"/>
        <geom type="box" size="0.55 0.25 0.12" mass="100" material="chassis_trl"/>

        <!-- Front Anchor Sites J1 (0.35m from ball_joint_1) -->
        <site name="j1_b_tl" pos="0.55  0.24  0.165" size="0.015"/>
        <site name="j1_b_tr" pos="0.55 -0.24  0.165" size="0.015"/>
        <site name="j1_b_bl" pos="0.55  0.24 -0.165" size="0.015"/>
        <site name="j1_b_br" pos="0.55 -0.24 -0.165" size="0.015"/>

        <!-- Wheels T2 -->
        <body name="t2_l" pos="0 0.32 -0.14"><joint name="j_t2_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        <body name="t2_r" pos="0 -0.32 -0.14"><joint name="j_t2_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

        <!-- Rear Drawbar T2 (350mm) + 4 Capstan Fairlead Sites J2 -->
        <geom type="capsule" fromto="-0.55 0 0 -0.90 0 0" size="0.025" material="drawbar"/>
        <site name="j2_a_tl" pos="-0.55  0.24  0.165" size="0.015"/>
        <site name="j2_a_tr" pos="-0.55 -0.24  0.165" size="0.015"/>
        <site name="j2_a_bl" pos="-0.55  0.24 -0.165" size="0.015"/>
        <site name="j2_a_br" pos="-0.55 -0.24 -0.165" size="0.015"/>

        <!-- BALL JOINT 2 -> TRAILER 3 -->
        <body name="trailer_3" pos="-1.80 0 0">
          <joint name="ball_joint_2" type="ball" pos="0.90 0 0" damping="12"/>
          <geom type="sphere" pos="0.90 0 0" size="0.032" material="cable_mat"/>
          <geom type="capsule" fromto="0.90 0 0 0.55 0 0" size="0.025" material="drawbar"/>
          <geom type="box" size="0.55 0.25 0.12" mass="100" material="chassis_lead"/>

          <!-- Front Anchor Sites J2 (0.35m from ball_joint_2) -->
          <site name="j2_b_tl" pos="0.55  0.24  0.165" size="0.015"/>
          <site name="j2_b_tr" pos="0.55 -0.24  0.165" size="0.015"/>
          <site name="j2_b_bl" pos="0.55  0.24 -0.165" size="0.015"/>
          <site name="j2_b_br" pos="0.55 -0.24 -0.165" size="0.015"/>

          <!-- Wheels T3 -->
          <body name="t3_l" pos="0 0.32 -0.14"><joint name="j_t3_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
          <body name="t3_r" pos="0 -0.32 -0.14"><joint name="j_t3_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 8 Spatial Tendons (4 per Ball Joint) with 150 N/mm Inline Die-Spring Stiffness -->
  <tendon>
    <spatial name="t1_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="200"><site site="j1_a_tl"/><site site="j1_b_tl"/></spatial>
    <spatial name="t1_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="200"><site site="j1_a_tr"/><site site="j1_b_tr"/></spatial>
    <spatial name="t1_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="200"><site site="j1_a_bl"/><site site="j1_b_bl"/></spatial>
    <spatial name="t1_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="200"><site site="j1_a_br"/><site site="j1_b_br"/></spatial>

    <spatial name="t2_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="200"><site site="j2_a_tl"/><site site="j2_b_tl"/></spatial>
    <spatial name="t2_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="200"><site site="j2_a_tr"/><site site="j2_b_tr"/></spatial>
    <spatial name="t2_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="200"><site site="j2_a_bl"/><site site="j2_b_bl"/></spatial>
    <spatial name="t2_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="200"><site site="j2_a_br"/><site site="j2_b_br"/></spatial>
  </tendon>

  <actuator>
    <!-- Standard Wheel Velocity Drives -->
    <velocity name="m_t1_fl" joint="j_t1_fl" kv="45"/>
    <velocity name="m_t1_fr" joint="j_t1_fr" kv="45"/>
    <velocity name="m_t1_rl" joint="j_t1_rl" kv="45"/>
    <velocity name="m_t1_rr" joint="j_t1_rr" kv="45"/>
    <velocity name="m_t2_l"  joint="j_t2_l"  kv="35"/>
    <velocity name="m_t2_r"  joint="j_t2_r"  kv="35"/>
    <velocity name="m_t3_l"  joint="j_t3_l"  kv="35"/>
    <velocity name="m_t3_r"  joint="j_t3_r"  kv="35"/>

    <!-- Capstan Position Actuators (Pull-only, uncapped up to 15,000 N so true peak is never truncated) -->
    <position name="cap1_tl" tendon="t1_tl" kp="150000" forcerange="-15000 0"/>
    <position name="cap1_tr" tendon="t1_tr" kp="150000" forcerange="-15000 0"/>
    <position name="cap1_bl" tendon="t1_bl" kp="150000" forcerange="-15000 0"/>
    <position name="cap1_br" tendon="t1_br" kp="150000" forcerange="-15000 0"/>

    <position name="cap2_tl" tendon="t2_tl" kp="150000" forcerange="-15000 0"/>
    <position name="cap2_tr" tendon="t2_tr" kp="150000" forcerange="-15000 0"/>
    <position name="cap2_bl" tendon="t2_bl" kp="150000" forcerange="-15000 0"/>
    <position name="cap2_br" tendon="t2_br" kp="150000" forcerange="-15000 0"/>
  </actuator>
</mujoco>
"""

def quat_to_zyx_euler(q):
    # q = [w, x, y, z] -> returns (yaw, pitch, roll) in radians
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll

def calc_cable_len(yaw, pitch, roll, y_sign, z_sign, L_bar=0.35, W_y=0.24, H_z=0.165):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy*cp)*bx + (cy*sp*sr - sy*cr)*by + (cy*sp*cr + sy*sr)*bz
    wy = (sy*cp)*bx + (sy*sp*sr + cy*cr)*by + (sy*sp*cr - cy*sr)*bz
    wz = (-sp)*bx   + (cp*sr)*by            + (cp*cr)*bz
    return np.linalg.norm([wx + L_bar, wy - by, wz - bz])

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

j1_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint_1")]
j2_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint_2")]

for i in range(8, 16):
    data.ctrl[i] = 0.698

signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
path_history = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    dist_traveled = 0.0
    last_print = time.time()
    interval_peak_n = 0.0

    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # Drive forward at ~1.1 m/s (6.1 rad/s) and execute smooth S-turns after t=5s
        base_omega = 6.1
        v_linear = base_omega * 0.18
        yaw_cmd = 0.0 if t < 5.0 else np.deg2rad(42.0 * np.sin(0.45 * (t - 5.0)))
        dist_traveled += v_linear * model.opt.timestep
        path_history.append((dist_traveled, yaw_cmd))

        def lookup_yaw(delay_m):
            target_d = dist_traveled - delay_m
            for d, y in reversed(path_history):
                if d <= target_d:
                    return y
            return 0.0

        # Joint 1 is 0.90m behind T1 center; Joint 2 is 2.70m behind T1 center
        y1 = lookup_yaw(0.90)
        y2 = lookup_yaw(2.70)

        # Read simulated IR Camera 3D relative pose for both ball joints
        meas_y1, meas_p1, meas_r1 = quat_to_zyx_euler(data.qpos[j1_qadr:j1_qadr+4])
        meas_y2, meas_p2, meas_r2 = quat_to_zyx_euler(data.qpos[j2_qadr:j2_qadr+4])

        # Ackermann-matched differential wheel speeds (track=0.64m, segment_len=1.80m -> kappa = 2*sin(yaw/2)/1.80)
        half_track = 0.32
        for idx in range(8):
            side = 1.0 if idx % 2 == 0 else -1.0  # +1 = Left wheel, -1 = Right wheel
            if idx < 4:
                kappa = (2.0 * np.sin(y1 * 0.5)) / 1.80
            elif idx < 6:
                kappa = (2.0 * np.sin((y1 + y2) * 0.25)) / 1.80
            else:
                kappa = (2.0 * np.sin(y2 * 0.5)) / 1.80
            data.ctrl[idx] = base_omega * (1.0 - side * half_track * kappa)

        # Command all 8 capstans with IR-camera measured pitch & roll compliance (minus 2mm preload)
        p1_in = meas_p1 if USE_IR_CAMERA_FEEDBACK else 0.0
        r1_in = meas_r1 if USE_IR_CAMERA_FEEDBACK else 0.0
        p2_in = meas_p2 if USE_IR_CAMERA_FEEDBACK else 0.0
        r2_in = meas_r2 if USE_IR_CAMERA_FEEDBACK else 0.0

        for k, (ys, zs) in enumerate(signs):
            data.ctrl[8 + k]  = calc_cable_len(y1, p1_in, r1_in, ys, zs) - 0.002
            data.ctrl[12 + k] = calc_cable_len(y2, p2_in, r2_in, ys, zs) - 0.002

        mujoco.mj_step(model, data)
        viewer.sync()

        # Track interval peak tension and reset every printout
        live_peak_n = float(np.max(np.abs(data.actuator_force[8:16])))
        if live_peak_n > interval_peak_n:
            interval_peak_n = live_peak_n

        if time.time() - last_print > 0.8:
            print(
                f"t={t:4.1f}s | J1 Yaw={np.rad2deg(y1):+5.1f}° (Roll={np.rad2deg(meas_r1):+4.1f}°) | "
                f"J2 Yaw={np.rad2deg(y2):+5.1f}° | Interval Peak={interval_peak_n:6.1f} N (Live={live_peak_n:6.1f} N)"
            )
            interval_peak_n = 0.0
            last_print = time.time()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
