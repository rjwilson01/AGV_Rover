import mujoco
import mujoco.viewer
import numpy as np
import time

USE_IR_CAMERA_FEEDBACK = True

XML = """
<mujoco model="agv_3trailer_uturn">
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
    <geom name="ground" type="plane" size="50 50 0.1" material="paddock" friction="0.9 0.01 0.001"/>

    <!-- Simulated Paddock Ruts -->
    <geom name="rut_1" type="box" pos="2.8  0.32 0.025" size="0.35 0.28 0.025" euler="0 4 0" material="paddock"/>
    <geom name="rut_2" type="box" pos="4.2 -0.32 0.030" size="0.40 0.28 0.030" euler="0 -5 0" material="paddock"/>

    <!-- TRAILER 1 (LEAD POWERED UNIT - Single central axle at x=0, rear plate at x=-0.35, ball joint at x=-0.70) -->
    <body name="trailer_1" pos="0 0 0.32">
      <freejoint name="root_jnt"/>
      <geom type="box" size="0.35 0.25 0.12" mass="110" material="chassis_lead"/>
      <body name="t1_l" pos="0  0.32 -0.14"><joint name="j_t1_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
      <body name="t1_r" pos="0 -0.32 -0.14"><joint name="j_t1_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

      <!-- Rear Drawbar T1 (350mm) + 4 Capstan Fairlead Sites -->
      <geom type="capsule" fromto="-0.35 0 0 -0.70 0 0" size="0.025" material="drawbar"/>
      <site name="j1_a_tl" pos="-0.35  0.24  0.165" size="0.015"/>
      <site name="j1_a_tr" pos="-0.35 -0.24  0.165" size="0.015"/>
      <site name="j1_a_bl" pos="-0.35  0.24 -0.165" size="0.015"/>
      <site name="j1_a_br" pos="-0.35 -0.24 -0.165" size="0.015"/>

      <!-- BALL JOINT 1 -> TRAILER 2 (CENTER TRAILER, pivot-to-pivot = 1.40m) -->
      <body name="trailer_2" pos="-1.40 0 0">
        <joint name="ball_joint_1" type="ball" pos="0.70 0 0" damping="12"/>
        <geom type="sphere" pos="0.70 0 0" size="0.032" material="cable_mat"/>
        <geom type="capsule" fromto="0.70 0 0 0.35 0 0" size="0.025" material="drawbar"/>
        <geom type="box" size="0.35 0.25 0.12" mass="100" material="chassis_trl"/>

        <site name="j1_b_tl" pos="0.35  0.24  0.165" size="0.015"/>
        <site name="j1_b_tr" pos="0.35 -0.24  0.165" size="0.015"/>
        <site name="j1_b_bl" pos="0.35  0.24 -0.165" size="0.015"/>
        <site name="j1_b_br" pos="0.35 -0.24 -0.165" size="0.015"/>

        <body name="t2_l" pos="0  0.32 -0.14"><joint name="j_t2_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        <body name="t2_r" pos="0 -0.32 -0.14"><joint name="j_t2_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>

        <!-- Rear Drawbar T2 + 4 Capstan Fairlead Sites J2 -->
        <geom type="capsule" fromto="-0.35 0 0 -0.70 0 0" size="0.025" material="drawbar"/>
        <site name="j2_a_tl" pos="-0.35  0.24  0.165" size="0.015"/>
        <site name="j2_a_tr" pos="-0.35 -0.24  0.165" size="0.015"/>
        <site name="j2_a_bl" pos="-0.35  0.24 -0.165" size="0.015"/>
        <site name="j2_a_br" pos="-0.35 -0.24 -0.165" size="0.015"/>

        <!-- BALL JOINT 2 -> TRAILER 3 (REAR TRAILER) -->
        <body name="trailer_3" pos="-1.40 0 0">
          <joint name="ball_joint_2" type="ball" pos="0.70 0 0" damping="12"/>
          <geom type="sphere" pos="0.70 0 0" size="0.032" material="cable_mat"/>
          <geom type="capsule" fromto="0.70 0 0 0.35 0 0" size="0.025" material="drawbar"/>
          <geom type="box" size="0.35 0.25 0.12" mass="100" material="chassis_lead"/>

          <site name="j2_b_tl" pos="0.35  0.24  0.165" size="0.015"/>
          <site name="j2_b_tr" pos="0.35 -0.24  0.165" size="0.015"/>
          <site name="j2_b_bl" pos="0.35  0.24 -0.165" size="0.015"/>
          <site name="j2_b_br" pos="0.35 -0.24 -0.165" size="0.015"/>

          <body name="t3_l" pos="0  0.32 -0.14"><joint name="j_t3_l" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
          <body name="t3_r" pos="0 -0.32 -0.14"><joint name="j_t3_r" type="hinge" axis="0 1 0"/><geom type="cylinder" size="0.18 0.05" euler="90 0 0" mass="8" material="tire"/></body>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- 8 Spatial Tendons with 150 N/mm Inline Die Springs -->
  <tendon>
    <spatial name="t1_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="250"><site site="j1_a_tl"/><site site="j1_b_tl"/></spatial>
    <spatial name="t1_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="250"><site site="j1_a_tr"/><site site="j1_b_tr"/></spatial>
    <spatial name="t1_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="250"><site site="j1_a_bl"/><site site="j1_b_bl"/></spatial>
    <spatial name="t1_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="250"><site site="j1_a_br"/><site site="j1_b_br"/></spatial>

    <spatial name="t2_tl" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="250"><site site="j2_a_tl"/><site site="j2_b_tl"/></spatial>
    <spatial name="t2_tr" width="0.003" rgba="0.9 0.2 0.2 1" stiffness="150000" damping="250"><site site="j2_a_tr"/><site site="j2_b_tr"/></spatial>
    <spatial name="t2_bl" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="250"><site site="j2_a_bl"/><site site="j2_b_bl"/></spatial>
    <spatial name="t2_br" width="0.003" rgba="0.2 0.7 0.9 1" stiffness="150000" damping="250"><site site="j2_a_br"/><site site="j2_b_br"/></spatial>
  </tendon>

  <actuator>
    <velocity name="m_t1_l" joint="j_t1_l" kv="45"/>
    <velocity name="m_t1_r" joint="j_t1_r" kv="45"/>
    <velocity name="m_t2_l" joint="j_t2_l" kv="40"/>
    <velocity name="m_t2_r" joint="j_t2_r" kv="40"/>
    <velocity name="m_t3_l" joint="j_t3_l" kv="40"/>
    <velocity name="m_t3_r" joint="j_t3_r" kv="40"/>

    <position name="cap1_tl" tendon="t1_tl" kp="150000" forcerange="-8000 0"/>
    <position name="cap1_tr" tendon="t1_tr" kp="150000" forcerange="-8000 0"/>
    <position name="cap1_bl" tendon="t1_bl" kp="150000" forcerange="-8000 0"/>
    <position name="cap1_br" tendon="t1_br" kp="150000" forcerange="-8000 0"/>

    <position name="cap2_tl" tendon="t2_tl" kp="150000" forcerange="-8000 0"/>
    <position name="cap2_tr" tendon="t2_tr" kp="150000" forcerange="-8000 0"/>
    <position name="cap2_bl" tendon="t2_bl" kp="150000" forcerange="-8000 0"/>
    <position name="cap2_br" tendon="t2_br" kp="150000" forcerange="-8000 0"/>
  </actuator>
</mujoco>
"""

def calc_cable_len(yaw, pitch, roll, y_sign, z_sign, L_bar=0.35, W_y=0.24, H_z=0.165):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    bx, by, bz = L_bar, y_sign * W_y, z_sign * H_z
    wx = (cy*cp)*bx + (cy*sp*sr - sy*cr)*by + (cy*sp*cr + sy*sr)*bz
    wy = (sy*cp)*bx + (sy*sp*sr + cy*cr)*by + (sy*sp*cr - cy*sr)*bz
    wz = (-sp)*bx   + (cp*sr)*by            + (cp*cr)*bz
    return np.linalg.norm([wx + L_bar, wy - by, wz - bz])

def quat_to_zyx(q):
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

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

j1_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint_1")]
j2_qpos_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint_2")]

for i in range(6, 14):
    data.ctrl[i] = 0.698

signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
path_history = []
L_SEG = 0.70
TRACK = 0.64
WHEEL_R = 0.18

with mujoco.viewer.launch_passive(model, data) as viewer:
    dist_traveled = 0.0
    last_print = time.time()
    interval_peak_n = 0.0

    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # Read global heading of Lead AGV (Trailer 1)
        lead_yaw, _, _ = quat_to_zyx(data.qpos[3:7])
        heading_deg = np.rad2deg(lead_yaw)

        # Drive straight over ruts until t=5.0s, then execute a sustained 180-deg orchard U-turn
        v_center = 1.0
        if t < 5.0:
            yaw_target = 0.0
        elif heading_deg < 174.0 and t < 16.0:
            ramp = min(1.0, (t - 5.0) / 1.2)
            yaw_target = np.deg2rad(42.0) * ramp
        else:
            yaw_target = 0.0

        dist_traveled += v_center * model.opt.timestep
        path_history.append((dist_traveled, yaw_target))

        def lookup_yaw(delay_m):
            target_d = dist_traveled - delay_m
            for d, y in reversed(path_history):
                if d <= target_d:
                    return y
            return 0.0

        y1 = lookup_yaw(0.35)
        y2 = lookup_yaw(1.75)

        # Exact articulated turning radius R = L_seg / tan(psi / 2) so tires roll with zero scrub
        def set_axle_vels(ctrl_l, ctrl_r, psi):
            if abs(psi) < 1e-4:
                vl = vr = v_center
            else:
                R = L_SEG / np.tan(psi * 0.5)
                vl = v_center * (1.0 - TRACK / (2.0 * R))
                vr = v_center * (1.0 + TRACK / (2.0 * R))
            data.ctrl[ctrl_l] = vl / WHEEL_R
            data.ctrl[ctrl_r] = vr / WHEEL_R

        set_axle_vels(0, 1, y1)
        set_axle_vels(2, 3, 0.5 * (y1 + y2))
        set_axle_vels(4, 5, y2)

        # Measured IR camera pitch and roll across both ball joints
        _, m_p1, m_r1 = quat_to_zyx(data.qpos[j1_qpos_adr:j1_qpos_adr + 4])
        _, m_p2, m_r2 = quat_to_zyx(data.qpos[j2_qpos_adr:j2_qpos_adr + 4])

        p1_fb = m_p1 if USE_IR_CAMERA_FEEDBACK else 0.0
        r1_fb = m_r1 if USE_IR_CAMERA_FEEDBACK else 0.0
        p2_fb = m_p2 if USE_IR_CAMERA_FEEDBACK else 0.0
        r2_fb = m_r2 if USE_IR_CAMERA_FEEDBACK else 0.0

        for k, (ys, zs) in enumerate(signs):
            data.ctrl[6 + k]  = calc_cable_len(y1, p1_fb, r1_fb, ys, zs) - 0.002
            data.ctrl[10 + k] = calc_cable_len(y2, p2_fb, r2_fb, ys, zs) - 0.002

        mujoco.mj_step(model, data)
        viewer.sync()

        live_peak_n = float(np.max(np.abs(data.actuator_force[6:14])))
        if live_peak_n > interval_peak_n:
            interval_peak_n = live_peak_n

        if time.time() - last_print > 0.8:
            print(
                f"t={t:4.1f}s | Hdg={heading_deg:5.1f}° | J1={np.rad2deg(y1):+4.1f}° | J2={np.rad2deg(y2):+4.1f}° | "
                f"Interval Peak={interval_peak_n:6.1f} N (Live={live_peak_n:6.1f} N)"
            )
            interval_peak_n = 0.0
            last_print = time.time()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
