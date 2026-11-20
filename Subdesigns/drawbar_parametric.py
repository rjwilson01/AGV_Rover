import math
import time
import mujoco
import mujoco.viewer
import numpy as np

# ==============================================================================
# 1. USER PARAMETERS: Set your drawbar span and end angles here
# ==============================================================================
DRAWBAR_SPAN = 0.80       # Total distance between Pivot 1 and Pivot 2 (meters)
ANGLE1_DEG   = 20.0       # Angle of Bar 1 relative to drawbar centerline (degrees)
ANGLE2_DEG   = 35.0       # Angle of Bar 2 relative to drawbar centerline (degrees)
HINGE_AXIS   = "0 0 1"    # "0 0 1" = Yaw (horizontal swing), "0 1 0" = Pitch (vertical)

# Initial external force applied to the Front End (Pivot 1 block) in Newtons [Fx, Fy, Fz]
INITIAL_FORCE = [200.0, 100.0, 0.0]
# ==============================================================================


def build_drawbar_xml(span: float, deg1: float, deg2: float, axis: str) -> str:
    """Computes exact triangle geometry so Bar 1 and Bar 2 meet at the ball joint without pre-stress."""
    a1 = math.radians(deg1)
    a2 = math.radians(deg2)
    a3 = math.pi - (a1 + a2)

    if a3 <= 0.05:
        raise ValueError("ANGLE1_DEG + ANGLE2_DEG must be less than 175 degrees so the triangle closes.")

    # Law of Sines: L1 / sin(a2) = L2 / sin(a1) = span / sin(a3)
    L1 = span * math.sin(a2) / math.sin(a3)
    L2 = span * math.sin(a1) / math.sin(a3)

    # Ball joint coordinates relative to Pivot 1 (at x=0, y=0)
    bx = L1 * math.cos(a1)
    by = L1 * math.sin(a1)

    return f"""
<mujoco model="parametric_drawbar_ball_joint">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  
  <!-- Gravity disabled so readouts show pure transmitted forces from your applied load -->
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast">
    <flag warmstart="enable"/>
  </option>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <scale forcewidth="0.025" contactwidth="0.04" contactheight="0.04" jointlength="0.12" jointwidth="0.012"/>
    <rgba force="1 0.2 0.2 1"/>
  </visual>

  <worldbody>
    <light pos="0.4 -1.2 2.0" dir="0 0.5 -0.8" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 -0.3" size="2 2 0.1" rgba="0.88 0.88 0.88 1"/>

    <!-- ================================================================= -->
    <!-- REAR BASE BLOCK (Fixed Reaction Anchor holding Pivot 2)           -->
    <!-- ================================================================= -->
    <body name="rear_block" pos="{span:.6f} 0 0.2">
      <geom name="rear_housing" type="box" pos="0.06 0 0" size="0.06 0.08 0.06" rgba="0.25 0.3 0.35 1" mass="50"/>
      
      <!-- Pivot 2: True 1-DoF Hinge mounted on the Rear Block -->
      <body name="pivot2_link" pos="0 0 0">
        <joint name="pivot2_hinge" type="hinge" axis="{axis}" damping="0.2"/>
        <site name="pivot2_sensor_site" pos="0 0 0" size="0.014" rgba="0 1 0 1"/>
        <geom name="pivot2_pin" type="cylinder" fromto="0 0 -0.05 0 0 0.05" size="0.018" rgba="0.8 0.3 0.2 1" mass="0.2"/>

        <!-- Bar 2 extends from Pivot 2 to the Ball Joint -->
        <geom name="bar2_geom" type="capsule" fromto="0 0 0  {bx - span:.6f} {by:.6f} 0" size="0.018" rgba="0.2 0.75 0.55 1" mass="1.0"/>
        <site name="ball_end_bar2" pos="{bx - span:.6f} {by:.6f} 0" size="0.014" rgba="0 1 1 1"/>
      </body>
    </body>

    <!-- ================================================================= -->
    <!-- FRONT LOAD BLOCK (Applies Forces into Pivot 1)                    -->
    <!-- ================================================================= -->
    <body name="front_block" pos="0 0 0.2">
      <!-- Free joint allows the front block to transfer all applied forces directly into the linkage -->
      <freejoint name="front_free"/>
      <geom name="front_housing" type="box" pos="-0.06 0 0" size="0.06 0.08 0.06" rgba="0.35 0.35 0.45 1" mass="2.0"/>
      <site name="force_input_site" pos="-0.06 0 0" size="0.016" rgba="1 0 0 1"/>

      <!-- Pivot 1: True 1-DoF Hinge mounted on the Front Block -->
      <body name="pivot1_link" pos="0 0 0">
        <joint name="pivot1_hinge" type="hinge" axis="{axis}" damping="0.2"/>
        <site name="pivot1_sensor_site" pos="0 0 0" size="0.014" rgba="1 0.6 0 1"/>
        <geom name="pivot1_pin" type="cylinder" fromto="0 0 -0.05 0 0 0.05" size="0.018" rgba="0.8 0.3 0.2 1" mass="0.2"/>

        <!-- Bar 1 extends from Pivot 1 to the Ball Joint -->
        <geom name="bar1_geom" type="capsule" fromto="0 0 0  {bx:.6f} {by:.6f} 0" size="0.018" rgba="0.2 0.6 0.85 1" mass="1.0"/>

        <!-- ============================================================= -->
        <!-- CENTRAL 3-DoF BALL JOINT                                      -->
        <!-- ============================================================= -->
        <body name="ball_socket_body" pos="{bx:.6f} {by:.6f} 0">
          <joint name="center_ball_joint" type="ball" damping="0.05"/>
          <geom name="ball_visual" type="sphere" pos="0 0 0" size="0.032" rgba="0.95 0.75 0.1 1" mass="0.3"/>
          <site name="ball_end_bar1" pos="0 0 0" size="0.014" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- =================================================================== -->
  <!-- LOOP CLOSURE: Welds Bar 2's tip to the Ball Joint's rotating socket -->
  <!-- =================================================================== -->
  <equality>
    <weld name="ball_socket_weld" site1="ball_end_bar1" site2="ball_end_bar2" solref="0.002 1" solimp="0.95 0.99 0.001"/>
  </equality>

  <!-- =================================================================== -->
  <!-- ACTUATORS: Apply forces to the Front End (Pivot 1 block)            -->
  <!-- =================================================================== -->
  <actuator>
    <general name="front_force_X" site="force_input_site" gear="1 0 0 0 0 0" ctrllimited="true" ctrlrange="-2000 2000"/>
    <general name="front_force_Y" site="force_input_site" gear="0 1 0 0 0 0" ctrllimited="true" ctrlrange="-2000 2000"/>
    <general name="front_force_Z" site="force_input_site" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="-2000 2000"/>
  </actuator>

  <!-- =================================================================== -->
  <!-- SENSORS: Read Forces & Torques at Front End, Ball Joint, & Rear End -->
  <!-- =================================================================== -->
  <sensor>
    <force  name="front_pivot1_force"  site="pivot1_sensor_site"/>
    <torque name="front_pivot1_torque" site="pivot1_sensor_site"/>

    <force  name="ball_joint_force"    site="ball_end_bar1"/>
    <torque name="ball_joint_torque"   site="ball_end_bar1"/>

    <force  name="rear_pivot2_force"   site="pivot2_sensor_site"/>
    <torque name="rear_pivot2_torque"  site="pivot2_sensor_site"/>
  </sensor>
</mujoco>
"""


xml_string = build_drawbar_xml(DRAWBAR_SPAN, ANGLE1_DEG, ANGLE2_DEG, HINGE_AXIS)
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Apply initial forces to the front end
data.actuator("front_force_X").ctrl = INITIAL_FORCE[0]
data.actuator("front_force_Y").ctrl = INITIAL_FORCE[1]
data.actuator("front_force_Z").ctrl = INITIAL_FORCE[2]

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    last_print = time.time()
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        if time.time() - last_print >= 1.0:
            f_in  = np.round(data.sensor("front_pivot1_force").data, 1)
            f_out = np.round(data.sensor("rear_pivot2_force").data, 1)
            t_out = np.round(data.sensor("rear_pivot2_torque").data, 2)
            print(f"Front Input [Fx,Fy,Fz]: {f_in} N  -->  Rear Pivot 2 Force: {f_out} N | Rear Torque: {t_out} Nm")
            last_print = time.time()

        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
