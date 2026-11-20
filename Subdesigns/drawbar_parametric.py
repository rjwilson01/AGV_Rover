import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("drawbar_ball_joint_loop.xml")
data = mujoco.MjData(model)

# Apply 500 N lateral force (Y) and -200 N vertical load (Z) at the ball joint
data.actuator("apply_force_Y").ctrl[0] = 500.0
data.actuator("apply_force_Z").ctrl[0] = -200.0

# Step simulation to reach static equilibrium
for _ in range(1000):
    mujoco.mj_step(model, data)

print("Pivot 1 Transmitted Force [Fx, Fy, Fz]:", np.round(data.sensor("pivot1_force").data, 2))
print("Ball Joint Force          [Fx, Fy, Fz]:", np.round(data.sensor("ball_joint_force").data, 2))
print("Pivot 2 Reaction Force    [Fx, Fy, Fz]:", np.round(data.sensor("pivot2_force").data, 2))
