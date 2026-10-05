from pathlib import Path
import math
import time

import mujoco
import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "joint_types.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# --------------------------------------------------
# Hinge:45°
# --------------------------------------------------

data.joint("hinge_joint").qpos[0] = np.deg2rad(45)

# --------------------------------------------------
# Slide:0.25
# --------------------------------------------------

data.joint("slide_joint").qpos[0] = 0.25

# --------------------------------------------------
# Free: XYZ + 45° around Z
# --------------------------------------------------

free_qpos = data.joint("free_joint").qpos

free_qpos[0:3] = [
    1.0,
    0.2,
    0.9,
]

angle = np.deg2rad(45)

half_angle = angle / 2.0

free_qpos[3:7] = [
    math.cos(half_angle),   # qw
    0.0,                     # qx
    0.0,                     # qy
    math.sin(half_angle),   # qz
]   

#--------------------------------------------------
# 根据qpos 更新运动学
#--------------------------------------------------

mujoco.mj_forward(model, data)

#--------------------------------------------------
# 可视化
#--------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)