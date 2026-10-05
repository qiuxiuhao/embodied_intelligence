from pathlib import Path
import math

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "joint_types.xml"


model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)


print("========== BEFORE ==========")

print("qpos:")
print(data.qpos)

# ==================================================
# 1. Hinge
# ==================================================

hinge_qpos = data.joint("hinge_joint").qpos

hinge_qpos[0] = np.deg2rad(45)

# ==================================================
# 2. Slide
# ==================================================

slide_qpos = data.joint("slide_joint").qpos

slide_qpos[0] = 0.25

# ==================================================
# 3. Free
# ==================================================

free_qpos = data.joint("free_joint").qpos


# XYZ
free_qpos[0:3] = [
    1.0,
    0.2,
    0.9,
]

# 绕 Z 轴旋转 45°
angle = np.deg2rad(45)

half_angle = angle / 2.0


free_qpos[3:7] = [
    math.cos(half_angle),   # qw
    0.0,                    # qx
    0.0,                    # qy
    math.sin(half_angle),   # qz
]

# ==================================================
# 更新派生状态
# ==================================================

mujoco.mj_forward(model, data)

print("\n========== AFTER ==========")

print("global qpos:")
print(data.qpos)


print("\nhinge qpos:")

print(
    data.joint("hinge_joint").qpos
)


print("\nslide qpos:")

print(
    data.joint("slide_joint").qpos
)


print("\nfree qpos:")

print(
    data.joint("free_joint").qpos
)