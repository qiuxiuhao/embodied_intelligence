from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "joint_types.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# 确保基于当前qpos计算完整运动学
mujoco.mj_forward(model, data)

body_names = [
    "frame_parent",
    "frame_child",
]

for body_name in body_names:

    model_body = model.body(body_name)

    data_body = data.body(body_name)

    print(f"========== {body_name} ==========")

    # XML 中的 body 坐标系位置(local position)
    print("local pos:", model_body.pos)

    # XML 中的 body 世界坐标系位置
    print("world pos:", data_body.xpos)

    # XML 中的 body 世界坐标系旋转
    print("world quat:", data_body.xquat)

    ## XML 中的 body 世界坐标系旋转矩阵
    print("world mat:", data_body.xmat.reshape(3, 3))
    
