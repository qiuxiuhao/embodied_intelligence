from pathlib import Path
import mujoco

# --------------------------------------------------
# 1. 找到项目根目录
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "pendulum.xml"

# --------------------------------------------------
# 2. 从 XML 创建 MjModel
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(xml_path))

# --------------------------------------------------
# 3. 创建当前仿真状态 MjData
# --------------------------------------------------

data = mujoco.MjData(model)

# --------------------------------------------------
# 4. 查看模型基本信息
# --------------------------------------------------

print("========== MODEL ==========")

print("model name:", model.names)
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("ngeom:", model.ngeom)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

print("timestep:", model.opt.timestep)


# --------------------------------------------------
# 5. 查看当前状态
# --------------------------------------------------

print("\n========== DATA ==========")

print("time:", data.time)

print("qpos:", data.qpos)
print("qvel:", data.qvel)

import numpy as np

print("\n========== CHANGE STATE ==========")

# 设置 hinge 的角度为 40 度
data.qpos[0] = np.deg2rad(40)

# 根据新的 qpos 重新计算场景中的派生物理量
mujoco.mj_forward(model,data)

print("qpos radians:",data.qpos)

print("qpos degrees:",np.rad2deg(data.qpos))

print("qvel:",data.qvel)