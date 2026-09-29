from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

# --------------------------------------------------
# 1. 加载模型
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "pendulum.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# --------------------------------------------------
# 2. 设置初始状态
# --------------------------------------------------

# 把摆杆拉开 40度
data.qpos[0] = np.deg2rad(40)

# 初始角速度为 0
data.qvel[0] = 0.0

# 更新派生状态
mujoco.mj_forward(model,data)

# --------------------------------------------------
# 3. 打开 Viewer
# --------------------------------------------------

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running() and data.time < 10.0:

        start_time = time.time()

        # ------------------------------------------
        # Physics simulation
        # ------------------------------------------

        mujoco.mj_step(model, data)

        # ------------------------------------------
        # 把新的状态同步给 Viewer
        # ------------------------------------------

        viewer.sync()

        # ------------------------------------------
        # 尽量让画面按照真实时间播放
        # ------------------------------------------

        elapsed = time.time() - start_time

        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)
