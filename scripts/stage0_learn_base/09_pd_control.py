from pathlib import Path
import  time

import mujoco
import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "actuated_joint.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# ==========================
# 初始状态
# ==========================
data.joint("hinge_joint").qpos[0] = 0.0
data.joint("hinge_joint").qvel[0] = 0.0

mujoco.mj_forward(model, data)

#==========================
# PD control parameters
#==========================

target_angle = np.deg2rad(-90.0)# 目标角度，单位为弧度

kp = 8.0

kd = 1.0

CTRL_MIN = -3.0
CTRL_MAX = 3.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    while viewer.is_running() and data.time < 6.0:
        start = time.time()

        #=====================================
        # Observation / State
        #=====================================

        q = data.joint("hinge_joint").qpos[0]
        qvel = data.joint("hinge_joint").qvel[0]

        #=====================================
        # PD control
        #=====================================

        error = target_angle - q

        action = (kp * error - kd * qvel)

        # 限制控制输入范围
        action = np.clip(action, CTRL_MIN, CTRL_MAX)

        # =====================================
        # Action -> mujoco control
        # =====================================
        data.ctrl[0] = action

        #====================================
        # Physics
        #====================================

        mujoco.mj_step(model, data)

        #====================================
        # log
        #====================================

        if step_count % 100 == 0:
            print(
                f"time: {data.time:.2f},  "
                f"target_angle: {np.rad2deg(target_angle):.1f} deg,  "
                f"q: {np.rad2deg(q):+.2f} deg,  "
                f"error: {np.rad2deg(error):+.2f} deg,  "
                f"qvel: {qvel:+.2f} rad/s,  "
                f"action: {action:+.2f}"
            )

        step_count += 1

        viewer.sync()

        elapsed = time.time() - start

        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)