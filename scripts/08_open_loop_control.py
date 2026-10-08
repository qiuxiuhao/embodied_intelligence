from pathlib import Path
import time
import mujoco
import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "actuated_joint.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# 初始状态
data.joint("hinge_joint").qpos[0] = 0.0
data.joint("hinge_joint").qvel[0] = 0.0

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    while viewer.is_running() and data.time < 4.0:
        start = time.time()

        #=====================================
        # action
        #=====================================

        if data.time < 1.0:
            action = 1.0
        elif data.time < 2.0:
            action = 0.0
        elif data.time < 3.0:
            action = -1.0
        else:
            action = 0.0

        # Action -> mujoco control
        data.ctrl[0] = action

        #====================================
        # Physics
        #====================================

        mujoco.mj_step(model, data)

        #====================================
        # log
        #====================================

        if step_count % 100 == 0:
            q  = data.joint("hinge_joint").qpos[0]
            qvel = data.joint("hinge_joint").qvel[0]
            actuator_force = data.actuator_force[0]
            joint_force = data.qfrc_actuator[0]
            print(f"time: {data.time:.2f},  action: {action:+.2f}, angle: {np.rad2deg(q):+.2f} deg,  velocity: {qvel:+.2f} rad/s")
            print(
                f"ctrl={data.ctrl[0]:+.2f}  "
                f"actuator_force={actuator_force:+.2f}  "
                f"joint_force={joint_force:+.2f}"
            )
        step_count += 1

        viewer.sync()

        elapsed = time.time() - start

        remaining = model.opt.timestep - elapsed
        if remaining > 0:
            time.sleep(remaining)