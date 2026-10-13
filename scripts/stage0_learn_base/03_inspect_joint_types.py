from pathlib import Path

import mujoco
import numpy as np


# --------------------------------------------------
# 1. 找项目根目录
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

xml_path = ROOT / "assets" / "joint_types.xml"

# --------------------------------------------------
# 2. 加载模型
# --------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(xml_path))

data = mujoco.MjData(model)

# --------------------------------------------------
# 3. Joint 类型名称表
# --------------------------------------------------

JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

# --------------------------------------------------
# 4. 打印整个模型的维度
# --------------------------------------------------

print("========== MODEL SIZE ==========")

print("nbody:", model.nbody)
print("njnt :", model.njnt)

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

print()

print("global qpos:")
print(data.qpos)

print()

print("global qvel:")
print(data.qvel)

# --------------------------------------------------
# 5. 分别检查每种 joint
# --------------------------------------------------

joint_names = [
    "hinge_joint",
    "slide_joint",
    "free_joint",
]

print("\n========== JOINT DETAILS ==========")


for joint_name in joint_names:

    # 根据名字得到 joint
    model_joint = model.joint(joint_name)

    data_joint = data.joint(joint_name)

    joint_id = model_joint.id


    # joint 类型
    joint_type_id = int(model.jnt_type[joint_id])

    joint_type_name = JOINT_TYPE_NAMES[joint_type_id]

    # 这个 joint 在全局 qpos/qvel 中的起始位置
    qpos_address = model.jnt_qposadr[joint_id]

    qvel_address = model.jnt_dofadr[joint_id]


    # Named Access
    joint_qpos = np.atleast_1d(data_joint.qpos)

    joint_qvel = np.atleast_1d(data_joint.qvel)


    print(f"\nJoint: {joint_name}")

    print("type:", joint_type_name)

    print("joint id:", joint_id)

    print("qpos address:", qpos_address)

    print("qvel address:", qvel_address)

    print("qpos:", joint_qpos)

    print("qpos shape:", joint_qpos.shape)

    print("qvel:", joint_qvel)

    print("qvel shape:", joint_qvel.shape)