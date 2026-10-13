from pathlib import Path

import mujoco
import numpy as np


# ============================================================
# 1. 文件路径
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

XML_PATH = ROOT / "assets" / "franka_panda" / "scene.xml"


# ============================================================
# 2. Panda 关节名称
# ============================================================

ARM_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

FINGER_JOINT_NAMES = [
    "finger_joint1",
    "finger_joint2",
]


# ============================================================
# 3. Joint 类型映射
# ============================================================

JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


# ============================================================
# 4. 加载模型
# ============================================================

def load_panda():

    if not XML_PATH.exists():

        raise FileNotFoundError(
            f"Cannot find Panda model: {XML_PATH}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(XML_PATH)
    )

    data = mujoco.MjData(model)

    return model, data


# ============================================================
# 5. 恢复 Home Pose
# ============================================================

def reset_to_home(model, data):

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_id == -1:

        raise ValueError(
            "The Panda model does not contain a 'home' keyframe."
        )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    return home_id


# ============================================================
# 6. 打印模型总体结构
# ============================================================

def inspect_model(model):

    print("\n========== MODEL ==========")

    print("nbody:", model.nbody)
    print("njnt :", model.njnt)
    print("ngeom:", model.ngeom)

    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    print("nkey:", model.nkey)

    print("timestep:", model.opt.timestep)


# ============================================================
# 7. 检查所有 Joint
# ============================================================

def inspect_joints(model, data):

    print("\n========== JOINTS ==========")

    all_joint_names = (
        ARM_JOINT_NAMES
        +
        FINGER_JOINT_NAMES
    )

    for joint_name in all_joint_names:

        model_joint = model.joint(joint_name)

        data_joint = data.joint(joint_name)

        joint_id = int(model_joint.id)

        joint_type_id = int(
            model.jnt_type[joint_id]
        )

        joint_type = JOINT_TYPE_NAMES[
            joint_type_id
        ]

        qpos_address = int(
            model.jnt_qposadr[joint_id]
        )

        qvel_address = int(
            model.jnt_dofadr[joint_id]
        )

        qpos = np.atleast_1d(
            data_joint.qpos
        ).copy()

        qvel = np.atleast_1d(
            data_joint.qvel
        ).copy()

        joint_range = model.jnt_range[
            joint_id
        ].copy()

        print(f"\nJoint: {joint_name}")

        print("  id:", joint_id)
        print("  type:", joint_type)

        print("  qpos address:", qpos_address)
        print("  qvel address:", qvel_address)

        print("  qpos:", qpos)
        print("  qvel:", qvel)

        print("  range:", joint_range)

        if joint_type == "hinge":

            print(
                "  angle (deg):",
                np.rad2deg(qpos[0]),
            )

        elif joint_type == "slide":

            print(
                "  displacement (m):",
                qpos[0],
            )


# ============================================================
# 8. 检查所有 Actuator
# ============================================================

def inspect_actuators(model, data):

    print("\n========== ACTUATORS ==========")

    for actuator_id in range(model.nu):

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        ctrl_range = model.actuator_ctrlrange[
            actuator_id
        ].copy()

        ctrl_value = float(
            data.ctrl[actuator_id]
        )

        transmission_type = int(
            model.actuator_trntype[actuator_id]
        )

        transmission_id = int(
            model.actuator_trnid[actuator_id, 0]
        )

        # ----------------------------------------
        # 判断 Actuator 控制的对象
        # ----------------------------------------

        if transmission_type == int(
            mujoco.mjtTrn.mjTRN_JOINT
        ):

            target_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                transmission_id,
            )

            transmission_name = "joint"

        elif transmission_type == int(
            mujoco.mjtTrn.mjTRN_TENDON
        ):

            target_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_TENDON,
                transmission_id,
            )

            transmission_name = "tendon"

        else:

            target_name = "other"
            transmission_name = "other"

        # ----------------------------------------
        # 输出
        # ----------------------------------------

        print(f"\nActuator: {actuator_name}")

        print("  id:", actuator_id)

        print(
            "  transmission type:",
            transmission_name,
        )

        print(
            "  transmission target:",
            target_name,
        )

        print(
            "  ctrl range:",
            ctrl_range,
        )

        print(
            "  current ctrl:",
            ctrl_value,
        )


# ============================================================
# 9. 检查 Home Pose
# ============================================================

def inspect_home(model, data, home_id):

    print("\n========== HOME POSE ==========")

    print("home keyframe id:", home_id)

    print("\nqpos:")

    print(data.qpos.copy())

    print("\nqvel:")

    print(data.qvel.copy())

    print("\nctrl:")

    print(data.ctrl.copy())


# ============================================================
# 10. 查看 Hand 世界位置
# ============================================================

def inspect_hand(data):

    print("\n========== HAND ==========")

    hand = data.body("hand")

    print("hand world position:")

    print(hand.xpos.copy())

    print("\nhand world quaternion:")

    print(hand.xquat.copy())


# ============================================================
# 11. 主程序
# ============================================================

def main():

    model, data = load_panda()

    home_id = reset_to_home(
        model,
        data,
    )

    inspect_model(model)

    inspect_joints(
        model,
        data,
    )

    inspect_actuators(
        model,
        data,
    )

    inspect_home(
        model,
        data,
        home_id,
    )

    inspect_hand(data)

    # ----------------------------------------
    # 基础结构断言
    # ----------------------------------------

    assert model.njnt == 9

    assert model.nq == 9

    assert model.nv == 9

    assert model.nu == 8

    print("\nPanda model inspection passed.")


if __name__ == "__main__":

    main()