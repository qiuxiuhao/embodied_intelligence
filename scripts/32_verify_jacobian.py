import numpy as np

from robotics.kinematics import (
    load_panda,
    reset_home,
    get_arm_qpos,
    set_arm_qpos,
    get_tcp_pose,
    get_tcp_jacobian,
)


# ============================================================
# 1. 实验配置
# ============================================================

EPSILON = 1e-5

POSITION_TOLERANCE = 1e-4

ROTATION_TOLERANCE = 1e-4

np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# 2. 通过有限差分计算 Jacobian
# ============================================================

def finite_difference_jacobian(
    model,
    data,
    q_reference,
    epsilon=EPSILON,
):

    # --------------------------------------------------------
    # 2.1 创建输出矩阵
    # --------------------------------------------------------

    jacp_fd = np.zeros(
        (3, 7),
        dtype=np.float64,
    )

    jacr_fd = np.zeros(
        (3, 7),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # 2.2 逐个扰动七个关节
    # --------------------------------------------------------

    for joint_index in range(7):

        q_plus = q_reference.copy()

        q_minus = q_reference.copy()

        # ------------------------------------
        # 当前关节 +epsilon
        # ------------------------------------

        q_plus[joint_index] += epsilon

        # ------------------------------------
        # 当前关节 -epsilon
        # ------------------------------------

        q_minus[joint_index] -= epsilon

        # ------------------------------------
        # 计算 +epsilon 对应的 FK
        # ------------------------------------

        set_arm_qpos(
            model,
            data,
            q_plus,
        )

        (
            position_plus,
            _,
            rotation_plus,
        ) = get_tcp_pose(data)

        # ------------------------------------
        # 计算 -epsilon 对应的 FK
        # ------------------------------------

        set_arm_qpos(
            model,
            data,
            q_minus,
        )

        (
            position_minus,
            _,
            rotation_minus,
        ) = get_tcp_pose(data)

        # ------------------------------------
        # 平移 Jacobian 的一列
        # ------------------------------------

        jacp_fd[:, joint_index] = (
            position_plus
            -
            position_minus
        ) / (
            2.0 * epsilon
        )

        # ------------------------------------
        # 旋转变化：世界坐标系表达
        # ------------------------------------

        relative_rotation = (
            rotation_plus
            @
            rotation_minus.T
        )

        # ------------------------------------
        # 提取近似角速度的反对称矩阵
        # ------------------------------------

        skew_matrix = (
            relative_rotation
            -
            relative_rotation.T
        ) / (
            4.0 * epsilon
        )

        # ------------------------------------
        # 从反对称矩阵提取旋转向量
        # ------------------------------------

        jacr_fd[:, joint_index] = np.array(
            [
                skew_matrix[2, 1],
                skew_matrix[0, 2],
                skew_matrix[1, 0],
            ]
        )

    # --------------------------------------------------------
    # 2.3 恢复参考 Configuration
    # --------------------------------------------------------

    set_arm_qpos(
        model,
        data,
        q_reference,
    )

    return (
        jacp_fd,
        jacr_fd,
    )


# ============================================================
# 3. 验证指定 Configuration
# ============================================================

def verify_configuration(
    model,
    data,
    q,
    name,
):

    print(f"\n========== {name} ==========")

    # --------------------------------------------------------
    # 3.1 设置参考姿态
    # --------------------------------------------------------

    set_arm_qpos(
        model,
        data,
        q,
    )

    # --------------------------------------------------------
    # 3.2 MuJoCo Analytical Jacobian
    # --------------------------------------------------------

    (
        jacp,
        jacr,
        _,
        _,
    ) = get_tcp_jacobian(
        model,
        data,
    )

    # --------------------------------------------------------
    # 3.3 Finite Difference Jacobian
    # --------------------------------------------------------

    (
        jacp_fd,
        jacr_fd,
    ) = finite_difference_jacobian(
        model,
        data,
        q,
        EPSILON,
    )

    # --------------------------------------------------------
    # 3.4 计算误差矩阵
    # --------------------------------------------------------

    position_error = (
        jacp
        -
        jacp_fd
    )

    rotation_error = (
        jacr
        -
        jacr_fd
    )

    # --------------------------------------------------------
    # 3.5 计算最大绝对误差
    # --------------------------------------------------------

    max_position_error = float(
        np.max(
            np.abs(position_error)
        )
    )

    max_rotation_error = float(
        np.max(
            np.abs(rotation_error)
        )
    )

    # --------------------------------------------------------
    # 3.6 打印检查结果
    # --------------------------------------------------------

    print("\nMuJoCo Jp:")
    print(jacp)

    print("\nFinite Difference Jp:")
    print(jacp_fd)

    print("\nMuJoCo Jr:")
    print(jacr)

    print("\nFinite Difference Jr:")
    print(jacr_fd)

    print("\nMax Position Jacobian Error:")
    print(max_position_error)

    print("\nMax Rotation Jacobian Error:")
    print(max_rotation_error)

    # --------------------------------------------------------
    # 3.7 验收
    # --------------------------------------------------------

    assert max_position_error < POSITION_TOLERANCE, (
        f"{name}: position Jacobian check failed"
    )

    assert max_rotation_error < ROTATION_TOLERANCE, (
        f"{name}: rotation Jacobian check failed"
    )

    print(f"\n{name} verification passed.")


# ============================================================
# 4. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 4.1 加载 Panda
    # --------------------------------------------------------

    model, data = load_panda()

    reset_home(model, data)

    # --------------------------------------------------------
    # 4.2 Home Configuration
    # --------------------------------------------------------

    q_home = get_arm_qpos(data)

    verify_configuration(
        model=model,
        data=data,
        q=q_home,
        name="HOME",
    )

    # --------------------------------------------------------
    # 4.3 第二个 Configuration
    # --------------------------------------------------------

    q_test = q_home.copy()

    q_test += np.deg2rad(
        [
            5.0,
            -5.0,
            7.0,
            4.0,
            -4.0,
            5.0,
            3.0,
        ]
    )

    verify_configuration(
        model=model,
        data=data,
        q=q_test,
        name="PERTURBED CONFIGURATION",
    )

    # --------------------------------------------------------
    # 4.4 恢复 Home
    # --------------------------------------------------------

    reset_home(
        model,
        data,
    )

    print("\n====================================")
    print("All Jacobian verification tests passed.")
    print("====================================")


if __name__ == "__main__":

    main()