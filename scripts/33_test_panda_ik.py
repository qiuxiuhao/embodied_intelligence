import mujoco
import numpy as np

from controllers.ik_controller import (
    PandaIKSolver,
)

from robotics.kinematics import (
    load_panda,
    reset_home,
    get_arm_qpos,
    set_arm_qpos,
    get_tcp_pose,
)


# ============================================================
# 1. 打印格式
# ============================================================

np.set_printoptions(
    precision=6,
    suppress=True,
)


# ============================================================
# 2. 生成已知可达的 TCP 目标
# ============================================================

def create_reachable_target(
    model,
    home_q,
    joint_offset_deg,
):

    # --------------------------------------------------------
    # 2.1 单独创建 FK Data
    # --------------------------------------------------------

    fk_data = mujoco.MjData(model)

    reset_home(
        model,
        fk_data,
    )

    # --------------------------------------------------------
    # 2.2 构造已知关节配置
    # --------------------------------------------------------

    q_known = (
        home_q
        +
        np.deg2rad(
            joint_offset_deg
        )
    )

    # --------------------------------------------------------
    # 2.3 计算对应的 FK
    # --------------------------------------------------------

    set_arm_qpos(
        model,
        fk_data,
        q_known,
    )

    target_position, _, _ = get_tcp_pose(
        fk_data
    )

    return (
        target_position,
        q_known,
    )


# ============================================================
# 3. 测试一个 IK 目标
# ============================================================

def test_one_target(
    solver,
    model,
    home_q,
    name,
    joint_offset_deg,
):

    print(f"\n========== {name} ==========")

    # --------------------------------------------------------
    # 3.1 已知可达目标
    # --------------------------------------------------------

    target_position, q_known = create_reachable_target(
        model,
        home_q,
        joint_offset_deg,
    )

    print("Target TCP XYZ:")
    print(target_position)

    print("\nKnown joint solution (deg):")
    print(np.rad2deg(q_known))

    # --------------------------------------------------------
    # 3.2 使用 IK 求解
    # --------------------------------------------------------

    result = solver.solve(
        target_position=target_position,
        initial_q=home_q,
    )

    # --------------------------------------------------------
    # 3.3 打印求解结果
    # --------------------------------------------------------

    print("\nIK Success:")
    print(result["success"])

    print("\nReason:")
    print(result["reason"])

    print("\nIterations:")
    print(result["iterations"])

    print("\nIK Joint Solution (deg):")
    print(
        np.rad2deg(
            result["q"]
        )
    )

    print("\nIK Final TCP XYZ:")
    print(result["final_position"])

    print("\nPosition Error (m):")
    print(result["position_error"])

    print("\nPosition Error (mm):")
    print(
        result["position_error"]
        *
        1000.0
    )

    # --------------------------------------------------------
    # 3.4 验收
    # --------------------------------------------------------

    assert result["success"], (
        f"{name}: IK failed, "
        f"reason={result['reason']}, "
        f"error={result['position_error']:.6f}"
    )

    assert result["position_error"] < 0.002

    print(f"\n{name} IK test passed.")


# ============================================================
# 4. 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 4.1 加载模型
    # --------------------------------------------------------

    model, data = load_panda()

    reset_home(
        model,
        data,
    )

    home_q = get_arm_qpos(
        data
    )

    # --------------------------------------------------------
    # 4.2 创建 IK Solver
    # --------------------------------------------------------

    solver = PandaIKSolver(
        model=model,
        damping=0.05,
        tolerance=0.002,
        max_iterations=200,
    )

    # --------------------------------------------------------
    # 4.3 测试三个不同目标
    # --------------------------------------------------------

    test_cases = {

        "TARGET_A": [
            8.0,
            -6.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],

        "TARGET_B": [
            -10.0,
            6.0,
            -4.0,
            4.0,
            4.0,
            -5.0,
            3.0,
        ],

        "TARGET_C": [
            5.0,
            3.0,
            -6.0,
            -3.0,
            0.0,
            7.0,
            -5.0,
        ],
    }

    for name, offset in test_cases.items():

        test_one_target(
            solver=solver,
            model=model,
            home_q=home_q,
            name=name,
            joint_offset_deg=offset,
        )

    # --------------------------------------------------------
    # 4.4 不可达目标测试
    # --------------------------------------------------------

    print("\n========== UNREACHABLE TARGET ==========")

    unreachable_target = np.array(
        [3.0, 3.0, 3.0],
        dtype=np.float64,
    )

    result = solver.solve(
        target_position=unreachable_target,
        initial_q=home_q,
    )

    print("Success:", result["success"])

    print("Reason:", result["reason"])

    print(
        "Final Error (m):",
        result["position_error"],
    )

    assert not result["success"]

    print("\nAll IK tests passed.")


if __name__ == "__main__":

    main()