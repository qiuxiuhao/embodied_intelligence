import numpy as np

from robotics.kinematics import (load_panda, reset_home, get_arm_qpos, get_arm_dof_indices, get_tcp_pose,get_tcp_jacobian, set_arm_qpos,)

# ============================================================
# 1. 打印格式
# ============================================================
np.set_printoptions( precision=6, suppress=True,)

# ============================================================
# 2. 主程序
# ============================================================
def main():

    # --------------------------------------------------------
    # 2.1 加载 Panda
    # --------------------------------------------------------
    model, data = load_panda()
    reset_home(model, data)

    # --------------------------------------------------------
    # 2.2 当前关节配置
    # --------------------------------------------------------
    q_home = get_arm_qpos(data)
    print("========== JOINT CONFIGURATION ==========")
    print("q (rad):")
    print(q_home)

    # --------------------------------------------------------
    # 2.3 当前 TCP Pose
    # --------------------------------------------------------
    tcp_position, _, _ = get_tcp_pose( data )
    print("\nTCP position:")
    print(tcp_position)

    # --------------------------------------------------------
    # 2.4 计算 Jacobian
    # --------------------------------------------------------
    (jacp,jacr,jacp_full,jacr_full) = get_tcp_jacobian(model,data)
    
    # --------------------------------------------------------
    # 2.5 检查 Shape
    # --------------------------------------------------------
    print("\n========== SHAPES ==========")
    print("model.nv:", model.nv)
    print("Jp full:", jacp_full.shape)
    print("Jr full:", jacr_full.shape)
    print("Jp arm:", jacp.shape)
    print("Jr arm:", jacr.shape)
    assert jacp.shape == (3, 7)
    assert jacr.shape == (3, 7)

    # --------------------------------------------------------
    # 2.6 打印 Jacobian
    # --------------------------------------------------------
    print("\n========== TRANSLATIONAL JACOBIAN ==========")
    print(jacp)
    print("\n========== ROTATIONAL JACOBIAN ==========")
    print(jacr)

    # --------------------------------------------------------
    # 2.7 拼接完整的 6x7 Jacobian
    # --------------------------------------------------------
    jacobian = np.vstack( [jacp, jacr] )
    print("\n========== SPATIAL JACOBIAN ==========")
    print(jacobian)
    print("Shape:", jacobian.shape)

    # --------------------------------------------------------
    # 2.8 检查手指自由度的影响
    # --------------------------------------------------------
    arm_indices = get_arm_dof_indices(model)
    finger_indices = [i for i in range(model.nv) if i not in arm_indices]

    print("\nFinger DOF indices:")
    print(finger_indices)
    print("\nFinger Jp columns:")
    print(jacp_full[:, finger_indices])
    print("\nFinger Jr columns:")
    print(jacr_full[:, finger_indices])

    # --------------------------------------------------------
    # 2.9 给定虚拟关节速度
    # --------------------------------------------------------
    qdot = np.array([  0.10,  -0.05,  0.08, 0.00,  0.05, -0.03, 0.02,], dtype=np.float64, )

    print("\n========== JOINT VELOCITY ==========")
    print("qdot (rad/s):")
    print(qdot)

    # --------------------------------------------------------
    # 2.10 计算末端线速度
    # --------------------------------------------------------
    linear_velocity = (jacp @ qdot)

    # --------------------------------------------------------
    # 2.11 计算末端角速度
    # --------------------------------------------------------
    angular_velocity = (jacr @ qdot)
    
    print("\nTCP linear velocity (m/s):")
    print(linear_velocity)
    print("\nTCP angular velocity (rad/s):")
    print(angular_velocity)

    # --------------------------------------------------------
    # 2.12 小角度位移预测
    # --------------------------------------------------------
    delta_q = np.zeros( 7, dtype=np.float64, )
    delta_q[0] = np.deg2rad(0.5)
    predicted_delta_position = (jacp @ delta_q)

    print("\n========== SMALL DISPLACEMENT ==========")
    print("Delta q:")
    print(delta_q)
    print("\nPredicted TCP displacement:")
    print(predicted_delta_position)

    # --------------------------------------------------------
    # 2.13 实际重新计算 FK
    # --------------------------------------------------------
    q_new = (q_home + delta_q)
    set_arm_qpos( model, data, q_new, )
    new_tcp_position, _, _ = get_tcp_pose(data)
    actual_delta_position = (new_tcp_position - tcp_position)
    prediction_error = (actual_delta_position - predicted_delta_position)

    print("\nActual TCP displacement:")
    print(actual_delta_position)
    print("\nPrediction error:")
    print(prediction_error)
    print("\nPrediction error norm:")
    print( np.linalg.norm( prediction_error ) )
    print("\nPanda Jacobian inspection passed.")

if __name__ == "__main__":
    main()

    





