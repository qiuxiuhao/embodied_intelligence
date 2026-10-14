import numpy  as np
from robotics.kinematics import (load_panda,reset_home,get_arm_qpos,set_arm_qpos,
                                 get_tcp_pose,get_hand_pose,pose_to_transform,TCP_OFFSET_LOCAL)

# ============================================================
# 1. 打印格式
# ============================================================
np.set_printoptions(precision=6, suppress=True,)

# ============================================================
# 2. 打印 Pose
# ============================================================
def print_pose(name, position, quaternion, rotation):

    print(f"\n========== {name} ==========")
    print("Position XYZ (m):")
    print(position)
    print("\nQuaternion (wxyz):")
    print(quaternion)
    print("\nRotation Matrix:")
    print(rotation)

    transform = pose_to_transform( position, rotation,)
    print("\nHomogeneous Transform:")
    print(transform)

# ============================================================
# 3. 主程序
# ============================================================
def main():
    # --------------------------------------------------------
    # 3.1 加载模型
    # --------------------------------------------------------
    model, data = load_panda()

    # --------------------------------------------------------
    # 3.2 恢复 Home
    # --------------------------------------------------------
    reset_home(model, data)

    # --------------------------------------------------------
    # 3.3 读取 Home 关节角
    # --------------------------------------------------------
    q_home = get_arm_qpos(data)
    print("========== HOME JOINTS ==========")
    print("q (rad):")
    print(q_home)
    print("\nq (deg):")
    print(np.rad2deg(q_home))

    # --------------------------------------------------------
    # 3.4 获取 Hand Pose
    # --------------------------------------------------------
    ( hand_position, hand_quaternion, hand_rotation, ) = get_hand_pose(data)
    print_pose("HAND POSE",hand_position,hand_quaternion,hand_rotation,)

    # --------------------------------------------------------
    # 3.5 获取 TCP Pose
    # --------------------------------------------------------
    ( tcp_position, tcp_quaternion, tcp_rotation, ) = get_tcp_pose(data)
    print_pose( "TCP POSE", tcp_position, tcp_quaternion, tcp_rotation,)

    # --------------------------------------------------------
    # 3.6 检查 Hand -> TCP 变换
    # --------------------------------------------------------
    expected_offset = (hand_rotation @ TCP_OFFSET_LOCAL )
    actual_offset = ( tcp_position - hand_position)

    print("\n========== FRAME CHECK ==========")
    print("Expected world offset:")
    print(expected_offset)
    print("\nActual world offset:")
    print(actual_offset)

    assert np.allclose( expected_offset, actual_offset, atol=1e-10,)

    # --------------------------------------------------------
    # 3.7 检查旋转矩阵
    # --------------------------------------------------------
    identity = np.eye(3)
    assert np.allclose( tcp_rotation @ tcp_rotation.T, identity, atol=1e-9, )
    assert np.isclose( np.linalg.det(tcp_rotation), 1.0, atol=1e-9,)

    # --------------------------------------------------------
    # 3.8 改变 Joint1 角度
    # --------------------------------------------------------
    q_new = q_home.copy()
    q_new[0] += np.deg2rad(10.0)

    set_arm_qpos(model,data,q_new)
    (new_tcp_position, new_tcp_quaternion, new_tcp_rotation) = get_tcp_pose(data)
    print_pose("TCP AFTER JOINT1 +10 DEG",new_tcp_position,new_tcp_quaternion,new_tcp_rotation,)

    # --------------------------------------------------------
    # 3.9 计算末端位置变化
    # --------------------------------------------------------
    delta_position = (new_tcp_position - tcp_position)
    distance = np.linalg.norm(delta_position)

    # --------------------------------------------------------
    # 3.10 计算末端方向变化
    # --------------------------------------------------------
    relative_rotation = ( new_tcp_rotation @ tcp_rotation.T)

    cos_angle = ( np.trace(relative_rotation) - 1.0 ) / 2.0
    cos_angle = np.clip(  cos_angle, -1.0, 1.0,)
    orientation_change = np.arccos( cos_angle )

    print("\n========== FK RESULT ==========")
    print("Joint1 change (deg): 10.0")
    print("TCP position change (m):")
    print(delta_position)
    print("TCP displacement magnitude (m):")
    print(distance)
    print("TCP orientation change (deg):")
    print(np.rad2deg(orientation_change))

    # --------------------------------------------------------
    # 3.11 恢复 Home 并检查
    # --------------------------------------------------------
    reset_home(model, data)
    q_restored = get_arm_qpos(data)
    assert np.allclose( q_restored, q_home, atol=1e-10, )
    print("\nPanda FK test passed.")

if __name__ == "__main__":
    main()










