from pathlib import Path
import mujoco 
import numpy as np

# ============================================================
# 1. Panda 模型路径
# ===========================================================
ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = (ROOT / "assets" / "franka_panda" / "scene.xml" )

# ============================================================
# 2. 七个机械臂关节
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

# ============================================================
# 3. TCP 在 Hand 局部坐标系中的位置
# ============================================================
TCP_OFFSET_LOCAL = np.array([0.0,0.0,0.10],dtype = np.float64)

# ============================================================
# 4. 加载 Panda 模型
# ============================================================
def load_panda():

    if not PANDA_XML.exists():
        raise FileNotFoundError( f"Panda XML not found: {PANDA_XML}" )
    model = mujoco.MjModel.from_xml_path( str(PANDA_XML) )
    data = mujoco.MjData(model)

    return model, data

# ============================================================
# 5. 恢复 Home Pose
# ============================================================
def reset_home(model, data):
    home_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,"home",)
    if home_id == -1:
        raise ValueError( "Home keyframe does not exist." )
    mujoco.mj_resetDataKeyframe( model,data,home_id,)
    mujoco.mj_forward( model, data, )

# ============================================================
# 6. 读取七个机械臂关节位置
# ============================================================
def get_arm_qpos(data):

    qpos = [float(data.joint(name).qpos[0]) for name in ARM_JOINT_NAMES ]

    return np.asarray(qpos,dtype = np.float64)

# ============================================================
# 7. 读取七个机械臂关节速度
# ============================================================
def get_arm_qvel(data):

    qvel = [ float( data.joint(name).qvel[0] ) for name in ARM_JOINT_NAMES ]

    return np.asarray( qvel, dtype=np.float64, )

# ============================================================
# 8. 获取机械臂关节在 qvel 中的索引
# ============================================================
def get_arm_dof_indices(model):

    indices = []
    for name in ARM_JOINT_NAMES:
        joint_id = int(model.joint(name).id)
        dof_index = int(model.jnt_dofadr[joint_id])
        indices.append(dof_index)

    return np.asarray(indices,dtype = np.int32)

# ============================================================
# 9. 手动设置七个关节角度
# ============================================================
def set_arm_qpos(model,data,qpos):

    qpos = np.asarray(qpos,dtype=np.float64)

    if qpos.shape != (7,):
        raise ValueError(f"Expected qpos shape (7,), got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos contains non-finite values")

    # --------------------------------------------------------
    # 检查关节范围
    # --------------------------------------------------------
    for  name,angle in zip(ARM_JOINT_NAMES,qpos):

        joint_id = int(model.joint(name).id)
        if model.jnt_limited[joint_id]:
            lower,upper = model.jnt_range[joint_id]
            if not (lower <= angle <= upper):
                raise ValueError(
                    f"{name}: angle {angle:.6f} "
                    f"outside range [{lower:.6f}, {upper:.6f}]")

    # --------------------------------------------------------
    # 修改关节位置
    # --------------------------------------------------------
    for name,angle in zip(ARM_JOINT_NAMES,qpos):
        data.joint(name).qpos[0] = float(angle)
        # --------------------------------------------------------
        # 重新计算当前 configuration 的运动学
        # --------------------------------------------------------
        mujoco.mj_forward(model,data,)

# ============================================================
# 10. 读取 Hand 世界位姿
# ============================================================
def get_hand_pose(data):

    hand = data.body("hand")
    position = hand.xpos.copy()
    quaternion = hand.xquat.copy()
    rotation_matrix = (hand.xmat.copy().reshape(3,3))

    return (position,quaternion,rotation_matrix)

# ============================================================
# 11. 计算 TCP 世界位姿
# ============================================================
def get_tcp_pose(data):

    (hand_position,hand_quaternion,hand_rotation) = get_hand_pose(data)

    # Hand 局部偏移 -> World 位移
    world_offset = (hand_rotation @ TCP_OFFSET_LOCAL)

    #TCP 世界位置
    tcp_position = (hand_position + world_offset)

    #TCP 与 Hand 方向一致
    tcp_quaternion = (hand_quaternion.copy())
    tcp_rotation = (hand_rotation.copy())

    return (tcp_position,tcp_quaternion,tcp_rotation)

# ============================================================
# 12. 构造 4x4 齐次变换矩阵
# ============================================================
def pose_to_transform(position,rotation_matrix):
    
    transform = np.eye(4, dtype = np.float64)
    transform[:3,:3] = rotation_matrix
    transform[:3,3] = position

    return transform

# ============================================================
# 13. 计算 TCP Jacobian
# ============================================================
def get_tcp_jacobian(model,data):
    # --------------------------------------------------------
    # TCP 当前世界位置
    # --------------------------------------------------------
    tcp_position , _, _ = get_tcp_pose(data)

    # --------------------------------------------------------
    # 获取 Hand Body ID
    # --------------------------------------------------------
    hand_id = int(model.body("hand").id)

    # --------------------------------------------------------
    # 为完整 Jacobian 创建数组
    # --------------------------------------------------------
    jacp_full = np.zeros((3,model.nv),dtype=np.float64,)
    jacr_full = np.zeros((3,model.nv),dtype=np.float64,)

    # --------------------------------------------------------
    # MuJoCo 计算 Jacobian
    # --------------------------------------------------------
    mujoco.mj_jac(model,data,jacp_full,jacr_full,tcp_position,hand_id)

    # --------------------------------------------------------
    # 提取七个机械臂关节的 DOF
    # --------------------------------------------------------
    arm_dof_indices = get_arm_dof_indices(model)
    jacp_arm = jacp_full[:,arm_dof_indices].copy()
    jacr_arm = jacr_full[:,arm_dof_indices].copy()
    return (jacp_arm,jacr_arm,jacp_full,jacr_full)







