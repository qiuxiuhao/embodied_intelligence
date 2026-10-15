import numpy as np
from robotics.kinematics import ( get_arm_qpos, )

# ============================================================
# 1. Panda Joint Position Controller
# ============================================================
class PandaJointPositionController:

    def __init__(self, model, max_joint_speed=0.5, ):
        self.model = model
        # 每个关节目标指令的最大变化速度
        # 单位 rad/s
        self.max_joint_speed = float( max_joint_speed)
        if self.max_joint_speed <= 0:
            raise ValueError(  "max_joint_speed must be positive." )
        # ----------------------------------------------------
        # 找到前七个机械臂执行器
        # ----------------------------------------------------
        self.actuator_ids = []
        for i in range(1, 8):

            actuator_name = f"actuator{i}"
            actuator_id = int( self.model.actuator(  actuator_name ).id )
            self.actuator_ids.append( actuator_id )

        self.actuator_ids = np.asarray( self.actuator_ids,  dtype=np.int32, )

        # ----------------------------------------------------
        # 读取控制范围
        # ----------------------------------------------------
        ctrl_ranges = ( self.model.actuator_ctrlrange[ self.actuator_ids ].copy() )
        self.ctrl_low = ctrl_ranges[:, 0]
        self.ctrl_high = ctrl_ranges[:, 1]

        # 当前发送给执行器的目标
        self.command_q = None

    # ========================================================
    # 2. 初始化控制器
    # ========================================================
    def reset(self, data):

        # 读取真实关节位置
        current_q = get_arm_qpos( data )
        # 限制在执行器允许范围内
        self.command_q = np.clip( current_q, self.ctrl_low, self.ctrl_high, )
        # 同步初始位置命令
        data.ctrl[ self.actuator_ids ] = self.command_q
        return self.command_q.copy()

    # ========================================================
    # 3. 发送一个控制步
    # ========================================================
    def step( self, data, target_q, dt,):

        if self.command_q is None:
            raise RuntimeError( "Call reset() before step().")

        target_q = np.asarray( target_q, dtype=np.float64, )
        if target_q.shape != (7,):
            raise ValueError( "target_q must have shape (7,).")
        if not np.all(np.isfinite(target_q)):
            raise ValueError( "target_q contains non-finite values." )
        dt = float(dt)
        if dt <= 0:
            raise ValueError( "dt must be positive." )

        # ----------------------------------------------------
        # 3.1 目标限幅
        # ----------------------------------------------------
        target_q = np.clip( target_q, self.ctrl_low, self.ctrl_high, )

        # ----------------------------------------------------
        # 3.2 计算目标指令误差
        # ----------------------------------------------------
        delta_q = ( target_q  - self.command_q )

        # ----------------------------------------------------
        # 3.3 单个控制步允许的最大变化
        # ----------------------------------------------------
        max_delta = ( self.max_joint_speed  * dt )

        # ----------------------------------------------------
        # 3.4 限制指令变化速度
        # ----------------------------------------------------
        limited_delta = np.clip( delta_q,  -max_delta, max_delta, )

        # ----------------------------------------------------
        # 3.5 更新当前关节命令
        # ----------------------------------------------------
        self.command_q = (   self.command_q + limited_delta  )

        # ----------------------------------------------------
        # 3.6 写入 Panda Position Actuators
        # ----------------------------------------------------
        data.ctrl[ self.actuator_ids ] = self.command_q

        return self.command_q.copy()




