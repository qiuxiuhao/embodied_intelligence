"""复用既有六维 IK 和关节控制器，执行有限长度的平滑位姿轨迹。"""

import numpy as np

from controllers.ik_controller import PandaIKSolver, rotation_error_world
from controllers.joint_controller import PandaJointPositionController
from robotics.kinematics import get_tcp_pose, set_arm_qpos


# 运动层数值参数集中在此；任务成功阈值由 PickPlaceConfig 单独管理。
IK_POSITION_TOLERANCE_M = 0.0003
IK_ORIENTATION_TOLERANCE_RAD = 0.005
MAX_TRACKING_BIAS_M = 0.015
MIN_SEGMENT_DURATION_S = 0.04
MAX_WAYPOINTS = 500


class MotionPlanningError(RuntimeError):
    """路点 IK 失败；轨迹尚未提交给真实执行器。"""


class PoseMotionController:
    """将世界坐标 TCP 目标转换为预先校验的 IK 路点和同步关节插值。

    该辅助层不推进物理、不决定任务状态。沿用固定朝下姿态；首个路点的
    姿态变化同样受关节插值限速，但不提供通用障碍物路径搜索。
    """

    def __init__(self, model, max_joint_speed_rad_s=0.35, waypoint_spacing_m=0.01):
        """创建原有求解器与控制器；位置单位 m、关节速度 rad/s。"""
        self.model = model
        self.arm = PandaJointPositionController(model, max_joint_speed=max_joint_speed_rad_s)
        self.solver = PandaIKSolver(model, tolerance=IK_POSITION_TOLERANCE_M, max_iterations=400)
        self.spacing = float(waypoint_spacing_m)
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("waypoint_spacing_m 必须为有限正数")
        self.bias_world_m = np.zeros(3)
        self.physical_target = None
        self.rotation = None
        self.segments = []
        self.elapsed_s = 0.0
        self.duration_s = 0.0
        self.ik_calls = 0

    def reset(self, data):
        """同步实际关节状态；不修改任何 qpos，不执行仿真重置。"""
        self.arm.reset(data)
        self.bias_world_m[:] = 0
        self.ik_calls = 0
        self.physical_target = None
        self.rotation = None
        self.hold()

    def hold(self):
        """停止推进轨迹，保持最后一个合法关节命令。"""
        self.segments = []
        self.elapsed_s = 0.0
        self.duration_s = 0.0

    @property
    def finished(self):
        """仅表示参考轨迹已走完；实际到达仍需任务层读取物理反馈确认。"""
        return bool(self.elapsed_s >= self.duration_s)

    def plan(self, data, target_position_m, target_rotation, speed_m_s):
        """预检整段轨迹，成功后原子替换；失败不写 ctrl 或改变旧轨迹。

        用上一段实测的伺服偏差修正 IK 目标，补偿 Day 11 已测得的重力稳态误差。
        起点采用最后命令的 FK，避免把实际跟踪偏差重复积分进参考轨迹。
        """
        target = np.asarray(target_position_m, dtype=float)
        rotation = np.asarray(target_rotation, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise MotionPlanningError("TCP 目标必须为有限的世界坐标三维向量")
        if not np.isfinite(speed_m_s) or speed_m_s <= 0:
            raise ValueError("speed_m_s 必须为有限正数")
        q = self.arm.command_q.copy()
        set_arm_qpos(self.model, self.solver.data, q)
        start = get_tcp_pose(self.solver.data)[0]
        ik_target = target - self.bias_world_m
        count = max(1, int(np.ceil(np.linalg.norm(ik_target - start) / self.spacing)))
        # 极端不可达输入也限制规划开销；由任务层返回明确 IK_FAILED。
        if count > MAX_WAYPOINTS:
            raise MotionPlanningError("目标距离超过路点规划上限")
        segments = []
        previous_position = start
        duration = 0.0
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            position = start + fraction * (ik_target - start)
            self.ik_calls += 1
            result = self.solver.solve(position, initial_q=q, target_rotation=rotation,
                                       orientation_tolerance=IK_ORIENTATION_TOLERANCE_RAD)
            candidate = np.asarray(result["q"])
            if (not result["success"] or candidate.shape != (7,)
                    or not np.all(np.isfinite(candidate))):
                raise MotionPlanningError(f"路点 IK 失败：{result['reason']}")
            # 再用 FK 独立验证返回值，拒绝错误地报告 success 的求解器结果。
            set_arm_qpos(self.model, self.solver.data, candidate)
            fk_position, _, fk_rotation = get_tcp_pose(self.solver.data)
            if (np.linalg.norm(fk_position - position) > self.solver.tolerance
                    or np.linalg.norm(rotation_error_world(rotation, fk_rotation))
                    > IK_ORIENTATION_TOLERANCE_RAD):
                raise MotionPlanningError("IK 返回值未通过独立 FK 验证")
            # cubic smoothstep 的最大导数为 1.5，按它计算时长保证参考速度上限。
            segment_s = max(MIN_SEGMENT_DURATION_S,
                            1.5 * np.linalg.norm(position - previous_position) / speed_m_s,
                            1.5 * np.max(np.abs(candidate - q)) / self.arm.max_joint_speed)
            segments.append((q.copy(), candidate.copy(), float(segment_s)))
            duration += segment_s
            q = candidate
            previous_position = position
        self.physical_target = target.copy()
        self.rotation = rotation.copy()
        self.ik_target = ik_target.copy()
        self.segments = segments
        self.elapsed_s = 0.0
        self.duration_s = duration

    def step(self, data, dt):
        """每个物理步仅写入限速的七关节 ctrl；不以轨迹完成冒充实际到达。"""
        command = self.arm.command_q.copy()
        if self.segments:
            self.elapsed_s += dt
            remaining = self.elapsed_s
            for start_q, end_q, duration in self.segments:
                if remaining <= duration:
                    u = np.clip(remaining / duration, 0., 1.)
                    command = start_q + u*u*(3 - 2*u)*(end_q - start_q)
                    break
                remaining -= duration
                command = end_q
        return self.arm.step(data, command, dt)

    def calibrate_tracking_bias(self, data):
        """在稳定到达后测量实际 TCP 减 IK 目标的偏差，供下一段前馈补偿。"""
        if self.physical_target is not None:
            error = get_tcp_pose(data)[0] - self.ik_target
            # 只补偿小范围稳态误差；大误差应由状态机判为到达失败。
            self.bias_world_m = np.clip(error, -MAX_TRACKING_BIAS_M, MAX_TRACKING_BIAS_M)
