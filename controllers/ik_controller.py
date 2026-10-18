import mujoco
import numpy as np

from robotics.kinematics import (
    ARM_JOINT_NAMES,
    reset_home,
    get_arm_qpos,
    get_tcp_pose,
    get_tcp_jacobian,
    set_arm_qpos,
)


def rotation_error_world(target_rotation, current_rotation):
    """返回世界坐标系下的最短 SO(3) 旋转误差向量，单位 rad。

    Jr 对应世界坐标系角速度，因此相对旋转矩阵为 R_target @ R_current.T。
    四元数对数映射能够处理 180° 误差，避免简单叉乘在该角度退化。
    """
    relative = np.ascontiguousarray(target_rotation @ current_rotation.T)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, relative.reshape(9))
    if quaternion[0] < 0:
        quaternion *= -1
    error = np.empty(3)
    mujoco.mju_quat2Vel(error, quaternion, 1.0)
    return error


# ============================================================
# 1. Panda 阻尼最小二乘逆运动学求解器
# ============================================================

class PandaIKSolver:

    def __init__(
        self,
        model,
        damping=0.05,
        tolerance=0.002,
        max_iterations=200,
        max_step_norm=0.08,
        joint_margin=0.01,
    ):

        self.model = model

        # DLS 阻尼系数
        self.damping = float(damping)

        # TCP 位置误差容许值，单位 m
        self.tolerance = float(tolerance)

        # 最大迭代次数
        self.max_iterations = int(max_iterations)

        # 每次迭代的最大关节变化范数，单位 rad
        self.max_step_norm = float(max_step_norm)

        # 避免贴住关节极限
        self.joint_margin = float(joint_margin)

        # ----------------------------------------------------
        # 独立的运动学计算数据
        # 不修改真实仿真中的 MjData
        # ----------------------------------------------------

        self.data = mujoco.MjData(
            self.model
        )

        reset_home(
            self.model,
            self.data,
        )

        self.home_q = get_arm_qpos(
            self.data
        )

        # ----------------------------------------------------
        # 保存七个关节的合法范围
        # ----------------------------------------------------

        lower_limits = []
        upper_limits = []

        for name in ARM_JOINT_NAMES:

            joint_id = int(
                self.model.joint(name).id
            )

            lower, upper = self.model.jnt_range[
                joint_id
            ]

            lower_limits.append(
                lower + self.joint_margin
            )

            upper_limits.append(
                upper - self.joint_margin
            )

        self.lower_limits = np.asarray(
            lower_limits,
            dtype=np.float64,
        )

        self.upper_limits = np.asarray(
            upper_limits,
            dtype=np.float64,
        )

        if not np.all(
            self.lower_limits < self.upper_limits
        ):

            raise ValueError(
                "Invalid joint limits after applying margin."
            )

        if self.damping <= 0:

            raise ValueError(
                "Damping must be positive."
            )

        if self.tolerance <= 0:

            raise ValueError(
                "Tolerance must be positive."
            )

        if self.max_step_norm <= 0:

            raise ValueError(
                "Maximum step norm must be positive."
            )

        if self.max_iterations <= 0:

            raise ValueError(
                "Maximum iterations must be positive."
            )

    # ========================================================
    # 2. 设置求解器中的关节配置
    # ========================================================

    def _set_configuration(self, q):

        set_arm_qpos(
            self.model,
            self.data,
            q,
        )

    # ========================================================
    # 3. 计算 Damped Least Squares 关节增量
    # ========================================================

    def _compute_dls_step(
        self,
        jacobian,
        position_error,
    ):

        """复用阻尼最小二乘法，求解位置任务（3 行）或加权位姿任务（6 行）。"""

        # ----------------------------------------
        # 计算雅可比矩阵与其转置的乘积：J * J^T
        # ----------------------------------------

        jjt = (
            jacobian
            @
            jacobian.T
        )

        # ----------------------------------------
        # 添加阻尼
        # ----------------------------------------

        damped_matrix = (
            jjt
            +
            self.damping ** 2
            *
            np.eye(jacobian.shape[0])
        )

        # ----------------------------------------
        # 解任务空间线性方程组（位置 3 维，位姿 6 维）
        # ----------------------------------------

        solution = np.linalg.solve(
            damped_matrix,
            position_error,
        )

        # ----------------------------------------
        # 映射回 7 个关节的增量
        # ----------------------------------------

        delta_q = (
            jacobian.T
            @
            solution
        )

        # ----------------------------------------
        # 限制单次迭代增量
        # ----------------------------------------

        step_norm = np.linalg.norm(
            delta_q
        )

        if step_norm > self.max_step_norm:

            delta_q = (
                delta_q
                /
                step_norm
                *
                self.max_step_norm
            )

        return delta_q

    # ========================================================
    # 4. IK 主函数
    # ========================================================

    def solve(
        self,
        target_position,
        initial_q=None,
        target_rotation=None,
        orientation_tolerance=0.01,
        orientation_weight=0.15,
    ):
        """求解位置或可选的完整位姿，仅修改求解器内部数据。

        位置使用世界坐标，单位 m；旋转矩阵将 TCP 局部坐标映射到世界坐标；角度容差单位 rad。
        orientation_weight 的单位为 m/rad，用于平衡阻尼最小二乘和线搜索中的平移、旋转误差。
        不传 target_rotation 时保持 Day 10 的位置逆运动学行为。
        """
        if target_rotation is not None:
            target_rotation = np.asarray(target_rotation, dtype=np.float64)
            if (target_rotation.shape != (3, 3)
                    or not np.all(np.isfinite(target_rotation))
                    or not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(target_rotation), 1.0, atol=1e-6)):
                raise ValueError("target_rotation must be a finite SO(3) matrix.")
            if (not np.isfinite(orientation_tolerance) or orientation_tolerance <= 0
                    or not np.isfinite(orientation_weight) or orientation_weight <= 0):
                raise ValueError("Orientation tolerance and weight must be finite and positive.")

        def residual():
            # 分别用 m、rad 判断位置和姿态收敛；线搜索的加权误差统一为 m。
            position, _, rotation = get_tcp_pose(self.data)
            ep = target_position - position
            er = (np.zeros(3) if target_rotation is None else
                  rotation_error_world(target_rotation, rotation))
            task_error = (ep if target_rotation is None else
                          np.concatenate([ep, orientation_weight * er]))
            return task_error, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

        # ----------------------------------------------------
        # 4.1 检查目标位置
        # ----------------------------------------------------

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        if target_position.shape != (3,):

            raise ValueError(
                "Target position must have shape (3,)."
            )

        if not np.all(
            np.isfinite(target_position)
        ):

            raise ValueError(
                "Target position contains non-finite values."
            )

        # ----------------------------------------------------
        # 4.2 初始化关节角
        # ----------------------------------------------------

        if initial_q is None:

            q = self.home_q.copy()

        else:

            q = np.asarray(
                initial_q,
                dtype=np.float64,
            ).copy()

            if q.shape != (7,):

                raise ValueError(
                    "initial_q must have shape (7,)."
                )

            if not np.all(np.isfinite(q)):

                raise ValueError(
                    "initial_q contains non-finite values."
                )

        # ----------------------------------------------------
        # 4.3 约束在关节范围内
        # ----------------------------------------------------

        q = np.clip(
            q,
            self.lower_limits,
            self.upper_limits,
        )

        # ----------------------------------------------------
        # 4.4 初始化求解状态
        # ----------------------------------------------------

        success = False
        reason = "maximum_iterations"

        iterations = 0

        # ====================================================
        # 5. IK 迭代
        # ====================================================

        for iteration in range(
            self.max_iterations
        ):

            iterations = iteration + 1

            # ----------------------------------------
            # 5.1 当前关节配置
            # ----------------------------------------

            self._set_configuration(q)

            # ----------------------------------------
            # 5.2 正向运动学（FK）
            # ----------------------------------------

            task_error, position_norm, orientation_norm = residual()
            error_norm = float(np.linalg.norm(task_error))

            if (position_norm < self.tolerance
                    and orientation_norm < orientation_tolerance):

                success = True
                reason = "converged"

                break

            # ----------------------------------------
            # 5.5 计算当前平移与旋转 Jacobian
            # ----------------------------------------

            jacp, jacr, _, _ = get_tcp_jacobian(
                self.model,
                self.data,
            )

            # ----------------------------------------
            # 5.6 DLS 关节增量
            # ----------------------------------------

            task_jacobian = (jacp if target_rotation is None else
                             np.vstack([jacp, orientation_weight * jacr]))
            delta_q = self._compute_dls_step(task_jacobian, task_error)

            # ----------------------------------------
            # 5.7 回溯线搜索
            # ----------------------------------------

            accepted = False

            for alpha in [
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
            ]:

                candidate_q = (
                    q
                    +
                    alpha * delta_q
                )

                # ------------------------------------
                # 关节限幅
                # ------------------------------------

                candidate_q = np.clip(
                    candidate_q,
                    self.lower_limits,
                    self.upper_limits,
                )

                # ------------------------------------
                # 计算候选关节配置的 FK
                # ------------------------------------

                self._set_configuration(
                    candidate_q
                )

                candidate_residual, _, _ = residual()
                candidate_error = float(np.linalg.norm(candidate_residual))

                # ------------------------------------
                # 只接受能减小任务误差（位置或加权位姿）的增量
                # ------------------------------------

                if candidate_error < error_norm - 1e-10:

                    q = candidate_q

                    accepted = True

                    break

            # ----------------------------------------
            # 5.8 无法继续减小误差
            # ----------------------------------------

            if not accepted:

                reason = "stalled"

                break

        # ====================================================
        # 6. 最终 FK 验证
        # ====================================================

        self._set_configuration(q)

        final_position, _, _ = get_tcp_pose(
            self.data
        )

        final_error = float(
            np.linalg.norm(
                target_position
                -
                final_position
            )
        )

        _, _, final_orientation_error = residual()
        success = (final_error < self.tolerance
                   and final_orientation_error < orientation_tolerance)
        if success:
            reason = "converged"

        # ====================================================
        # 7. 输出求解结果
        # ====================================================

        result = {

            "success": bool(success),

            "reason": reason,

            "iterations": int(iterations),

            "q": q.copy(),

            "target_position": (
                target_position.copy()
            ),

            "final_position": (
                final_position.copy()
            ),

            "position_error": final_error,
            "orientation_error": final_orientation_error,  # 单位 rad；仅求解位置时为零

        }

        return result
