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


# ============================================================
# 1. Panda DLS IK Solver
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

        # Jp: (3, 7)
        # error: (3,)

        # ----------------------------------------
        # J * J^T
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
            np.eye(3)
        )

        # ----------------------------------------
        # 解一个 3 维线性方程组
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
    ):

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
            # 5.2 FK
            # ----------------------------------------

            current_position, _, _ = get_tcp_pose(
                self.data
            )

            # ----------------------------------------
            # 5.3 TCP 位置误差
            # ----------------------------------------

            position_error = (
                target_position
                -
                current_position
            )

            error_norm = float(
                np.linalg.norm(
                    position_error
                )
            )

            # ----------------------------------------
            # 5.4 判断是否收敛
            # ----------------------------------------

            if error_norm < self.tolerance:

                success = True
                reason = "converged"

                break

            # ----------------------------------------
            # 5.5 计算当前平移 Jacobian
            # ----------------------------------------

            jacp, _, _, _ = get_tcp_jacobian(
                self.model,
                self.data,
            )

            # ----------------------------------------
            # 5.6 DLS 关节增量
            # ----------------------------------------

            delta_q = self._compute_dls_step(
                jacp,
                position_error,
            )

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

                candidate_position, _, _ = get_tcp_pose(
                    self.data
                )

                candidate_error = float(
                    np.linalg.norm(
                        target_position
                        -
                        candidate_position
                    )
                )

                # ------------------------------------
                # 只接受能减小位置误差的增量
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

        if final_error < self.tolerance:

            success = True
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
        }

        return result