"""Day 13：四维增量控制、基于真实状态与接触的 Gymnasium 操作环境。

不调用 Scripted Agent，也不在环境内部替策略选择抓取或运输阶段。
长度 m、角度 rad、时间 s、力 N；全部动态物体写位置仅限 reset。
"""

from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from controllers.gripper_controller import PandaGripperController
from controllers.ik_controller import PandaIKSolver, rotation_error_world
from controllers.joint_controller import PandaJointPositionController
from robotics.contacts import PandaContactDetector
from robotics.grasp_experiment import cube_bottom_clearance, load_cube_scene
from robotics.kinematics import get_arm_qpos, get_arm_qvel, get_tcp_pose, set_arm_qpos


TASK_MODES = ("reach", "grasp", "lift", "pick_place")


@dataclass(frozen=True)
class RewardWeights:
    """无量纲奖励权重；进度项为势函数差，成本率乘以控制步长 s。"""

    reach_progress: float = 2.0
    reach_distance_rate: float = 0.2
    approach_progress: float = 1.0
    approach_distance_rate: float = 0.05
    grasp_progress: float = 0.5
    lift_progress: float = 2.0
    place_progress: float = 2.0
    release_progress: float = 1.0
    orientation_rate: float = 0.1
    action_rate: float = 0.02
    action_change: float = 0.01
    clipped_action: float = 0.1
    ik_failure: float = 1.0
    collision_rate: float = 1.0
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    shaping_discount: float = 0.99


@dataclass(frozen=True)
class PickPlaceEnvConfig:
    """控制、随机初始化、奖励归一化尺度与物理验收阈值集中配置。"""

    frame_skip: int = 20
    action_scale_m: float = 0.01
    max_episode_steps: int = 1500
    max_simulation_time_s: float = 60.0
    max_joint_speed_rad_s: float = 0.5
    ik_position_tolerance_m: float = 0.0005
    ik_orientation_tolerance_rad: float = 0.005
    ik_max_iterations: int = 60
    max_ik_failures: int = 5
    workspace_low: tuple = (0.32, -0.19, 0.215)
    workspace_high: tuple = (0.68, 0.19, 0.65)
    cube_random_xy_half_range_m: tuple = (0.025, 0.025)
    cube_yaw_half_range_rad: float = 0.20
    target_random_xy_half_range_m: tuple = (0.035, 0.025)
    minimum_place_separation_m: float = 0.06
    pregrasp_clearance_m: float = 0.12
    lift_goal_height_m: float = 0.10
    tcp_position_tolerance_m: float = 0.009
    tcp_orientation_tolerance_rad: float = 0.035
    arm_velocity_tolerance_rad_s: float = 0.10
    cube_linear_velocity_tolerance_m_s: float = 0.02
    cube_angular_velocity_tolerance_rad_s: float = 0.20
    lift_clearance_m: float = 0.04
    placement_xy_tolerance_m: float = 0.012
    placement_z_tolerance_m: float = 0.004
    release_separation_m: float = 0.08
    open_feedback_min_m: float = 0.07
    contact_min_force_N: float = 0.05
    reach_hold_s: float = 0.40
    grasp_hold_s: float = 0.50
    lift_hold_s: float = 0.50
    place_hold_s: float = 0.50
    dropped_clearance_m: float = -0.03
    numerical_abs_limit: float = 1e6
    viewer_open_timeout_s: float = 2.0
    viewer_retry_interval_s: float = 0.02
    distance_scale_m: float = 0.20
    terminate_on_finger_table: bool = True
    rewards: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self):
        """尽早拒绝无效时间、步数、范围和奖励配置，避免难诊断的训练回合。"""
        integers = ("frame_skip", "max_episode_steps", "ik_max_iterations", "max_ik_failures")
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name, value in vars(self).items():
            if isinstance(value, float) and (not np.isfinite(value) or
                    (value <= 0 and name != "dropped_clearance_m")):
                raise ValueError(f"{name} 必须为有限正数")
        low, high = np.asarray(self.workspace_low), np.asarray(self.workspace_high)
        if (low.shape != (3,) or high.shape != (3,) or not np.all(np.isfinite([low, high]))
                or np.any(low >= high)):
            raise ValueError("工作空间必须为有限、有序的三维边界")
        for bounds in (self.cube_random_xy_half_range_m, self.target_random_xy_half_range_m):
            if np.shape(bounds) != (2,) or not np.all(np.isfinite(bounds)) or np.any(np.asarray(bounds) < 0):
                raise ValueError("随机 XY 半范围必须为非负二维向量")
        if self.dropped_clearance_m >= 0 or not 0 < self.open_feedback_min_m <= 0.08:
            raise ValueError("掉落高度应为负值，开度门槛应在 (0, 0.08] m 内")
        if any(not np.isfinite(v) or v < 0 for v in vars(self.rewards).values()):
            raise ValueError("奖励权重必须有限且非负")
        if not 0 < self.rewards.shaping_discount <= 1:
            raise ValueError("shaping_discount 必须在 (0, 1] 内，训练 gamma 应与它一致")


# 物理连续量保留 SI 原值，不截断异常速度或关节软限位越界；姿态用前两列表示。
# 命令、上一动作与计时器是环境自身已发生的控制状态，不是专家动作或未来信息。
OBSERVATION_FIELDS = (
    ("arm_qpos", 7, "rad"), ("arm_qvel", 7, "rad/s"),
    ("tcp_position", 3, "m"), ("tcp_rotation_6d", 6, "1"),
    ("cube_position", 3, "m"), ("cube_rotation_6d", 6, "1"),
    ("cube_velocity", 6, "m/s,rad/s"), ("finger_positions", 2, "m"),
    ("finger_velocities", 2, "m/s"),
    ("gripper_opening", 1, "m"), ("tcp_to_cube", 3, "m"),
    ("cube_to_target", 3, "m"), ("target_position", 3, "m"),
    ("finger_contacts", 3, "0/1"), ("cube_table_contact", 1, "0/1"),
    ("task_one_hot", 4, "0/1"), ("tcp_command", 3, "m"),
    ("joint_command", 7, "rad"), ("joint_target", 7, "rad"),
    ("previous_action", 4, "[-1,1]"),
    ("target_rotation_6d", 6, "1"), ("stable_timers", 4, "[0,1]"),
    ("has_lifted", 1, "0/1"), ("ik_failure_fraction", 1, "[0,1]"),
    ("remaining_fraction", 1, "[0,1]"),
)
OBSERVATION_SLICES = {}
_offset = 0
for _name, _size, _unit in OBSERVATION_FIELDS:
    OBSERVATION_SLICES[_name] = slice(_offset, _offset + _size)
    _offset += _size
OBSERVATION_SIZE = _offset


class PickPlaceEnv(gym.Env):
    """四任务共用的 94 维观测、4 维动作环境；没有内置任务决策状态机。"""

    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, task_mode="reach", render_mode=None, config=None):
        """每个实例独立加载模型、数据和控制器；无界面模式不导入 Viewer。"""
        super().__init__()
        if task_mode not in TASK_MODES or render_mode not in (None, "human"):
            raise ValueError("不支持的 task_mode 或 render_mode")
        self.config = config or PickPlaceEnvConfig()
        self.default_task_mode = task_mode
        self.task_mode = task_mode
        self.render_mode = render_mode
        self.model, self.data = load_cube_scene()
        self.physics_dt = float(self.model.opt.timestep)
        self.dt = self.physics_dt * self.config.frame_skip
        self.metadata = {**type(self).metadata, "render_fps": max(1, round(1/self.dt))}
        self.arm = PandaJointPositionController(self.model, self.config.max_joint_speed_rad_s)
        self.gripper = PandaGripperController(self.model)
        self.ik = PandaIKSolver(self.model, tolerance=self.config.ik_position_tolerance_m,
                                max_iterations=self.config.ik_max_iterations)
        self.detector = PandaContactDetector(self.model, self.config.contact_min_force_N)
        self.action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)
        low = np.full(OBSERVATION_SIZE, -np.inf, dtype=np.float32)
        high = np.full(OBSERVATION_SIZE, np.inf, dtype=np.float32)
        for name, _, unit in OBSERVATION_FIELDS:
            if name.endswith("rotation_6d") or unit == "[-1,1]":
                low[OBSERVATION_SLICES[name]], high[OBSERVATION_SLICES[name]] = -1, 1
            elif unit in ("0/1", "[0,1]"):
                low[OBSERVATION_SLICES[name]], high[OBSERVATION_SLICES[name]] = 0, 1
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.viewer = None
        self._needs_reset = True
        self.cube_qadr = int(self.model.jnt_qposadr[self.model.joint("cube_free").id])
        self.half_size = self.model.geom_size[self.model.geom("cube_geom").id].copy()
        self.table_half_size = self.model.geom_size[self.model.geom("table").id].copy()
        self.table_center = self.data.geom("table").xpos.copy()
        self.table_top = float(self.table_center[2] + self.table_half_size[2])

    @staticmethod
    def _xyz(value, name):
        """对外部 XYZ 选项做严格形状与有限性检查，不悄悄改变用户初始条件。"""
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} 必须为有限的世界坐标 XYZ（m）")
        return result.copy()

    def _on_table(self, position, yaw=0.):
        """检查整个方块在桌面内，并使用偏航后的投影半尺寸避免边缘悬空。"""
        cs, sn = abs(np.cos(yaw)), abs(np.sin(yaw))
        extent = np.array([cs*self.half_size[0] + sn*self.half_size[1],
                           sn*self.half_size[0] + cs*self.half_size[1]])
        return bool(np.all(np.abs(position[:2] - self.table_center[:2]) + extent
                           + self.config.placement_xy_tolerance_m < self.table_half_size[:2]))

    def reset(self, *, seed=None, options=None):
        """用完整场景关键帧复位；所有采样只使用 Gymnasium 的 np_random。"""
        super().reset(seed=seed)
        self._needs_reset = True
        options = {} if options is None else dict(options)
        allowed = {"task_mode", "cube_position", "cube_yaw", "target_position", "start_pose"}
        if set(options) - allowed:
            raise ValueError(f"未知 reset 选项：{set(options) - allowed}")
        self.task_mode = options.get("task_mode", self.default_task_mode)
        if self.task_mode not in TASK_MODES:
            raise ValueError("无效 task_mode")
        self._needs_reset = True
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("day11_home").id)
            cube = np.array([.50, 0., self.table_top + self.half_size[2]])
            cube[:2] += self.np_random.uniform(-np.asarray(self.config.cube_random_xy_half_range_m),
                                              self.config.cube_random_xy_half_range_m)
            cube = self._xyz(options.get("cube_position", cube), "cube_position")
            yaw = float(options.get("cube_yaw", self.np_random.uniform(
                -self.config.cube_yaw_half_range_rad, self.config.cube_yaw_half_range_rad)))
            if (not np.isfinite(yaw) or not self._on_table(cube, yaw)
                    or not np.isclose(cube[2], self.table_top + self.half_size[2], atol=1e-9, rtol=0)):
                raise ValueError("初始 Cube 必须完整位于桌面内，且底面恰好在台面上")
            # 仅 reset 可定义初始物体位姿；step 中绝不写 Cube 的 qpos/qvel。
            self.data.qpos[self.cube_qadr:self.cube_qadr+3] = cube
            self.data.qpos[self.cube_qadr+3:self.cube_qadr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            self.target_rotation = np.array([[-np.sin(yaw), np.cos(yaw), 0.],
                                             [np.cos(yaw), np.sin(yaw), 0.], [0., 0., -1.]])
            if self.task_mode == "reach":
                target = self.np_random.uniform([.46, -.08, .42], [.60, .08, .55])
            elif self.task_mode == "pick_place":
                for _ in range(100):
                    target = np.array([.50, .12, self.table_top + self.half_size[2]])
                    target[:2] += self.np_random.uniform(-np.asarray(self.config.target_random_xy_half_range_m),
                                                        self.config.target_random_xy_half_range_m)
                    if self._on_table(target, yaw) and np.linalg.norm(target[:2]-cube[:2]) >= self.config.minimum_place_separation_m:
                        break
                else:
                    raise ValueError("随机范围无法生成安全且分离的放置目标")
            else:
                target = cube + [0, 0, self.config.lift_goal_height_m if self.task_mode == "lift" else 0]
            self.target_position = self._xyz(options.get("target_position", target), "target_position")
            if self.task_mode in ("grasp", "lift") and not np.allclose(self.target_position, target, atol=1e-9, rtol=0):
                raise ValueError("grasp/lift 的参考目标由 Cube 初始位置与配置高度派生，不能另设无关 XYZ")
            if (np.any(self.target_position < self.config.workspace_low)
                    or np.any(self.target_position > self.config.workspace_high)):
                raise ValueError("目标超出配置的工作空间")
            if self.task_mode == "pick_place" and (not self._on_table(self.target_position, yaw)
                    or abs(self.target_position[2] - cube[2]) > 1e-9):
                raise ValueError("放置目标必须是同一桌面上可容纳完整 Cube 的中心位置")
            mujoco.mj_forward(self.model, self.data)
            start_pose = options.get("start_pose", "home")
            if start_pose not in ("home", "pre_grasp"):
                raise ValueError("start_pose 只能为 home 或 pre_grasp")
            if start_pose == "pre_grasp":
                solution = self.ik.solve(cube + [0, 0, self.config.pregrasp_clearance_m],
                                         initial_q=get_arm_qpos(self.data), target_rotation=self.target_rotation,
                                         orientation_tolerance=self.config.ik_orientation_tolerance_rad)
                if not solution["success"]:
                    raise ValueError("初始预抓取位姿 IK 不可达")
                set_arm_qpos(self.model, self.data, solution["q"])
            self.arm.reset(self.data)
            self.gripper.command(self.data, self.gripper.width_range[1])
            mujoco.mj_forward(self.model, self.data)
            self.tcp_command = get_tcp_pose(self.data)[0].copy()
            self.joint_target = self.arm.command_q.copy()
            self.previous_action = np.array([0., 0., 0., 1.], dtype=np.float32)
            self.timers = np.zeros(4)
            self.hold_durations = np.array([self.config.reach_hold_s, self.config.grasp_hold_s,
                                            self.config.lift_hold_s, self.config.place_hold_s])
            self.has_lifted = False
            self.ik_failures = 0
            self.episode_step = 0
            self.episode_return = 0.0
            self._needs_reset = False
            state = self._state()
            self.previous_potentials = self._potentials(state)
            observation = self._observation(state)
            info = self._info(state, {}, False, None, None, physics_steps=0)
        if self.render_mode == "human":
            self.render()
        return observation, info

    def _state(self):
        """读取当前已 mj_forward 的物理状态，不使用跨步缓存的位姿或接触。"""
        tcp, _, rotation = get_tcp_pose(self.data)
        contact = self.detector.read(self.data)
        spatial_velocity = np.zeros(6)
        # MuJoCo 此 API 返回“角速度、线速度”，flag=0 统一为世界坐标；
        # 不直接把自由关节局部角速度混入世界坐标观测。Cube 的质心就在 body 原点。
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 int(self.model.body("cube").id), spatial_velocity, 0)
        velocity = np.r_[spatial_velocity[3:], spatial_velocity[:3]]
        feedback = self.gripper.feedback(self.data)
        cube = self.data.body("cube").xpos.copy()
        return {"tcp": tcp, "rotation": rotation, "cube": cube,
                "cube_rotation": self.data.body("cube").xmat.reshape(3, 3).copy(),
                "cube_velocity": velocity, "fingers": feedback["finger_position_m"],
                "finger_velocities": feedback["finger_velocity_m_s"],
                "opening": feedback["opening_m"], "contact": contact,
                "clearance": cube_bottom_clearance(self.model, self.data),
                "linear_speed": float(np.linalg.norm(velocity[:3])),
                "angular_speed": float(np.linalg.norm(velocity[3:])),
                "arm_speed": float(np.max(np.abs(get_arm_qvel(self.data)))),
                "tcp_error": float(np.linalg.norm(tcp - self.target_position)),
                "cube_error": float(np.linalg.norm(cube - self.target_position)),
                "orientation_error": float(np.linalg.norm(rotation_error_world(self.target_rotation, rotation)))}

    def _physical_conditions(self, state):
        """独立物理条件，不要求任何 Scripted Agent 状态序列。"""
        c, contact = self.config, state["contact"]
        still = (state["linear_speed"] < c.cube_linear_velocity_tolerance_m_s
                 and state["angular_speed"] < c.cube_angular_velocity_tolerance_rad_s)
        quiet_arm = state["arm_speed"] < c.arm_velocity_tolerance_rad_s
        reach = (state["tcp_error"] < c.tcp_position_tolerance_m
                 and state["orientation_error"] < c.tcp_orientation_tolerance_rad and quiet_arm)
        grasp = contact.grasp_candidate and still and quiet_arm
        lift = (grasp and not contact.cube_table and not contact.cube_floor
                and state["clearance"] >= c.lift_clearance_m)
        position = (np.linalg.norm(state["cube"][:2] - self.target_position[:2]) < c.placement_xy_tolerance_m
                    and abs(state["cube"][2] - self.target_position[2]) < c.placement_z_tolerance_m
                    and abs(state["clearance"]) < c.placement_z_tolerance_m)
        place = (self.has_lifted and position and still and contact.cube_table and not contact.cube_floor
                 and not contact.finger_contact and state["opening"] >= c.open_feedback_min_m
                 and np.linalg.norm(state["tcp"] - state["cube"]) >= c.release_separation_m)
        return np.array([reach, grasp, lift, place], dtype=bool)

    def _update_timers(self, state):
        """每个物理子步累计连续证据，抬升历史只能由真实稳定持物置位。"""
        valid = self._physical_conditions(state)
        self.timers = np.where(valid, np.minimum(self.timers+self.physics_dt, self.hold_durations), 0.)
        self.has_lifted |= bool(self.timers[2] >= self.config.lift_hold_s)

    def _potentials(self, state):
        """以当前物理证据构造有界势；无双侧夹持的弹起不产生高度势。"""
        c, w = self.config, self.config.rewards
        reach = np.clip(state["tcp_error"]/c.distance_scale_m, 0, 5)
        approach = np.clip(np.linalg.norm(state["tcp"]-state["cube"])/c.distance_scale_m, 0, 5)
        return {"reach_progress": w.reach_progress/(1+float(reach)),
                "approach_progress": w.approach_progress/(1+float(approach)),
                "grasp_progress": w.grasp_progress*float(self.timers[1]/c.grasp_hold_s),
                "lift_progress": w.lift_progress*float(np.clip(state["clearance"]/c.lift_goal_height_m, 0, 1))
                * float(state["contact"].grasp_candidate and not state["contact"].cube_table),
                "place_progress": w.place_progress*float(self.has_lifted)
                * float(1-np.clip(state["cube_error"]/c.distance_scale_m, 0, 1)),
                "release_progress": w.release_progress*float(self.timers[3]/c.place_hold_s)}

    def _reward(self, state, action, ik_valid, clipped, success, failure, elapsed_s):
        """返回可审计分项；无效 IK/物理失败步禁用全部正向密集奖励。"""
        c, w = self.config, self.config.rewards
        current = self._potentials(state)
        active = {"reach": ("reach_progress",), "grasp": ("approach_progress", "grasp_progress"),
                  "lift": ("approach_progress", "grasp_progress", "lift_progress"),
                  "pick_place": ("approach_progress", "grasp_progress", "lift_progress", "place_progress", "release_progress")}
        # 使用与未来训练 gamma 匹配的折扣势整形，避免折扣后靠接触循环刷分。
        # 真正终止时后继势为零；外部时间截断保留后继势，便于 SB3 正确 bootstrap。
        terminal = success or failure is not None
        terms = {name: float(w.shaping_discount*(0. if terminal else current[name])
                             - self.previous_potentials[name]) for name in active[self.task_mode]}
        distance = (state["tcp_error"] if self.task_mode == "reach" else
                    float(np.linalg.norm(state["tcp"]-state["cube"])))
        rate = w.reach_distance_rate if self.task_mode == "reach" else w.approach_distance_rate
        terms["distance_cost"] = -rate*min(distance/c.distance_scale_m, 5)*elapsed_s
        terms["orientation_cost"] = -w.orientation_rate*min(state["orientation_error"]/np.pi, 1)*elapsed_s
        terms["action_cost"] = -w.action_rate*float(np.mean(action**2))*elapsed_s
        terms["action_change_cost"] = -w.action_change*float(np.mean((action-self.previous_action)**2))
        terms["clipped_action_cost"] = -w.clipped_action*float(clipped)
        terms["ik_failure_cost"] = -w.ik_failure*float(not ik_valid)
        terms["collision_cost"] = -w.collision_rate*float(state["contact"].finger_table)*elapsed_s
        terms["success_bonus"] = w.success_bonus*float(success and ik_valid and failure is None)
        terms["failure_penalty"] = -w.failure_penalty*float(failure is not None)
        if not ik_valid or failure is not None:
            terms = {name: min(0., float(value)) for name, value in terms.items()}
        self.previous_potentials = current
        return terms

    def _observation(self, state):
        """构建原值物理观测与有界控制记忆；任何非有限输出均为实现错误。"""
        c, contact = self.config, state["contact"]
        remaining = max(0., 1-max(self.episode_step/c.max_episode_steps,
                                  self.data.time/c.max_simulation_time_s))
        values = (get_arm_qpos(self.data), get_arm_qvel(self.data), state["tcp"],
                  state["rotation"][:, :2].T.ravel(), state["cube"],
                  state["cube_rotation"][:, :2].T.ravel(), state["cube_velocity"], state["fingers"],
                  state["finger_velocities"],
                  [state["opening"]], state["cube"]-state["tcp"], self.target_position-state["cube"],
                  self.target_position, [contact.left_force_N > 0, contact.right_force_N > 0, contact.grasp_candidate],
                  [contact.cube_table], np.eye(4)[TASK_MODES.index(self.task_mode)], self.tcp_command,
                  self.arm.command_q, self.joint_target, self.previous_action, self.target_rotation[:, :2].T.ravel(),
                  np.clip(self.timers/self.hold_durations, 0, 1), [self.has_lifted],
                  [min(1., self.ik_failures/c.max_ik_failures)], [remaining])
        observation = np.concatenate(values).astype(np.float32)
        if not np.all(np.isfinite(observation)) or not self.observation_space.contains(observation):
            raise FloatingPointError("观测超出空间或存在非有限值")
        return observation

    def _info(self, state, terms, success, failure, reason, **extra):
        """指标保留明确单位，兼容 JSON 和 SB3 Monitor；不直接返回 NumPy 数组。"""
        info = {"task_mode": self.task_mode, "is_success": bool(success),
                "tcp_position_error": state["tcp_error"], "cube_position_error": state["cube_error"],
                "cube_bottom_clearance": state["clearance"],
                "grasp_candidate": bool(state["contact"].grasp_candidate),
                "cube_table_contact": bool(state["contact"].cube_table), "gripper_opening": state["opening"],
                "tcp_orientation_error_rad": state["orientation_error"],
                "cube_linear_speed_m_s": state["linear_speed"], "cube_angular_speed_rad_s": state["angular_speed"],
                "cube_position_m": state["cube"].tolist(), "tcp_position_m": state["tcp"].tolist(),
                "tcp_command_m": self.tcp_command.tolist(), "target_position_m": self.target_position.tolist(),
                "finger_positions_m": state["fingers"].tolist(), "has_lifted": bool(self.has_lifted),
                "stable_times_s": self.timers.tolist(), "reward_terms": terms,
                "episode_step": self.episode_step, "simulation_time": float(self.data.time),
                "failure_reason": failure, "termination_reason": reason,
                "control_dt_s": self.dt, "metric_units": {"distance": "m", "time": "s", "angle": "rad"},
                **extra}
        return info

    def _numerical_failure(self, detail, physics_steps=0):
        """损坏状态不能用于 IK/观测；返回明确标记的中性有限终止观测，不伪造恢复。"""
        self.episode_step += 1
        reward = -self.config.rewards.failure_penalty
        self.episode_return += reward
        self._needs_reset = True
        sim_time = float(self.data.time) if np.isfinite(self.data.time) else 0.
        info = {"task_mode": self.task_mode, "is_success": False, "failure_reason": "NUMERICAL_ERROR",
                "termination_reason": "NUMERICAL_ERROR", "failure_detail": detail,
                "tcp_orientation_error_rad": None,
                "observation_sanitized": True, "tcp_position_error": None, "cube_position_error": None,
                "cube_bottom_clearance": None, "grasp_candidate": False, "cube_table_contact": False,
                "gripper_opening": None, "reward_terms": {"failure_penalty": reward},
                "episode_step": self.episode_step, "simulation_time": sim_time, "physics_steps": physics_steps,
                "episode": {"r": self.episode_return, "l": self.episode_step, "t": sim_time,
                            "is_success": False}}
        return np.zeros(self.observation_space.shape, dtype=np.float32), float(reward), True, False, info

    def _task_success(self, valid, failure):
        """Shared success hook; subclasses may latch a completed physical hold."""
        return bool(valid and failure is None and self.timers[TASK_MODES.index(self.task_mode)]
                    >= self.hold_durations[TASK_MODES.index(self.task_mode)])

    def _finite_dynamics(self):
        """积分前后检查原始状态和警告，避免 MuJoCo 自动重置掩盖严重物理异常。"""
        values = np.concatenate((self.data.qpos, self.data.qvel, self.data.ctrl, [self.data.time]))
        return bool(np.all(np.isfinite(values)) and np.max(np.abs(values)) < self.config.numerical_abs_limit
                    and not any(w.number for w in self.data.warning))

    def step(self, action):
        """执行一条四维动作并推进固定 frame_skip 步；仅数值损坏时提前停止积分。"""
        if self._needs_reset:
            raise RuntimeError("请先 reset；回合结束后不能继续 step")
        try:
            raw = np.asarray(action, dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError("action 必须是有限实数四维向量") from error
        if raw.shape != (4,) or not np.all(np.isfinite(raw)) or np.iscomplexobj(action):
            raise ValueError("action 必须具有 shape=(4,) 且不含 NaN/Inf/复数")
        applied = np.clip(raw, -1, 1).astype(np.float32)
        clipped = bool(np.any(raw != applied.astype(float))) if np.any(np.abs(raw) > 1) else False
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            if not self._finite_dynamics():
                return self._numerical_failure("积分前状态损坏或存在 MuJoCo 警告")
            requested = self.tcp_command + self.config.action_scale_m*applied[:3].astype(float)
            desired = np.clip(requested, self.config.workspace_low, self.config.workspace_high)
            valid, ik_reason = False, None
            try:
                solution = self.ik.solve(desired, initial_q=self.arm.command_q,
                                         target_rotation=self.target_rotation,
                                         orientation_tolerance=self.config.ik_orientation_tolerance_rad)
                q = np.asarray(solution["q"])
                valid = bool(solution["success"] and q.shape == (7,) and np.all(np.isfinite(q))
                             and np.all(q >= self.ik.lower_limits) and np.all(q <= self.ik.upper_limits)
                             and np.isfinite(solution["position_error"]) and np.isfinite(solution["orientation_error"])
                             and solution["position_error"] < self.config.ik_position_tolerance_m
                             and solution["orientation_error"] < self.config.ik_orientation_tolerance_rad)
                ik_reason = solution["reason"]
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                ik_reason = str(error)
            if valid:
                self.joint_target = q.copy()
                self.tcp_command = desired.copy()
                self.ik_failures = 0
            else:
                self.ik_failures += 1
            width = float((applied[3]+1)*0.5*self.gripper.width_range[1])
            failure = None
            for physics_step in range(self.config.frame_skip):
                self.arm.step(self.data, self.joint_target, self.physics_dt)
                self.gripper.command(self.data, width)
                mujoco.mj_step(self.model, self.data)
                if not self._finite_dynamics():
                    return self._numerical_failure("积分后状态损坏或存在 MuJoCo 警告", physics_step+1)
                mujoco.mj_forward(self.model, self.data)
                state = self._state()
                self._update_timers(state)
                # 锁存物理失败，但仍完成本控制步的固定子步数；只有数值异常会提前停止。
                if failure is None:
                    if state["contact"].cube_floor or (state["clearance"] < self.config.dropped_clearance_m
                                                       and not state["contact"].cube_table):
                        failure = "OBJECT_DROPPED"
                    elif state["contact"].finger_table and self.config.terminate_on_finger_table:
                        failure = "FINGER_TABLE_COLLISION"
            if self.ik_failures >= self.config.max_ik_failures:
                failure = failure or "IK_FAILED"
            self.episode_step += 1
            success = self._task_success(valid, failure)
            terminated = bool(success or failure is not None)
            timeout = (self.episode_step >= self.config.max_episode_steps
                       or self.data.time >= self.config.max_simulation_time_s-1e-12)
            truncated = bool(timeout and not terminated)
            reason = ("SUCCESS" if success else failure) or ("TIME_LIMIT" if truncated else None)
            terms = self._reward(state, applied, valid, clipped, success, failure,
                                 (physics_step+1)*self.physics_dt)
            reward = float(sum(terms.values()))
            self.episode_return += reward
            self.previous_action = applied.copy()
            observation = self._observation(state)
            info = self._info(state, terms, success, failure, reason, ik_success=valid,
                              ik_reason=ik_reason, consecutive_ik_failures=self.ik_failures,
                              action_clipped=clipped, applied_action=applied.tolist(),
                              workspace_clipped=bool(np.any(desired != requested)), physics_steps=physics_step+1,
                              elapsed_control_time_s=(physics_step+1)*self.physics_dt)
            self._needs_reset = terminated or truncated
            if self._needs_reset:
                info["episode"] = {"r": self.episode_return, "l": self.episode_step,
                                   "t": float(self.data.time), "is_success": success,
                                   "final_tcp_error_m": state["tcp_error"], "final_cube_error_m": state["cube_error"]}
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        """只同步当前状态；不积分、不改变任务计时，也不自动 reset。"""
        if self.render_mode != "human":
            return None
        if self._needs_reset and not hasattr(self, "episode_step"):
            raise RuntimeError("render 前需要 reset")
        if self.viewer is None:
            import mujoco.viewer

            # macOS 的 close() 仅请求 UI 线程退出，下一回合立即打开可能撞上
            # 尚未释放的单窗口槽位。仅对此已知错误做有时限的重试，不修改物理。
            deadline = time.monotonic() + self.config.viewer_open_timeout_s
            while True:
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    break
                except RuntimeError as error:
                    if ("another MuJoCo viewer is already open" not in str(error)
                            or time.monotonic() >= deadline):
                        raise
                    time.sleep(self.config.viewer_retry_interval_s)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [.45, .05, .3]
                self.viewer.cam.distance = 1.5
        self.viewer.sync()
        return None

    def close(self):
        """释放 Viewer；允许随后 reset 重用同一环境实例。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._needs_reset = True
