"""Day 12：根据仿真真值与真实接触反馈执行抓取、运输和放置的有限状态机。"""

from dataclasses import asdict, dataclass
from enum import Enum

import mujoco
import numpy as np

from controllers.gripper_controller import PandaGripperController
from controllers.ik_controller import rotation_error_world
from controllers.pose_motion import MotionPlanningError, PoseMotionController
from robotics.contacts import PandaContactDetector
from robotics.grasp_experiment import cube_bottom_clearance
from robotics.kinematics import get_arm_qvel, get_tcp_pose


class PickPlaceState(str, Enum):
    """终态之外，每个状态均由物理反馈转移，并受阶段与整回合超时约束。"""

    IDLE = "IDLE"
    PRE_GRASP = "PRE_GRASP"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    TRANSPORT = "TRANSPORT"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


@dataclass(frozen=True)
class PickPlaceConfig:
    """集中配置 SI 单位阈值；默认值以 Day 11 实测误差和接触能力为依据。

    Day 11 TCP 静差约 6.5 mm、角误差约 0.016 rad，因此到达门槛为
    9 mm / 0.035 rad；抬升仍需 4 cm 离桌并连续确认 0.5 s。
    """

    tcp_position_tolerance_m: float = 0.009
    tcp_orientation_tolerance_rad: float = 0.035
    arm_velocity_tolerance_rad_s: float = 0.10
    placement_xy_tolerance_m: float = 0.012
    placement_z_tolerance_m: float = 0.004
    cube_linear_velocity_tolerance_m_s: float = 0.02
    cube_angular_velocity_tolerance_rad_s: float = 0.20
    reach_stable_s: float = 0.25
    grasp_stable_s: float = 0.25
    holding_stable_s: float = 0.50
    placement_stable_s: float = 0.50
    lost_grasp_grace_s: float = 0.12
    max_relative_slip_m: float = 0.018
    stage_timeout_s: float = 15.0
    gripper_timeout_s: float = 3.0
    total_timeout_s: float = 90.0
    hover_clearance_m: float = 0.12
    lift_distance_m: float = 0.12
    minimum_lift_clearance_m: float = 0.04
    retreat_distance_m: float = 0.12
    retreat_separation_m: float = 0.08
    place_preload_m: float = 0.001
    transit_speed_m_s: float = 0.06
    approach_speed_m_s: float = 0.035
    place_speed_m_s: float = 0.025
    max_joint_speed_rad_s: float = 0.35
    waypoint_spacing_m: float = 0.01
    open_width_m: float = 0.08
    open_feedback_min_m: float = 0.07
    contact_min_force_N: float = 0.05

    def __post_init__(self):
        """拒绝非有限或非正参数，避免无效计时与无界控制回合。"""
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("所有配置参数必须为有限正数，并遵循字段标注的单位")
        if self.open_feedback_min_m > self.open_width_m or self.open_width_m > 0.08:
            raise ValueError("夹爪开度阈值必须满足 0 < 反馈门槛 <= 命令 <= 0.08 m")


class ScriptedPickPlaceAgent:
    """单步任务控制器：读取 MjData 真值，输出到既有八个执行器的 ctrl。

    reset(data, target_place_position) 开始新任务而不重置物理；step(data)
    不调用 mj_step。外部驱动者推进一步物理并 mj_forward 后再次调用即可。
    目标 XYZ 表示同一水平台面上 Cube 的中心，单位 m，不是桌面接触点。
    """

    def __init__(self, model, config=None):
        """绑定官方 Panda、Cube 与水平台面；创建可单独测试的运动辅助层。"""
        self.model = model
        self.config = config or PickPlaceConfig()
        self.motion = PoseMotionController(model, self.config.max_joint_speed_rad_s,
                                           self.config.waypoint_spacing_m)
        self.gripper = PandaGripperController(model)
        self.detector = PandaContactDetector(model, self.config.contact_min_force_N)
        table = model.geom("table")
        cube = model.geom("cube_geom")
        if (model.geom_type[cube.id] != mujoco.mjtGeom.mjGEOM_BOX
                or model.geom_type[table.id] != mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("当前实现要求 Cube 与桌面均为 box")
        self.half_size = model.geom_size[cube.id].copy()
        self.table_half_size = model.geom_size[table.id].copy()
        self.state = PickPlaceState.IDLE
        self.started = False

    @property
    def done(self):
        """只在 SUCCESS 或 FAILURE 时停止外部物理循环。"""
        return self.state in (PickPlaceState.SUCCESS, PickPlaceState.FAILURE)

    def observe(self, data):
        """采集同一仿真时刻的接触、TCP、Cube、速度和实际夹爪反馈。"""
        tcp, _, rotation = get_tcp_pose(data)
        velocity = data.joint("cube_free").qvel.copy()
        return {"tcp": tcp, "rotation": rotation,
                "cube": data.body("cube").xpos.copy(),
                "cube_rotation": data.body("cube").xmat.reshape(3, 3).copy(),
                "cube_linear_speed": float(np.linalg.norm(velocity[:3])),
                "cube_angular_speed": float(np.linalg.norm(velocity[3:])),
                "arm_speed": float(np.max(np.abs(get_arm_qvel(data)))),
                "opening": self.gripper.feedback(data)["opening_m"],
                "contact": self.detector.read(data),
                "clearance": cube_bottom_clearance(self.model, data)}

    def reset(self, data, target_place_position, seed=0):
        """接收当前物理状态与放置中心 XYZ；seed 仅记录，当前策略完全确定性。"""
        target = np.asarray(target_place_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("目标必须为有限世界坐标 XYZ（m）")
        mujoco.mj_forward(self.model, data)
        self.motion.reset(data)
        self.target_place_position = target.copy()
        self.seed = int(seed)
        self.start_time_s = float(data.time)
        self.enter_time_s = float(data.time)
        self.stable_s = 0.0
        self.lost_s = 0.0
        self.failure_reason = None
        self.failure_detail = None
        self.failure_stage = None
        self.last_step_time_s = None
        self.state = PickPlaceState.IDLE
        self.state_history = []
        self.events = {name: False for name in ("reach_success", "grasp_candidate", "lift_success",
                                                "transport_complete", "release_success")}
        self.target_tcp = get_tcp_pose(data)[0].copy()
        self.target_rotation = get_tcp_pose(data)[2].copy()
        self.opening_target_m = self.config.open_width_m
        self.grasp_offset_local = None
        self.max_lift_clearance_m = 0.0
        self.started = True
        self.table_center = data.geom("table").xpos.copy()
        if not np.allclose(data.geom("table").xmat.reshape(3, 3), np.eye(3)):
            raise ValueError("当前实现仅支持水平且轴对齐的静态桌面")
        self.table_top_m = self.table_center[2] + self.table_half_size[2]
        self.place_center = target.copy()
        self.place_center[2] = self.table_top_m + self.half_size[2]
        self._record(data, "任务初始化；保持当前合法关节命令并打开夹爪")
        if (abs(target[2] - self.place_center[2]) > self.config.placement_z_tolerance_m
                or np.any(np.abs(target[:2] - self.table_center[:2])
                          + self.half_size[:2] + self.config.placement_xy_tolerance_m
                          > self.table_half_size[:2])):
            self._fail(data, "PLACE_FAILED", "目标中心不在桌面安全放置范围或高度不匹配")

    def _record(self, data, reason):
        """转移日志保留物理证据，时间为本次任务开始后的仿真秒数。"""
        self.state_history.append({"state": self.state.value,
                                   "time_s": float(data.time - self.start_time_s),
                                   "reason": reason, "snapshot": self.status(data)})

    def _fail(self, data, reason, detail):
        """锁存可定位的失败原因；保留最后合法命令，不尝试自动重抓或瞬移。"""
        if self.done:
            return
        self.failure_reason = reason
        self.failure_detail = detail
        self.failure_stage = self.state.value
        if self.state_history[-1]["state"] != self.state.value:
            self._record(data, "进入阶段时目标规划失败")
        self.motion.hold()
        self.state = PickPlaceState.FAILURE
        self.enter_time_s = float(data.time)
        self._record(data, detail)

    def _move(self, data, target, speed):
        """提交位姿轨迹；任何路点 IK 失败均在发送轨迹前转为 FAILURE。"""
        self.target_tcp = np.asarray(target, dtype=float).copy()
        try:
            self.motion.plan(data, self.target_tcp, self.target_rotation, speed)
        except (MotionPlanningError, ValueError, np.linalg.LinAlgError) as error:
            self._fail(data, "IK_FAILED", str(error))

    def _transition(self, data, state, reason):
        """统一处理状态进入动作：目标生成、保持命令、夹爪目标与计时器清零。"""
        completed_state = self.state.value
        completed_stable_s = self.stable_s
        self.state = state
        self.enter_time_s = float(data.time)
        self.stable_s = 0.0
        self.lost_s = 0.0
        obs = self.observe(data)
        c = self.config
        if state == PickPlaceState.PRE_GRASP:
            # 取 Cube 水平 x 轴作为闭合轴，同时保持 TCP 的 +z 朝下。
            closing = obs["cube_rotation"][:, 0].copy()
            if np.linalg.norm(closing[:2]) < 0.5:
                closing = obs["cube_rotation"][:, 1].copy()
            closing[2] = 0
            closing /= np.linalg.norm(closing)
            self.target_rotation = np.column_stack((np.cross(closing, [0, 0, -1]),
                                                    closing, [0, 0, -1]))
            self.pick_center = obs["cube"].copy()
            self._move(data, self.pick_center + [0, 0, c.hover_clearance_m], c.transit_speed_m_s)
        elif state == PickPlaceState.APPROACH:
            self.pick_center = obs["cube"].copy()
            self._move(data, self.pick_center, c.approach_speed_m_s)
        elif state in (PickPlaceState.CLOSE_GRIPPER, PickPlaceState.VERIFY_GRASP):
            self.motion.hold()
            self.opening_target_m = 0.0
        elif state == PickPlaceState.LIFT:
            self.grasp_offset_local = obs["rotation"].T @ (obs["cube"] - obs["tcp"])
            self._move(data, obs["tcp"] + [0, 0, c.lift_distance_m], c.approach_speed_m_s)
        elif state in (PickPlaceState.TRANSPORT, PickPlaceState.PLACE):
            # 每段开始重新测量 TCP->Cube 偏移，补偿实际夹持偏心及可容忍的小量下滑。
            self.grasp_offset_local = obs["rotation"].T @ (obs["cube"] - obs["tcp"])
            center = self.place_center.copy()
            center[2] += c.lift_distance_m if state == PickPlaceState.TRANSPORT else -c.place_preload_m
            target = center - self.target_rotation @ self.grasp_offset_local
            speed = c.transit_speed_m_s if state == PickPlaceState.TRANSPORT else c.place_speed_m_s
            self._move(data, target, speed)
        elif state == PickPlaceState.RELEASE:
            self.motion.hold()
            self.opening_target_m = c.open_width_m
        elif state == PickPlaceState.RETREAT:
            self._move(data, obs["tcp"] + [0, 0, c.retreat_distance_m], c.approach_speed_m_s)
        elif state == PickPlaceState.SUCCESS:
            self.motion.hold()
        if self.state != PickPlaceState.FAILURE:
            self._record(data, reason)
            self.state_history[-1].update(completed_state=completed_state,
                                          completed_stable_s=completed_stable_s)

    def _stable(self, condition, duration_s, dt):
        """只累计连续满足物理条件的时长，任何一次不满足都清零。"""
        self.stable_s = self.stable_s + dt if condition else 0.0
        return self.stable_s >= duration_s

    def _reached(self, obs):
        """轨迹结束后同时验证实际 TCP 位置、姿态和关节速度。"""
        c = self.config
        return (self.motion.finished
                and np.linalg.norm(obs["tcp"] - self.target_tcp) < c.tcp_position_tolerance_m
                and np.linalg.norm(rotation_error_world(self.target_rotation, obs["rotation"]))
                < c.tcp_orientation_tolerance_rad
                and obs["arm_speed"] < c.arm_velocity_tolerance_rad_s)

    def _placement_position_ok(self, obs):
        """目标为 Cube 中心；同时约束 XY、中心高度及旋转方块最低点。"""
        c = self.config
        return bool(np.linalg.norm(obs["cube"][:2] - self.target_place_position[:2])
                < c.placement_xy_tolerance_m
                and abs(obs["cube"][2] - self.target_place_position[2]) < c.placement_z_tolerance_m
                and abs(obs["clearance"]) < c.placement_z_tolerance_m)

    def _cube_still(self, obs):
        """使用自由关节实际线速度和角速度；不以桌面接触代替静止条件。"""
        return (obs["cube_linear_speed"] < self.config.cube_linear_velocity_tolerance_m_s
                and obs["cube_angular_speed"] < self.config.cube_angular_velocity_tolerance_rad_s)

    def _released_and_placed(self, obs):
        """真实打开、脱离手指、目标内静止并由桌面支撑，才构成释放成功证据。"""
        contact = obs["contact"]
        return (self._placement_position_ok(obs) and self._cube_still(obs)
                and contact.cube_table and not contact.cube_floor and not contact.finger_contact
                and obs["opening"] >= self.config.open_feedback_min_m)

    def _check_slip(self, data, obs, dt):
        """在持物阶段检测持续丢失双侧接触或 TCP 相对偏移异常，不改变物体状态。"""
        offset = obs["rotation"].T @ (obs["cube"] - obs["tcp"])
        lost = (not obs["contact"].grasp_candidate
                or np.linalg.norm(offset - self.grasp_offset_local) > self.config.max_relative_slip_m)
        self.lost_s = self.lost_s + dt if lost else 0.0
        if self.lost_s >= self.config.lost_grasp_grace_s:
            self._fail(data, "OBJECT_SLIPPED", "持物阶段持续丢失双侧夹持或相对位移超限")

    def step(self, data):
        """消费当前反馈并发出下一物理步控制；所有循环均由外部有限回合驱动。"""
        if not self.started:
            raise RuntimeError("必须先调用 reset(data, target_place_position)")
        if self.done:
            return
        control_dt = float(self.model.opt.timestep)
        if self.last_step_time_s is not None and data.time == self.last_step_time_s:
            return
        # 同一时刻的重复调用不会累积稳定时长；首次尚无物理间隔，计时为零。
        dt = (0.0 if self.last_step_time_s is None else
              min(control_dt, float(data.time - self.last_step_time_s)))
        if dt < 0:
            self._fail(data, "CLOCK_RESET", "物理时钟倒退，请重新 reset 智能体")
            return
        self.last_step_time_s = float(data.time)
        obs = self.observe(data)
        c = self.config
        self.max_lift_clearance_m = max(self.max_lift_clearance_m, obs["clearance"])
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            self._fail(data, "NUMERICAL_ERROR", "仿真状态出现非有限数值")
        elif any(w.number for w in data.warning):
            self._fail(data, "MUJOCO_WARNING", "MuJoCo 发出数值或约束警告")
        elif data.time - self.start_time_s >= c.total_timeout_s:
            self._fail(data, "TASK_TIMEOUT", "总仿真时间超限")
        elif obs["contact"].finger_table:
            self._fail(data, "COLLISION", "手指与桌面发生有效碰撞")
        if self.done:
            return
        if self.state in (PickPlaceState.LIFT, PickPlaceState.TRANSPORT):
            self._check_slip(data, obs, dt)
        elif self.state == PickPlaceState.PLACE and not obs["contact"].cube_table:
            self._check_slip(data, obs, dt)
        if self.done:
            return

        state = self.state
        if state == PickPlaceState.IDLE:
            ready = (obs["contact"].cube_table and self._cube_still(obs)
                     and obs["arm_speed"] < c.arm_velocity_tolerance_rad_s
                     and obs["opening"] >= c.open_feedback_min_m)
            if self._stable(ready, c.reach_stable_s, dt):
                self._transition(data, PickPlaceState.PRE_GRASP, "初始 Cube 已稳定受桌面支撑")
        elif state in (PickPlaceState.PRE_GRASP, PickPlaceState.APPROACH):
            if self._stable(self._reached(obs) and obs["opening"] >= c.open_feedback_min_m,
                            c.reach_stable_s, dt):
                self.events["reach_success"] = True
                self.motion.calibrate_tracking_bias(data)
                next_state = (PickPlaceState.APPROACH if state == PickPlaceState.PRE_GRASP
                              else PickPlaceState.CLOSE_GRIPPER)
                self._transition(data, next_state, "实际 TCP 位置、姿态和速度连续达标")
        elif state == PickPlaceState.CLOSE_GRIPPER:
            if self._stable(obs["contact"].grasp_candidate, c.grasp_stable_s, dt):
                self.events["grasp_candidate"] = True
                self._transition(data, PickPlaceState.VERIFY_GRASP, "双侧接触及相向法向力连续成立")
        elif state == PickPlaceState.VERIFY_GRASP:
            if self._stable(obs["contact"].grasp_candidate and self._cube_still(obs),
                            c.grasp_stable_s, dt):
                self._transition(data, PickPlaceState.LIFT, "夹持候选持续稳定，记录实际 TCP 到 Cube 偏移")
        elif state == PickPlaceState.LIFT:
            lifted = (self._reached(obs) and obs["contact"].grasp_candidate
                      and not obs["contact"].cube_table and not obs["contact"].cube_floor
                      and obs["clearance"] >= c.minimum_lift_clearance_m and self._cube_still(obs))
            if self._stable(lifted, c.holding_stable_s, dt):
                self.events["lift_success"] = True
                self.motion.calibrate_tracking_bias(data)
                self._transition(data, PickPlaceState.TRANSPORT, "离桌、双侧夹持及实际静止连续达标")
        elif state == PickPlaceState.TRANSPORT:
            centered = np.linalg.norm(obs["cube"][:2] - self.place_center[:2]) < c.placement_xy_tolerance_m
            transported = (self._reached(obs) and centered and obs["contact"].grasp_candidate
                           and obs["clearance"] >= c.minimum_lift_clearance_m
                           and not obs["contact"].cube_table and self._cube_still(obs))
            if self._stable(transported, c.holding_stable_s, dt):
                self.events["transport_complete"] = True
                self.motion.calibrate_tracking_bias(data)
                self._transition(data, PickPlaceState.PLACE, "Cube 在目标上方稳定持握，按实测偏移生成放置目标")
        elif state == PickPlaceState.PLACE:
            supported = (self._reached(obs) and self._placement_position_ok(obs)
                         and obs["contact"].cube_table and self._cube_still(obs))
            if self._stable(supported, c.placement_stable_s, dt):
                self.motion.calibrate_tracking_bias(data)
                self._transition(data, PickPlaceState.RELEASE, "目标区域内 Cube 已由桌面稳定支撑")
        elif state == PickPlaceState.RELEASE:
            if self._stable(self._released_and_placed(obs), c.placement_stable_s, dt):
                self.events["release_success"] = True
                self._transition(data, PickPlaceState.RETREAT, "实际开度、脱离接触及目标内稳定落桌连续达标")
        elif state == PickPlaceState.RETREAT:
            separated = np.linalg.norm(obs["tcp"] - obs["cube"]) >= c.retreat_separation_m
            final = self._reached(obs) and separated and self._released_and_placed(obs)
            if self._stable(final, c.placement_stable_s, dt):
                self._transition(data, PickPlaceState.SUCCESS, "机械臂已撤离，最终位置、速度、支撑和释放均持续达标")

        if not self.done:
            limit = c.gripper_timeout_s if self.state in (
                PickPlaceState.CLOSE_GRIPPER, PickPlaceState.VERIFY_GRASP) else c.stage_timeout_s
            if data.time - self.enter_time_s >= limit:
                reasons = {PickPlaceState.IDLE: "INITIAL_STATE_FAILED",
                           PickPlaceState.PRE_GRASP: "REACH_TIMEOUT", PickPlaceState.APPROACH: "REACH_TIMEOUT",
                           PickPlaceState.CLOSE_GRIPPER: "GRASP_FAILED", PickPlaceState.VERIFY_GRASP: "GRASP_FAILED",
                           PickPlaceState.LIFT: "LIFT_FAILED", PickPlaceState.TRANSPORT: "TRANSPORT_FAILED",
                           PickPlaceState.PLACE: "PLACE_FAILED", PickPlaceState.RELEASE: "RELEASE_FAILED",
                           PickPlaceState.RETREAT: "PLACE_FAILED"}
                self._fail(data, reasons[self.state], "阶段超时，真实反馈未连续满足转移条件")
        if not self.done:
            self.motion.step(data, control_dt)
            self.gripper.command(data, self.opening_target_m)

    def status(self, data):
        """返回可 JSON 序列化的实时观测，供日志、Viewer 和未来环境层使用。"""
        obs = self.observe(data)
        return {"state": self.state.value, "simulation_time_s": float(data.time - self.start_time_s),
                "state_time_s": float(data.time - self.enter_time_s),
                "current_tcp": obs["tcp"].tolist(), "target_tcp": self.target_tcp.tolist(),
                "cube_position": obs["cube"].tolist(), "gripper_opening_m": obs["opening"],
                "finger_contact": obs["contact"].finger_contact,
                "grasp_candidate": obs["contact"].grasp_candidate, "cube_table": obs["contact"].cube_table,
                "left_finger_force_N": obs["contact"].left_force_N,
                "right_finger_force_N": obs["contact"].right_force_N,
                "tcp_position_error_m": float(np.linalg.norm(obs["tcp"] - self.target_tcp)),
                "tcp_orientation_error_rad": float(np.linalg.norm(
                    rotation_error_world(self.target_rotation, obs["rotation"]))),
                "cube_linear_speed_m_s": obs["cube_linear_speed"],
                "cube_angular_speed_rad_s": obs["cube_angular_speed"],
                "lift_height_m": obs["clearance"], "stable_time_s": self.stable_s,
                "reference_finished": self.motion.finished, "failure_reason": self.failure_reason}

    def abort(self, data, reason="CANCELLED", detail="外部驱动器请求停止"):
        """允许 Viewer 关闭或外部驱动器到达上限时明确终止，而非无限等待。"""
        self._fail(data, reason, detail)

    def result(self, data):
        """返回完整任务结果；最终成功必须仍满足实际放置与释放条件。"""
        obs = self.observe(data)
        error = obs["cube"] - self.target_place_position
        valid = self._released_and_placed(obs)
        success = self.state == PickPlaceState.SUCCESS and valid
        reason = self.failure_reason
        if self.state == PickPlaceState.SUCCESS and not valid:
            reason = "FINAL_STATE_INVALID"
        return {"success": bool(success), "failure_reason": reason,
                "failure_detail": self.failure_detail, "final_state": self.state.value,
                "failure_stage": self.failure_stage,
                "total_simulation_time": float(data.time - self.start_time_s), "seed": self.seed,
                "state_history": self.state_history,
                "final_cube_position": obs["cube"].tolist(),
                "target_place_position": self.target_place_position.tolist(),
                "final_placement_error": float(np.linalg.norm(error)),
                "final_placement_xy_error_m": float(np.linalg.norm(error[:2])),
                "final_cube_linear_speed_m_s": obs["cube_linear_speed"],
                "final_cube_angular_speed_rad_s": obs["cube_angular_speed"],
                "placement_position_success": self._placement_position_ok(obs),
                "full_task_success": bool(success), "max_lift_clearance_m": self.max_lift_clearance_m,
                "grasp_offset_local_m": (None if self.grasp_offset_local is None else
                                          self.grasp_offset_local.tolist()),
                "configuration": asdict(self.config), "physics_timestep_s": float(self.model.opt.timestep),
                "mujoco_warning_count": int(sum(w.number for w in data.warning)),
                "final_status": self.status(data),
                "ik_calls": self.motion.ik_calls, **self.events}
