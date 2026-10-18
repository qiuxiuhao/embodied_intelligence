"""Day 11 固定场景物理实验，用于验证基础抓取、抬升与释放。

运动仅由机械臂位置执行器和夹爪腱传动驱动。自由 Cube 只通过 XML 关键帧初始化一次，
之后完全按动力学演化。本模块不实现通用抓取与放置智能体。
距离单位 m，角度 rad，关节角速度 rad/s，力 N，时间 s。
"""

from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np

from controllers.gripper_controller import PandaGripperController
from controllers.ik_controller import PandaIKSolver, rotation_error_world
from controllers.joint_controller import PandaJointPositionController
from robotics.contacts import PandaContactDetector
from robotics.kinematics import get_arm_qpos, get_arm_qvel, get_tcp_pose


SCENE_XML = Path(__file__).resolve().parents[1] / "assets/franka_panda/scene_cube.xml"
# TCP 局部 +z 轴朝下；局部 +y 轴（闭合轴）沿世界 +x 方向。
TOP_DOWN_ROTATION = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
LIFT_CLEARANCE_M = 0.04
LIFT_HOLD_S = 0.50
RELEASE_HOLD_S = 0.20


def load_cube_scene():
    """加载包含官方 Panda 的独立场景，并用完整的场景关键帧初始化。"""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("day11_home").id)
    mujoco.mj_forward(model, data)
    return model, data


def cube_bottom_clearance(model, data):
    """精确计算旋转方块最低点相对于水平台面的高度，单位 m。

    使用旋转后方块在竖直方向的支撑范围，而非中心高度减去固定半边长；
    释放后的 Cube 可能旋转，此时某个角点会低于后一种近似结果。
    """
    cube_id = int(model.geom("cube_geom").id)
    table_id = int(model.geom("table").id)
    rotation = data.geom_xmat[cube_id].reshape(3, 3)
    extent_z = np.abs(rotation[2]) @ model.geom_size[cube_id]
    table_top = data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2]
    return float(data.geom_xpos[cube_id, 2] - extent_z - table_top)


class GraspMonitor:
    """结合持续时间判定物理状态；仅有接触不能说明抬升成功。"""

    def __init__(self):
        """初始化单次实验的计时器及事件记录；时间单位 s，高度单位 m。"""
        self.finger_contact = False
        self.grasp_candidate = False
        self.lift_success = False
        self.release = False
        self.lift_time_s = 0.0
        self.max_lift_time_s = 0.0
        self.release_time_s = 0.0
        self.lifted_center_z_m = None

    def update(self, contact, clearance_m, center_z_m, opening_m, releasing, dt):
        """累计连续满足条件的时长；条件丢失时立即清零对应计时器。

        抬升要求双指相向受力、无支撑接触，且底部离桌至少 0.04 m，连续保持 0.5 s。
        释放要求此前已确认抬升；发出打开命令后，实测开度至少 0.07 m、无手指接触，
        且 Cube 下落至少 0.05 m，上述释放条件连续保持 0.2 s。
        这些是保守的实验阈值，不构成完整的力闭合证明。
        """
        self.finger_contact |= contact.finger_contact
        self.grasp_candidate |= contact.grasp_candidate
        airborne_grasp = (contact.grasp_candidate and not contact.cube_table
                          and not contact.cube_floor and clearance_m >= LIFT_CLEARANCE_M)
        self.lift_time_s = self.lift_time_s + dt if airborne_grasp else 0.0
        self.max_lift_time_s = max(self.max_lift_time_s, self.lift_time_s)
        if self.lift_time_s >= LIFT_HOLD_S:
            self.lift_success = True
            self.lifted_center_z_m = center_z_m
        released = (releasing and self.lift_success and opening_m >= 0.07
                    and not contact.finger_contact
                    and self.lifted_center_z_m - center_z_m >= 0.05)
        self.release_time_s = self.release_time_s + dt if released else 0.0
        self.release |= self.release_time_s >= RELEASE_HOLD_S


def plan_experiment(model, data):
    """在独立 MjData 中复用现有逆运动学，依次以上一个解为初值求解三个朝下位姿。

    沿用的 TCP 位于 hand 局部 z=0.10 m；主指垫中心位于
    0.0584+0.0445=0.1029 m。因此 TCP 对准 Cube 中心时，指垫接近方块中部，
    指尖仍高于 z=0.20 m 的桌面，无需重新定义 TCP。
    """
    solver = PandaIKSolver(model, tolerance=0.0005, max_iterations=400)
    q = get_arm_qpos(data)
    solutions = {}
    for name, position in (("hover", [0.50, 0., 0.35]),
                           ("grasp", [0.50, 0., 0.225]),
                           ("lift", [0.50, 0., 0.345])):
        result = solver.solve(position, initial_q=q, target_rotation=TOP_DOWN_ROTATION,
                              orientation_tolerance=0.005)
        if not result["success"]:
            raise RuntimeError(f"{name} pose IK failed: {result}")
        solutions[name] = result
        q = result["q"]
    return solutions


def run_experiment(model, data, viewer=None, verbose=True):
    """仅通过执行器完成一次有时限的抓取、抬升、释放实验，返回国际单位制指标。

    固定阶段时长便于复现；若实际接近位姿或夹持条件不满足，则终止后续阶段。
    无界面模式与 Viewer 模式使用完全相同的物理仿真和验收逻辑。
    """
    plan = plan_experiment(model, data)
    arm = PandaJointPositionController(model, max_joint_speed=0.35)
    gripper = PandaGripperController(model)
    detector = PandaContactDetector(model)
    monitor = GraspMonitor()
    arm.reset(data)
    home_q = get_arm_qpos(data)
    phases = (("settle", 1.0, home_q, 0.08, None),
              ("approach", 4.0, plan["hover"]["q"], 0.08, "hover"),
              ("descend", 3.0, plan["grasp"]["q"], 0.08, "grasp"),
              ("close", 2.0, plan["grasp"]["q"], 0.0, "grasp"),
              ("lift", 3.0, plan["lift"]["q"], 0.0, "lift"),
              ("hold", 1.0, plan["lift"]["q"], 0.0, "lift"),
              ("release", 2.5, plan["lift"]["q"], 0.08, "lift"))
    dt = float(model.opt.timestep)
    summaries = {}
    max_clearance = cube_bottom_clearance(model, data)
    finger_table_seen = False
    single_finger_seen = False
    contact_pairs = set()
    failed_reason = None
    release_start_z = None
    min_release_vz = 0.0
    landed_time = 0.0
    steps = 0
    for phase, duration, target_q, opening, pose_name in phases:
        start_command = arm.command_q.copy()
        if phase == "release":
            release_start_z = float(data.body("cube").xpos[2])
        if verbose:
            print(f"phase={phase} time={data.time:.3f}s opening_target={opening:.3f}m")
        for phase_step in range(round(duration / dt)):
            wall_start = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                failed_reason = "viewer_closed"
                break
            with viewer.lock() if viewer is not None else nullcontext():
                # 使用同步平滑关节插值，避免下降时腕部先于肘部到达目标。
                # 各关节独立限速时，即使起点和终点都安全，
                # 中间过程仍可能使指尖短暂压入桌面。
                u = min(1.0, (phase_step + 1) * dt / max(dt, duration - 0.5))
                blend = u * u * (3.0 - 2.0 * u)
                command = start_command + blend * (target_q - start_command)
                arm.step(data, command, dt)
                gripper.command(data, opening)
                mujoco.mj_step(model, data)
                # mj_step 在计算接触后才积分更新 qpos，因此需刷新完整计算流程，
                # 使位姿、接触坐标系和约束力都对应同一个更新后的状态；
                # 仅调用 mj_kinematics 会使接触力仍停留在上一状态。
                mujoco.mj_forward(model, data)
                contact = detector.read(data)
                feedback = gripper.feedback(data)
                clearance = cube_bottom_clearance(model, data)
                z = float(data.body("cube").xpos[2])
                monitor.update(contact, clearance, z, feedback["opening_m"],
                               phase == "release", dt)
                max_clearance = max(max_clearance, clearance)
                finger_table_seen |= contact.finger_table
                single_finger_seen |= (contact.finger_contact and not contact.grasp_candidate
                                       and (contact.left_force_N == 0 or contact.right_force_N == 0))
                contact_pairs.update(tuple(sorted((r.geom1, r.geom2)))
                                     for r in contact.cube_contacts)
                if phase == "release":
                    cube_velocity = data.joint("cube_free").qvel
                    vz = float(cube_velocity[2])
                    min_release_vz = min(min_release_vz, vz)
                    landed = (contact.cube_table and not contact.finger_contact
                              and np.linalg.norm(cube_velocity[:3]) < 0.05
                              and np.linalg.norm(cube_velocity[3:]) < 0.2)
                    landed_time = landed_time + dt if landed else 0.0
                steps += 1
            if viewer is not None:
                if steps % 10 == 0:
                    viewer.sync()
                remaining = dt - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
        if failed_reason:
            break
        tcp, _, rotation = get_tcp_pose(data)
        target = plan[pose_name]["target_position"] if pose_name else tcp
        position_error = float(np.linalg.norm(target - tcp))
        angular_error = float(np.linalg.norm(rotation_error_world(TOP_DOWN_ROTATION, rotation)))
        summaries[phase] = {
            "time_s": float(data.time), "cube_center_z_m": z,
            "bottom_clearance_m": clearance,
            "finger_position_m": feedback["finger_position_m"].tolist(),
            "opening_m": feedback["opening_m"],
            "left_force_N": contact.left_force_N, "right_force_N": contact.right_force_N,
            "cube_table": contact.cube_table, "finger_contact": contact.finger_contact,
            "grasp_candidate": contact.grasp_candidate,
            "tcp_error_m": position_error, "orientation_error_rad": angular_error,
            "max_arm_speed_rad_s": float(np.max(np.abs(get_arm_qvel(data))))}
        if verbose:
            print(summaries[phase])
        if (phase in ("approach", "descend") and
                (position_error > 0.01 or angular_error > 0.04
                 or np.max(np.abs(get_arm_qvel(data))) > 0.10)):
            failed_reason = f"{phase}_physical_pose_not_reached"
        if phase == "close" and not contact.grasp_candidate:
            failed_reason = "no_bilateral_grasp_at_close"
        if phase == "hold" and monitor.lift_time_s < LIFT_HOLD_S:
            failed_reason = "lift_not_sustained"
        if finger_table_seen:
            failed_reason = "finger_table_collision"
        if failed_reason:
            break
    warnings = int(sum(w.number for w in data.warning))
    if failed_reason is None:
        if warnings:
            failed_reason = "mujoco_warning"
        elif not monitor.release:
            failed_reason = "release_not_confirmed"
        elif landed_time < 0.3:
            failed_reason = "cube_not_stably_landed"
    success = (monitor.lift_success and monitor.release and landed_time >= 0.3
               and not finger_table_seen and warnings == 0 and failed_reason is None)
    return {"success": bool(success), "failure_reason": failed_reason,
            "finger_contact": monitor.finger_contact, "grasp_candidate": monitor.grasp_candidate,
            "lift_success": monitor.lift_success, "release": monitor.release,
            "max_lift_hold_s": monitor.max_lift_time_s,
            "max_bottom_clearance_m": float(max_clearance),
            "release_drop_m": (None if release_start_z is None else
                               release_start_z - float(data.body("cube").xpos[2])),
            "min_release_velocity_m_s": min_release_vz,
            "landed_stable_s": landed_time, "finger_table_collision": finger_table_seen,
            "single_finger_contact_seen": single_finger_seen,
            "cube_contact_geom_pairs": sorted(contact_pairs), "warning_count": warnings,
            "simulation_time_s": float(data.time), "phases": summaries,
            "ik": {n: {k: r[k] for k in ("position_error", "orientation_error", "iterations")}
                   for n, r in plan.items()}}
