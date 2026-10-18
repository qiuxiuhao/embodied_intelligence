"""Day 12 真实物理验收及失败注入；不写 Cube 的动态 qpos 或 qvel。

运行：PYTHONPATH=. python scripts/37_test_pick_place.py --report 新结果路径.json
故障只作用于夹爪执行器、真实接触摩擦参数或已注明单位的物理外力。
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

from agents.scripted_pick_place import PickPlaceConfig, PickPlaceState, ScriptedPickPlaceAgent
from controllers.pose_motion import MotionPlanningError
from robotics.grasp_experiment import SCENE_XML, load_cube_scene
from robotics.pick_place_runtime import run_pick_place


EPISODES = []


def trial(name, target=(0.50, 0.12, 0.225), config=None, fault=None, seed=17,
          scene_factory=load_cube_scene):
    """每次重新加载物理模型，记录完整终态和转移证据以便复现。"""
    model, data = scene_factory()
    agent = ScriptedPickPlaceAgent(model, config)
    agent.reset(data, target, seed=seed)
    result = run_pick_place(agent, data, verbose=False, before_physics=fault)
    EPISODES.append({"case": name, "result": result})
    print(f"\n{name}: success={result['success']} failure={result['failure_reason']} "
          f"stage={result['failure_stage']} t={result['total_simulation_time']:.3f}s "
          f"error={result['final_placement_error']:.6f}m", flush=True)
    return agent, data, result


class PickPlaceTests(unittest.TestCase):
    """同时测试真实 MuJoCo 动力学和控制接口的失败原子性。"""

    def test_normal_and_seed_reproduction(self):
        """完整正常任务重复两次；相同初始条件和 seed 应给出一致状态历史。"""
        first_agent, data, first = trial("normal_seed17_first")
        # 重置整个物理回合后复用同一个 Agent，验证没有遗留计时器、轨迹或统计量。
        model = first_agent.model
        mujoco.mj_resetDataKeyframe(model, data, model.key("day11_home").id)
        mujoco.mj_forward(model, data)
        first_agent.reset(data, [.5, .12, .225], seed=17)
        second = run_pick_place(first_agent, data, verbose=False)
        EPISODES.append({"case": "normal_seed17_repeat_same_agent", "result": second})
        print(f"\nnormal_seed17_repeat_same_agent: success={second['success']} "
              f"error={second['final_placement_error']:.6f}m", flush=True)
        expected = [state.value for state in PickPlaceState if state != PickPlaceState.FAILURE]
        self.assertEqual([e["state"] for e in first["state_history"]], expected)
        for result in (first, second):
            self.assertTrue(result["success"], result)
            for flag in ("reach_success", "grasp_candidate", "lift_success", "transport_complete",
                         "placement_position_success", "release_success", "full_task_success"):
                self.assertTrue(result[flag], flag)
            self.assertLess(result["final_placement_error"], 0.004)
            self.assertGreater(result["max_lift_clearance_m"], 0.10)
            self.assertGreaterEqual(result["state_history"][-1]["completed_stable_s"], 0.50)
        self.assertEqual(first, second)
        obs = first_agent.observe(data)
        self.assertTrue(obs["contact"].cube_table)
        self.assertFalse(obs["contact"].finger_contact)
        self.assertGreater(obs["opening"], 0.07)
        self.assertGreater(np.linalg.norm(obs["tcp"] - obs["cube"]), 0.08)

    def test_another_place_target(self):
        """用不同的世界坐标放置目标，排除固定终点轨迹冒充任务接口。"""
        _, _, result = trial("alternate_target", target=(0.56, -0.08, 0.225))
        self.assertTrue(result["success"], result)
        self.assertLess(result["final_placement_error"], 0.004)

    def test_changed_initial_cube_pose(self):
        """在编译前定义不同初始位置及 20° 偏航，验证真值输入和朝向适配。"""
        def initial_scene():
            # 仅定义场景初始条件，再由 MuJoCo 关键帧初始化；任务中不改物体状态。
            spec = mujoco.MjSpec.from_file(str(SCENE_XML))
            position = [.48, -.04, .225]
            yaw = np.deg2rad(20.)
            quaternion = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
            spec.body("cube").pos = position
            spec.body("cube").quat = quaternion
            key = spec.key("day11_home")
            initial_configuration = np.asarray(key.qpos).copy()
            initial_configuration[9:12] = position
            initial_configuration[12:16] = quaternion
            key.qpos = initial_configuration
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("day11_home").id)
            mujoco.mj_forward(model, data)
            return model, data

        _, _, result = trial("changed_cube_xyz_yaw20deg", target=(.52, .10, .225),
                             scene_factory=initial_scene)
        self.assertTrue(result["success"], result)
        self.assertLess(result["final_placement_error"], .004)
        EPISODES[-1]["initial_cube_pose"] = {"position_m": [.48, -.04, .225], "yaw_rad": float(np.deg2rad(20))}

    def test_initial_grasp_failure(self):
        """夹爪执行器卡在全开，不能仅因已经发出闭合命令就通过抓取。"""
        def stuck_open(agent, data):
            # 只覆盖执行器 ctrl，手指和 Cube 均继续真实物理运动。
            if agent.state in (PickPlaceState.CLOSE_GRIPPER, PickPlaceState.VERIFY_GRASP):
                agent.gripper.command(data, 0.08)

        _, _, result = trial("gripper_stuck_open", fault=stuck_open)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "GRASP_FAILED")
        self.assertEqual(result["failure_stage"], "CLOSE_GRIPPER")
        self.assertFalse(result["lift_success"])

    def test_actual_ik_failure_blocks_commands(self):
        """真实 IK 不可达时进入 FAILURE；整段预检不向执行器发送无效路点。"""
        config = replace(PickPlaceConfig(), hover_clearance_m=3.0)
        _, _, result = trial("unreachable_pregrasp", config=config)
        self.assertEqual(result["failure_reason"], "IK_FAILED")
        self.assertEqual(result["failure_stage"], "PRE_GRASP")
        self.assertFalse(result["reach_success"])
        model, data = load_cube_scene()
        agent = ScriptedPickPlaceAgent(model)
        agent.reset(data, [.5, .12, .225])
        before_ctrl = data.ctrl.copy()
        before_command = agent.motion.arm.command_q.copy()
        with self.assertRaises(MotionPlanningError):
            agent.motion.plan(data, [3., 3., 3.], np.diag([1., -1., -1.]), .06)
        np.testing.assert_array_equal(data.ctrl, before_ctrl)
        np.testing.assert_array_equal(agent.motion.arm.command_q, before_command)
        self.assertEqual(agent.motion.segments, [])

    def test_friction_loss_after_lift(self):
        """已稳定抬升后降低接触摩擦，靠重力真实滑落，必须识别 OBJECT_SLIPPED。"""
        evidence = {}

        def friction_loss(agent, data):
            if agent.state == PickPlaceState.TRANSPORT and not evidence:
                ids = [int(agent.model.geom("cube_geom").id)]
                ids.extend(np.flatnonzero(np.isin(agent.model.geom_bodyid,
                                                  [agent.detector.left, agent.detector.right])).tolist())
                # 动态材质故障：接触双方滑动摩擦降到 0.0001，无量纲。
                agent.model.geom_friction[ids, 0] = 0.0001
                evidence.update(time_s=float(data.time), clearance_m=agent.observe(data)["clearance"],
                                closed_ctrl=float(data.ctrl[agent.gripper.actuator_id]))

        _, _, result = trial("friction_loss_after_verified_lift", fault=friction_loss)
        self.assertTrue(result["lift_success"])
        self.assertGreater(evidence["clearance_m"], 0.04)
        self.assertAlmostEqual(evidence["closed_ctrl"], 0.)
        self.assertEqual(result["failure_reason"], "OBJECT_SLIPPED")
        self.assertEqual(result["failure_stage"], "TRANSPORT")
        self.assertFalse(result["success"])
        EPISODES[-1]["fault_evidence"] = evidence

    def test_final_placement_rejected_after_disturbance(self):
        """释放已确认后施加短时物理外力；撤离末端目标外的 Cube 不能判成功。"""
        evidence = {}

        def push_after_release(agent, data):
            body_id = int(agent.model.body("cube").id)
            data.xfrc_applied[body_id] = 0
            if agent.state == PickPlaceState.RETREAT:
                age = float(data.time - agent.enter_time_s)
                # 等工具上升离开物体，再沿世界 +x 施加 2 N，持续 0.06 s。
                if 2.5 <= age < 2.56:
                    data.xfrc_applied[body_id, 0] = 2.0
                    evidence.update(force_N=2.0, duration_s=0.06, release_already_verified=True)

        _, _, result = trial("post_release_physical_push", fault=push_after_release)
        self.assertTrue(evidence)
        self.assertTrue(result["release_success"])
        self.assertFalse(result["placement_position_success"])
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_reason"], "PLACE_FAILED")
        self.assertEqual(result["failure_stage"], "RETREAT")
        EPISODES[-1]["fault_evidence"] = evidence

    def test_release_failure(self):
        """放置后夹爪执行器仍保持闭合，不能因 Cube 接触桌面而认为已释放。"""
        def stuck_closed(agent, data):
            if agent.state == PickPlaceState.RELEASE:
                agent.gripper.command(data, 0.)

        _, _, result = trial("release_stuck_closed", fault=stuck_closed)
        self.assertTrue(result["transport_complete"])
        self.assertFalse(result["release_success"])
        self.assertEqual(result["failure_reason"], "RELEASE_FAILED")

    def test_timeouts_and_invalid_target(self):
        """验证总超时、到达超时及桌面外目标都返回有限且明确的失败结果。"""
        _, _, total = trial("total_timeout", config=replace(PickPlaceConfig(), total_timeout_s=.10))
        self.assertEqual(total["failure_reason"], "TASK_TIMEOUT")
        _, _, reach = trial("reach_timeout", config=replace(PickPlaceConfig(), stage_timeout_s=.30))
        self.assertEqual(reach["failure_reason"], "REACH_TIMEOUT")
        _, _, invalid = trial("invalid_place_height", target=(.5, .12, .4))
        self.assertEqual(invalid["failure_reason"], "PLACE_FAILED")
        self.assertEqual(invalid["total_simulation_time"], 0.)

    def test_same_time_step_is_idempotent(self):
        """不推进物理时重复调用 step，不能累计稳定计时或推进参考轨迹。"""
        model, data = load_cube_scene()
        agent = ScriptedPickPlaceAgent(model)
        agent.reset(data, [.5, .12, .225])
        agent.step(data)
        before = data.ctrl.copy()
        for _ in range(1000):
            agent.step(data)
        self.assertEqual(agent.stable_s, 0.)
        self.assertEqual(agent.state, PickPlaceState.IDLE)
        np.testing.assert_array_equal(data.ctrl, before)


def main():
    """运行可选筛选的 unittest，并以新文件保存真实成功与失败任务的证据。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path)
    args, remaining = parser.parse_known_args()
    if args.report is not None and args.report.exists():
        parser.error("报告路径已存在，请更换文件名")
    program = unittest.main(argv=[sys.argv[0], *remaining], verbosity=2, exit=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = {"passed": program.result.wasSuccessful(), "test_count": program.result.testsRun,
                  "failures": len(program.result.failures), "errors": len(program.result.errors),
                  "episodes": EPISODES}
        with args.report.open("x") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    raise SystemExit(0 if program.result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
