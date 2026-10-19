"""Day 13 环境验收；动作驱动物理、随机性、任务条件和标准 API 均实际执行。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import unittest
import warnings

from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from envs.pick_place_env import (OBSERVATION_FIELDS, OBSERVATION_SIZE, OBSERVATION_SLICES,
                                PickPlaceEnv, PickPlaceEnvConfig, TASK_MODES)


EVIDENCE = []
FIXED = {"cube_position": [.5, 0., .225], "cube_yaw": 0.}


def assert_transition(test, env, transition):
    """检查每次真实 step 的空间、类型、有限奖励、分项总和及 JSON 序列化。"""
    observation, reward, terminated, truncated, info = transition
    test.assertEqual(observation.dtype, np.float32)
    test.assertTrue(env.observation_space.contains(observation))
    test.assertTrue(np.all(np.isfinite(observation)))
    test.assertIsInstance(reward, float)
    test.assertIsInstance(terminated, bool)
    test.assertIsInstance(truncated, bool)
    test.assertFalse(terminated and truncated)
    test.assertTrue(np.isfinite(reward))
    test.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
    json.dumps(info, allow_nan=False)


def physical_action_probe(test, task):
    """独立测试用真值路点策略，只调用四维 env.step，不写底层 ctrl 或物体状态。

    这不是 Day 12 Agent，也不是训练策略；用外部有限探针证明四维接口可以产生
    真实抓取、抬升与放置。所有阶段反馈来自当前物理状态，不属于环境内部决策。
    """
    env = PickPlaceEnv(task_mode=task)
    options = {**FIXED, "start_pose": "pre_grasp"}
    if task == "pick_place":
        options["target_position"] = [.5, .12, .225]
    env.reset(seed=4, options=options)
    phase, total, supported_while_closed = 0, 0., False
    for index in range(400):
        state = env._state()
        if phase == 0:
            goal, grip = np.array([.5, 0., .231]), 1.
            if (np.linalg.norm(env.tcp_command-goal) < .001
                    and np.linalg.norm(state["tcp"]-goal) < .009 and state["arm_speed"] < .1):
                phase = 1
        elif phase == 1:
            goal, grip = np.array([.5, 0., .231]), -1.
            if env.timers[1] >= env.config.grasp_hold_s:
                phase = 2
        elif phase == 2:
            goal, grip = np.array([.5, 0., .35]), -1.
            if env.has_lifted:
                phase = 3
        elif phase == 3:
            goal, grip = np.array([.5, .12, .35]), -1.
            if np.linalg.norm(state["tcp"]-goal) < .009 and state["arm_speed"] < .1:
                phase = 4
        elif phase == 4:
            goal, grip = np.array([.5, .12, .235]), -1.
            if (state["contact"].cube_table and np.linalg.norm(state["tcp"]-goal) < .009
                    and state["arm_speed"] < .1):
                supported_while_closed = True
                # 已抬升且目标处受支撑，但尚夹持，必须仍未满足放置成功。
                test.assertFalse(env._physical_conditions(state)[3])
                phase = 5
        elif phase == 5:
            goal, grip = np.array([.5, .12, .235]), 1.
            if state["opening"] >= .07 and not state["contact"].finger_contact:
                phase = 6
        else:
            goal, grip = np.array([.5, .12, .35]), 1.
        action = np.r_[np.clip((goal-env.tcp_command)/env.config.action_scale_m, -1, 1), grip].astype(np.float32)
        transition = env.step(action)
        assert_transition(test, env, transition)
        _, reward, terminated, truncated, info = transition
        total += reward
        if terminated or truncated:
            break
    else:
        test.fail(f"{task} 真实四维动作探针未在有限步数内结束")
    test.assertTrue(info["is_success"], info)
    test.assertTrue(terminated)
    test.assertFalse(truncated)
    test.assertGreater(info["reward_terms"]["success_bonus"], 0)
    if task == "pick_place":
        test.assertTrue(supported_while_closed)
        test.assertTrue(info["has_lifted"])
        test.assertFalse(info["grasp_candidate"])
        test.assertTrue(info["cube_table_contact"])
        test.assertLess(info["cube_position_error"], env.config.placement_xy_tolerance_m)
        test.assertGreaterEqual(info["gripper_opening"], .07)
    evidence = {"case": f"four_dimensional_{task}", "steps": index+1, "return": total,
                "info": info, "action_interface": "仅 env.step([dx,dy,dz,gripper])"}
    EVIDENCE.append(evidence)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)
    env.reset(seed=4, options=options)
    test.assertFalse(env.has_lifted)
    np.testing.assert_array_equal(env.timers, 0)
    test.assertEqual(env.data.time, 0.)
    env.close()


class PickPlaceEnvTests(unittest.TestCase):
    """不训练 PPO；通过 Gymnasium/SB3 检查器与真实 MuJoCo 动力学验收。"""

    def test_api_and_spaces(self):
        """四任务使用统一 Box、正确返回类型，并通过两个独立检查器。"""
        for task in TASK_MODES:
            with self.subTest(task=task):
                env = PickPlaceEnv(task_mode=task)
                with warnings.catch_warnings(record=True) as captured:
                    check_env(env, skip_render_check=True)
                EVIDENCE.append({"case": f"gym_checker_{task}", "warnings": [str(w.message) for w in captured]})
                sb3_check_env(env, warn=True)
                observation, info = env.reset(seed=1)
                self.assertEqual(observation.shape, (94,))
                self.assertEqual(OBSERVATION_SIZE, 94)
                self.assertEqual(env.action_space.shape, (4,))
                self.assertEqual(env.action_space.dtype, np.float32)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(len(info["stable_times_s"]), 4)
                self.assertIsNone(env.viewer)
                env.close()

    def test_sb3_monitor_and_vector_autoreset(self):
        """验证后续 PPO 常用包装器的数据通路，不创建策略也不执行训练。"""
        config = replace(PickPlaceEnvConfig(), max_episode_steps=2)
        vector = DummyVecEnv([lambda: Monitor(PickPlaceEnv(config=config)),
                              lambda: Monitor(PickPlaceEnv(config=config))])
        vector.seed(42)
        observation = vector.reset()
        self.assertEqual(observation.shape, (2, OBSERVATION_SIZE))
        action = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
        vector.step(action)
        observation, reward, done, infos = vector.step(action)
        self.assertTrue(np.all(done))
        for info in infos:
            self.assertTrue(info["TimeLimit.truncated"])
            self.assertEqual(info["terminal_observation"].shape, (OBSERVATION_SIZE,))
            self.assertIn("episode", info)
        self.assertTrue(all(env.unwrapped.data.time == 0 for env in vector.envs))
        vector.close()

    def test_reset_seed_randomization_and_isolation(self):
        """相同 seed 的动作轨迹复现；不同 seed 真正改变初始状态，实例互不串扰。"""
        first, second = PickPlaceEnv(), PickPlaceEnv()
        a, ai = first.reset(seed=9)
        b, bi = second.reset(seed=9)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ai, bi)
        other, _ = second.reset(seed=10)
        self.assertFalse(np.array_equal(a, other))
        self.assertFalse(np.allclose(first.data.joint("cube_free").qpos, second.data.joint("cube_free").qpos))
        self.assertIsNot(first.model, second.model)
        self.assertIsNot(first.data, second.data)
        second.reset(seed=9)
        actions = np.random.default_rng(100).uniform(-.3, .3, (25, 4)).astype(np.float32)
        for action in actions:
            x, y = first.step(action), second.step(action)
            np.testing.assert_array_equal(x[0], y[0])
            self.assertEqual(x[1:], y[1:])
            assert_transition(self, first, x)
        before = second.data.qpos.copy()
        first.step(np.array([1, 0, 0, 1], np.float32))
        np.testing.assert_array_equal(second.data.qpos, before)
        for seed in range(20):
            observation, info = first.reset(seed=seed, options={"task_mode": "pick_place"})
            self.assertGreaterEqual(info["cube_bottom_clearance"], -1e-9)
            self.assertTrue(first._on_table(np.array(info["cube_position_m"])))
            self.assertTrue(first._on_table(first.target_position))
        first.close()
        second.close()

    def test_clean_reset_options_and_close(self):
        """完整 keyframe 重置所有物理与控制记忆；只在 reset 使用指定初始位姿。"""
        env = PickPlaceEnv()
        options = {**FIXED, "cube_yaw": .15, "task_mode": "lift", "start_pose": "pre_grasp"}
        initial, _ = env.reset(seed=3, options=options)
        for _ in range(5):
            env.step([0, 0, .2, -1])
        reset, info = env.reset(seed=3, options=options)
        np.testing.assert_array_equal(initial, reset)
        self.assertEqual(env.data.time, 0.)
        self.assertEqual(env.episode_step, 0)
        self.assertEqual(env.episode_return, 0)
        self.assertEqual(env.ik_failures, 0)
        self.assertFalse(env.has_lifted)
        np.testing.assert_array_equal(env.data.qvel, np.zeros(env.model.nv))
        np.testing.assert_array_equal(env.data.xfrc_applied, 0)
        self.assertEqual(info["stable_times_s"], [0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            env.reset(options={"cube_position": [.5, 0, .20]})
        with self.assertRaises(ValueError):
            env.reset(options={"task_mode": "pick_place", "target_position": [.5, .12, .4]})
        with self.assertRaises(ValueError):
            env.reset(options={"task_mode": "grasp", "target_position": [.6, .12, .225]})
        env.reset(seed=3)
        env.close()
        env.close()
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4, np.float32))
        env.reset(seed=3)
        self.assertTrue(env.observation_space.contains(env.step([0, 0, 0, 1])[0]))
        env.close()

    def test_increment_semantics_movement_and_frame_skip(self):
        """增量从上一合法命令累计，零位移保持；真实 TCP 会移动，时间精确对应子步。"""
        env = PickPlaceEnv(config=replace(PickPlaceEnvConfig(), frame_skip=10))
        env.reset(seed=2, options=FIXED)
        original = env.tcp_command.copy()
        tcp = env._state()["tcp"].copy()
        first = env.step([1, 0, 0, 1])
        np.testing.assert_allclose(env.tcp_command, original + [.01, 0, 0], atol=1e-10)
        env.step([1, 0, 0, 1])
        np.testing.assert_allclose(env.tcp_command, original + [.02, 0, 0], atol=1e-10)
        for _ in range(30):
            transition = env.step([0, 0, 0, 1])
            assert_transition(self, env, transition)
        np.testing.assert_allclose(env.tcp_command, original + [.02, 0, 0], atol=1e-10)
        self.assertGreater(env._state()["tcp"][0]-tcp[0], .01)
        self.assertAlmostEqual(env.data.time, 32*10*.002)
        self.assertEqual(first[4]["physics_steps"], 10)
        env.close()

    def test_gripper_feedback_and_action_validation(self):
        """全闭、全开和零动作半开均需真实手指反馈；坏格式不推进时间。"""
        env = PickPlaceEnv()
        env.reset(seed=2, options=FIXED)
        for grip, expected in ((-1., 0.), (1., .08), (0., .04)):
            for _ in range(50):
                transition = env.step([0, 0, 0, grip])
            self.assertAlmostEqual(transition[4]["gripper_opening"], expected, delta=.0002)
            np.testing.assert_allclose(env.gripper.feedback(env.data)["finger_position_m"], expected/2, atol=.0002)
        now = env.data.time
        for action in ([1, 2], [[0, 0, 0, 0]], [np.nan, 0, 0, 0], [np.inf, 0, 0, 0]):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.data.time, now)
        clipped = env.step([2., -2., 0., 3.])
        self.assertTrue(clipped[4]["action_clipped"])
        self.assertEqual(clipped[4]["applied_action"], [1., -1., 0., 1.])
        self.assertLess(clipped[4]["reward_terms"]["clipped_action_cost"], 0)
        assert_transition(self, env, clipped)
        env.close()

    def test_reach_success_reward_direction_and_end_guard(self):
        """相同物理初态下向目标移动优于远离；成功需实际位姿和稳定时长。"""
        toward, away = PickPlaceEnv(), PickPlaceEnv()
        for env in (toward, away):
            env.reset(seed=1, options={**FIXED, "target_position": [.60, 0., .52]})
        good, bad = toward.step([1, 0, 0, 1]), away.step([-1, 0, 0, 1])
        self.assertLess(good[4]["tcp_position_error"], bad[4]["tcp_position_error"])
        self.assertGreater(good[1], bad[1])
        EVIDENCE.append({"case": "reach_reward_direction", "toward_reward": good[1], "away_reward": bad[1]})
        toward.close()
        away.close()
        env = PickPlaceEnv()
        env.reset(seed=1, options=FIXED)
        target = env._state()["tcp"].copy()
        env.reset(seed=1, options={**FIXED, "target_position": target})
        for _ in range(50):
            transition = env.step([0, 0, 0, 1])
            assert_transition(self, env, transition)
            if transition[2] or transition[3]:
                break
        self.assertTrue(transition[4]["is_success"])
        self.assertGreaterEqual(env.data.time, env.config.reach_hold_s)
        self.assertTrue(transition[2])
        self.assertFalse(transition[3])
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0, 1])
        env.close()

    def test_real_grasp_lift_and_pick_place(self):
        """三个任务均通过四维动作接口获得真实成功，放置必须打开并撤离。"""
        for task in ("grasp", "lift", "pick_place"):
            with self.subTest(task=task):
                physical_action_probe(self, task)

    def test_no_lift_or_place_false_positive(self):
        """初始就位不能算完成；非夹持弹起也不获得抬升势或成功。"""
        env = PickPlaceEnv(task_mode="pick_place")
        env.reset(seed=1, options={**FIXED, "target_position": [.5, 0, .225]})
        for _ in range(30):
            transition = env.step([0, 0, 0, 1])
        self.assertFalse(transition[4]["is_success"])
        self.assertFalse(transition[4]["has_lifted"])
        self.assertEqual(env.timers[3], 0.)
        env.close()
        env = PickPlaceEnv(task_mode="lift")
        env.reset(seed=1, options=FIXED)
        body = int(env.model.body("cube").id)
        peak = 0.
        for index in range(12):
            env.data.xfrc_applied[body, 2] = 2.0 if index < 3 else 0.
            transition = env.step([0, 0, 0, 1])
            peak = max(peak, transition[4]["cube_bottom_clearance"])
            self.assertFalse(transition[4]["has_lifted"])
            self.assertFalse(transition[4]["is_success"])
            self.assertEqual(env._potentials(env._state())["lift_progress"], 0.)
        self.assertGreater(peak, .04)
        EVIDENCE.append({"case": "ungrasped_physical_launch", "peak_clearance_m": peak, "lift_success": False})
        env.close()

    def test_timeout_ik_and_numerical_failure(self):
        """时间限制是 truncated；真实 IK 连续失败和坏物理状态是 terminated。"""
        for config in (replace(PickPlaceEnvConfig(), max_episode_steps=2),
                       replace(PickPlaceEnvConfig(), max_simulation_time_s=.08)):
            env = PickPlaceEnv(config=config)
            env.reset(seed=1, options=FIXED)
            env.step([0, 0, 0, 1])
            transition = env.step([0, 0, 0, 1])
            self.assertEqual(transition[2:4], (False, True))
            self.assertEqual(transition[4]["termination_reason"], "TIME_LIMIT")
            self.assertIsNone(transition[4]["failure_reason"])
            assert_transition(self, env, transition)
            env.close()
        config = replace(PickPlaceEnvConfig(), action_scale_m=3., workspace_high=(4., 4., 4.), max_ik_failures=2)
        env = PickPlaceEnv(config=config)
        env.reset(seed=1, options=FIXED)
        command = env.tcp_command.copy()
        joint_target = env.joint_target.copy()
        for _ in range(2):
            transition = env.step([1, 0, 0, 1])
            self.assertFalse(transition[4]["ik_success"])
            self.assertLessEqual(transition[1], -env.config.rewards.ik_failure)
            np.testing.assert_array_equal(env.tcp_command, command)
            np.testing.assert_array_equal(env.joint_target, joint_target)
        self.assertEqual(transition[2:4], (True, False))
        self.assertEqual(transition[4]["failure_reason"], "IK_FAILED")
        EVIDENCE.append({"case": "actual_ik_failure", "info": transition[4]})
        env.close()
        env = PickPlaceEnv()
        env.reset(seed=1)
        # 故意破坏机器人状态以测试防护；这不是任务成功测试，不操作 Cube。
        env.data.joint("joint1").qvel[0] = np.nan
        transition = env.step([0, 0, 0, 1])
        self.assertEqual(transition[2:4], (True, False))
        self.assertEqual(transition[4]["failure_reason"], "NUMERICAL_ERROR")
        self.assertTrue(transition[4]["observation_sanitized"])
        assert_transition(self, env, transition)
        env.reset(seed=1)
        self.assertTrue(np.all(np.isfinite(env.data.qvel)))
        env.close()

    def test_physical_drop_and_reward_cycle(self):
        """真实推落会失败；反复建立/解除双侧接触的势奖励会互相抵消。"""
        env = PickPlaceEnv(task_mode="lift")
        env.reset(seed=1, options=FIXED)
        body = int(env.model.body("cube").id)
        for index in range(60):
            env.data.xfrc_applied[body, 0] = 5. if index < 5 else 0.
            transition = env.step([0, 0, 0, 1])
            assert_transition(self, env, transition)
            if transition[2] or transition[3]:
                break
        self.assertEqual(transition[4]["failure_reason"], "OBJECT_DROPPED")
        self.assertEqual(transition[4]["physics_steps"], env.config.frame_skip)
        self.assertEqual(transition[2:4], (True, False))
        self.assertLessEqual(transition[1], -env.config.rewards.failure_penalty)
        EVIDENCE.append({"case": "physical_drop", "info": transition[4]})
        env.close()
        env = PickPlaceEnv(task_mode="lift")
        env.reset(seed=4, options={**FIXED, "start_pose": "pre_grasp"})
        goal = np.array([.5, 0, .231])
        grasp_sum, discounted_sum, action_index = 0., 0., 0
        for grip, count in ((1., 30), (-1., 40), (1., 30)):
            for _ in range(count):
                action = np.r_[np.clip((goal-env.tcp_command)/env.config.action_scale_m, -1, 1), grip]
                transition = env.step(action)
                assert_transition(self, env, transition)
                grasp_sum += transition[4]["reward_terms"]["grasp_progress"]
                discounted_sum += (env.config.rewards.shaping_discount**action_index
                                   * transition[4]["reward_terms"]["grasp_progress"])
                action_index += 1
                self.assertFalse(transition[2] or transition[3])
            if grip == -1:
                self.assertGreaterEqual(env.timers[1], env.config.grasp_hold_s)
        self.assertFalse(env.has_lifted)
        self.assertLessEqual(grasp_sum, 0.)
        self.assertAlmostEqual(discounted_sum, 0., places=10)
        EVIDENCE.append({"case": "physical_contact_cycle", "grasp_progress_sum": grasp_sum,
                         "discounted_grasp_progress_sum": discounted_sum})
        env.close()

    def test_render_no_physics_and_headless_no_gui(self):
        """无界面 render 为纯空操作；显示调用的无积分约束由假 Viewer 验证。"""
        env = PickPlaceEnv()
        env.reset(seed=1)
        env.step([0, 0, 0, 1])
        before = (env.data.time, env.episode_step, env.timers.copy(), env.data.qpos.copy())
        env.render()
        self.assertIsNone(env.viewer)

        class ViewerProbe:
            """仅验证 render 资源契约，不把该对象当作 GUI 实跑证据。"""
            def sync(self):
                """记录同步次数。"""
                self.synced = True

            def close(self):
                """记录资源释放。"""
                self.closed = True

        probe = ViewerProbe()
        env.render_mode = "human"
        env.viewer = probe
        env.render()
        self.assertTrue(probe.synced)
        self.assertEqual((env.data.time, env.episode_step), before[:2])
        np.testing.assert_array_equal(env.timers, before[2])
        np.testing.assert_array_equal(env.data.qpos, before[3])
        env.close()
        self.assertTrue(probe.closed)


def main():
    """运行测试并可选新建 JSON 报告；不覆盖以前的测试证据。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path)
    args, remaining = parser.parse_known_args()
    if args.report is not None and args.report.exists():
        parser.error("报告已存在，请使用新路径")
    started = time.perf_counter()
    program = unittest.main(argv=[sys.argv[0], *remaining], verbosity=2, exit=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as output:
            json.dump({"passed": program.result.wasSuccessful(), "tests": program.result.testsRun,
                       "failures": len(program.result.failures), "errors": len(program.result.errors),
                       "wall_time_s": time.perf_counter()-started, "evidence": EVIDENCE},
                      output, ensure_ascii=False, indent=2, allow_nan=False)
    raise SystemExit(0 if program.result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
