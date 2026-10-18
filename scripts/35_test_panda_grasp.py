"""Day 11 回归测试：PYTHONPATH=. python scripts/35_test_panda_grasp.py。

无需渲染器、Viewer、pytest 或其他新增依赖。物理测试不写入 Cube 的 qpos 或外力；
构造的接触快照仅用于验证状态监测器能否拒绝不足的成功证据。
"""

from dataclasses import replace
import unittest

import mujoco
import numpy as np

from controllers.gripper_controller import PandaGripperController
from controllers.ik_controller import PandaIKSolver, rotation_error_world
from robotics.contacts import ContactSnapshot, PandaContactDetector
from robotics.grasp_experiment import (GraspMonitor, cube_bottom_clearance,
                                       load_cube_scene, run_experiment)
from robotics.kinematics import (get_arm_qpos, get_tcp_jacobian, get_tcp_pose,
                                 load_panda, reset_home, set_arm_qpos)


def advance(model, data, duration_s):
    """保持控制输入推进动力学，随后同步位姿与接触力。"""
    for _ in range(round(duration_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


class Day11Tests(unittest.TestCase):
    """验证物理过程、执行器映射、坐标系约定和成功判据。"""

    def test_scene_structure(self):
        """检查自由刚体惯量，并确认 Cube 没有附加约束或执行器。"""
        model, data = load_cube_scene()
        self.assertEqual((model.nq, model.nv, model.nu), (16, 15, 8))
        self.assertEqual(model.nmocap, 0)
        joint = model.joint("cube_free")
        self.assertEqual(model.jnt_type[joint.id], mujoco.mjtJoint.mjJNT_FREE)
        self.assertAlmostEqual(model.body_mass[model.body("cube").id], 0.10)
        np.testing.assert_allclose(model.body_inertia[model.body("cube").id],
                                   np.full(3, 0.10 * 0.05**2 / 6))
        np.testing.assert_allclose(data.body("cube").xpos, [0.50, 0., 0.225])
        self.assertEqual(model.neq, 1)  # 仅保留官方模型中的手指等式约束。
        self.assertEqual(model.eq_type[0], mujoco.mjtEq.mjEQ_JOINT)
        finger_ids = {model.joint("finger_joint1").id, model.joint("finger_joint2").id}
        self.assertEqual({int(model.eq_obj1id[0]), int(model.eq_obj2id[0])}, finger_ids)
        original, original_data = load_panda()
        reset_home(original, original_data)
        self.assertEqual((original.nq, original.nv, original.nu), (9, 9, 8))
        print("\nXML: nq=16 nv=15 nu=8, cube=0.10kg, freejoint, official scene still loads")

    def test_gripper_open_close_feedback(self):
        """在机械臂初始位姿下，测试真实闭合、中间开度和重新全开。"""
        model, data = load_cube_scene()
        gripper = PandaGripperController(model)
        for width, native_ctrl in ((0.0, 0.0), (0.04, 127.5), (0.08, 255.0)):
            self.assertAlmostEqual(gripper.command(data, width), native_ctrl, places=6)
            advance(model, data, 2.0)
            feedback = gripper.feedback(data)
            np.testing.assert_allclose(feedback["finger_position_m"], width / 2, atol=0.0002)
            np.testing.assert_allclose(feedback["finger_velocity_m_s"], 0, atol=0.001)
            print(f"\nopening_target={width:.3f}m ctrl={native_ctrl:g} "
                  f"actual_fingers_m={feedback['finger_position_m']}")
        self.assertEqual(gripper.command(data, -1.0), 0.0)
        self.assertAlmostEqual(gripper.command(data, 1.0), 255.0, places=6)
        with self.assertRaises(ValueError):
            gripper.command(data, np.nan)

    def test_table_contact_is_not_grasp(self):
        """真实的 Cube 与桌面承重接触不能被识别成手指接触或抓取。"""
        model, data = load_cube_scene()
        advance(model, data, 1.0)
        contact = PandaContactDetector(model).read(data)
        self.assertGreater(data.ncon, 0)
        self.assertTrue(contact.cube_table)
        self.assertFalse(contact.cube_floor)
        self.assertFalse(contact.finger_contact)
        self.assertFalse(contact.grasp_candidate)
        self.assertFalse(contact.finger_table)
        normal_force = sum(c.normal_force_N for c in contact.cube_contacts)
        self.assertAlmostEqual(normal_force, 0.981, delta=0.01)
        self.assertLess(abs(cube_bottom_clearance(model, data)), 0.001)
        print(f"\nCube/table normal force={normal_force:.6f}N; ncon={data.ncon}; grasp=False")

    def test_rotation_error_matches_world_jacobian(self):
        """通过中心差分验证世界坐标系下旋转残差的符号和各坐标轴方向。"""
        model, data = load_cube_scene()
        q = get_arm_qpos(data) + np.deg2rad([5, -5, 7, 4, -4, 5, 3])
        set_arm_qpos(model, data, q)
        _, jacr, _, _ = get_tcp_jacobian(model, data)
        fd = np.zeros_like(jacr)
        eps = 1e-6  # 扰动角度单位 rad
        for i in range(7):
            offset = np.eye(7)[i] * eps
            set_arm_qpos(model, data, q + offset)
            r_plus = get_tcp_pose(data)[2]
            set_arm_qpos(model, data, q - offset)
            r_minus = get_tcp_pose(data)[2]
            fd[:, i] = rotation_error_world(r_plus, r_minus) / (2 * eps)
        np.testing.assert_allclose(fd, jacr, atol=1e-8)
        pi_error = rotation_error_world(np.diag([1., -1., -1.]), np.eye(3))
        self.assertAlmostEqual(np.linalg.norm(pi_error), np.pi)
        print(f"\nWorld rotation Jacobian max error={np.max(np.abs(fd - jacr)):.3e}")

    def test_pose_ik_requires_orientation(self):
        """保持位置不变并改变偏航角，验证姿态求解及真实仿真数据的隔离。"""
        model, data = load_cube_scene()
        original_qpos = data.qpos.copy()
        position, _, rotation = get_tcp_pose(data)
        yaw = np.deg2rad(30.0)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                       [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        target_rotation = rz @ rotation
        solver = PandaIKSolver(model, tolerance=0.0005, max_iterations=400)
        result = solver.solve(position, initial_q=get_arm_qpos(data),
                              target_rotation=target_rotation, orientation_tolerance=0.005)
        self.assertTrue(result["success"], result)
        self.assertLess(result["position_error"], 0.0005)
        self.assertLess(result["orientation_error"], 0.005)
        np.testing.assert_array_equal(data.qpos, original_qpos)
        with self.assertRaises(ValueError):
            solver.solve(position, target_rotation=np.diag([1., 1., -1.]))
        print(f"\n30deg yaw IK: position={result['position_error']:.6f}m "
              f"angle={result['orientation_error']:.6f}rad; live data unchanged")

    def test_state_monitor_rejects_insufficient_evidence(self):
        """检查单侧接触、仍有支撑、抬升中断和释放条件不足的反例。"""
        none = ContactSnapshot((), False, False, False, False, False, 0., 0.)
        single = replace(none, finger_contact=True, left_force_N=1.)
        bilateral = replace(single, grasp_candidate=True, right_force_N=1.)
        monitor = GraspMonitor()
        monitor.update(single, 0.1, 0.325, 0.05, False, 1.0)
        self.assertTrue(monitor.finger_contact)
        self.assertFalse(monitor.grasp_candidate)
        self.assertFalse(monitor.lift_success)
        monitor.update(replace(bilateral, cube_table=True), 0.1, 0.325, 0.05, False, 1.0)
        self.assertFalse(monitor.lift_success)
        monitor.update(bilateral, 0.1, 0.325, 0.05, False, 0.3)
        monitor.update(none, 0.1, 0.325, 0.05, False, 0.01)
        monitor.update(bilateral, 0.1, 0.325, 0.05, False, 0.3)
        self.assertFalse(monitor.lift_success)
        monitor.update(bilateral, 0.1, 0.325, 0.05, False, 0.21)
        self.assertTrue(monitor.lift_success)
        monitor.update(none, 0., 0.225, 0.08, False, 1.0)
        self.assertFalse(monitor.release)  # 尚未发出释放命令。
        monitor.update(bilateral, 0., 0.225, 0.08, True, 1.0)
        self.assertFalse(monitor.release)  # 接触仍在夹持 Cube。
        monitor.update(none, 0., 0.225, 0.05, True, 1.0)
        self.assertFalse(monitor.release)  # 手指的实际开度尚未达到要求。
        monitor.update(none, 0.1, 0.325, 0.08, True, 1.0)
        self.assertFalse(monitor.release)  # 尚未观察到真实下落。
        monitor.update(none, 0., 0.225, 0.08, True, 0.21)
        self.assertTrue(monitor.release)

    def test_physical_grasp_lift_release(self):
        """重新初始化并重复三次完整实验，断言真实接触、持续抬升和自由下落。"""
        for trial in range(3):
            model, data = load_cube_scene()
            result = run_experiment(model, data, verbose=False)
            self.assertTrue(result["success"], result)
            for flag in ("finger_contact", "grasp_candidate", "lift_success", "release"):
                self.assertTrue(result[flag], flag)
            self.assertGreaterEqual(result["max_lift_hold_s"], 0.5)
            self.assertGreater(result["max_bottom_clearance_m"], 0.10)
            self.assertGreater(result["release_drop_m"], 0.10)
            self.assertLess(result["min_release_velocity_m_s"], -0.5)
            self.assertGreater(result["landed_stable_s"], 0.3)
            self.assertFalse(result["finger_table_collision"])
            self.assertTrue(result["single_finger_contact_seen"])
            self.assertEqual(result["warning_count"], 0)
            phases = result["phases"]
            self.assertFalse(phases["descend"]["finger_contact"])
            self.assertTrue(phases["close"]["cube_table"])
            self.assertTrue(phases["close"]["grasp_candidate"])
            self.assertFalse(phases["hold"]["cube_table"])
            self.assertTrue(phases["hold"]["grasp_candidate"])
            self.assertGreater(phases["hold"]["left_force_N"], 0.05)
            self.assertGreater(phases["hold"]["right_force_N"], 0.05)
            self.assertTrue(phases["release"]["cube_table"])
            self.assertFalse(phases["release"]["finger_contact"])
            self.assertGreater(phases["release"]["opening_m"], 0.07)
            print(f"\nTrial {trial + 1}: clearance={result['max_bottom_clearance_m']:.6f}m "
                  f"lift_hold={result['max_lift_hold_s']:.3f}s "
                  f"drop={result['release_drop_m']:.6f}m "
                  f"fall_vz={result['min_release_velocity_m_s']:.3f}m/s "
                  f"landed={result['landed_stable_s']:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
