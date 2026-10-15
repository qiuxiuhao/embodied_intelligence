from pathlib import Path
from contextlib import nullcontext

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers.ik_controller import (
    PandaIKSolver,
)

from controllers.joint_controller import (
    PandaJointPositionController,
)

from robotics.kinematics import (
    load_panda,
    reset_home,
    get_arm_qpos,
    get_arm_qvel,
    get_tcp_pose,
    set_arm_qpos,
)

# ============================================================
# 1. 实验配置
# ============================================================
SIM_DURATION = 60.0
RENDER_EVERY = 10
PRINT_INTERVAL = 0.5
POSITION_TOLERANCE = 0.015
VELOCITY_TOLERANCE = 0.10
STABLE_DURATION = 0.4
MAX_JOINT_SPEED = 0.5

# ============================================================
# 2. 命令行参数
# ============================================================
def parse_args():

    parser = argparse.ArgumentParser(description="Day 10 Panda IK Reach Demo")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo Viewer.", )
    parser.add_argument( "--target", type=float,  nargs=3, default=None, metavar=("X", "Y", "Z"),  help="Target TCP XYZ in world coordinates, meters.", )

    return parser.parse_args()

def create_default_target(model,  home_q,):
    # --------------------------------------------------------
    # 3.1 单独创建 FK Data
    # --------------------------------------------------------
    fk_data = mujoco.MjData(model)
    reset_home(  model, fk_data, )
    # --------------------------------------------------------
    # 3.2 创建一个接近 Home 的合法关节配置
    # --------------------------------------------------------
    joint_offset_deg = np.array( [  8.0, -6.0, 4.0, 0.0, 0.0, 0.0, 0.0,],dtype=np.float64,)
    q_known = (  home_q + np.deg2rad( joint_offset_deg ))

    # --------------------------------------------------------
    # 3.3 计算对应的 TCP 位置
    # --------------------------------------------------------
    set_arm_qpos( model,fk_data,q_known,)
    target_position, _, _ = get_tcp_pose(  fk_data )
    return target_position

# ============================================================
# 4. 在 Viewer 中绘制目标和当前 TCP
# ============================================================
def update_markers( viewer, target_position, current_position,):

    scene = viewer.user_scn
    if scene.maxgeom < 2:
        return
    # 清空之前添加的自定义可视化几何体
    scene.ngeom = 0
    # --------------------------------------------------------
    # 4.1 目标位置：红色球
    # --------------------------------------------------------
    mujoco.mjv_initGeom(
        scene.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array(
            [0.025, 0.0, 0.0]
        ),
        pos=np.asarray(
            target_position,
            dtype=np.float64,
        ),
        mat=np.eye(3).reshape(-1),
        rgba=np.array(
            [1.0, 0.1, 0.1, 0.9]
        ),
    )
    # --------------------------------------------------------
    # 4.2 当前 TCP：绿色球
    # --------------------------------------------------------
    mujoco.mjv_initGeom(
        scene.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array(
            [0.015, 0.0, 0.0]
        ),
        pos=np.asarray(
            current_position,
            dtype=np.float64,
        ),
        mat=np.eye(3).reshape(-1),
        rgba=np.array(
            [0.1, 1.0, 0.1, 0.9]
        ),
    )
    scene.ngeom = 2

# ============================================================
# 5. 执行物理仿真
# ============================================================
def run_simulation( model, data, controller, target_q, target_position, viewer=None,):
    physics_dt = float(model.opt.timestep)
    step_index = 0
    next_print_time = 0.0
    stable_time = 0.0
    reached = False

    # --------------------------------------------------------
    # 5.1 初始化控制器
    # --------------------------------------------------------
    controller.reset(data)
    print("\n========== PHYSICS CONTROL ==========")
    print("Physics dt:", physics_dt)
    print("Target TCP XYZ:")
    print(target_position)

    # ========================================================
    # 6. 物理循环
    # ========================================================
    while data.time < SIM_DURATION:
        if viewer is not None:
            if not viewer.is_running():
                break
        wall_start = time.perf_counter()
        step_index += 1
        # ----------------------------------------------------
        # 6.1 计算是否需要本轮可视化和日志
        # ----------------------------------------------------
        need_update = ( step_index % RENDER_EVERY == 0 )
        # ----------------------------------------------------
        # 6.2 对共享状态加锁
        # ----------------------------------------------------
        lock_context = (
            viewer.lock()
            if viewer is not None
            else nullcontext()
        )
        with lock_context:

            # ----------------------------------------
            # 6.3 生成平滑关节目标指令
            # ----------------------------------------
            command_q = controller.step(data=data,target_q=target_q, dt=physics_dt,)

            # ----------------------------------------
            # 6.4 推进 MuJoCo 物理
            # ----------------------------------------

            mujoco.mj_step(model, data,)

            # ----------------------------------------
            # 6.5 定期刷新运动学状态
            # ----------------------------------------
            if need_update:
                mujoco.mj_kinematics( model, data, )
                # 当前实际 TCP 位置
                current_position, _, _ = get_tcp_pose( data )
                # TCP 误差
                position_error = float( np.linalg.norm( target_position  - current_position) )
                # 当前真实关节角度
                actual_q = get_arm_qpos( data)
                # 当前真实关节速度
                actual_qvel = get_arm_qvel( data )
                max_velocity = float( np.max( np.abs(actual_qvel) ) )
                current_time = float( data.time )

                # ------------------------------------
                # Viewer 可视化
                # ------------------------------------
                if viewer is not None:
                    update_markers(viewer,target_position,current_position, )

        # ====================================================
        # 7. Viewer Sync
        # ====================================================

        if viewer is not None and need_update:
            viewer.sync()

        # ====================================================
        # 8. 状态与成功判定
        # ====================================================
        if need_update:
            # ----------------------------------------
            # 8.1 成功条件
            # ----------------------------------------
            position_ok = (position_error < POSITION_TOLERANCE )
            velocity_ok = ( max_velocity < VELOCITY_TOLERANCE )
            # ----------------------------------------
            # 8.2 连续稳定计时
            # ----------------------------------------
            if position_ok and velocity_ok:
                stable_time += ( RENDER_EVERY * physics_dt )

            else:
                stable_time = 0.0

            # ----------------------------------------
            # 8.3 打印日志
            # ----------------------------------------
            if current_time >= next_print_time:
                joint_tracking_error = float(  np.linalg.norm( command_q - actual_q ))

                print(
                    f"time={current_time:5.2f}s  "
                    f"tcp_error={position_error * 1000:7.2f}mm  "
                    f"max_qvel={max_velocity:6.3f}rad/s  "
                    f"joint_tracking_error="
                    f"{joint_tracking_error:7.4f}rad"
                )
                next_print_time += (  PRINT_INTERVAL )

            # ----------------------------------------
            # 8.4 判定持续稳定成功
            # ----------------------------------------
            if stable_time >= STABLE_DURATION:
                reached = True
                print(
                    "\nTCP target reached and stabilized."
                )
                break

        # ====================================================
        # 9. Viewer 实时播放节奏
        # ====================================================
        if viewer is not None:
            elapsed = (  time.perf_counter()   -   wall_start)
            remaining = ( physics_dt - elapsed)
            if remaining > 0:
                time.sleep(remaining)

    # ========================================================
    # 10. 最终状态检查
    # ========================================================
    mujoco.mj_kinematics(model,data,)

    final_position, _, _ = get_tcp_pose( data )
    final_error = float( np.linalg.norm(  target_position  - final_position ) )

    print("\n========== FINAL RESULT ==========")
    print("Target position:")
    print(target_position)
    print("\nActual position:")
    print(final_position)
    print("\nFinal position error (mm):")
    print(final_error * 1000.0)
    print("\nReached and stabilized:")
    print(reached)

    return {
        "success": reached,
        "target_position": ( target_position.copy() ),
        "final_position": ( final_position.copy() ),
        "final_error": final_error,
        "simulation_time": float( data.time ),
    }


# ============================================================
# 11. 主程序
# ===========================================================
def main():
    args = parse_args()
    np.set_printoptions(precision=6,suppress=True,)

    # --------------------------------------------------------
    # 11.1 加载 Panda
    # --------------------------------------------------------

    model, data = load_panda()

    reset_home(
        model,
        data,
    )

    # --------------------------------------------------------
    # 11.2 Home Joint Configuration
    # --------------------------------------------------------

    home_q = get_arm_qpos(
        data
    )

    # --------------------------------------------------------
    # 11.3 目标 TCP XYZ
    # --------------------------------------------------------

    if args.target is None:

        target_position = create_default_target(
            model,
            home_q,
        )

    else:

        target_position = np.asarray(
            args.target,
            dtype=np.float64,
        )

    print("========== TARGET ==========")

    print("Target TCP XYZ:")
    print(target_position)

    # --------------------------------------------------------
    # 11.4 创建 IK Solver
    # --------------------------------------------------------

    ik_solver = PandaIKSolver(
        model=model,
        damping=0.05,
        tolerance=0.002,
        max_iterations=200,
    )

    # --------------------------------------------------------
    # 11.5 求解 IK
    # --------------------------------------------------------

    ik_result = ik_solver.solve(
        target_position=target_position,
        initial_q=home_q,
    )

    print("\n========== IK ==========")

    print("Success:", ik_result["success"])

    print("Reason:", ik_result["reason"])

    print("Iterations:", ik_result["iterations"])

    print(
        "Position error (mm):",
        ik_result["position_error"] * 1000.0,
    )

    # --------------------------------------------------------
    # 11.6 IK 求解失败则停止
    # --------------------------------------------------------

    if not ik_result["success"]:

        raise RuntimeError(
            "IK failed. "
            "The target will not be sent to the robot."
        )

    target_q = ik_result["q"]

    print("\nTarget joint angles (deg):")
    print(np.rad2deg(target_q))

    # --------------------------------------------------------
    # 11.7 创建关节控制器
    # --------------------------------------------------------

    controller = PandaJointPositionController(
        model=model,
        max_joint_speed=MAX_JOINT_SPEED,
    )

    # --------------------------------------------------------
    # 11.8 Headless
    # --------------------------------------------------------

    if args.headless:

        result = run_simulation(
            model=model,
            data=data,
            controller=controller,
            target_q=target_q,
            target_position=target_position,
            viewer=None,
        )

    # --------------------------------------------------------
    # 11.9 Viewer
    # --------------------------------------------------------

    else:

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            result = run_simulation(
                model=model,
                data=data,
                controller=controller,
                target_q=target_q,
                target_position=target_position,
                viewer=viewer,
            )

    # --------------------------------------------------------
    # 11.10 结果总结
    # --------------------------------------------------------

    print("\n========== DAY 10 RESULT ==========")

    print(
        "IK solved:",
        ik_result["success"],
    )

    print(
        "Physical reach success:",
        result["success"],
    )

    print(
        "Final TCP error (mm):",
        result["final_error"] * 1000.0,
    )


if __name__ == "__main__":

    main()


