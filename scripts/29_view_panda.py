from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 1. 基本配置
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "assets" / "franka_panda" / "scene.xml"

SIM_DURATION = 60.0
PRINT_INTERVAL = 1

# ============================================================
# 2. 加载 Panda
# ============================================================
def load_panda():

    model = mujoco.MjModel.from_xml_path( str(XML_PATH) )
    data = mujoco.MjData(model)

    return model, data

# ============================================================
# 3. 重置到 Home Pose
# ============================================================
def reset_home(model,data):

    home_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,"home",)
    if home_id == -1:
        raise ValueError("Home keyframe not found.")

    mujoco.mj_resetDataKeyframe(model,data,home_id,)
    mujoco.mj_forward(model,data,)

# ============================================================
# 4. 计算当前目标 Control
# ============================================================
def compute_target_control(model,home_ctrl,sim_time,):
    
    target_ctrl = home_ctrl.copy()

    # --------------------------------------------------------
    # Joint 1
    # --------------------------------------------------------
    joint1_offset = (0.35 * np.sin(2.0 * np.pi *0.2*sim_time))
    target_ctrl[0] = (home_ctrl[0]+joint1_offset)

    # --------------------------------------------------------
    # Joint 2
    # --------------------------------------------------------
    joint2_offset = (0.20 * np.sin(2.0 * np.pi * 0.15 * sim_time))
    target_ctrl[1] = (home_ctrl[1]+joint2_offset)

    # --------------------------------------------------------
    # Gripper
    # --------------------------------------------------------
    gripper_command = (127.5 * (1.0 + np.cos(2.0*np.pi*sim_time/8.0)))
    target_ctrl[7] = gripper_command

    # --------------------------------------------------------
    # 限制所有 Actuator Control
    # --------------------------------------------------------
    ctrl_low= model.actuator_ctrlrange[:,0]
    ctrl_high=model.actuator_ctrlrange[:,1]
    target_ctrl = np.clip(target_ctrl,ctrl_low,ctrl_high)

    return target_ctrl

# ============================================================
# 5. 打印关节状态
# ============================================================
def print_robot_state(data):

    joint1_q= float(data.joint("joint1").qpos[0])
    joint1_qvel = float(data.joint("joint1").qvel[0])
    joint2_q = float(data.joint("joint2").qpos[0])
    joint2_qvel = float(data.joint("joint2").qvel[0])

    finger1 = float(data.joint("finger_joint1").qpos[0])
    finger2 = float(data.joint("finger_joint2").qpos[0])

    target_joint1 = float(data.ctrl[0])
    target_joint2 = float(data.ctrl[1])

    gripper_ctrl = float(data.ctrl[7])

    print(f"\ntime = {data.time:.2f} s")
    print(
        f"joint1: "
        f"target={np.rad2deg(target_joint1):+.2f} deg, "
        f"actual={np.rad2deg(joint1_q):+.2f} deg, "
        f"velocity={joint1_qvel:+.3f} rad/s"
    )

    print(
        f"joint2: "
        f"target={np.rad2deg(target_joint2):+.2f} deg, "
        f"actual={np.rad2deg(joint2_q):+.2f} deg, "
        f"velocity={joint2_qvel:+.3f} rad/s"
    )

    print(
        f"gripper: "
        f"ctrl={gripper_ctrl:.1f}, "
        f"left={finger1:.4f} m, "
        f"right={finger2:.4f} m"
    )

# ============================================================
# 6. 主程序
# ============================================================
def main():
    # --------------------------------------------------------
    # 6.1 加载机器人
    # --------------------------------------------------------
    model, data = load_panda()

    # --------------------------------------------------------
    # 6.2 恢复 Home Pose
    # --------------------------------------------------------
    reset_home(model, data)

    # --------------------------------------------------------
    # 6.3 保存 Home Control
    # --------------------------------------------------------
    home_ctrl = data.ctrl.copy()

    # --------------------------------------------------------
    # 6.4 物理时间步长
    # --------------------------------------------------------
    physics_dt = float(model.opt.timestep)

    print("Panda Viewer Demo")
    print("Physics timestep:", physics_dt)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("\nOpening MuJoCo Viewer...")

    # --------------------------------------------------------
    # 6.5 启动 Viewer
    # --------------------------------------------------------
    next_print_time = 0.0

    with mujoco.viewer.launch_passive(model,data,) as viewer:
        while(viewer.is_running() and data.time < SIM_DURATION):
            wall_start = time.perf_counter()

            # ================================================
            # 读取仿真时间
            # ================================================
            sim_time = float(data.time)
            target_ctrl = compute_target_control(model,home_ctrl,sim_time,)

            # ================================================
            # 同步修改状态并推进物理
            # ================================================
            with viewer.lock():
                data.ctrl[:] = target_ctrl
                mujoco.mj_step(model,data)

            # ================================================
            # 更新 Viewer
            # ================================================
            viewer.sync()

            # ================================================
            # 打印日志
            # ================================================
            if data.time >= next_print_time:
                print_robot_state(data)
                next_print_time += PRINT_INTERVAL

            # ================================================
            # 近似实时播放
            # ================================================
            elapsed = (time.perf_counter() - wall_start)
            remaining = physics_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    print("\nPanda Viewer Demo Finished.")
        
# ============================================================
# 7. 程序入口
# ============================================================
if __name__ == "__main__":
    main()



