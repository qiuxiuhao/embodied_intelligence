"""Day 12 演示：无界面使用 python；macOS Viewer 使用 mjpython。"""

import argparse
import json
from pathlib import Path

from agents.scripted_pick_place import ScriptedPickPlaceAgent
from robotics.grasp_experiment import load_cube_scene
from robotics.pick_place_runtime import run_pick_place


def main():
    """从 Home 开始运行一次任务；可将完整结果写入新的 JSON 文件。"""
    parser = argparse.ArgumentParser(description="Day 12 真实物理抓取、运输、放置和撤离")
    parser.add_argument("--headless", action="store_true", help="不加载 Viewer")
    parser.add_argument("--target", type=float, nargs=3, default=[0.50, 0.12, 0.225],
                        metavar=("X", "Y", "Z"), help="放置后 Cube 中心的世界坐标，单位 m")
    parser.add_argument("--seed", type=int, default=0, help="结果记录用种子；当前策略不采样随机动作")
    parser.add_argument("--output", type=Path, help="可选：新建结果 JSON，拒绝覆盖已有文件")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output 已存在，请使用新文件名以保留实验结果")
    model, data = load_cube_scene()
    agent = ScriptedPickPlaceAgent(model)
    agent.reset(data, args.target, seed=args.seed)
    if args.headless:
        result = run_pick_place(agent, data)
    else:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.43, 0.05, 0.30]
                viewer.cam.distance = 1.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            result = run_pick_place(agent, data, viewer=viewer)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(serialized + "\n")
    if not result["success"]:
        raise SystemExit(f"任务失败：{result['failure_reason']}，阶段：{result['failure_stage']}")


if __name__ == "__main__":
    main()
