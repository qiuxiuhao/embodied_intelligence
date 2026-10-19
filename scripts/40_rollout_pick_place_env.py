"""Day 13 随机/全零四维动作回合；只评估环境接口，不创建或训练 PPO。"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from envs.pick_place_env import OBSERVATION_SIZE, PickPlaceEnv, PickPlaceEnvConfig, TASK_MODES


def rollout(task, policy, seed, config, render_mode=None, start_pose="home"):
    """运行完整回合；环境初始随机性与策略动作随机性分别使用明确种子。"""
    env = PickPlaceEnv(task_mode=task, render_mode=render_mode, config=config)
    try:
        observation, reset_info = env.reset(seed=seed, options={"start_pose": start_pose})
        env.action_space.seed(seed + 10000)
        total_return, step_wall_s, ik_failures = 0., 0., 0
        started = time.perf_counter()
        for _ in range(config.max_episode_steps):
            action = env.action_space.sample() if policy == "random" else np.zeros(4, dtype=np.float32)
            wall_start = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action)
            elapsed = time.perf_counter() - wall_start
            step_wall_s += elapsed
            total_return += reward
            ik_failures += int(not info.get("ik_success", True))
            if not env.observation_space.contains(observation) or not np.isfinite(reward):
                raise RuntimeError("运行时发现非法观测或非有限奖励")
            if render_mode == "human":
                # 只有演示驱动器控制墙钟播放节奏；render() 本身不额外推进物理。
                delay = env.dt - elapsed
                if delay > 0:
                    time.sleep(delay)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("环境未在配置的完整回合上限内结束")
        return {"task_mode": task, "policy": policy, "seed": seed, "start_pose": start_pose,
                "episode_return": float(total_return), "episode_steps": info["episode_step"],
                "success": info["is_success"], "terminated": terminated, "truncated": truncated,
                "termination_reason": info["termination_reason"], "simulation_time_s": info["simulation_time"],
                "wall_time_s": time.perf_counter()-started, "step_wall_time_s": step_wall_s,
                "steps_per_second": info["episode_step"]/step_wall_s,
                "ik_failure_steps": ik_failures, "initial_info": reset_info, "final_info": info}
    finally:
        env.close()


def main():
    """按任务与策略组合统计回报、成功率和实测性能，并可新建 JSON 记录。"""
    parser = argparse.ArgumentParser(description="Day 13 Random / Zero Policy 环境基线")
    parser.add_argument("--tasks", nargs="+", choices=TASK_MODES, default=list(TASK_MODES))
    parser.add_argument("--policy", choices=("random", "zero", "both"), default="both")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--start-pose", choices=("home", "pre_grasp"), default="home")
    parser.add_argument("--human", action="store_true", help="macOS 使用 mjpython，约实时播放")
    parser.add_argument("--output", type=Path, help="只新建结果文件，不覆盖已有结果")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes 必须为正整数")
    if args.output is not None and args.output.exists():
        parser.error("输出已存在，请使用新路径")
    config = replace(PickPlaceEnvConfig(), max_episode_steps=args.max_steps, frame_skip=args.frame_skip)
    policies = ("random", "zero") if args.policy == "both" else (args.policy,)
    episodes, summaries = [], []
    for task in args.tasks:
        for policy in policies:
            group = []
            for index in range(args.episodes):
                result = rollout(task, policy, args.seed+index, config,
                                 "human" if args.human else None, args.start_pose)
                episodes.append(result)
                group.append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ("initial_info", "final_info")},
                                 ensure_ascii=False, allow_nan=False), flush=True)
            summaries.append({"task_mode": task, "policy": policy, "episodes": len(group),
                              "mean_return": float(np.mean([r["episode_return"] for r in group])),
                              "mean_steps": float(np.mean([r["episode_steps"] for r in group])),
                              "success_rate": float(np.mean([r["success"] for r in group])),
                              "mean_simulation_time_s": float(np.mean([r["simulation_time_s"] for r in group])),
                              "aggregate_steps_per_second": sum(r["episode_steps"] for r in group)
                              / sum(r["step_wall_time_s"] for r in group)})
    report = {"configuration": asdict(config), "observation_size": OBSERVATION_SIZE,
              "summary": summaries, "episodes": episodes,
              "note": "Zero 为四维全零：位移目标保持，夹爪目标为 0.04 m；没有专家或 PPO 动作"}
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, allow_nan=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
