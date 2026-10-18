"""Day 12 的有限步数物理驱动与 Viewer 标记；任务条件由 Agent 独立负责。"""

from contextlib import nullcontext
import math
import time

import mujoco
import numpy as np


def update_markers(viewer, agent):
    """以红球标记当前 TCP 目标，以绿球标记目标 Cube 中心；尺寸单位 m。"""
    scene = viewer.user_scn
    if scene.maxgeom < 2:
        return
    scene.ngeom = 0
    for i, (position, color) in enumerate(((agent.target_tcp, [1., .1, .1, .8]),
                                         (agent.target_place_position, [.1, 1., .1, .6]))):
        mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([.008, 0., 0.]), np.asarray(position),
                           np.eye(3).reshape(9), np.asarray(color))
    scene.ngeom = 2


def run_pick_place(agent, data, viewer=None, verbose=True, before_physics=None, log_interval_s=1.0):
    """推进已 reset 的 Agent，返回结构化结果；不会直接操作 Cube 状态。

    before_physics(agent, data) 仅供测试注入执行器故障或明确的物理外力，
    生产演示不传该参数。物理步数还受独立上限保护，避免异常时钟导致死循环。
    """
    if not agent.started:
        raise RuntimeError("请先 reset 智能体并指定目标中心 XYZ")
    dt = float(agent.model.opt.timestep)
    steps_limit = math.ceil(agent.config.total_timeout_s / dt) + 2
    next_log_s = 0.0
    history_count = 0
    for index in range(steps_limit):
        wall_start = time.perf_counter()
        if viewer is not None and not viewer.is_running():
            agent.abort(data, "VIEWER_CLOSED", "用户提前关闭 Viewer")
            break
        with viewer.lock() if viewer is not None else nullcontext():
            agent.step(data)
            changed = len(agent.state_history) != history_count
            if verbose and (changed or data.time >= next_log_s):
                status = agent.status(data)
                if changed:
                    for entry in agent.state_history[history_count:]:
                        print(f"TRANSITION t={entry['time_s']:.3f}s -> {entry['state']} "
                              f"completed={entry.get('completed_state')} {entry['reason']}", flush=True)
                print(f"STATUS {status}", flush=True)
                next_log_s = float(data.time) + log_interval_s
            history_count = len(agent.state_history)
            if agent.done:
                break
            if before_physics is not None:
                before_physics(agent, data)
            mujoco.mj_step(agent.model, data)
            # 完整刷新使观测中的位姿、接触法向与求解力对应同一时刻。
            mujoco.mj_forward(agent.model, data)
            if viewer is not None and index % 10 == 0:
                update_markers(viewer, agent)
        if viewer is not None:
            if index % 10 == 0:
                viewer.sync()
            delay = dt - (time.perf_counter() - wall_start)
            if delay > 0:
                time.sleep(delay)
    if not agent.done:
        agent.abort(data, "TASK_TIMEOUT", "物理步数达到独立回合上限")
    return agent.result(data)
