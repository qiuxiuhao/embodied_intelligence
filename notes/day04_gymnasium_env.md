# day4 - gymnasium environment
## environment
environment 封装：
- dynamics
- observation
- action
- reward
- reset
- termination
- truncation
Agent 不应该直接依赖Mujoco内部状态

## observation
obs = [q,qvel,target]
observation_space 描述 observation 的合法范围

## action
action = [motor_control]
action_space = Box([-3],[3])

## physics step vs environment step
Mujoco timestep = 0.002s
physics frequency = 500 hz
frame_skip = 10
control frequency =50 hz
一个 env.step(action)对应10个mj_step
## Reward
reward = reward =
    - error^2
    - 0.01 * qvel^2
    - 0.001 * action^2

Reward 是 task definition，不是 physics。

## Terminated

任务真正成功或失败导致 Episode 结束。

当前：

angle error < 2 degree
and
velocity < 0.05 rad/s

## Truncated

由于时间限制等原因结束。
当前：

250 env steps
= 5 seconds

## agent-environment loop
obs -> policy -> action -> env.step(action) -> next_obs, reward, terminated, truncated