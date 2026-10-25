# 具身智能仿真环境：从 MuJoCo 到 Gymnasium

统合日期：2026-10-25。本文以结项后的代码为准，整合 Day0–5、Day8、Day13 的环境知识。项目已经结项，文中复现方式用于回顾已有成果。

本文负责仿真模型、状态、动作、时间尺度和环境接口；运动学推导见 [机器人运动学](02_robot_kinematics.md)，物理抓放判定见 [Pick & Place](05_pick_and_place.md)。

## 1. 仿真环境在整套系统中的位置

具身智能策略必须通过动作影响物理世界，再观察动作的后果。本项目的链条是：

```text
观测 → 策略 → 动作 → 控制器/IK → 执行器 → MuJoCo 动力学
 ↑                                                  ↓
 └──────── 传感/状态读取、奖励与任务判定 ←───────────────┘
```

环境定义世界如何变化以及什么算完成任务；策略决定下一步做什么。把抓放状态机藏进环境、再让策略输出无关动作，不能证明策略学会抓放。当前 PickPlaceEnv 没有内置专家阶段决策。

项目使用状态观测，而不是 RGB 图像输入。MuJoCo 里的精确位姿与接触信息是当前仿真任务的可用状态，不等于真实机器人已具备相应感知能力。

## 2. MjModel、MjData 与 MJCF

| 对象 | 含义 | 项目中的例子 |
| --- | --- | --- |
| MJCF/XML | 对物理世界的声明 | 连杆、关节、惯性、几何、执行器、场景 |
| MjModel | XML 编译后的模型和参数 | 关节范围、质量、传动、时间步长 |
| MjData | 某个仿真实例当前的状态和计算结果 | qpos、qvel、ctrl、位姿、接触、仿真时间 |
| body | 刚体节点及其局部坐标系 | Panda 连杆、手掌、Cube |
| joint | 子刚体相对父节点允许的运动 | hinge、slide、ball、free |
| geom | 外形、碰撞及相关属性 | 连杆网格、手指碰撞体、桌面 |
| actuator | 对关节或腱施加控制作用的机构 | 关节位置伺服、夹爪 tendon actuator |
| keyframe | 一组预定义状态 | Home、含 Cube 的初始状态 |

`mj_step(model, data)` 推进一个物理步。`mj_forward(model, data)` 在给定配置下刷新派生量、接触和力相关计算，不负责推进仿真时间。修改初始 qpos 后，不能假定此前缓存的世界位姿仍有效。

每个环境实例拥有独立的模型、数据和控制状态；训练与评估也应隔离。IK 另持有用于运动学求解的 MjData，不通过试探关节角直接移动真实仿真。

## 3. 关节状态：为什么 nq 不一定等于 nv

| 关节 | 运动自由度 | qpos 维数 | qvel 维数 |
| --- | ---: | ---: | ---: |
| hinge | 1 | 1 | 1 |
| slide | 1 | 1 | 1 |
| ball | 3 | 4 | 3 |
| free | 6 | 7 | 6 |

`nq` 是配置表示维数，`nv` 是速度/自由度维数，`nu` 是控制输入维数。姿态用四元数表示需要四个数，但只有三个独立旋转自由度，因此不能把 qpos、qvel 和 ctrl 当作同形数组。

Panda 不含自由 Cube 时，七个臂关节加两个手指关节得到 nq=9、nv=9；七个臂执行器加一个夹爪腱执行器得到 nu=8。加入自由 Cube 后，抓放场景变为 nq=16、nv=15、nu=8。Cube 没有专用控制输入，运动依靠真实接触、重力与动力学。

统一单位：位置 m，角度 rad，线速度 m/s，角速度 rad/s，时间 s，力 N。显示成 mm 或 degree 时，只改变展示，不改变计算单位。

## 4. 动作不是 qpos，也不一定就是 ctrl

单关节环境里，策略输出直接成为受限的电机控制输入。Panda 操作环境中，动作经过多层转换：

```text
归一化 XYZ 增量 → TCP 命令 → IK 关节目标
             → 关节参考限速 → 位置执行器 ctrl → 真实运动
夹爪动作     → 名义开度 → 腱执行器 ctrl → 手指真实位移/接触
```

命令位置与实际位置不同。伺服需要时间跟踪，还受重力、惯性、接触和力限制影响。给出关闭夹爪命令，也不能直接判定已经抓住物体。

基础反馈 PD 为：

\[
u=K_p e-K_d\dot q,\qquad
e=\operatorname{atan2}(\sin(q_{\text{target}}-q),\cos(q_{\text{target}}-q)).
\]

P 项拉向目标，D 项抑制速度和振荡；最后仍需按执行器 ctrlrange 限幅。Panda 的位置执行器与单关节力矩输入不同，不能把相同数值解释成相同物理控制量。

## 5. 物理步与控制步

\[
\Delta t_{\text{control}}=\text{frame_skip}\times\Delta t_{\text{physics}}.
\]

| 环境/过程 | 物理步长 | frame_skip | 控制周期 |
| --- | ---: | ---: | ---: |
| 单关节 Gymnasium | 0.002 s | 10 | 0.02 s，50 Hz |
| Panda 学习环境 | 0.002 s | 20 | 0.04 s，25 Hz |
| Day12 规则运行器 | 0.002 s | 每次运行器循环推进一个物理步 | 物理步级反馈 |

抓放环境在物理子步检查接触、稳定时间和故障，避免只在控制步末看到一瞬间的合格状态。render 只同步画面，不能额外推进物理；Viewer 实时节奏不等于仿真时间定义。

## 6. Gymnasium 的基本契约

```python
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

- observation_space/action_space 声明形状、类型和取值范围，不能代替运行时有限值检查。
- reset 开始新回合，重置随机源、物理状态、控制记忆、计时器和累计统计。
- step 推进动作并返回下一观测；结束后应重新 reset。
- info 保存诊断证据，不应在评估中绕过观测直接操纵物体。
- 同一环境 seed 配合相同动作序列可用于重现；随机策略的动作空间也需要独立设置 seed。

单关节环境观测为 `[q, qvel, target]`，奖励为：

\[
r=-e^2-0.01\dot q^2-0.001u^2.
\]

角度误差与速度分别小于 5°、5°/s 即成功；默认 250 控制步是 5 s。该基础例程没有 Panda 抓放那样的连续稳定计时，不能把两者的成功标准混用。

## 7. 本项目三种 Panda 环境版本

| 版本 | 观测/动作 | XYZ 每轴每步最大增量 | 时间上限 | 超时含义 |
| --- | --- | ---: | --- | --- |
| Day13 PickPlaceEnv 默认 | 94D / 4D | 0.01 m | 由基类配置决定 | 外部截断 |
| Reach v2 当前配置 | 46D / 3D | 0.005 m | 250 步 / 10 s | 有限任务终止 |
| Day15 BC 配置 | 94D / 4D | 0.005 m | 750 步 / 30 s | 外部截断 |

Day13 基类支持 reach、grasp、lift、pick_place。Reach v2 固定夹爪张开，专门优化 Reach；Day15 环境固定为 pick_place 并加入重力前馈。相同基类不代表观测、动作、奖励、超时协议可以互换。

## 8. 四维动作与内部控制记忆

Day13/Day15 的动作：

\[
a=[a_x,a_y,a_z,a_g]\in[-1,1]^4.
\]

\[
p_{\text{request}}=p_{\text{command,prev}}+s\,a_{xyz},
\qquad w_{\text{gripper}}=0.04(a_g+1)\ \mathrm m.
\]

新位置先限制在工作空间，再进行位姿 IK；只有求解有效才提交为下一合法目标。增量相对上一合法命令，不是当前真实 TCP。这样零增量可以保持目标，也可能积累命令跟踪间隙，所以观测里必须提供命令状态。

四维零动作意味着 XYZ 命令不变、夹爪目标开度为 0.04 m；不表示“夹爪维持原开度”。Reach v2 三维零动作则维持 XYZ 命令，夹爪始终全开。

单次 IK 失败保持上一合法手臂目标，夹爪仍可响应；连续失败达到配置上限才终止。环境不会把非法 IK 关节解强行写入真实 qpos。

## 9. 94 维观测速查

切片由代码中的 `OBSERVATION_FIELDS/OBSERVATION_SLICES` 生成，训练代码应复用定义。

| 切片 | 内容 | 维数 |
| --- | --- | ---: |
| 0:14 | 臂关节位置、速度 | 14 |
| 14:23 | TCP 世界位置、姿态 6D | 9 |
| 23:38 | Cube 世界位置、姿态 6D、线/角速度 | 15 |
| 38:43 | 两指位置、速度、实际名义开度 | 5 |
| 43:52 | TCP→Cube、Cube→Target、目标位置 | 9 |
| 52:60 | 手指接触/夹持候选、桌面接触、任务 one-hot | 8 |
| 60:77 | TCP 命令、关节限速命令、IK 目标 | 17 |
| 77:87 | 上一动作、工具目标姿态 6D | 10 |
| 87:94 | 四个稳定计时比例、has_lifted、IK 失败与剩余比例 | 7 |

姿态 6D 是旋转矩阵前两列，不是六个旋转自由度。Cube 速度读取后重排成“线速度在前、角速度在后”，均表达在世界系。控制记忆、计时与 has_lifted 是已发生的信息，不是专家阶段或未来信息。

物理量保留原始 SI 数值；神经网络归一化交给训练层。Box 的无限边界不代表允许 NaN/Inf。

## 10. reset、结束语义与奖励

抓放 reset 恢复完整关键帧，在安全桌面内采样 Cube/目标和偏航，再清除计时、动作、合法命令、失败计数和势函数历史。只有初始化阶段可以设置 Cube 初始 qpos；step 中禁止靠瞬移、mocap 跟随或临时焊接完成抓放。

`terminated` 表示任务定义内的成功、失败或有限期限结束；`truncated` 表示外部截断。Day15 外部超时和 Reach v2 有限期限结束采用不同标志，是任务定义差别，不应机械地把所有超时都写成同一种。

奖励指导学习，成功条件核验物理事实。Day13 采用任务相关势差与成本：

\[
F_t=\gamma\Phi(s_{t+1})-\Phi(s_t).
\]

真实终止将后继势置零；外部截断保留后继势并按学习器语义自举。Reach v2 使用另一套近且静密集奖励，详见 [PPO](04_reinforcement_learning_ppo.md)。BC 的监督损失不使用环境奖励更新 actor，详见 [BC](03_imitation_learning_bc.md)。

## 11. 当前代码与可运行检查

| 代码 | 阅读重点 |
| --- | --- |
| [single_joint_env.py](../envs/single_joint_env.py) | 最小环境封装 |
| [pick_place_env.py](../envs/pick_place_env.py) | reset→动作→物理子步→状态→奖励与结束 |
| [reach_env.py](../envs/reach_env.py) | 从公共物理环境构造专用任务 |
| [pick_place_learning_env.py](../envs/pick_place_learning_env.py) | BC 环境与重力补偿 |
| [00_smoke_test.py](../scripts/stage0_learn_base/00_smoke_test.py) | model/data/step 最小例子 |
| [12_check_env.py](../scripts/stage0_learn_base/12_check_env.py) | Gymnasium 标准接口检查 |

在项目根目录、已有 embodied 环境执行：

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied
PYTHONPATH=. python scripts/stage0_learn_base/00_smoke_test.py
PYTHONPATH=. python scripts/stage0_learn_base/12_check_env.py
```

03、05、07、09 号例程仍存在搬迁后的资源根路径错误，当前以源码阅读为主；本文未修复代码。Day05/06/07 产物也已按用户要求删除，17 号例程缺少原轨迹，25 号比较器缺少原模型。requirements-lock.txt 不是完整可移植的训练依赖锁。

## 12. 自查问题与来源

能够解释：nq/nv/nu 为什么不同；action 如何变成真实运动；reset 能做什么、step 不能做什么；物理步与控制步有何区别；命令记忆为何进入观测；terminated/truncated 如何影响价值自举。

原 Day0、MJCF、关节与坐标系三份基础笔记已删除，内容已统合至本文。其他历史来源（旧笔记已清理）：控制、Gymnasium、轨迹、Panda 模型、Day13 实现、Day13 知识笔记。旧笔记中的已删除脚本命令不作为本文当前入口。
