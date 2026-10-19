# Day 13 — Gymnasium PickPlaceEnv

## 范围、开发前检查与文件

2026-10-18 在本地重新检查 Git，工作区初始干净；重新读取 `single_joint_env.py`、Day 9～12 的运动学/控制器/接触、Agent、运行器、场景、测试与笔记。环境没有调用 Day 12 Agent、没有内置其有限状态机，也没有创建或训练 PPO。

实际环境：Python 3.12.14、MuJoCo 3.14.0、Gymnasium 1.3.0、Stable-Baselines3 2.9.0，使用已有 `embodied` Conda 环境，没有安装新依赖。

仅新增：

| 文件 | 职责 |
|---|---|
| `envs/pick_place_env.py` | `PickPlaceEnv`、集中配置、94 维观测定义、物理任务判据、奖励与 Gymnasium API |
| `scripts/39_test_pick_place_env.py` | 12 项自动化验收，含四维动作物理成功探针、错误处理和 SB3 包装器 |
| `scripts/40_rollout_pick_place_env.py` | Random / Zero 完整回合、统计和 human 演示；不含训练算法 |
| `notes/day13_pick_place_env.md` | 本说明、公式、测试与实验记录 |

模型复用 `scene_cube.xml` 的完整 `day11_home` keyframe，物理步长 `0.002 s`，`nq=16/nv=15/nu=8`。TCP 仍是 hand 局部 `[0,0,0.10] m`。夹爪控制仍按官方 `actuator8 → split tendon → 双指` 映射，原生 ctrl 为 `0～255`；接触复用 `PandaContactDetector` 的 geom/body 匹配和法向力判据。

低层复用 `PandaIKSolver`、`PandaJointPositionController`、`PandaGripperController`，不重复实现 DLS、Jacobian、位置 servo 或 tendon 传动。Day 12 的整段路点规划用于规则基线；RL 每步只求解一个目标，不调用 `PoseMotionController.plan()`。

没有修改 `single_joint_env.py` 或 Day 0～12 代码，没有 commit/push，没有覆盖 Day 11/12 的实验数据。

## 环境 API

```python
from envs.pick_place_env import PickPlaceEnv, PickPlaceEnvConfig

env = PickPlaceEnv(task_mode="reach", render_mode=None,
                   config=PickPlaceEnvConfig())
observation, info = env.reset(seed=1234)
observation, reward, terminated, truncated, info = env.step(action)
env.render()
env.close()
```

- `task_mode`：`reach/grasp/lift/pick_place`。
- `render_mode`：`None` 或 `"human"`；metadata 声明 human，默认 25 fps，根据控制周期更新实例 metadata。
- observation 为固定 `(94,)` float32，reward 为 Python float，terminated/truncated 为 Python bool。
- Episode 结束或 close 后必须 reset；重复 step 会明确报错。
- reset 不偷偷推进物理；物理时间和速度从零开始。
- 各环境实例分别持有独立 `MjModel/MjData`、IK 数据和控制器，配置为不可变 dataclass。

## 四维 Action

`Box(-1, 1, shape=(4,), dtype=float32)`：

| 维度 | 意义 | 默认真实量 |
|---|---|---|
| 0 | 世界坐标 x 增量 | `0.01 * dx m` |
| 1 | 世界坐标 y 增量 | `0.01 * dy m` |
| 2 | 世界坐标 z 增量 | `0.01 * dz m` |
| 3 | 绝对名义夹爪开度 | `0.04 * (gripper + 1) m` |

位置增量**相对于上一条合法 TCP 命令目标**，不是相对于实际 TCP：

```text
requested_target = previous_valid_tcp_command + action_scale_m * action[:3]
desired_target = clip(requested_target, workspace_low, workspace_high)
```

这样零位移动作能保持目标，不会把重力静差每步重新加进目标。代价是存在命令积压，故观测同时包含实际 TCP、命令 TCP、关节 servo 当前命令与目标。测试确认连续两次 `dx=1` 累计 0.02 m；后续零位移动作保持该命令，真实 TCP 确实移动。

工作空间默认下界 `[0.32,-0.19,0.215] m`、上界 `[0.68,0.19,0.65] m`，它是任务边界，不保证其中每个角点的六维 IK 都可达。位置范围饱和在 info 的 `workspace_clipped` 中可见。

夹爪 `-1 → 0 m`、`0 → 0.04 m`、`+1 → 0.08 m`。因此 **Zero Policy 不是夹爪保持原开度，而是目标半开**。实际两指 qpos/qvel 单独读取，不用命令替代反馈。

每个 RL step：

1. 严格检查 shape=(4,) 和有限实数；NaN/Inf、复数或错误形状抛 ValueError，不推进时间。
2. 有限越界动作限幅到 [-1,1]，记录 `action_clipped/applied_action` 并扣分。
3. 调用已有六维 IK，以当前关节命令为初值，固定朝下并与初始 Cube 偏航对齐。IK 最多 60 次迭代，容差 0.0005 m / 0.005 rad。
4. 只有 success、有限关节解、合法关节范围、位置和姿态误差均通过时才提交 TCP/关节目标。
5. 每个子步经原 Joint Controller 限制参考关节速度为 0.5 rad/s，再控制夹爪、积分、完整 mj_forward 并读取真实接触。

默认 `frame_skip=20`，控制周期 `20 * 0.002 = 0.04 s`，25 Hz。正常、IK 失败、物理掉落/碰撞步都完成固定子步数；物理失败一旦出现就锁存。只有状态非有限或 MuJoCo 数值警告时为避免继续积分而提前终止，info 保留实际 `physics_steps`。

单次 IK 失败保持上一合法目标，夹爪仍能响应其合法动作；该步不发非法关节命令、不瞬移，扣除 IK 成本且禁用正向奖励。连续 5 次失败才以 `IK_FAILED` terminated，允许策略在此前恢复。

## Observation：完整 94 维表

切片使用 Python 的左闭右开索引。代码导出 `OBSERVATION_FIELDS/OBSERVATION_SLICES/OBSERVATION_SIZE`，供后续策略或测试统一读取。

所有位置和线速度统一为世界坐标。Cube 角速度也通过 `mj_objectVelocity(..., flg_local=0)` 转为世界坐标，避免混用自由关节原生局部角速度；API 返回顺序为角/线，观测调整为线/角。姿态 6D 是旋转矩阵前两列各 3 个分量，避免四元数正负双覆盖。

| 切片 | 内容 | Shape | 单位 | Space 范围/处理 |
|---|---|---|---|---|
| 0:7 | 七关节角度 | 7 | rad | SI 原值，±∞；名义限位来自模型，允许 soft limit 的微小越界 |
| 7:14 | 七关节速度 | 7 | rad/s | 原值，±∞ |
| 14:17 | TCP 世界位置 | 3 | m | 原值，±∞ |
| 17:23 | TCP 姿态 6D | 6 | 无量纲 | [-1,1] |
| 23:26 | Cube 世界位置 | 3 | m | 原值，±∞ |
| 26:32 | Cube 姿态 6D | 6 | 无量纲 | [-1,1] |
| 32:38 | Cube 世界线速度、角速度 | 6 | m/s、rad/s | 原值，±∞ |
| 38:40 | 左右手指实际位置 | 2 | m | 原值，±∞；名义各 [0,0.04] |
| 40:42 | 左右手指实际速度 | 2 | m/s | 原值，±∞ |
| 42:43 | 两指位移之和 | 1 | m | 原值，±∞；名义 [0,0.08] |
| 43:46 | Cube − TCP | 3 | m | 原值，±∞ |
| 46:49 | Target − Cube | 3 | m | 原值，±∞ |
| 49:52 | 当前任务目标位置 | 3 | m | 原值，±∞ |
| 52:55 | 左指有效接触、右指有效接触、Grasp Candidate | 3 | 0/1 | [0,1] |
| 55:56 | Cube–桌面有效接触 | 1 | 0/1 | [0,1] |
| 56:60 | reach/grasp/lift/pick_place one-hot | 4 | 0/1 | [0,1] |
| 60:63 | 上一合法 TCP 命令目标 | 3 | m | 原值，±∞ |
| 63:70 | 关节 servo 当前限速命令 | 7 | rad | 原值，±∞ |
| 70:77 | 上一合法 IK 关节目标 | 7 | rad | 原值，±∞ |
| 77:81 | 上一条限幅后的四维动作 | 4 | 无量纲 | [-1,1] |
| 81:87 | 目标工具姿态 6D | 6 | 无量纲 | [-1,1] |
| 87:91 | reach/grasp/lift/place 连续计时器 ÷ 各自门槛 | 4 | 无量纲 | [0,1] |
| 91:92 | 曾经真实稳定抬升 | 1 | 0/1 | [0,1] |
| 92:93 | 连续 IK 失败数 ÷ 失败上限 | 1 | 无量纲 | [0,1] |
| 93:94 | 剩余步数/时间限制比例的较小者 | 1 | 无量纲 | [0,1] |

连续物理量不在环境内做隐藏归一化或截断，推荐以后用 VecNormalize 处理观测。Box 的无限上下界表示没有伪造物理硬界；所有实际正常输出仍须有限且 `observation_space.contains(obs)` 为 true。检查器会对此发出范围建议警告，记录在报告中，不是 API 失败。

这些控制记忆和历史计时均为过去/当前信息，不含专家动作、Day 12 阶段、未来状态或奖励值。每次观察都使用刚完整刷新过的 MuJoCo 数据。

数值损坏是唯一特殊路径：返回带 `observation_sanitized=True` 的中性有限终止向量，无法定义的误差为 JSON null，terminated=True，禁止 bootstrap。没有复用伪装成当前状态的旧观测，也没有在 episode 内自动重置或修复物体。

## 四种任务与成功条件

固定工具姿态的角误差使用 SO(3) 最短旋转向量范数。所有稳定计时逐 **0.002 s 物理子步** 更新，一次不满足立即清零，而不是每 0.04 s 只看一个瞬态。

| 模式 | 目标位置含义 | 真实成功条件 |
|---|---|---|
| reach | TCP 世界位置 | 实际位置误差 <0.009 m、姿态误差 <0.035 rad、最大实际关节速度 <0.10 rad/s，连续 0.40 s |
| grasp | 初始 Cube 中心参考 | 真正双侧合理接触，Cube 线速度 <0.02 m/s、角速度 <0.20 rad/s，机械臂速度 <0.10 rad/s，连续 0.50 s |
| lift | 初始 Cube 中心上方 0.10 m 的参考点 | grasp 的受力/速度条件、无桌面/地面接触、旋转方块最低点离桌 ≥0.04 m，连续 0.50 s |
| pick_place | 放置后 Cube 中心 | 曾满足上述稳定 lift；最终 XY 误差 <0.012 m、Z 误差 <0.004 m、最低点与台面差绝对值 <0.004 m、真实桌面支撑、无地面/手指接触、实际开度 ≥0.07 m、Cube 速度合格、TCP–Cube 距离 ≥0.08 m，连续 0.50 s |

没有“必须先到某个 Scripted 状态”的要求。`has_lifted` 只是经过物理验证的历史事实，不能通过单次高度、命令闭合或弹起置位。最终放置也不能因为初始 Cube 已位于目标处就成功。

lift 的硬成功门槛由 `lift_clearance_m` 指定，0.10 m 为密集高度势的饱和尺度/参考高度；不要求恰好在某个高度停住。需要更高的硬门槛时调整该配置，不要只改变观测里的参考 XYZ。

阈值依据：Day 11/12 实测约 6.5 mm 的未补偿 TCP 静差、0.016 rad 的姿态误差；保留已验证的 4 cm / 0.5 s 抬升判据、12 mm XY / 4 mm Z 放置判据。环境没有靠放宽成功定义使测试通过。

## Reward 公式与防刷分

定义 `S=0.20 m`，`dR=||TCP-target||`、`dA=||TCP-Cube||`、`dP=||Cube-target||`，`h` 为旋转方块最低点相对台面的高度。`g/p` 为连续 grasp/place 时间除门槛，均限幅到 [0,1]。

默认势函数（无量纲）：

```text
Phi_R = 2 / (1 + clip(dR/S, 0, 5))
Phi_A = 1 / (1 + clip(dA/S, 0, 5))
Phi_G = 0.5 * g
Phi_H = 2 * clip(h/0.10m, 0, 1) * I(GraspCandidate 且无桌面接触)
Phi_P = 2 * I(has_lifted) * (1 - clip(dP/S, 0, 1))
Phi_release = 1 * p

F_i = gamma * Phi_i(next) - Phi_i(current), gamma = 0.99
```

真正 terminated 的后继势置零；TIME_LIMIT truncated 保留后继势，供 SB3 正确 bootstrap。训练时 PPO 的 gamma 应与 `rewards.shaping_discount` 匹配。以下组合均另外加公共成本与成功/失败项：

| 任务 | Dense 势整形 |
|---|---|
| reach | F_R |
| grasp | F_A + F_G |
| lift | F_A + F_G + F_H |
| pick_place | F_A + F_G + F_H + F_P + F_release |

公共项：

```text
distance_cost    = -rate * min(d/S, 5) * elapsed_s
                  reach: rate=0.2, d=dR；其余: rate=0.05, d=dA
orientation_cost = -0.1 * min(angle_error/pi, 1) * elapsed_s
action_cost      = -0.02 * mean(action^2) * elapsed_s
change_cost      = -0.01 * mean((action-previous_action)^2)
clipped_cost     = -0.1 * I(原始有限动作越界)
ik_failure_cost  = -1.0 * I(IK 未通过)
collision_cost   = -1.0 * I(手指接触桌面) * elapsed_s
success_bonus    = +10，仅真实成功终止且控制有效时给一次
failure_penalty  = -10，仅明确物理/控制失败；普通超时不额外扣该项
```

`elapsed_s` 是实际积分时长，通常 0.04 s。权重全部在 `RewardWeights`。每步 info 提供所有有效分项，总和严格等于 reward。

IK 失败或物理失败步把全部正向分项截为零，因此不能利用惯性续动刷正向奖励。严重数值异常没有可用势函数，直接返回 -10 与明确诊断。

Grasp/height/place 不给“每停留一步就加固定正分”，而是折扣势差。解除接触会失去对应势；真实闭合→稳定夹持→重新打开的循环中，抓取势未折扣总和为 **-0.15828**，折扣净和约 **-5.55e-17**，没有正收益。未夹持 Cube 被物理外力弹起到约 **0.1495 m** 离桌时，抬升势始终为 0，也不能置位 has_lifted。

这不是 PPO 收敛证明；不同任务的权重、等待与碰撞策略仍需后续训练监测。不要把不同任务的原始 Return 直接当成同一种难度评分。

## Reset、随机性与结束语义

reset 调用 `super().reset(seed=seed)`；环境所有随机采样使用 `self.np_random`。选项支持：

| options 键 | 含义 |
|---|---|
| `task_mode` | 本回合四任务之一，不改变 observation/action 维度 |
| `cube_position` | 固定初始 Cube 世界 XYZ；底面必须恰好在台面上且完整处于桌内 |
| `cube_yaw` | 初始偏航 rad；默认均匀采样 [-0.20,0.20] |
| `target_position` | reach 的 TCP 目标或 pick_place 的 Cube 放置中心；grasp/lift 仅接受与由 Cube/配置派生参考一致的值，拒绝无关且不会生效的 XYZ |
| `start_pose` | `home`（默认）或 `pre_grasp`；后者仅 reset 时用 IK 初始化机械臂到 Cube 上方 0.12 m |

默认 Cube 在 `[0.50,0,0.225] m` 附近，XY 半范围 [0.025,0.025] m；放置目标在 `[0.50,0.12,0.225] m` 附近，XY 半范围 [0.035,0.025] m，与初始 Cube XY 至少相隔 0.06 m。抽样有有限次数上限，整个旋转方块的投影均须落在桌内并留 0.012 m 安全余量。

reach 目标在 `[0.46,-0.08,0.42]` 至 `[0.60,0.08,0.55] m` 的简化可达区域。默认范围在当前模型中经过运行验证，但没有保证任意自定义范围都可达。

每次 reset 重置完整 keyframe、物理时间、全部速度、外力、控制器状态、上一动作、累计奖励、连续失败计数、全部稳定计时及 has_lifted。只有这里允许写初始 Cube qpos。step 没有 weld、mocap、物体位置写入或自动重抓。

| 情况 | terminated | truncated | 原因 |
|---|---|---|---|
| 真实成功 | true | false | SUCCESS |
| Cube 地面接触，或无桌面支撑且最低点比桌面低 0.03 m 以上 | true | false | OBJECT_DROPPED |
| 手指–桌面有效接触，默认禁止 | true | false | FINGER_TABLE_COLLISION |
| 连续 5 次 IK 不可执行 | true | false | IK_FAILED |
| 非有限/极端状态或 MuJoCo warning | true | false | NUMERICAL_ERROR |
| 达到最大步数或仿真时间 | false | true | TIME_LIMIT |

真正终态优先于时间限制，不同时返回两种 true。时间限制在固定控制步结束时检查，非整控制周期的限制最多多执行不到一个控制周期。

info 包含要求的 task、成功、TCP/Cube 误差（m）、底部离桌高度（m）、接触、实际开度（m）、reward_terms、步数、仿真时间（s）和结束原因，并增加 IK 诊断、限幅状态、实际物理步数、两指位置、控制目标和稳定计时。终止时 `info.episode` 包含 Return、步数、仿真时长、成功及最终误差。

## 实际测试与保存结果

全部结果放在新的 `outputs/day13_20261018_3s2qzed3/`，按既有 gitignore 忽略但本机实际保留：

- 最终验收：`acceptance_final.json` / `acceptance_final.log`。
- 最终 16 回合 Random/Zero：`baselines_acceptance.json` / `.log`。
- Day 11 回归：`day11_regression.log`；Day 12 回归：`day12_regression.json` / `.log`。
- 最终 GUI：`viewer_acceptance.json` / `.log`；相同动作 Headless 参考：`viewer_reference_acceptance.json` / `.log`。
- `tests*`、`acceptance*`、`baselines*`、`viewer_verified*` 等较早调试记录也保留，没有覆盖。

实际执行命令（已有报告路径不宜重复使用；日常命令见下一节）：

```bash
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python -m compileall -q envs/pick_place_env.py scripts/39_test_pick_place_env.py scripts/40_rollout_pick_place_env.py

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/39_test_pick_place_env.py --report outputs/day13_20261018_3s2qzed3/acceptance_final.json > outputs/day13_20261018_3s2qzed3/acceptance_final.log 2>&1

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/40_rollout_pick_place_env.py --episodes 2 --max-steps 250 --output outputs/day13_20261018_3s2qzed3/baselines_acceptance.json > outputs/day13_20261018_3s2qzed3/baselines_acceptance.log 2>&1

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/35_test_panda_grasp.py > outputs/day13_20261018_3s2qzed3/day11_regression.log 2>&1
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/37_test_pick_place.py --report outputs/day13_20261018_3s2qzed3/day12_regression.json > outputs/day13_20261018_3s2qzed3/day12_regression.log 2>&1

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/mjpython scripts/40_rollout_pick_place_env.py --tasks reach --policy random --episodes 2 --max-steps 100 --human --output outputs/day13_20261018_3s2qzed3/viewer_acceptance.json > outputs/day13_20261018_3s2qzed3/viewer_acceptance.log 2>&1
git diff --check
```

验收结果：

| 检查 | 实际结果 |
|---|---|
| Gymnasium check_env + SB3 check_env | 四种任务全部通过；每种有两条物理 Box 无限范围的建议警告，无错误 |
| Monitor + 双实例 DummyVecEnv | 通过；2 步截断，TimeLimit.truncated、terminal_observation、自动 reset 正常 |
| Space、dtype、contains、每步奖励分项与 JSON | 通过，94 维 float32，全正常观测有限 |
| 同 seed 初始化与相同动作轨迹 | 两独立环境逐字段复现；不同 seed 真正改变 Cube、偏航和目标 |
| Reset、固定 options、独立实例 | 通过，20 个随机初态无穿透/桌外目标，实际成功后 reset 不遗留历史 |
| 增量、真实移动、frame_skip | 通过，2 次 dx=1 累积 0.02 m；实际 TCP 移动 >0.01 m；frame_skip=10 的 32 步仿真时间 0.64 s |
| 夹爪真实反馈 | 全闭≈0 m、全开≈0.08 m、零动作≈0.04 m；分别读取两指位置 |
| 坏动作/限幅/真实不可达 IK | 通过；非法形状和 NaN/Inf 报错且时间不变；越界可诊断；不可达目标保留合法命令并最终 IK_FAILED |
| Reach 稳定成功与奖励方向 | 通过；相同初态下向目标优于远离，成功必须稳定 ≥0.4 s |
| 四维动作真实 Grasp / Lift / PickPlace | 全部成功，详见下表；不写物体状态，不使用 Day 12 底层 ctrl |
| 假阳性、弹起与奖励循环 | 通过；初始就位无 lift 历史不成功；149.5 mm 弹起不算 lift；折扣接触循环无收益 |
| 掉落、超时、数值异常 | 真实物理推落返回 terminated；时间限制返回 truncated；损坏机器人速度返回明确数值失败 |
| Render 契约与资源 | 单元测试确认不推进物理；真实 GUI 连续两回合成功，见下文 |
| Day 11 / Day 12 | 原有 7 项 / 10 项全部通过 |

Day 13 最终 **12 项全部通过**。这里既有接口单元测试，也有大量真实 MuJoCo 推进；没有仅凭 Mock 声称物理成功。

测试中的外部真值路点探针只通过四维 `env.step()` 操作：

| 任务 | RL steps | 仿真时长 s | Episode Return | 物理证据 |
|---|---:|---:|---:|---|
| grasp | 36 | 1.44 | 8.9918 | 双侧受力及静止累计 0.5 s，实际开度约 0.04996 m |
| lift | 69 | 2.76 | 8.1177 | 底部离桌约 0.11544 m，无桌面接触，连续稳定持握 0.5 s |
| pick_place | 134 | 5.36 | 5.9239 | 最终中心误差 4.271 mm，真实打开/脱离/落桌/撤离全部成立 |

这是用于验收控制通路的独立探针，不是 Random Policy 成功，也不是训练结果。环境内部没有该探针的路点或阶段逻辑。

## Random / Zero 基线

每个 task/policy 组合使用 seed=1234、1235，Home 初态，250 步上限、frame_skip=20；总计 16 个完整回合。所有回合均 TIME_LIMIT 截断，250 步、10.00 s，成功率 0%。随机动作 RNG 由 action_space 单独显式播种，环境采样仍只用 np_random。

| Task | Policy | Mean Return | Mean steps | Success | Headless steps/s |
|---|---|---:|---:|---:|---:|
| reach | Random | -6.3469 | 250 | 0/2 | 510.5 |
| reach | Zero | -4.3308 | 250 | 0/2 | 586.4 |
| grasp | Random | -3.4668 | 250 | 0/2 | 509.4 |
| grasp | Zero | -1.7516 | 250 | 0/2 | 569.5 |
| lift | Random | -3.4668 | 250 | 0/2 | 502.1 |
| lift | Zero | -1.7516 | 250 | 0/2 | 577.1 |
| pick_place | Random | -3.4668 | 250 | 0/2 | 510.6 |
| pick_place | Zero | -1.7516 | 250 | 0/2 | 581.6 |

性能为单进程 `env.step` 调用实测，包含 IK、20 子步物理、观测和奖励，不含模型加载/reset、动作采样、打印和未来策略推理。Zero 的 IK 通常立即收敛，因此更快。Home 上方随机短回合没有形成夹持，所以 grasp/lift/pick_place 的新增势没有触发，后三种 Return 相同是预期结果。

## Viewer 与日常命令

human 首次实际测试发现 Cocoa 窗口异步关闭导致第二回合报 `another MuJoCo viewer is already open`，失败原始日志保留在 `viewer.log`，没有把它记为通过。已修复：只对该特定窗口槽位错误最多等待 2 s，每次间隔 0.02 s；不推进物理或任务时间，其他错误照常抛出。

修复后实跑两个完整 100 步 human 回合，每回合仿真 4.0 s，完成 Viewer 关闭与下一窗口重开，最终报告成功生成。最终两个回合的物理 info 和 Return 与同 seed、同动作的 Headless 参考逐字段相等。GUI 已在沙箱外获准运行；未做截图级视觉布局核验。

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 无界面验收
PYTHONPATH=. python scripts/39_test_pick_place_env.py

# 四任务 Random / Zero 完整回合
PYTHONPATH=. python scripts/40_rollout_pick_place_env.py --episodes 2 --max-steps 250

# Mac human；Python 单独启动 human 不满足 MuJoCo 的 Cocoa 主线程要求
PYTHONPATH=. mjpython scripts/40_rollout_pick_place_env.py --tasks reach --policy random --episodes 2 --max-steps 100 --human
```

`--output 新路径.json` 只新建文件，拒绝覆盖已有结果。Headless 不导入 `mujoco.viewer`；实际检查同时确认没有导入 `agents.scripted_pick_place`。

## 限制、瓶颈与 Day 14～16 建议

- 当前四维动作不能在 episode 内直接调节工具朝向；工具姿态按初始 Cube 偏航固定。只验证了当前 Panda、50 mm/0.10 kg Cube、小范围偏航和水平台面，不保证复杂旋转物体或任意工作空间角点可控。
- 这是增量 IK 控制，没有全臂避障规划。碰撞失败目前覆盖手指–桌面；没有把全部机械臂自碰撞或与台面接触都纳入任务失败。
- 原位置 servo 有重力静差，环境没有自动执行 Day 12 的阶段偏移补偿；策略应通过实际 TCP 反馈修正目标。命令可领先实际机器人，观测已暴露该差异。
- 每个物理子步完整 mj_forward 和接触检测、每个控制步 Python IK 是主要计算开销。现有约 500～590 steps/s 足以开始单机接口实验，但不是 PPO 端到端训练吞吐保证。
- 连续观测采用 SI 原值，无穷 Box 边界是有意设计；建议后续 VecNormalize，而非为消除检查器警告截断物理反馈。数值异常终止向量只用于坏状态的吸收终态，不能当作有效训练状态继续控制。
- 初始随机策略 0% 成功不代表环境不能完成任务，也不保证 PPO 会很快学会；必须分任务与课程逐步验证。

Day 14 Reach 的建议起点：

| 参数 | 建议 | 依据 |
|---|---|---|
| task/start_pose | reach/home | 先学固定姿态的末端位置控制 |
| action_scale_m | 0.005 m | 比环境通用默认 0.01 m 更保守，减少参考目标领先 servo；每轴参考最高 0.125 m/s |
| frame_skip | 20 | 已验证的 0.04 s / 25 Hz 控制周期 |
| horizon | 250 steps / 10 s | 足以覆盖简化 Reach 运动和 0.4 s 稳定窗口，基线已实跑 |
| PPO gamma | 0.99 | 必须与 shaping_discount 匹配 |
| 其他 PPO 起点 | lr=3e-4、n_steps=2048、batch_size=64、gae_lambda=0.95、clip_range=0.2 | 仅是后续实验起点，本日未创建或训练 PPO |
| 归一化 | 优先观测 VecNormalize，先保留原始 reward 便于核对分项 | 避免单位差异掩盖任务指标 |

Day 15 可先在每次 reset 注入固定 Cube、较小偏航、`start_pose="pre_grasp"`，使用 0.003～0.005 m 增量，分别验证 grasp/lift 再扩大随机性；可从 500/750 步上限起步。Day 16 pick_place 可保留 1500 步 / 60 s，先固定目标再扩展。不会因学习日安排就承诺复杂操作能在一天内收敛。
