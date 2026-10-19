# Day 13｜PickPlaceEnv 核心知识复习笔记

> 项目：[`qiuxiuhao/embodied_intelligence`](https://github.com/qiuxiuhao/embodied_intelligence)  
> 阶段：Stage 02 — State-based Reinforcement Learning  
> 核对日期：2026-10-19；依据 GitHub `main` 的 Day 13 实际源码及实验记录。后续如有修改，以当前 Git 提交为准。  
> 目标：忘记代码时，用 **问题 → 公式 → 对应函数 → 常见误区** 快速恢复理解。  
> **Day 13 仅构建 Gymnasium RL 环境及物理验收，没有训练 PPO。**

---

## 0｜一分钟速览（先读这里）

**Day 12 与 Day 13 的本质区别：**

- **Day 12：** `Observation → ScriptedPickPlaceAgent（人工 FSM）→ Controller → MuJoCo`。程序预先规定抓取、运输、放置的决策顺序。
- **Day 13：** `Observation → Policy（暂用 random/zero，未来 PPO）→ Action → PickPlaceEnv → MuJoCo → Observation + Reward + Done`。环境不替策略决定抓取阶段。

**必须记住的数值：**

| 项目 | 当前默认 | 代码位置 |
|---|---:|---|
| Action | `Box(-1,1, shape=(4,))` | `__init__()` |
| Observation | `Box(shape=(94,), dtype=float32)` | `OBSERVATION_FIELDS` / `_observation()` |
| 物理步长 | `0.002 s` | MuJoCo 模型 |
| `frame_skip` | `20` | `PickPlaceEnvConfig` |
| RL 控制周期 | `20 × 0.002 = 0.04 s`（25 Hz） | `self.dt` |
| 位置 Action 尺度 | `0.01 m / 维 / RL step` | `action_scale_m` |
| 夹爪目标开度 | `0～0.08 m` | `step()` → `gripper.command()` |
| `max_episode_steps` | `1500` | `PickPlaceEnvConfig` |
| 最长仿真时间 | `60 s` | `max_simulation_time_s` |
| IK 数值容差 | `0.0005 m` + `0.005 rad` | `PickPlaceEnvConfig` |
| 最大连续 IK 失败 | `5` | `max_ik_failures` |
| Shaping 折扣因子 | `0.99` | `RewardWeights.shaping_discount` |

**三条主公式：**

1. **位置命令增量**  
   $$\mathbf p_{cmd,t+1}=\operatorname{clip}_{workspace}(\mathbf p_{cmd,t}+0.01\mathbf a_{xyz,t})$$
2. **夹爪目标开度**  
   $$w_t=0.08\,\frac{a_{g,t}+1}{2}\quad(\mathrm m)$$
3. **折扣势函数塑形**  
   $$F_t=\gamma\Phi(s_{t+1})-\Phi(s_t),\qquad\gamma=0.99$$

**一句话：** 环境定义 *动作怎样产生真实运动、怎样观测、怎样奖励、何时结束*，策略负责 *具体选择什么动作*。

---

## 1｜文件与整体架构

| 文件 | 读代码时重点关注 |
|---|---|
| [`envs/pick_place_env.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/envs/pick_place_env.py) | `PickPlaceEnvConfig`、`OBSERVATION_FIELDS`、`reset()`、`_state()`、`_observation()`、`_physical_conditions()`、`_update_timers()`、`_potentials()`、`_reward()`、`step()` |
| [`scripts/39_test_pick_place_env.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/scripts/39_test_pick_place_env.py) | Gymnasium/SB3 接口、四维动作物理探针、负例、Reset/Seed/数值测试 |
| [`scripts/40_rollout_pick_place_env.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/scripts/40_rollout_pick_place_env.py) | Random/Zero 基线、Episode 指标、Steps/s、Viewer |
| [`notes/day13_pick_place_env.md`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/notes/day13_pick_place_env.md) | Codex 当日实验记录、配置依据、测试结果和局限 |

**复用旧模块而非重造：**

- `PandaIKSolver`：目标 TCP 位姿 → 七关节角；使用 DLS，Day 11 已支持位置 + 姿态。
- `PandaJointPositionController`：关节目标指令限速与伺服目标更新。
- `PandaGripperController`：开度（米）映射到真实 `actuator8`。
- `PandaContactDetector`：真实手指—Cube 接触和 `grasp_candidate`。
- `load_cube_scene()` / `cube_bottom_clearance()`：加载物理场景、计算 Cube 最低点离桌高度。

```text
Policy（未来 PPO）
   │  94维 Observation → 4维 Action
   ▼
PickPlaceEnv.step(action)
   ├─ Action 校验与工作空间限幅
   ├─ 单次六维 IK → 合法七关节目标
   ├─ Gripper 目标开度
   └─ for physics_step in range(frame_skip):
          Joint Controller + Gripper Controller
          mj_step → mj_forward
          _state → _update_timers（实际接触/成功条件）
   ▼
Reward + 94维新 Observation + terminated/truncated + info
```

**特别区别：** Day 12 的 `PoseMotionController.plan()` 会预检整条路点轨迹；Day 13 每个 RL Step 只计算一次新的 TCP 目标 IK，**没有把 Day 12 的 FSM 或整段路点策略放进环境内部**。

---

## 2｜Action：4 维控制怎样变成真实运动？

### 2.1 动作定义

$$\mathbf a=[a_x,a_y,a_z,a_g]\in[-1,1]^4$$

| 维度 | 语义 | 默认范围/换算 |
|---|---|---|
| `a[0:3]` | 世界坐标 TCP 目标增量 | 每轴 `0.01 × a[i] m` |
| `a[3]` | **绝对**夹爪名义开度 | `(a[3]+1)/2 × 0.08 m` |

**前 3 维是增量目标，最后 1 维是绝对目标，不能混淆。**

### 2.2 为什么是相对于上一条“合法命令”，而不是实际 TCP？

代码逻辑：

```python
requested = self.tcp_command + action_scale_m * applied[:3]
desired = np.clip(requested, workspace_low, workspace_high)
```

`self.tcp_command` 保存最后一次 **IK 成功并被接受** 的 TCP 参考位置。假设其世界 X 为 0.50 m，连续输入两次 `dx=+1`，位置参考依次为 0.51、0.52 m；这时物理 TCP 仍可能落后于参考。`dx=0` 表示**保持当前 TCP 参考目标**，不是刹停真实运动。

该设计能避免把执行器重力静差每步重复加到目标上，但会形成 **命令领先实际关节** 的跟踪差。Observation 特地暴露 `tcp_command`、`joint_command` 和 `joint_target`，用于描述控制器状态。

### 2.3 `action=np.zeros(4)` 究竟是什么意思？

位置三维 `0` 表示保持 TCP 位置命令；但是：

$$w=(0+1)/2\times0.08=0.04\mathrm m$$

**Zero Policy 会把夹爪目标设为 4 cm 半开，不是“全部执行器保持不变”。**

### 2.4 IK 失败会怎样？

1. `PandaIKSolver.solve(desired, initial_q=self.arm.command_q, target_rotation=...)` 返回候选关节角。
2. 环境验证 `success`、关节维度/有限性/限位、位置/姿态残差。
3. **成功：** 更新 `joint_target` 和 `tcp_command`，连续失败计数清零。
4. **失败：** 不提交新的位置/关节目标，保持上一次合法目标；夹爪仍可响应当前合法开度命令；记录 `ik_reason` 并给出惩罚。
5. 连续失败 `≥5`，以 `IK_FAILED` 终止回合。

工作空间 `workspace_low/high` 只是轴对齐的 XYZ 范围，**不等于六维 IK 的全部可达空间**。

---

## 3｜Observation：94 维速查表

**构建流程：**

```text
MuJoCo 当前状态
  └─ _state() → 有名称的内部物理字典（含 ContactSnapshot）
       └─ _observation() → 固定 94 维 float32 向量
            └─ MlpPolicy
```

`OBSERVATION_SLICES` 自动由 `OBSERVATION_FIELDS` 计算，不建议自己在训练代码里到处硬编码切片。

| Python 切片 | 字段 | 维度 | 单位 / 作用 |
|---|---|---:|---|
| `0:7` | `arm_qpos` | 7 | 七关节角度，rad |
| `7:14` | `arm_qvel` | 7 | 七关节角速度，rad/s |
| `14:17` | `tcp_position` | 3 | TCP 世界 XYZ，m |
| `17:23` | `tcp_rotation_6d` | 6 | TCP 姿态 6D |
| `23:26` | `cube_position` | 3 | Cube 世界 XYZ，m |
| `26:32` | `cube_rotation_6d` | 6 | Cube 姿态 6D |
| `32:38` | `cube_velocity` | 6 | **线速度在前、角速度在后**（世界系） |
| `38:40` | `finger_positions` | 2 | 两指实际位置，m |
| `40:42` | `finger_velocities` | 2 | 两指速度，m/s |
| `42:43` | `gripper_opening` | 1 | 两指名义总开度，m |
| `43:46` | `tcp_to_cube` | 3 | `cube - tcp`，m |
| `46:49` | `cube_to_target` | 3 | `target - cube`，m |
| `49:52` | `target_position` | 3 | 任务目标 XYZ，m |
| `52:55` | `finger_contacts` | 3 | 左/右有效接触、双侧夹持候选（0/1） |
| `55:56` | `cube_table_contact` | 1 | Cube 是否有桌面受力接触（0/1） |
| `56:60` | `task_one_hot` | 4 | `reach / grasp / lift / pick_place` |
| `60:63` | `tcp_command` | 3 | 上一次合法 TCP 参考目标，m |
| `63:70` | `joint_command` | 7 | 关节限速器当前命令，rad |
| `70:77` | `joint_target` | 7 | 上一次合法 IK 关节目标，rad |
| `77:81` | `previous_action` | 4 | 前一四维 Action |
| `81:87` | `target_rotation_6d` | 6 | 工具目标姿态 |
| `87:91` | `stable_timers` | 4 | 四个连续稳定计时的归一化比例 |
| `91:92` | `has_lifted` | 1 | 曾经真实稳定抬升过（0/1） |
| `92:93` | `ik_failure_fraction` | 1 | 连续 IK 失败比例 |
| `93:94` | `remaining_fraction` | 1 | 剩余 Episode 预算比例 |
| **总计** | | **94** | `float32` |

### 3.1 为什么姿态使用 6D？

由旋转矩阵 `R` 的前两列组成：

$$r_{6D}=[R[:,0];\ R[:,1]]\in\mathbb R^6$$

四元数有 `q` 和 `-q` 对应同一旋转的表示歧义；6D 表示在神经网络输入上更连续。**6D 表示不意味着姿态有六个独立自由度。**

### 3.2 `mj_objectVelocity()` 为什么要重排？

MuJoCo 此 API 给 Cube Body 的六维速度时，先角速度后线速度；`flg_local=0` 表示用世界坐标系表达。环境重新拼成：

$$[v_x,v_y,v_z,\omega_x,\omega_y,\omega_z]$$

避免把 MuJoCo 自由关节内部速度表达与世界坐标观测混淆。

### 3.3 为什么 Observation 包含历史量？

下一步控制目标取决于 **前一合法命令**，成功判定取决于 **连续稳定时间与 `has_lifted`**。如果只给位置和速度，策略不能完整推断这些内部任务状态。历史变量来自已经发生的事实，不是未来信息，也不是 Scripted Agent 的专家动作。

**注意：** 连续物理量采用 SI 原值，`observation_space` 中部分 Box 上下界为 ±∞；这不是说允许 NaN/Inf。`_observation()` 会拒绝所有非有限值。Day 14 可用 `VecNormalize` 做归一化，但不能随意把真实速度等物理量截断以迎合检查器。

---

## 4｜Reset：一个新 Episode 必须清除哪些信息？

实际接口：

```python
obs, info = env.reset(seed=1234, options={"start_pose": "home"})
```

1. `super().reset(seed=seed)`：建立 Gymnasium 随机源，随机采样统一使用 `self.np_random`。
2. 恢复含 Cube 自由关节的完整 `day11_home` keyframe；场景 `nq=16, nv=15, nu=8`。
3. 在 **Reset 阶段**设置 Cube 初始 XYZ 和绕世界 Z 轴的 Quaternion：
   $$q_{yaw}=[\cos(\psi/2),0,0,\sin(\psi/2)]$$
4. 在安全桌面内随机采样 Cube 偏移、偏航及对应任务目标；检查 Cube 全部占地投影不越桌。
5. 选择 `home` 或 `pre_grasp` 起点。后者可在 Reset 内用 IK 设定预抓取初始关节状态，**不是执行了一段物理轨迹**。
6. 重置伺服命令、上一 Action、四个稳定计时器、`has_lifted`、IK 连败计数、回合步数、累计 Return、势函数历史。
7. 返回 `(obs, info)`；Reset 自身不推进物理仿真时间。

可用 `options`：`task_mode`、`cube_position`、`cube_yaw`、`target_position`、`start_pose`。其中 `grasp/lift` 的目标由 Cube 和配置推导，不接受无关的任意位置目标。

**记忆点：** `reset()` 可以修改 Cube 的初始 `qpos`，但 `step()` 中的 Cube 只能通过真实 MuJoCo 接触和动力学移动，不能瞬移、weld 或跟随 mocap。

---

## 5｜Step：一次 RL Action 是怎样执行的？

`step(action)` 的真实主线：

```text
校验 action：shape=(4,)、有限实数
     ↓
clip 到 [-1,1]；累计 TCP 参考位置
     ↓
尝试六维 IK（固定目标工具方向）
     ↓
通过才更新合法 joint_target / tcp_command
     ↓
夹爪 action[3] → 开度目标
     ↓
for 20 个 MuJoCo 物理子步：
    arm.step → gripper.command → mj_step → mj_forward
    _state → _update_timers → 物理故障检查
     ↓
当前任务成功? 物理失败? 回合超时?
     ↓
_reward → _observation → _info
     ↓
obs, reward, terminated, truncated, info
```

### 5.1 RL 时间尺度与物理时间尺度

$$\Delta t_{physics}=0.002s,\quad N=20$$

$$\boxed{\Delta t_{RL}=N\Delta t_{physics}=0.04s,\quad f_{RL}=25Hz}$$

- 策略 25 Hz 决策；低层物理 500 Hz 推进。
- **连续稳定时间在每个物理子步累加**，避免一个 RL Step 内接触中断却被忽略。
- 常规失败也尽量完成固定子步；仅严重数值损坏会提前停止并返回明确 `NUMERICAL_ERROR`。

### 5.2 `mj_step()` 后为什么再 `mj_forward()`？

`mj_step()` 积分推进状态；当前实现在积分后 `mj_forward()` 更新 FK、接触和动力学派生量，再由 `_state()` 读取最新物理反馈。两者承担不同职责。

### 5.3 为什么 `_state()` 和 `_observation()` 分开？

- `_state()`：便于程序按**字段名称**读取真实物理状态和 ContactSnapshot。
- `_observation()`：把字段按固定索引打包给神经网络。
- `_info()`：输出可诊断且 JSON 友好的指标，`reward_terms` 能核对各项奖励。

---

## 6｜四种任务模式：什么才算真正成功？

在 `_physical_conditions()` 中只判断**物理条件**，不调用 Day 12 的 FSM。

| Task | 当前真实条件 | 必须连续保持 |
|---|---|---:|
| **Reach** | TCP 位置误差 `<9 mm`；姿态误差 `<0.035 rad`；七关节最大速度 `<0.10 rad/s` | `0.40 s` |
| **Grasp** | `grasp_candidate=True`（双侧几何与接触力符合 Day 11 判据）；Cube 线速度 `<0.02 m/s`、角速度 `<0.20 rad/s`；机械臂速度 `<0.10 rad/s` | `0.50 s` |
| **Lift** | 同时满足 Grasp；无桌面/地面接触；**Cube 最低点离桌 ≥4 cm** | `0.50 s` |
| **PickPlace** | 曾真实稳定 Lift；Cube 最终 XY 误差 `<12 mm`、Z 误差 `<4 mm`、最低点与桌面高度差 `<4 mm`；桌面支撑、无地面/手指接触；夹爪开度 `≥7 cm`；TCP 离 Cube `≥8 cm`；Cube 速度合格 | `0.50 s` |

连续计时：

$$T_{t+1}=\begin{cases}\min(T_t+\Delta t_{physics},T_{required}),&\mathrm{condition}=True\\0,&\mathrm{condition}=False\end{cases}$$

`has_lifted` 只有在 **Lift 连续稳定达标** 时才置 `True`。单次碰撞把 Cube 弹起、夹爪只闭合但没有真实受力、物体尚未释放等，都不能当作 PickPlace 成功。

**易混淆：** `lift_goal_height_m=0.10` 是参考目标/奖励高度势饱和尺度，**硬性 Lift Success 门槛仍是 `lift_clearance_m=0.04`**。

---

## 7｜Reward Shaping：为什么要用势函数差？

对应 `_potentials()`、`_reward()` 和 `RewardWeights`。

### 7.1 问题：只有成功奖励为什么难训练？

初期随机策略极难抓取，若只有最终成功 `+10`，多数轨迹中间奖励全为 0。**密集奖励**需要告诉 PPO 动作是否让状态逐步接近目标，同时避免重复接触刷分。

### 7.2 当前势函数（快速公式表）

令：

- `S = 0.20 m`：距离归一化尺度。
- `dR = ||TCP - reach_target||`、`dA = ||TCP - Cube||`、`dP = ||Cube - place_target||`。
- `h`：Cube 最低点离桌高度；`g`、`p`：Grasp 和 Place 的稳定进度（都在 `[0,1]`）。
- `I(condition)`：条件为真取 1，否则取 0。

$$\Phi_R=\frac{2}{1+\operatorname{clip}(d_R/S,0,5)}$$

$$\Phi_A=\frac{1}{1+\operatorname{clip}(d_A/S,0,5)}$$

$$\Phi_G=0.5g$$

$$\Phi_H=2\operatorname{clip}(h/0.10,0,1)\cdot I(\text{GraspCandidate 且 Cube 不接触桌面})$$

$$\Phi_P=2I(\text{has_lifted})\left[1-\operatorname{clip}(d_P/S,0,1)\right]$$

$$\Phi_{release}=p$$

任务启用的塑形：

| Task | 势函数项 |
|---|---|
| `reach` | Reach |
| `grasp` | Approach + Grasp |
| `lift` | Approach + Grasp + Lift |
| `pick_place` | Approach + Grasp + Lift + Place + Release |

### 7.3 为什么不是单纯 `Phi(next)-Phi(current)`？

当前实现：

$$\boxed{F_t=\gamma\Phi(s_{t+1})-\Phi(s_t),\quad\gamma=0.99}$$

若上一时刻势为 1.0，下一时刻势为 1.2：

$$F=0.99\times1.2-1.0=0.188$$

当状态不进步、势始终为 1.0 时：

$$F=0.99-1.0=-0.01$$

这样可以抑制只保持某个局部状态却不断获得正奖励的行为。在 PPO 使用相同 `gamma=0.99` 时，折扣势差会望远镜式抵消：

$$\sum_{t=0}^{T-1}\gamma^tF_t=-\Phi(s_0)+\gamma^T\Phi(s_T)$$

对**真正终止**的下一状态势取 0；对**时间截断**不当作真实终止，保留下一状态势，用于合理 Bootstrap。

> 注意：这属于势函数塑形的设计。当前 Reward 还包含动作/距离/碰撞等其他成本，因此不能据此保证整个 Reward 对最优策略完全无影响。

### 7.4 还加了哪些 Reward？

| 项 | 意义 |
|---|---|
| `distance_cost` | 持续远离目标的距离成本，乘实际物理控制时长 |
| `orientation_cost` | 工具姿态偏差成本 |
| `action_cost` | 动作幅度成本 |
| `action_change_cost` | 动作相对上一动作变化过大 |
| `clipped_action_cost` | 有限动作越界被裁剪 |
| `ik_failure_cost` | 当前 IK 无效 |
| `collision_cost` | 手指与桌面接触 |
| `success_bonus` | 真正成功 `+10`，一次性 |
| `failure_penalty` | 明确失败 `-10`，超时不额外扣该项 |

`reward = sum(info["reward_terms"].values())`。IK 失败或明确物理失败时，正向 Reward 分项会被截为非正值，防止无效指令借惯性移动得到正奖励。

---

## 8｜`terminated` 与 `truncated`：不要合并为一个 `done`

| 结束情形 | `terminated` | `truncated` | 原因 |
|---|---|---|---|
| 达到真实任务成功条件 | `True` | `False` | `SUCCESS` |
| Cube 跌落地面/超出台面下方阈值 | `True` | `False` | `OBJECT_DROPPED` |
| 手指与桌面发生被禁止的有效接触 | `True` | `False` | `FINGER_TABLE_COLLISION` |
| 连续 5 次 IK 失败 | `True` | `False` | `IK_FAILED` |
| 严重非有限状态或 MuJoCo 数值警告 | `True` | `False` | `NUMERICAL_ERROR` |
| 达到最大步数/仿真时间 | `False` | `True` | `TIME_LIMIT` |

**核心区别：**

- `terminated`：环境定义的真正终态，后续没有常规任务价值。
- `truncated`：外部时间/步数上限截断，**不应简单把未来价值当零**。

当前实现优先真实终态，`terminated` 和 `truncated` 不会同时为 `True`。结束后必须再次 `reset()`，否则 `step()` 报错。

---

## 9｜已验证结果 vs 尚未证明的事情

以下是仓库 `notes/day13_pick_place_env.md` 中的 **Codex 实验记录**，不是本次生成笔记时重新执行的测试。

### 9.1 环境验收

- 12 项最终验收通过；四种任务的 Gymnasium/SB3 `check_env`、固定 Seed、独立环境实例、Observation 合法性、Reward 分项、Reset、物理推进、故障注入、Viewer 等完成验证。
- Day 11/12 既有测试也记录通过。
- 使用**环境外部的真值动作探针**通过四维 `env.step()` 完成：

| 物理探针任务 | RL Steps | 仿真时间 | 说明 |
|---|---:|---:|---|
| Grasp | 36 | 1.44 s | 双侧接触稳定 |
| Lift | 69 | 2.76 s | 最低点离桌约 0.11544 m，稳定持握 |
| PickPlace | 134 | 5.36 s | 最终 Cube 中心误差约 4.271 mm，实际释放与撤离 |

这里的探针在**环境外部**提供动作顺序，只证明「四维控制通路可以完成任务」，**不是 PPO 或 Random Policy 学会了任务**。

### 9.2 Random / Zero 基线

- `4 tasks × 2 policies × 2 seeds = 16` 个回合。
- 每回合设置 `250` RL Steps，等于 `250 × 0.04 = 10 s` 仿真时间。
- **16/16 回合均以 TIME_LIMIT 截断，成功率 0%。**
- 记录的 Headless 单进程环境吞吐约 `500～590 env steps/s`，包含 IK/物理/观测/奖励，不包含后续 PPO 更新开销。

**不要混淆：** 环境完成接口验收 ≠ 环境随机策略表现良好 ≠ PPO 训练已收敛。

---

## 10｜下次遗忘时最该重读的 12 个问题

1. **Day 12 和 Day 13 最大区别？** 前者 FSM 选动作，后者环境只执行/评估动作，动作交给 Policy。
2. **4 维 Action 是什么？** 世界 XYZ **增量目标** + 夹爪**绝对开度**。
3. **增量相对于实际 TCP 吗？** 不是，是上一条**合法 `tcp_command`**。
4. **Zero Policy 等于保持夹爪不动吗？** 不是，`a_g=0` 意味着目标开度 0.04 m。
5. **为什么固定工具姿态？** 当前任务简化了 Action，不让 PPO 控制旋转；姿态由初始 Cube 偏航决定。
6. **为什么 94 维包含命令和计时器？** 环境内部控制状态会影响未来行为，观测需要表达它们。
7. **为什么 Rotation 6D？** 减少 Quaternion `q/-q` 表示歧义，不是六个旋转自由度。
8. **为什么 `frame_skip=20`？** PPO 25 Hz 决策，MuJoCo 500 Hz 物理；每动作推进 0.04 s。
9. **为什么 GraspCandidate 不能直接算 Lift？** 还须物体真的离桌、没有桌面支撑并持续稳定。
10. **为什么 `has_lifted` 要锁存？** 完整放置必须能证明 Cube 过去真正抬起过，避免初始就在目标位置直接成功。
11. **为什么 Reward 用折扣势差？** 通过状态变化提供密集学习信号，同时抑制反复接触刷奖励；训练 `gamma` 必须对齐。
12. **为什么 `truncated` 和 `terminated` 不一样？** 时间截断可能仍有后续价值，应按 RL 算法规则 Bootstrap。

---

## 11｜代码阅读顺序（10～20 分钟复习）

1. `PickPlaceEnvConfig` + `OBSERVATION_FIELDS`：理解四任务与全部配置。
2. `__init__()`：看 MuJoCo、IK/Servo/Gripper、`action_space`、`observation_space`。
3. `reset()`：检查 Keyframe、Seed、起始姿态、目标和记忆清零。
4. `step()`：沿 Action → IK → 20×Physics → Reward → Obs 读一遍。
5. `_state()` + `_observation()`：核对 94 维切片来源。
6. `_physical_conditions()` + `_update_timers()`：核对四种任务真实成功标准。
7. `_potentials()` + `_reward()`：重点复习 `γΦ(next)-Φ(current)`。
8. 测试脚本：区分四维物理探针与 Random/Zero 基线。

---

## 12｜复现命令（在仓库根目录）

```bash
conda activate embodied

# Day 13 Gymnasium / SB3 和真实 MuJoCo 环境验收
PYTHONPATH=. python scripts/39_test_pick_place_env.py

# Random + Zero，四任务，每种策略各 2 回合，250 步上限
PYTHONPATH=. python scripts/40_rollout_pick_place_env.py --episodes 2 --max-steps 250

# Mac 上的 Viewer：必须由 mjpython 启动
PYTHONPATH=. mjpython scripts/40_rollout_pick_place_env.py \
  --tasks reach --policy random --episodes 2 --max-steps 100 --human
```

不要在未检查既有结果时复用实验输出路径；`40_rollout_pick_place_env.py --output` 会拒绝覆盖已存在的结果文件。

---

## 13｜接下来 Day 14（PPO Reach）要关注什么？

- 先只训练 `reach`，不要一开始把四种任务一起混训。
- 使用 `MlpPolicy`、合适的 `VecMonitor` 和可选 `VecNormalize`；保存/复用归一化统计量。
- 配置 PPO `gamma=0.99`，与环境 `shaping_discount=0.99` 一致。
- Day 13 笔记建议 Reach 起点：`action_scale_m=0.005`、`frame_skip=20`、`horizon=250`；这**是建议的训练超参，不是 Day 13 的通用默认值**。
- 分别记录：Success Rate、距离误差、Episode Length、Return、IK Failure Rate、Headless Steps/s；不要只看平均 Return。
- 当前 Action 不能调整 TCP 朝向，且没有全臂避障规划，不能把特定桌面 Cube 的成功外推为任意物体抓取能力。

### 最终记忆锚点

> **Day 13 把已有的机器人控制与真实物理接触，封装为标准 RL 接口：`94维 Observation → 4维 Action → IK/Servo/MuJoCo → Reward/Success/Done`。**
>
> **环境“能够完成任务”由外部物理探针证明；策略“已经学会任务”必须在后续训练和独立评估中证明。**

