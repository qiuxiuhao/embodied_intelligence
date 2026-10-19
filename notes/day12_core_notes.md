# Day 12｜Scripted Pick-and-Place Agent 核心复习笔记

> 项目：[`qiuxiuhao/embodied_intelligence`](https://github.com/qiuxiuhao/embodied_intelligence)  
> 阅读依据：2026-10-18 再次获取的 GitHub `main` 分支 Day 12 源码与实验记录。**这不是独立重跑仿真所得的测试报告**。若以后源码更新，请以新版本为准。  
> 目标：忘记代码时，先用本文快速恢复“模块怎么连接、算法为什么这样设计、真实成功如何判断”的知识。

## 0. 十分钟速查：先记住这张图

```text
scripts/38_panda_pick_place.py        # 命令行、目标、Headless / Viewer
               │
               ▼
robotics/pick_place_runtime.py        # 推进物理、同步数据、日志
               │
               ▼
agents/scripted_pick_place.py       # FSM：观察 → 决策 → 写控制命令
        ┌──────┴────────┐
        ▼               ▼
controllers/       controllers/
pose_motion.py     gripper_controller.py
路点 + 六维 IK +      名义开度 → actuator8
关节平滑插值
        └──────┬────────┘
               ▼
          MuJoCo physics
               │
               ▼
FK / Cube state / Contact / Gripper feedback
               │
               └──────────────► 下一物理步的 Agent
```

**四个不等号：**

1. **IK 成功 ≠ 实际 TCP 到达。** IK 是运动学解，伺服物理跟踪另有误差。
2. **轨迹计时结束 ≠ TCP 达标。** `motion.finished` 只说明参考轨迹执行完。
3. **Grasp Candidate ≠ 稳定抬升。** 双侧接触还要经受重力、摩擦、时间检验。
4. **RELEASE 成功 ≠ 整个 Pick-and-Place 成功。** 必须撤离并再次检查 Cube。

**核心的三个等式：**

\[
s(u)=3u^2-2u^3, \qquad 0\leq u\leq1
\]

\[
\mathbf d^{TCP}=R_{TCP}^{T}(\mathbf p_{cube}-\mathbf p_{TCP})
\]

\[
\boxed{\mathbf p_{TCP,target}=\mathbf p_{cube,target}-R_{TCP,target}\mathbf d^{TCP}}
\]

---

## 1. 源文件导航：出现疑问先定位哪个文件？

| 文件 | 负责什么 | 优先阅读的入口 |
|---|---|---|
| [`agents/scripted_pick_place.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/agents/scripted_pick_place.py) | 任务参数、FSM、成功/失败判断 | `PickPlaceConfig`、`observe`、`_transition`、`step`、`result` |
| [`controllers/pose_motion.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/controllers/pose_motion.py) | TCP 路点、六维 IK、平滑的关节参考轨迹 | `plan`、`step`、`calibrate_tracking_bias` |
| [`robotics/pick_place_runtime.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/robotics/pick_place_runtime.py) | 有限步物理循环、日志、Viewer | `run_pick_place` |
| [`scripts/37_test_pick_place.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/scripts/37_test_pick_place.py) | 完整任务和物理故障注入验收 | `PickPlaceTests` |
| [`scripts/38_panda_pick_place.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/scripts/38_panda_pick_place.py) | 命令行与运行入口 | `main` |
| [`notes/day12_scripted_pick_place.md`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/notes/day12_scripted_pick_place.md) | Day 12 的完整工程记录、测试证据 | “状态机与物理转移”“实际测试命令与结果” |

**依赖链：** Day 12 `PoseMotionController` 复用 Day 10 的 `PandaIKSolver` 与 `PandaJointPositionController`；Agent 又复用 Day 11 的 `PandaGripperController`、`PandaContactDetector` 和 `cube_bottom_clearance`。**Day 12 没有重新实现 FK/IK、夹爪传动或 Contact 检测。**

---

## 2. FSM（有限状态机）：做什么、何时才允许进入下一步？

```text
IDLE → PRE_GRASP → APPROACH → CLOSE_GRIPPER → VERIFY_GRASP
     → LIFT → TRANSPORT → PLACE → RELEASE → RETREAT → SUCCESS

所有非终态：物理异常 / IK 失败 / 阶段超时等 → FAILURE
```

| State | 进入时的动作 | 向下一状态转移的真实证据 |
|---|---|---|
| `IDLE` | 保持合法关节指令，打开夹爪 | Cube 在桌面、Cube/机械臂静止、夹爪实际开度达标，持续 0.25 s |
| `PRE_GRASP` | 规划 Cube 上方约 0.12 m 的朝下位姿 | 参考轨迹结束 **且**实际 TCP 位置、姿态、速度、夹爪开度持续达标 |
| `APPROACH` | 沿路点缓慢下降到抓取 TCP 目标 | 与 PRE_GRASP 同类的实际到达条件持续达标 |
| `CLOSE_GRIPPER` | 保持手臂命令，闭合夹爪 | `grasp_candidate` 连续 0.25 s |
| `VERIFY_GRASP` | 继续夹持、暂不抬起 | `grasp_candidate` + Cube 近似静止，连续 0.25 s |
| `LIFT` | 记录 TCP–Cube 偏移，抬高 TCP 约 0.12 m | TCP 到达、双侧夹持、Cube 无桌/地支撑、最低点离桌 ≥ 0.04 m、Cube 近似静止，连续 0.5 s |
| `TRANSPORT` | 按 Cube 目标中心反算 TCP 目标，运到放置点上方 | Cube XY 接近目标，仍被夹住、离桌、静止，持续 0.5 s |
| `PLACE` | 重新估计夹持偏移、缓慢下放 | Cube 在目标区域、真正受桌面支撑、Cube 近似静止，持续 0.5 s |
| `RELEASE` | 保持手臂、打开夹爪 | 实际开度达标、已脱离手指、由桌面支撑、落点正确且静止，持续 0.5 s |
| `RETREAT` | 机械臂上撤约 0.12 m | TCP 已到撤离位姿，距 Cube ≥ 0.08 m，且释放/放置条件**再次**稳定成立 |
| `SUCCESS` | 终态 | `result()` 还会做一次最终物理状态复核 |
| `FAILURE` | 终态 | 保存 `failure_reason/failure_stage`，不自动重抓 |

**容易忘的概念：状态转移的“进入动作”在 `_transition()`，持续判断逻辑在 `step()`。** 例如进入 `LIFT` 时规划抬升轨迹；随后 `step()` 每个物理步检查“是否真的抬起”。

### `_stable()`：为什么不能只看某一帧？

累计方式：

\[
T_{k+1}=\begin{cases}
T_k+\Delta t,&\text{条件持续为真}\\
0,&\text{条件为假}
\end{cases}
\]

只有 `T >= duration_s` 才通过。物理接触可能瞬时抖动，因此不能因为某一帧 `grasp_candidate=True` 就宣告稳定抓取。

`step()` 还会拒绝在**同一仿真时刻**重复计时；计时依赖 `data.time` 而不是函数调用次数。这防止重复调用 `step()` 人为累计“稳定时间”。

### `_reached()`：四个条件必须同时成立

\[
\begin{aligned}
&\text{motion.finished}\\
&\|p_{actual}-p_{target}\|<0.009\ \mathrm m\\
&\|e_R\|<0.035\ \mathrm{rad}\\
&\max_i|\dot q_i|<0.10\ \mathrm{rad/s}
\end{aligned}
\]

其中姿态误差复用 `rotation_error_world()`：

\[
e_R=\mathrm{Log}(R_{target}R_{current}^{T})^\vee
\]

**参考运动结束与真实运动到达是两件事。**

---

## 3. PoseMotionController：为什么要在 IK 外再加一层？

### 3.1 `plan()`：起点 → TCP 路点 → IK → 关节轨迹

Day 10 的 IK 只会给出一个目标关节配置。Day 12 还需要沿途运动，所以先把末端路径切为小段。

\[
N=\max\left(1,\left\lceil\frac{\|p_{end}-p_{start}\|}{d}\right\rceil\right)
\]

其中 `d=0.01 m`（路点间距）；每个路点：

\[
p_i=p_{start}+\frac{i}{N}(p_{end}-p_{start})
\]

1. 起点由**上一条合法关节命令**做 FK 得到，而不直接以瞬时实际位置重新生成参考轨迹。
2. 对每个 TCP 路点调用**六维 IK**：同时指定 XYZ 与目标旋转矩阵。
3. 对 IK 结果再独立做一次 FK 检验位置与姿态误差。
4. **只有全部路点合格，才用新的轨迹替换旧轨迹。** 任何路点 IK 失败，抛 `MotionPlanningError`，不会发送半条新轨迹。

这一点是“**先预检，后提交**”；但它不是全臂避障或全局运动规划。

### 3.2 Cubic Smoothstep：为什么是 `u²(3−2u)`？

每段关节插值：

\[
q(u)=q_0+s(u)(q_1-q_0),\qquad s(u)=3u^2-2u^3
\]

\[
s(0)=0,\ s(1)=1,\ s'(0)=0,\ s'(1)=0
\]

即起点/终点处的参考速度为零，减小接缝处突变。导数：

\[
s'(u)=6u(1-u),\qquad \max_{u\in[0,1]}s'(u)=1.5
\]

所以参考速度上限：

\[
|\dot q_j|_{max}=\frac{1.5|q_{1,j}-q_{0,j}|}{T}
\]

为限制参考速度，段时长必须至少满足：

\[
T\ge\max\left(0.04,\frac{1.5\|p_i-p_{i-1}\|}{v_{TCP}},\frac{1.5\max_j|\Delta q_j|}{v_{joint}}\right)
\]

这解释 `segment_s` 中两次出现的 **1.5**。当前速度参数：关节参考最大速度 0.35 rad/s；TCP 运输/接近/放置参考速度分别为 0.06/0.035/0.025 m/s。

**易错点：** 路点的 TCP 位置近似沿直线分布，但相邻路点之间是**关节空间**平滑插值，真实 TCP 路径并非数学上严格直线。限速的是参考指令，实际关节速度需另行观测。

### 3.3 静态跟踪偏差 `calibrate_tracking_bias()`

Day 11 已记录约 6.5 mm 的 TCP 静态跟踪偏差。Day 12 在一个阶段稳定到达后估计：

\[
b=p_{actual}-p_{IK,target}
\]

下一段用：

\[
\boxed{p_{IK,target}=p_{desired}-b}
\]

直觉：命令到 0.40 m，实际只能到 0.394 m，则 `b=-0.006m`；下一次求解 0.406 m 的参考目标，尝试抵消稳态误差。当前按分量限幅 ±0.015 m。

**注意：** 只是相似工况下的小误差前馈补偿，不是动力学补偿，也不会直接改变 Cube 的位置。

---

## 4. 抓取方向与 TCP–Cube 坐标变换

### 4.1 目标抓取姿态如何生成？

进入 `PRE_GRASP` 时：

- 从 Cube 旋转矩阵取一个局部水平轴投影作为夹爪**闭合方向** `closing`；若所选轴近乎竖直，改用另一个轴。
- 令工具局部 `+Z` 朝世界 `-Z`（向下）。
- 构造一个右手正交工具姿态：

\[
\mathbf y_{tool}=\mathrm{closing},\qquad
\mathbf z_{tool}=(0,0,-1)^T,\qquad
\mathbf x_{tool}=\mathbf y_{tool}\times\mathbf z_{tool}
\]

\[
R_{target}=[\mathbf x_{tool},\ \mathbf y_{tool},\ \mathbf z_{tool}]
\]

这样不是只要求 TCP 到 XYZ，也要求夹爪以可接触 Cube 两侧的姿态接近。当前实现仍是针对盒形物体的专用抓取姿态规则。

### 4.2 为什么不能把目标 Cube 中心直接当成目标 TCP？

真实夹持后，Cube 中心和 TCP 原点往往不重合。首先在**世界系**计算：

\[
d^W=p_{cube}^W-p_{TCP}^W
\]

然后将该相对向量转换到 TCP 局部坐标系：

\[
\boxed{d^{TCP}=R_{TCP}^{T}(p_{cube}^W-p_{TCP}^W)}
\]

`R.T` 是逆旋转（旋转矩阵满足 `R^{-1}=R.T`），表示从世界系变换回 TCP 系。

若希望 Cube 中心到达 `p_cube,target`，且 TCP 目标方向为 `R_target`，由：

\[
p_{cube,target}=p_{TCP,target}+R_{target}d^{TCP}
\]

得到：

\[
\boxed{p_{TCP,target}=p_{cube,target}-R_{target}d^{TCP}}
\]

**核心区别：** 目标是物体的中心，控制对象是机器人 TCP。必须做一次坐标转换，不能忽略中间的相对偏移。

### 4.3 为什么在 LIFT、TRANSPORT、PLACE 重新测量偏移？

Cube 依靠摩擦夹持，可能存在微小下滑。每段重新记录偏移，可以减轻累积位置误差；但并非允许无限滑动。持物期间 `_check_slip()` 仍然约束偏移变化及夹持接触状态。

### 4.4 放置中心高度为什么为 0.225 m？

场景桌面顶面 `z=0.20m`，Cube 半高 `0.025m`：

\[
\boxed{z_{cube,place}=z_{table,top}+h_{cube,half}=0.225m}
\]

目标是**Cube 中心**，不是 TCP，也不是桌面高度。下放阶段使用少量 0.001 m 的预压参考以建立桌面支撑接触，但最终仍必须由真实位置、速度和受力验证。

---

## 5. 接触、滑脱、抬升与最终成功：必须分层验证

### 5.1 四个状态不能等同

| 概念 | 意味着什么 | 不意味着什么 |
|---|---|---|
| `finger_contact` | 至少一个手指与 Cube 有有效受力接触 | 已形成稳定夹持 |
| `grasp_candidate` | 左右手指双侧有效接触、位置与法向符合夹持几何 | Cube 一定能被长期提起 |
| `lift_success` | Cube 真实离桌、未滑脱、低速且持续稳定 | 已运到目标位置 |
| `release_success` | Cube 落在目标内、桌面支撑、夹爪真实打开且不再接触 | 撤离之后也一定合格 |

Day 12 **没有**因为控制命令已发送就直接判成功。

### 5.2 `_check_slip()`：滑脱双证据

每次读取当前局部偏移：

\[
d_{now}^{TCP}=R_{TCP}^{T}(p_{cube}-p_{TCP})
\]

若满足任意条件：

1. 双侧夹持候选消失；
2. \(\|d_{now}^{TCP}-d_{reference}^{TCP}\|>0.018\,m\)；

就累计 `lost_s`，**持续 ≥ 0.12 s** 才报 `OBJECT_SLIPPED`。短暂 Contact 波动不立即视为失败。

在 `PLACE` 且 Cube 已接触桌面后，正常释放时手指可以失去接触，因此此时不再按“空中滑脱”处理。

### 5.3 LIFT Success：为什么还需要 `clearance`？

不能只看 Cube 中心 Z 是否升高。Cube 可能翻转或部分仍接触桌面。Day 11 的 `cube_bottom_clearance()` 计算 Cube 旋转后**最低点**相对于桌面的离地高度。

LIFT 主要要求：

- 真实 TCP 到达目标位置及姿态；
- 仍有合理双侧夹持；
- Cube 不再与桌面/地面发生有效支撑接触；
- `clearance >= 0.04 m`；
- Cube 线速度、角速度低于阈值；
- 上述条件连续 `0.50 s`。

### 5.4 RELEASE 为什么不等于 SUCCESS？

`_released_and_placed(obs)` 要求：

- Cube XY/Z 和底部 clearance 满足放置门槛；
- Cube 线速度 `< 0.02 m/s`、角速度 `< 0.20 rad/s`；
- Cube 受桌面支撑，**不**受地面支撑；
- 无手指有效接触；
- 夹爪实际开度 `>= 0.07 m`。

即使 RELEASE 已经通过，**RETREAT 期间**机械臂还可能意外推走 Cube。只有机械臂撤离且最终物理条件继续稳定，才进入 SUCCESS。`result()` 又会进行一次独立复核；如终态已不合法，返回 `FINAL_STATE_INVALID`。

---

## 6. Runtime：谁负责执行 `mj_step()`？

**不是** Agent，也**不是** `PoseMotionController`。

`run_pick_place()` 的每个循环按以下顺序：

```text
agent.step(data)         # 观察当前物理状态、判断状态机、写下一步 ctrl
        ↓
可选 before_physics()    # 只用于物理故障注入测试；正常演示不传
        ↓
mujoco.mj_step()         # 真正推进 MuJoCo 物理一步
        ↓
mujoco.mj_forward()      # 刷新位姿、接触、约束与力相关数据
        ↓
下一次 agent.step(data)
```

这样 `observe()` 能在一致的仿真状态上读取 TCP、Cube 及 Contact。外部物理循环还设置 `ceil(total_timeout / timestep)+2` 步的独立上限，避免失效时死循环。

**Headless 和 Viewer 使用相同的物理逻辑。** Viewer 只是额外同步渲染、添加目标标记并适当 `sleep`。因此其职责是展示，不承担状态机决策。

---

## 7. 常用配置与验收阈值速查

| 名称 | 当前默认值 | 实际物理含义 |
|---|---:|---|
| TCP 到达位置容差 | 0.009 m | 9 mm |
| TCP 姿态容差 | 0.035 rad | 约 2.0° |
| TCP 到达最大关节速度 | 0.10 rad/s | 实际速度 |
| 夹持候选稳定 | 0.25 s | 连续成立 |
| 持物/放置稳定 | 0.50 s | 连续成立 |
| 夹持相对位移上限 | 0.018 m | 18 mm |
| 滑脱宽限 | 0.12 s | 连续异常后失败 |
| 抬升最低离桌高度 | 0.04 m | 4 cm |
| 最终 XY 容差 | 0.012 m | 12 mm |
| 最终 Z 容差 | 0.004 m | 4 mm |
| Cube 最大线/角速度 | 0.02 m/s / 0.20 rad/s | 静止门槛 |
| 参考关节最大速度 | 0.35 rad/s | 指令约束，不是物理硬限制 |
| 路点间距 | 0.01 m | 每约 1 cm 一点 |
| IK 数值位置/姿态容差 | 0.0003 m / 0.005 rad | 运动层内部，严于物理到达 |
| 阶段 / 抓手 / 总超时 | 15 s / 3 s / 90 s | 有限任务回合 |

请记住**三种不同阈值来源**：IK 数学求解精度、实际机器人 TCP 到达精度、最终 Cube 放置精度。不能把它们混成同一个 `tolerance`。

---

## 8. 测试与故障注入：什么证据才算有效？

测试文件：`scripts/37_test_pick_place.py`。它运行真实 MuJoCo 物理而非只 Mock 状态机。

| 案例 | 期待验证的能力 | 仓库记录 |
|---|---|---|
| 默认放置任务 | 完整状态链路 | SUCCESS；仿真 33.958 s，Cube 最终位置误差约 1.148 mm |
| 同一 Agent reset 后重跑 | 计时器、参考轨迹、事件正确清空 | 与首次结构化结果相同 |
| 不同放置目标 | 非硬编码终点 | SUCCESS |
| 初始 Cube 平移并偏航 20° | 抓取轴与目标动态适应 | SUCCESS |
| 夹爪故障：卡在全开 | 不把闭合命令等于抓取成功 | `GRASP_FAILED` |
| 预抓取目标不可达 | IK 失败不改合法执行器命令 | `IK_FAILED` |
| 成功抬升后降低摩擦 | 真实物理滑落被识别 | `OBJECT_SLIPPED` |
| 释放后外力推 Cube | 防止 RELEASE 后提前报全任务成功 | `PLACE_FAILED`（RETREAT） |
| 放置后夹爪卡闭合 | Cube 到位但未释放，不能成功 | `RELEASE_FAILED` |
| 无效高度或超时 | 明确失败、有限回合 | `PLACE_FAILED` / `TASK_TIMEOUT` / `REACH_TIMEOUT` |

仓库笔记写明：**10 个测试方法、12 个任务案例，正常设计场景 4/4 成功。** 这是预设的确定性场景覆盖，**不是**不同随机初始位置、材质、重量上的泛化成功率，也不是本次笔记编写时重新执行的测试结果。

示例运行：

```bash
conda activate embodied
# 在仓库根目录执行
PYTHONPATH=. python scripts/37_test_pick_place.py
PYTHONPATH=. python scripts/38_panda_pick_place.py --headless --target 0.50 0.12 0.225 --seed 17
PYTHONPATH=. mjpython scripts/38_panda_pick_place.py --target 0.50 0.12 0.225 --seed 17
```

`--target` 表示 **最终 Cube 中心的世界 XYZ（米）**，不是 TCP 目标。当前 `--seed` 主要被记录，不能单靠改变它就获得不同随机场景。

---

## 9. 常见疑问速答（下次忘记先看这里）

**Q1：为什么需要 PoseMotionController？**  
A：IK 给一个目标关节解，但不能自动保证中间轨迹平滑或可达。Motion Controller 先做路点 IK 预检，再按限速的平滑关节轨迹执行。

**Q2：为什么 Smoothstep 有两次 `1.5`？**  
A：`s'(u)` 的最大值为 1.5；需同时约束参考 TCP 运动速度和关节运动速度。

**Q3：为什么路点都可解，还可能撞到物体？**  
A：IK 检查的是关节限位和末端位姿，不等价于碰撞检测或完整避障。

**Q4：为什么 PRE_GRASP 的目标在 Cube 上方？**  
A：先安全接近，再按朝下姿态降低高度，避免从远处直接向 Cube 中心运动。

**Q5：为什么要把 Cube 相对 TCP 的向量乘 `R.T`？**  
A：`R` 是 TCP 局部→世界的旋转，`R.T` 将世界向量转换回 TCP 局部。这样物体随夹爪旋转后，偏移仍能按局部参考表示。

**Q6：为什么目标 Cube 位置要减去旋转后的偏移？**  
A：因为 `Cube中心 = TCP位置 + 旋转后的局部偏移`。移项才能求真正应给 IK 的 TCP 目标。

**Q7：为什么抓取候选必须连续成立？**  
A：单帧接触或受力可能是碰撞瞬态；持续稳定、实际抬升与静止条件提供更强证据。

**Q8：为什么 `motion.finished` 不能直接进入下一状态？**  
A：它只指参考轨迹运行完；实际伺服可能仍有位置/姿态误差，或者机械臂仍在运动。

**Q9：为什么进入 TRANSPORT / PLACE 又测量偏移？**  
A：真实摩擦抓取可能有少量滑移。重新估计小偏差有利于放置；大滑移仍由 `_check_slip()` 判失败。

**Q10：为什么 Cube 受桌面支撑不等于 RELEASE 成功？**  
A：手指可能还夹着 Cube；还要检查夹爪真实开度、手指是否脱离、Cube 是否静止和放置位置是否合格。

**Q11：为什么 RELEASE 后还要 RETREAT？**  
A：机械臂撤离时可能碰到、推走 Cube。必须撤离完成并复核放置状态才算全任务成功。

**Q12：Agent 为什么不调用 `mj_step()`？**  
A：Agent 负责控制决策，Runtime 负责物理推进。分离后 Headless/Viewer 和故障注入测试能复用同一个 Agent。

**Q13：失败后会自动重试吗？**  
A：目前不会。进入 FAILURE、保留原因和已合法指令，不自动重抓。恢复策略不是 Day 12 的范围。

---

## 10. 当前边界：不要误以为已经解决了什么

- **不是通用物体抓取器**：主要面向水平轴对齐桌面、5 cm 立方体和有限的可达位置/初始偏航。
- **不是碰撞感知全局规划器**：路点 IK 和关节插值不能替代 RRT、全臂避障或接触规划。
- **不保证刚性抓取**：夹持依赖真实接触与摩擦；允许小量下滑，过大的滑脱触发失败。
- **不是闭环视觉 Agent**：当前使用 MuJoCo 仿真真值，未使用 Camera / BC / ACT / LLM / PPO。
- **不是 Gymnasium 训练环境**：Day 13 才定义规范的 `Observation`、`Action`、`Reward`、`terminated` 和 `truncated`。

**Day 13 衔接：** Scripted Agent 可作为规则专家/基线，但 `PickPlaceEnv` 应允许 RL 策略通过统一动作空间直接控制机器人，不应强迫 PPO 遵循这套有限状态机。

---

## 11. 下次读代码的推荐顺序（约 20～30 分钟）

1. **`38_panda_pick_place.py`**：确认输入 `--target` 是 Cube 中心，以及如何构造 Agent 和 Runtime。
2. **`pick_place_runtime.py`**：搞清 `agent.step → mj_step → mj_forward` 的物理循环。
3. **`pose_motion.py`**：看 `plan` 的路点 IK 预检、`step` 的 Smoothstep、`calibrate_tracking_bias`。
4. **`scripted_pick_place.py`**：先看 `PickPlaceState` 和 `PickPlaceConfig`，再依序看 `_transition`、`step`、`_check_slip`、`result`。
5. **`37_test_pick_place.py`**：用正常和失败注入案例确认成功判据是否真的有效。

> 一句话复习：**Day 12 用 FSM 决定做什么，用路点 IK 与平滑关节轨迹决定怎么动，用 Contact/Cube/伺服反馈判断是否真正完成，最后通过撤离后的物理复核确认成功。**
