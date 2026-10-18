# Day 12 — Scripted Pick-and-Place Agent

## 开发前核对

2026-10-18 重新读取本地 Git 状态、模型、Day 9～11 控制代码和测试，未依据聊天中的旧版本推断接口。开发开始时只有用户的 `notes/day11_core_notes.md` 未跟踪，本次没有修改该文件。

- 七关节仍为 `joint1`～`joint7`，由 `actuator1`～`actuator7` 控制。第八执行器 `actuator8` 仍通过 fixed tendon `split` 驱动双指，原生 `ctrl` 范围 `0～255`；`PandaGripperController` 负责实际传动映射。
- `PandaIKSolver.solve(target_position, initial_q=None, target_rotation=None, orientation_tolerance=0.01, orientation_weight=0.15)` 返回 `success/reason/iterations/q/target_position/final_position/position_error/orientation_error`。内部使用独立 `MjData`，不会给真实仿真写关节位置。
- TCP 沿用 hand 局部 `[0,0,0.10] m`。旋转矩阵是世界坐标从 TCP 坐标的变换。Day 12 把 Cube 的水平局部轴投影作为闭合轴，并保持 TCP 的局部 `+z` 指向世界 `-z`，适应初始偏航。
- 接触继续使用 `ContactSnapshot`。Grasp Candidate 要求左右手指真实受力、接触点在闭合轴两侧、法向相向；不是 `data.ncon > 0`。
- 开发前重新运行 Day 11：7 项全部通过，3 次物理抓取各抬升约 `116.529 mm`，连续合格持握约 `2.898 s`。Day 11 仍是固定时序实验，没有运输、目标区域放置和完整状态机。
- Day 11 已知约 `6.5 mm` TCP 重力静差、约 `0.016 rad` 姿态误差及缓慢下滑，成为本日阈值和偏移修正的依据。

## 文件和依赖

本次只增加文件，不修改 Day 0～11 的任何已有模型或模块，不增加 Python 依赖。

| 新增文件 | 职责 |
|---|---|
| `agents/scripted_pick_place.py` | 状态枚举、集中配置、反馈判定、有限状态机和结构化结果 |
| `controllers/pose_motion.py` | 复用既有六维 IK、FK 和 Joint Controller；路点预检、同步平滑插值、实测跟踪偏差前馈 |
| `robotics/pick_place_runtime.py` | 有限步数的物理循环、日志、Viewer 标记、测试故障注入接口 |
| `scripts/37_test_pick_place.py` | 10 项验收测试，包含正常任务、复现、初始位姿变化和真实物理故障 |
| `scripts/38_panda_pick_place.py` | Headless / macOS Viewer 共用入口，支持目标 XYZ、seed 和新建结果 JSON |
| `notes/day12_scripted_pick_place.md` | 本说明与实际测试记录 |

依赖方向：演示脚本 → Agent + 物理驱动 → `PoseMotionController` → Day 10 `PandaIKSolver/PandaJointPositionController` → Day 9 FK/Jacobian。Agent 另复用 Day 11 `PandaGripperController/PandaContactDetector/cube_bottom_clearance`，场景初始化直接复用 `load_cube_scene()`。

本次新建、保存在本机的实验文件位于 `outputs/day12_20261018_wwy2zywj/`：

| 文件 | 内容 |
|---|---|
| `headless.log` / `headless_result.json` | 单次正常任务的逐状态运行日志与结构化结果 |
| `viewer.log` / `viewer_result.json` | 实际 macOS Viewer 完整运行的日志与结果 |
| `tests.log` / `tests.json` | 第一轮完整 10 项验收记录，保留不覆盖 |
| `tests_final.log` / `tests_final.json` | 最终验收；进一步验证同一 Agent 在回合重置后也能精确复现 |

这些输出按仓库既有 `.gitignore` 被忽略，但文件实际保存在上述目录。没有删除或覆盖以前的实验结果，没有执行 commit/push。

## 核心接口

| 类或函数 | 输入 | 输出与副作用 |
|---|---|---|
| `PickPlaceConfig` | 带单位的阈值和超时配置 | 不可变配置；拒绝非有限/非正参数 |
| `ScriptedPickPlaceAgent(model, config=None)` | 真实 MuJoCo 模型、可选配置 | 创建任务控制器；目前要求水平、轴对齐 box 桌面和 box Cube |
| `agent.reset(data, target_place_position, seed=0)` | 当前完整物理状态；目标 **Cube 中心** 世界 XYZ（m） | 初始化 IDLE 和本回合统计，保留物理状态；同步合法机械臂 ctrl。目标不适合桌面时直接 FAILURE |
| `agent.step(data)` | 本物理时刻、已经完整 `mj_forward` 的 `MjData` | 状态判定并写下一步的七关节/夹爪 ctrl；不调用 `mj_step`。同一仿真时刻重复调用不累计时间或推进轨迹 |
| `agent.observe(data)` | 当前物理数据 | 含 TCP/Cube 位姿、真实速度、接触快照、实际开度和底部离桌高度的字典 |
| `agent.status(data)` | 当前数据 | 可 JSON 序列化的实时状态、当前/目标 TCP、Cube XYZ、开度、接触、左右指法向力、误差、稳定计时和失败原因 |
| `agent.result(data)` | 终止时物理数据 | 完整结果，含要求的七个字段及细化指标、失败阶段、配置、最终状态、偏移、IK 调用次数 |
| `agent.abort(data, reason, detail)` | 外部取消或运行器异常 | 明确进入 FAILURE，保持最后合法关节命令，不做自动恢复 |
| `PoseMotionController.plan(data, target_position_m, target_rotation, speed_m_s)` | TCP 的物理目标、世界旋转矩阵、参考线速度 | 在独立数据上预检所有 IK 路点；成功才原子替换参考轨迹。失败抛 `MotionPlanningError`，不发送无效关节目标 |
| `motion.step(data, dt)` | 数据和物理步长 s | 经原 Joint Controller 限速写入 ctrl；返回七关节命令 |
| `motion.calibrate_tracking_bias(data)` | 已稳定到达的物理数据 | 记录小量实际 TCP 与 IK 目标的偏差，供下一段前馈修正，不施加额外物体力 |
| `run_pick_place(agent, data, viewer=None, verbose=True, before_physics=None)` | 已 reset 的 Agent；可选 Viewer、测试钩子 | 有限步数推进 MuJoCo 并返回任务结果；Viewer 只改变墙钟播放节奏 |

目标 XYZ 是 Cube 最终中心，不是桌面点或夹爪 TCP。例如桌面顶面 `z=0.20 m`，Cube 半边长 `0.025 m`，目标高度应为 `0.225 m`。其他高度或桌面边缘无法容纳完整方块的目标会明确失败。

## 状态机与物理转移

正常链路：

```text
IDLE → PRE_GRASP → APPROACH → CLOSE_GRIPPER → VERIFY_GRASP
     → LIFT → TRANSPORT → PLACE → RELEASE → RETREAT → SUCCESS
任一非终态遇到对应故障、超时或数值异常 → FAILURE
```

每个状态的进入条件是前一状态真实反馈连续达标。下表的稳定时间均来自实际仿真时间；不使用固定 sleep 或固定步数作为成功证据。

| 状态 | 进入后控制目标、每步行为 | 成功转移条件 | 超时与失败 |
|---|---|---|---|
| IDLE | 保持当前合法关节命令，打开夹爪 | Cube 在桌面稳定、机械臂速度合格、实际开度合格连续 0.25 s | 15 s：`INITIAL_STATE_FAILED`；不支持的放置目标：`PLACE_FAILED` |
| PRE_GRASP | 从当前 Cube 真值生成其上方 0.12 m 的朝下位姿；轨迹预检后平滑执行 | 参考走完、实际 TCP 位置/姿态/关节速度与开度连续达标 0.25 s | 15 s：`REACH_TIMEOUT`；任一路点不可达：`IK_FAILED` |
| APPROACH | 保持全开，以 0.035 m/s 参考速度沿细分笛卡尔路点下降到当前 Cube 中心 | 与预抓取相同的实际位姿/速度/开度条件连续 0.25 s | 15 s：`REACH_TIMEOUT`；路点失败：`IK_FAILED` |
| CLOSE_GRIPPER | 保持机械臂合法命令，发送名义开度 0 m | 合理双侧接触与受力连续 0.25 s | 3 s：`GRASP_FAILED` |
| VERIFY_GRASP | 保持闭合并继续观察，暂不抬升 | 双侧候选与 Cube 实际静止连续 0.25 s | 3 s：`GRASP_FAILED` |
| LIFT | 测量局部 TCP→Cube 偏移，垂直提高实际 TCP 目标 0.12 m | 实际到达、双侧接触、无支撑接触、最低点离桌 ≥0.04 m、Cube 静止连续 0.50 s | 15 s：`LIFT_FAILED`；持续接触丢失/相对滑移超限：`OBJECT_SLIPPED` |
| TRANSPORT | 重测偏移，把 Cube 中心运输到目标上方 0.12 m | 实际 TCP 到达、Cube XY 位于目标内、仍合理持握/离桌/静止连续 0.50 s | 15 s：`TRANSPORT_FAILED`；滑脱：`OBJECT_SLIPPED` |
| PLACE | 重测偏移，以 0.025 m/s 缓慢下降；中心高度按桌面和半尺寸计算，预压 0.001 m | 实际到达，Cube 真实位置合格、静止且有桌面支撑连续 0.50 s | 15 s：`PLACE_FAILED`；尚未支撑时滑脱：`OBJECT_SLIPPED` |
| RELEASE | 保持机械臂，发送开度 0.08 m | 实际开度 ≥0.07 m、无手指接触、目标内静止、由桌面支撑连续 0.50 s | 15 s：`RELEASE_FAILED` |
| RETREAT | 开爪，垂直撤离 0.12 m | 实际位姿到达，TCP–Cube 间距 ≥0.08 m，Cube 的释放/位置/速度/支撑仍合格连续 0.50 s | 15 s：`PLACE_FAILED`，不会因曾经 RELEASE 成功就直接认定任务成功 |
| SUCCESS | 停止轨迹推进并返回结果 | 终态；`result()` 再检查真实释放和放置状态 | 终态之后数据被外部破坏时，`result()` 可返回 `FINAL_STATE_INVALID` |
| FAILURE | 记录失败阶段和原因，停止继续规划 | 终态，供调用方定位；不自动重抓 | 不再执行仿真循环 |

全局保护：总时长 90 s → `TASK_TIMEOUT`；独立物理步数上限也会终止；时钟倒退 → `CLOCK_RESET`；手指–桌面有效碰撞 → `COLLISION`；非有限状态或 MuJoCo warning 均终止。Viewer 提前关闭 → `VIEWER_CLOSED`。

转移记录保存 `completed_state` 和 `completed_stable_s`，表示刚完成的状态及物理稳定时长。新状态的实时 `stable_time_s` 会归零，这是正常进入行为。`reference_finished` 只说明参考轨迹结束，绝不单独构成 Reach Success。

## 运动学、偏移与阈值依据

1. **路点预检**：先从最后合法关节命令计算 FK，再把起点到终点按至多 0.01 m 间距细分；各路点调用既有六维 IK，并独立用 FK 验证返回解。整段通过后才提交。相邻关节解用 cubic smoothstep 同步插值，其最大导数为 1.5，因此时长同时受笛卡尔参考速度和七关节参考速度约束。
2. **姿态**：Cube 的水平局部 x 轴作为闭合轴，若该轴接近竖直则选局部 y 轴；工具的局部 z 始终朝下。首段姿态变化也经过平滑关节限速，不能把位置到达视为姿态到达。
3. **伺服静差**：稳定到达后测量 `bias = actual_tcp - ik_target`；下一段求解目标为 `desired_physical_tcp - bias`。偏差限幅 ±0.015 m，只补偿小量稳态误差；较大跟踪误差由实际 Reach 条件拒绝。
4. **真实夹持偏移**：`offset_local = R_tcp.T @ (cube_center - tcp_position)`；运输和放置的目标 TCP 为 `desired_cube_center - R_target @ offset_local`。偏移在 LIFT、TRANSPORT、PLACE 进入时重新测量，持物过程中还监控相对变化，不能把物体中心默认为 TCP。
5. **放置高度**：`table_top + cube_half_height`。在该中心高度基础上用 0.001 m 的小量预压确保真实承重接触，仍必须通过位置与速度验收。旋转方块的最低点复用 Day 11 的精确支撑范围计算。
6. **滑脱**：持物时失去 Grasp Candidate，或局部相对偏移变化超过 0.018 m，连续 0.12 s 即失败；短暂接触跳变不立即判失败。PLACE 已由桌面支撑后可自然失去夹持，不再按空中滑脱处理。

关键默认配置均位于 `PickPlaceConfig`：

| 参数 | 值与单位 | 依据 |
|---|---|---|
| TCP 位置/姿态容差 | 0.009 m / 0.035 rad | 包含 Day 11 实测约 6.5 mm / 0.016 rad 的误差；测试另要求最终放置误差 <4 mm |
| 到达时最大实际关节速度 | 0.10 rad/s | 沿用 Day 10/11 的稳定验收速度 |
| 抬升最低点离桌门槛 | 0.04 m | 沿用 Day 11 已验证的物理抬升定义 |
| Placement XY / Z 容差 | 0.012 m / 0.004 m | XY 小于 50 mm Cube 边长的 1/4；Z 区分离桌悬停和实际落桌，允许约 0.108 mm soft contact 压入 |
| Cube 线/角速度容差 | 0.02 m/s / 0.20 rad/s | 线速度比 Day 11 落桌门槛更严格，角速度沿用其门槛 |
| 持物/放置连续稳定时间 | 0.50 s / 0.50 s | 沿用持续物理确认，拒绝弹起或瞬态碰撞 |
| 参考关节最大速度 | 0.35 rad/s | Day 11 已验证的控制幅度；实际速度仍单独测量 |
| transit / approach / place 参考速度 | 0.06 / 0.035 / 0.025 m/s | 接近与放置更慢，降低接触冲击 |

运动层的 IK 位置/姿态容差分别为 0.0003 m / 0.005 rad，数值参数集中于 `controllers/pose_motion.py` 顶部。物理成功定义不会随失败测试自动放宽。

## 实际测试命令与结果

环境：macOS、已有 `embodied` Conda 环境，Python 3.12.14、MuJoCo 3.14.0、NumPy 2.5.3。以下从仓库根目录实际执行；记录中的输出路径已经存在，重跑请省略输出参数或指定新的文件名。

```bash
# 语法与导入：退出 0
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python -m compileall -q agents/scripted_pick_place.py controllers/pose_motion.py robotics/pick_place_runtime.py scripts/37_test_pick_place.py scripts/38_panda_pick_place.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python -c 'from agents.scripted_pick_place import ScriptedPickPlaceAgent, PickPlaceConfig; from controllers.pose_motion import PoseMotionController; from robotics.pick_place_runtime import run_pick_place; print("Day 12 imports passed")'

# 最终 Day 12 验收：10 项、12 个案例，退出 0
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/37_test_pick_place.py --report outputs/day12_20261018_wwy2zywj/tests_final.json > outputs/day12_20261018_wwy2zywj/tests_final.log 2>&1

# 单次 Headless：退出 0
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/38_panda_pick_place.py --headless --seed 17 --output outputs/day12_20261018_wwy2zywj/headless_result.json > outputs/day12_20261018_wwy2zywj/headless.log 2>&1

# Day 9～11 回归：均退出 0
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/32_verify_jacobian.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/33_test_panda_ik.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/34_panda_reach.py --headless
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/35_test_panda_grasp.py
git diff --check

# macOS Viewer：在沙箱外获准实际运行，退出 0
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/mjpython scripts/38_panda_pick_place.py --seed 17 --output outputs/day12_20261018_wwy2zywj/viewer_result.json > outputs/day12_20261018_wwy2zywj/viewer.log 2>&1
```

10 项测试覆盖 12 个任务案例，其中 11 个实际推进 MuJoCo 动力学；高度非法的目标在零步输入校验时失败。没有只用 Mock 的任务测试。

| 案例 | 结果 | 仿真时长 s / 关键证据 |
|---|---|---|
| 默认目标、seed=17 | SUCCESS | 33.958；中心误差 1.148 mm |
| 同一初始条件和 seed 重复 | SUCCESS | 33.958；最终验收复用同一 Agent 并重置物理，整个结构化结果与第一次完全相等 |
| 放置目标 `[0.56,-0.08,0.225] m` | SUCCESS | 33.426；中心误差 1.382 mm |
| 初始 Cube `[0.48,-0.04,0.225] m`、偏航 20°，目标 `[0.52,0.10,0.225] m` | SUCCESS | 36.068；误差 1.117 mm。只在编译前定义初始场景与关键帧，任务中不写 Cube 状态 |
| 夹爪卡在全开 | `GRASP_FAILED` / CLOSE_GRIPPER | 13.360；只覆盖执行器 ctrl，Cube 保持真实动力学，不能抬升 |
| 不可达预抓取高度 3 m | `IK_FAILED` / PRE_GRASP | 0.250；真实 IK 求解失败。另断言不可达整段规划前后 ctrl、合法关节命令与空轨迹均未改变 |
| 已验证抬升后，Cube/双指滑动摩擦降为 0.0001 | `OBJECT_SLIPPED` / TRANSPORT | 16.742；故障前确有 ≥4 cm 离桌高度、夹爪 ctrl 仍为闭合；在重力下真实滑落 |
| 释放后、撤离途中沿世界 +x 施加 2 N，持续 0.06 s | `PLACE_FAILED` / RETREAT | 43.330；虽已 RELEASE 成功，最终误差 29.331 mm，任务不能成功 |
| 释放时夹爪仍被卡闭合 | `RELEASE_FAILED` / RELEASE | 42.698；运输已完成，但实际开度/脱离不满足 |
| 总超时 0.10 s | `TASK_TIMEOUT` / IDLE | 0.100；循环按时终止 |
| 阶段超时 0.30 s | `REACH_TIMEOUT` / PRE_GRASP | 0.550；不因轨迹正在执行而无限等待 |
| 放置中心高度 0.40 m | `PLACE_FAILED` / IDLE | 0.000；不把悬空目标悄悄改为合法任务 |

附加测试在不推进物理时重复调用 `step()` 1000 次，确认不能累计稳定时长、推进参考或改变 ctrl。正常场景为 4/4 成功，这是确定性的设计案例覆盖，不是随机场景统计成功率。

回归：Day 9 HOME/Perturbed Jacobian 最大误差仍约 1e-11～1e-10；Day 10 三个可达 IK 目标及不可达目标测试通过，物理 Reach 最终误差 6.173 mm；Day 11 7 项全部通过。未运行旧的训练类实验，也未声称覆盖全部 Day 0～8 脚本。

### 默认任务实际状态日志摘录

```text
t=0.000   IDLE
t=0.250   PRE_GRASP
t=5.202   APPROACH
t=10.360  CLOSE_GRIPPER
t=10.688  VERIFY_GRASP
t=10.938  LIFT
t=16.566  TRANSPORT
t=20.130  PLACE
t=27.698  RELEASE
t=28.330  RETREAT
t=33.958  SUCCESS
```

最终 Cube 中心约 `[0.498860963, 0.119901278, 0.224892245] m`，目标 `[0.50,0.12,0.225] m`；XY 误差约 1.143 mm，总位置误差约 1.148 mm。实际开度约 0.080 m，无手指接触，桌面支撑为 true，线/角速度分别约 1.52e-10 m/s / 1.44e-9 rad/s，MuJoCo warning 为 0。

最终 TCP 约 `[0.500029445,0.120137364,0.352255297] m`，已撤离 Cube。记录的局部 TCP→Cube 偏移约 `[-0.00002783,0.00000082,0.00837319] m`，可见不能假定二者重合。

Viewer 实际执行完上述完整任务，退出码 0；其结构化结果与相同 seed 的 Headless 结果逐字段完全一致。红球为当前 TCP 目标、绿球为放置中心。未进行截图级视觉布局核验，GUI 测试结论限于真实 Viewer 运行和任务指标通过。

## 日常运行

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 自动化验收
PYTHONPATH=. python scripts/37_test_pick_place.py

# 单次无界面任务，目标为 Cube 中心（m）
PYTHONPATH=. python scripts/38_panda_pick_place.py --headless --target 0.50 0.12 0.225 --seed 17

# Mac 可视化，约 34 秒仿真时长；末尾自动关闭窗口并输出 JSON
PYTHONPATH=. mjpython scripts/38_panda_pick_place.py --target 0.50 0.12 0.225 --seed 17
```

`--output 新路径.json` 可保存完整结果，程序使用独占创建并拒绝覆盖已有文件。`--seed` 被记录到结果中；当前没有随机策略或随机初始化，改变 seed 本身不会改变物理初始条件。

## 限制与 Day 13 衔接

- 仅面向当前 50 mm、0.10 kg Cube、水平静态桌面及其可达区域。已经覆盖不同初始 XY、20° 偏航和另一目标位置，尚未做大规模随机物体/姿态/摩擦评估。
- 细分路点与 IK 是局部运动生成，不是全臂避障、全局 IK 或接触规划器。只有现有 detector 能识别的手指–桌面碰撞参与全局碰撞失败判断；没有新增全臂自碰撞规划。
- 夹持仍存在小量下滑，本次最长持物时间内可由测量偏移和阈值监控处理；未证明长期力闭合。超限时失败，不伪造恢复。
- 跟踪偏差前馈只适用于小量准静态误差，不能替代完整动力学补偿。没有调高官方力矩上限、改变正常模型摩擦或放宽物理成功定义来隐藏失败。
- 失败时保持最后合法命令并返回，没有恢复或自动重抓；物理循环结束后的继续仿真/复位由调用方负责。
- 默认数据使用仿真真值，当前不包含 RL、视觉、LLM、BC 或 Gymnasium 环境。

Day 13 可由 `PickPlaceEnv.reset()` 负责 MuJoCo 场景初始化，再调用专家 Agent 的 `reset()`；在每个物理子步上使用 `step(data)` 作为确定性 baseline。环境观测可基于 `observe()` 中的关节/TCP/Cube/接触信息构造稳定数组。环境的奖励、terminated/truncated 和任务成功指标应单独定义并复用本日经过验证的物理条件，不能要求学习策略必须走本状态机。`result()/status()` 可作为 `info` 和评估日志来源；本日没有提前创建环境。
