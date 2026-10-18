# Day 11 — Panda Gripper、Contact、Cube Grasp & Lift

## 范围与文件

本日实现固定场景下的一次基础物理实验：打开夹爪、从上方接近、闭合、抬升、保持、打开并让 Cube 自由下落。它是验证运动学、位置执行器和接触动力学的实验夹具，不包含通用 Pick-and-Place Agent、任务规划、PPO、VLM 或视觉模型。

| 文件 | 变更 | 职责与接口 |
|---|---|---|
| `assets/franka_panda/scene_cube.xml` | 新增 | 包含原 `scene.xml`；增加桌面、自由 Cube 和完整的 `day11_home` keyframe |
| `controllers/ik_controller.py` | 修改 | `PandaIKSolver.solve(..., target_rotation=None, orientation_tolerance=0.01, orientation_weight=0.15)` 可选姿态约束；`rotation_error_world()` 返回世界坐标旋转误差向量 |
| `controllers/gripper_controller.py` | 新增 | `PandaGripperController(model)` 校验传动；`command(data, opening_m)` 发送夹爪目标；`feedback(data)` 返回两指实测位置、速度、tendon 长度、执行器力 |
| `robotics/contacts.py` | 新增 | `PandaContactDetector(model).read(data)` 返回 `ContactSnapshot`；保留 Cube 的实际 geom 接触对、接触位置、法向与法向力 |
| `robotics/grasp_experiment.py` | 新增 | `load_cube_scene()`、`plan_experiment()`、`cube_bottom_clearance()`、`GraspMonitor.update()`、`run_experiment()`；Headless/Viewer 共用同一实验与验收逻辑 |
| `scripts/35_test_panda_grasp.py` | 新增 | 7 项 unittest，包含三次重新初始化的完整物理实验 |
| `scripts/36_panda_grasp.py` | 新增 | `--headless` 或 macOS Viewer；打印阶段指标与 JSON，失败以非零退出码结束 |
| `notes/day11_gripper_contact_grasp.md` | 新增 | 本文件：接口、单位、判定、实测报告和限制 |

`panda.xml`、原 `scene.xml`、Day 9 的 `robotics/kinematics.py` 和 Day 10 的 `controllers/joint_controller.py` 均保持原样。没有添加依赖。

依赖链：`36_panda_grasp.py` → `grasp_experiment.py` → Day 10 `PandaIKSolver` / `PandaJointPositionController` + 新夹爪控制器与接触检测器。IK 继续调用 Day 9 的 `get_tcp_pose()`、`get_tcp_jacobian()`、`set_arm_qpos()`，只在自己的 scratch `MjData` 上迭代；真实仿真中的机械臂与 Cube 均由 `mj_step()` 推进。

## 场景、初始化与单位

所有 XYZ 均为世界坐标，单位 m；关节角 rad；关节角速度 rad/s；手指位移 m、速度 m/s；力 N；质量 kg；惯量 kg·m²；仿真时间 s。Viewer 用墙钟时间近似实时播放，验收计时始终使用物理步长。

- 桌面 geom：中心 `[0.50, 0, 0.10] m`，box 半尺寸 `[0.18, 0.22, 0.10] m`，上表面 `z=0.20 m`。
- Cube：初始中心 `[0.50, 0, 0.225] m`，边长 `0.05 m`，质量 `0.10 kg`，三轴惯量均为 `m*a²/6 = 0.0000416666667 kg·m²`。`cube_free` 提供 6 个自由速度量、7 个位置量。
- 场景 `nq=16, nv=15, nu=8`；原场景仍为 `9, 9, 8`。
- Cube 与桌面摩擦：滑动系数 `1`（无量纲），扭转参数 `0.005 m`，滚动参数 `0.0001 m`；Cube 使用 `condim=3`。机器人碰撞、手指 pad 和 tendon servo 均保留官方配置。
- 重力 `[0, 0, -9.81] m/s²`，步长 `0.002 s`。

新增自由关节后，官方 `home` 的 9 个 `qpos` 值不能初始化 Cube 到桌面，补齐后的 Cube 平移会是零。场景提供独立的完整 `day11_home`，由 `load_cube_scene()` 一次性 reset。请不要对本场景直接使用 Day 9 的 `reset_home()` 来初始化真实实验。IK 内部的 `reset_home()` 仅用于读取机器人初始关节，并不推进或操作真实 Cube。

除 XML keyframe 初始化以外，实验不会写 Cube 的 `qpos`，也不会写外力、创建 weld、mocap 或新的等式约束。唯一的 equality 是官方两手指关节的耦合。

## 夹爪传动

初始化控制器时按模型实际内容核对：第八执行器 `actuator8`、fixed tendon `split`、两 slide joint、tendon 系数、gear、固定 gain、affine bias、无 activation dynamics 以及控制范围。

当前模型：

```text
finger_joint1, finger_joint2 ∈ [0, 0.04] m
split length = 0.5*q_left + 0.5*q_right
actuator length = gear * split length       (gear = 1)
force = gain*ctrl + bias0 + bias1*length + bias2*velocity
gain = 0.01568627451; bias0 = 0; bias1 = -100; bias2 = -10
ctrl ∈ [0, 255]                            （原生控制量，不是 rad 或 m）
```

`opening_m` 定义为两 slide 位置之和，控制器用零速度的 servo 平衡式反解 `ctrl`，再与真实 joint/ctrl 范围求交并限幅：

```text
ctrl = (-bias1 * gear * 0.5 * opening_m - bias0) / gain
opening 0.00 m → ctrl 0
opening 0.04 m → ctrl 127.5
opening 0.08 m → ctrl 255
```

这是名义关节开度，不是所有 pad 表面之间的精密标定距离。`feedback()` 直接分别读取两指的 `qpos/qvel`，不会因 equality 耦合就假设二者完全相等。夹持时即使命令为 `0 m`，实测开度仍约 `0.04991 m`；这是 Cube 阻挡手指的正常物理结果。`actuator_force_N` 是 tendon 执行器的广义力，不能直接当作每个接触点的法向力；后者由接触求解器读取。

## 最小姿态 IK 扩展与接近路径

`target_rotation` 是 `world_from_TCP` 的 SO(3) 矩阵。未提供时保持 Day 10 的位置 IK 行为。提供时，将位置残差与旋转残差堆叠，继续复用原 DLS、关节限位、最大步长与回溯线搜索。

旋转误差取 `log(R_target @ R_current.T)` 的最短旋转向量，与 Day 9 世界坐标旋转 Jacobian 一致。使用 MuJoCo 的矩阵转四元数、四元数转旋转向量函数，避免简单叉乘在 180° 处得到零误差。位置和姿态分别验收；`orientation_weight=0.15 m/rad` 仅用于 DLS 和线搜索的尺度平衡。

固定抓取方向：TCP 局部 `+z` 朝世界 `-z`，局部 `+y`（夹爪闭合轴）朝世界 `+x`。沿用局部 TCP `[0,0,0.10] m`；官方主 pad 中心沿 hand z 轴约为 `0.0584+0.0445=0.1029 m`，因此 TCP 对准 Cube 中心即可使 pad 接近 Cube 中部。

| 参数 | 数值与单位 |
|---|---|
| Hover TCP | `[0.50, 0, 0.35] m` |
| Grasp TCP | `[0.50, 0, 0.225] m` |
| Lift TCP | `[0.50, 0, 0.345] m` |
| IK 位置容差 | `0.0005 m` |
| 实验 IK 姿态容差 | `0.005 rad` |
| 关节命令最大速度 | `0.35 rad/s` |
| 接近/下降末端验收 | 位置误差 `<0.01 m`，姿态误差 `<0.04 rad`，最大实际关节速度 `<0.10 rad/s` |
| 阶段时长 | settle 1、approach 4、descend 3、close 2、lift 3、hold 1、release 2.5 s；合计 16.5 s |

轨迹在关节空间用同步 cubic smoothstep 插值，每段最后留 `0.5 s` 稳定时间，然后交给原关节位置控制器限速。第一轮实现发现独立饱和 ramp 会使腕部先到、中途手指碰桌；同步插值解决了该固定路径的问题。它不是通用碰撞规划器，也不保证任意端点之间的 Cartesian 直线路径。

## 接触与状态判定

`PandaContactDetector` 遍历 `data.contact`，按实际 geom ID 匹配 Cube、table、floor；官方手指碰撞 geom 没有名字，按 `geom_bodyid` 归属到 `left_finger/right_finger`。仅接受存在约束行、且 `mj_contactForce()` 法向力至少 `0.05 N` 的接触。接触位置和法向统一到世界坐标，法向符号统一成指向 Cube 的受力方向。

每次 `mj_step()` 后调用完整 `mj_forward()`，使新位置、接触帧和求解后的接触力对应同一个状态。只调用 `mj_kinematics()` 无法同步这些接触力。API 依据见 [MuJoCo 官方 mj_contactForce 文档](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)。

| 状态 | 判定 |
|---|---|
| Finger Contact | 左指或右指与 Cube 存在上述有效接触 |
| Grasp Candidate | 左右指均接触；接触点沿闭合轴分别位于 Cube 中心正侧、负侧，距中心投影各超过 `0.005 m`；法向沿相应向内闭合方向投影超过 `0.5`，且左右法向点积 `<-0.5` |
| Lift Success | 连续 `0.50 s` 保持 Grasp Candidate，Cube 无 table/floor 接触，Cube 最低点高于桌面至少 `0.04 m` |
| Release | 曾确认 Lift；发出释放命令后，两指实测开度至少 `0.07 m`，无有效指–Cube 接触，Cube 中心较最后确认夹持高度下落至少 `0.05 m`；这些释放条件连续保持 `0.20 s` |
| 整体实验成功 | 上述 Lift、Release 均成立；最终 Cube 与桌面接触、无手指接触，线速度范数 `<0.05 m/s`、角速度范数 `<0.2 rad/s` 连续至少 `0.30 s`；全程无有效手指–桌面碰撞、无 MuJoCo warning |

Cube 最低点使用旋转 box 的精确支撑范围 `center_z - abs(R_world_from_cube[2,:]) @ half_sizes`，不把倾斜后的 Cube 简化成中心高度减固定半边长。高度是相对于本场景水平桌面的几何量，不是任意地形的距离传感器。

`ContactSnapshot` 是瞬时状态。最终结果顶层 `finger_contact/grasp_candidate/lift_success/release` 是本次实验曾确认的事件；`phases` 保存每个阶段末端的瞬时反馈。因此释放结束后，顶层 `grasp_candidate` 仍可为 true，而 `phases.release.grasp_candidate` 为 false。

## 实际验证记录（2026-10-16）

环境：macOS，Python `3.12.14`，MuJoCo `3.14.0`，NumPy `2.5.3`。系统 `python3` 没装 MuJoCo，实际采用已有 Conda `embodied` 环境，没有安装或更新软件。

以下命令均从仓库根目录实际执行：

```bash
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python -m compileall -q controllers/ik_controller.py controllers/gripper_controller.py robotics/contacts.py robotics/grasp_experiment.py scripts/35_test_panda_grasp.py scripts/36_panda_grasp.py

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python -c 'import controllers.gripper_controller, controllers.ik_controller, robotics.contacts, robotics.grasp_experiment; print("Day 11 imports passed")'

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/35_test_panda_grasp.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/36_panda_grasp.py --headless
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/32_verify_jacobian.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/33_test_panda_ik.py
PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/python scripts/34_panda_reach.py --headless
git diff --check

PYTHONPATH=. /Users/qiu_star/miniforge3/envs/embodied/bin/mjpython scripts/36_panda_grasp.py
```

| 检查 | 实际结果与关键指标 |
|---|---|
| 语法、导入、diff 空白检查 | 通过，退出 0 |
| XML 与结构 | 通过；Cube free joint、质量、惯量、完整初始化及唯一的官方手指 equality 均断言验证；原场景仍可加载 |
| 夹爪开合 | 通过；闭合两指约 `[4.78e-8, -9.39e-10] m`，中间开度两指约 `[0.02000002, 0.01999998] m`，全开约 `[0.04, 0.03999995] m`。接近零处存在 soft joint limit 的微小数值越界 |
| 真实 Cube–桌面接触 | 通过；`ncon=4`、总法向力 `0.981000 N`，Finger Contact 与 Grasp Candidate 均为 false |
| 姿态误差/Jacobian | 通过；世界旋转 Jacobian 中心差分最大绝对误差约 `1.419e-10`，180° 旋转误差测试通过 |
| 固定位置改变 yaw 30° 的姿态 IK | 通过；最终位置误差 `0.000178 m`，角误差 `0.004733 rad`；真实仿真数据不变 |
| 状态判定反例 | 通过；单侧接触、仍有桌面支撑、抬升中断、未发释放命令、开度不足、仍夹持或尚未下落都不能被误判为对应成功 |
| Day 11 unittest | 7 项通过，含 3 次重新初始化的物理抓取；每次都观察到单侧接触瞬态，随后才形成双侧候选 |
| Day 11 完整 Headless | 通过，仿真 `16.5 s`，指标见下表 |
| Day 9 Jacobian 回归 | 通过；HOME/Perturbed 最大平移误差分别约 `1.283e-11 / 1.201e-11`，最大旋转误差约 `7.167e-11 / 7.436e-11` |
| Day 10 位置 IK 回归 | 通过；3 个可达目标位置误差 `0.289 / 0.096 / 0.230 mm`；不可达目标按预期返回失败 |
| Day 10 物理 Reach 回归 | 通过；最终 TCP 误差 `6.173 mm`，达到并稳定 |
| macOS Viewer | 沙箱内启动退出 `134`；获准在沙箱外执行同一命令后，完整运行结束退出 `0`，`success=true`，物理指标与 Headless 一致。独立窗口读取超时，未完成截图级视觉核验，不把该项标记为通过 |

三次物理测试在同一初始条件下均得到：

| 指标 | 数值 |
|---|---|
| Cube 底部最高离桌 | `0.116529 m` |
| 满足抬升条件的最长连续时间 | `2.898 s` |
| Hold 末端底部离桌 | `0.115496 m` |
| Hold 左/右指法向力总和 | `1.253094 / 1.242393 N` |
| Hold 左/右指实际位置 | `0.02495495 / 0.02495479 m` |
| 释放后中心下落 | `0.115901 m` |
| 释放阶段最小竖直速度 | `-1.500659 m/s` |
| 末端连续稳定落桌 | `2.262 s` |
| 手指–桌面碰撞 | 无有效接触 |
| MuJoCo warning | `0` |
| Hover/Grasp/Lift IK 位置误差 | `0.159 / 0.102 / 0.135 mm` |
| Hover/Grasp/Lift IK 姿态误差 | `0.002882 / 0.002099 / 0.002353 rad` |

## 运行方式

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 自动化验收，包括 3 次完整物理实验
PYTHONPATH=. python scripts/35_test_panda_grasp.py

# 单次 Headless，输出逐阶段反馈和最终 JSON
PYTHONPATH=. python scripts/36_panda_grasp.py --headless

# Mac 本地 Viewer，按约实时节奏运行，结束后自动关闭窗口
PYTHONPATH=. mjpython scripts/36_panda_grasp.py
```

不需要显示服务或 OpenGL 的自动化测试不会导入 `mujoco.viewer`。Viewer 可手动提前关闭，此时结果为 `viewer_closed`，不报告实验成功。

## 当前限制

1. 这是固定尺寸、质量、摩擦、初始位置和方向的 bench 实验。三次重复验证确定性，不代表随机初始条件下的抓取成功率；没有通用避障和轨迹规划。
2. 原关节位置 servo 在重力下存在约 `6.0–6.5 mm` TCP 稳态误差；实际姿态误差约 `0.011–0.016 rad`。IK 收敛指标与物理追踪指标分开报告；没有额外添加重力补偿。
3. Grasp Candidate 是双侧几何和受力启发式，不是完整 force-closure / 摩擦锥证明。抬升保持 0.5 s 是本实验门槛，没有验证长期持物；Hold 的 1 s 中能观察到约 `0.84 mm` 的缓慢下滑。
4. MuJoCo soft contact 允许小量压入；Cube 静置时底部约低于名义桌面 `0.108 mm`，不作为抬升证据。接触力小于 `0.05 N` 的轻触不计入本实现的有效接触。
5. 位姿 IK 是局部 DLS 求解，不保证远距离目标、奇异位姿或关节限位附近收敛。姿态扩展通过 yaw 变化、180° 误差与旋转 Jacobian 验证，但不是全局 IK 求解器。
6. Viewer 程序已实跑通过；截图级视觉布局核验尚未完成。可用上述 `mjpython` 命令在 Mac 手动观察。

未执行 `git commit`、`git push`；未删除或覆盖已有实验结果。
