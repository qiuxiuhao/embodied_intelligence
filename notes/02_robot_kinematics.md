# 机器人运动学：正向运动学、Jacobian 与逆运动学

统合日期：2026-10-25。以 Panda 七自由度机械臂为对象，整合 Day8–12 的运动学与控制内容。本文说明现有实现，不扩展新的规划或控制路线。

环境与状态基础见 [仿真环境](01_simulation_environment.md)；夹持物体后的目标转换及任务执行见 [Pick & Place](05_pick_and_place.md)。

## 1. 三个问题要分开

| 问题 | 输入→输出 | 项目实现 |
| --- | --- | --- |
| FK | 关节配置 q → TCP 位姿 | MuJoCo 运动学 + kinematics.py |
| IK | TCP 目标位姿 → 候选关节配置 | PandaIKSolver，DLS 迭代 |
| 动力学跟踪 | 关节参考 → 执行器 → 实际运动 | 关节控制器 + mj_step |

FK/IK 计算可在独立数据上试探关节配置；真实抓放必须通过执行器和动力学发生。IK 成功不表示真实 TCP 已到达，更不表示整条路径无碰撞或物体已稳定抓住。

本项目没有单独建立 DH 符号 FK 引擎；由 Panda MJCF 描述运动学树，MuJoCo 计算世界位姿和几何 Jacobian。

## 2. 位姿、旋转与坐标系

用 W 表示世界系、H 表示 Hand、T 表示 TCP：

\[
{}^WT_H=
\begin{bmatrix}{}^WR_H & {}^Wp_H\\0&1\end{bmatrix}.
\]

旋转矩阵满足 `RᵀR=I`、`det(R)=1`。MuJoCo 四元数顺序为 wxyz；q 与 -q 表示同一旋转，不能把两者直接相减当姿态误差。

模型中的 body.pos 是父坐标系下的局部量；data 中 body.xpos/xmat 是当前世界位姿。读取任何位姿时先确认“哪个点、相对哪个坐标系、使用什么方向表示”。

## 3. 正向运动学与 TCP 偏移

机械臂关节：

\[
q=[q_1,\ldots,q_7]^T,\qquad T_{\text{TCP}}=FK(q).
\]

项目 TCP 定义为 Hand 局部 `[0,0,0.10] m`，方向与 Hand 相同：

\[
{}^Wp_T={}^Wp_H+{}^WR_H\,{}^Hp_T,
\qquad {}^WR_T={}^WR_H.
\]

这 0.10 m 是工具参考点定义，不是一个额外关节，也不保证 TCP 与夹住的 Cube 中心重合。不能把局部偏移直接加到世界位置而忽略旋转。

示例接口：

```python
from robotics.kinematics import load_panda, reset_home, get_tcp_pose
model, data = load_panda()
reset_home(model, data)
position, quaternion, rotation = get_tcp_pose(data)
```

`set_arm_qpos()` 用于检查或独立运动学求解，会验证形状、有限值和关节范围，并刷新 FK；它不是物理运动控制器。

## 4. Jacobian：局部运动映射

\[
v_{\text{TCP}}=J_p(q)\dot q,\qquad
\omega_{\text{TCP}}=J_r(q)\dot q,\qquad
J=\begin{bmatrix}J_p\\J_r\end{bmatrix}.
\]

七关节机械臂的 `Jp/Jr` 为 3×7，完整位姿 J 为 6×7。它随当前 q 改变，只描述局部一阶关系：

\[
\Delta p\approx J_p\Delta q.
\]

MuJoCo 对整个模型生成 3×nv 的 Jacobian。代码按七个臂关节的 DOF 索引取列，不能因为模型前面看起来排列整齐就把 qpos 索引当 qvel/DOF 索引。TCP 不在 Hand 原点时，应计算 TCP 点的 Jacobian。

七关节多于完整位姿任务的六个约束，通常存在冗余解；奇异位形、关节限位和碰撞约束又会缩小实际可用解空间。本项目没有实现完整的冗余姿态优化或全局避障。

## 5. 用有限差分验证 Jacobian

中心差分验证平移列：

\[
J_{p,i}^{FD}=\frac{p(q+\epsilon e_i)-p(q-\epsilon e_i)}{2\epsilon}.
\]

旋转列需要从相邻旋转矩阵的相对旋转提取，不能直接差分四元数。32 号脚本使用 ε=1e-5，在 Home 与扰动配置比较平移/旋转 Jacobian，容差各为 1e-4。

它验证的是实现及坐标表达的一致性，不是“所有目标都能达到”或“整条轨迹无碰撞”的证明。扰动后恢复配置也很重要，避免验证代码悄悄改变下一次计算的起点。

## 6. 世界系姿态误差

希望用世界系左乘修正当前方向：

\[
R_t=R_\Delta R_c
\quad\Rightarrow\quad
R_\Delta=R_tR_c^T.
\]

对应旋转向量：

\[
e_r=\operatorname{Log}(R_tR_c^T)^\vee.
\]

代码先转四元数，选较短旋转表示，再得到三维旋转向量。这里的单位是 rad；调用 `mju_quat2Vel(...,1.0)` 是表示转换，不是在测量实际机器人速度。

`R_cᵀR_t` 对应局部系右乘误差，不能与世界系几何旋转 Jacobian 混用。接近 180° 时轴角表示存在歧义；大角度误差的局部线性化也更不准确。

## 7. 位置 IK 与六维位姿 IK

位置任务：

\[
e_p=p_t-p_c,\qquad J_{\text{task}}=J_p.
\]

位姿任务：

\[
e_{\text{task}}=\begin{bmatrix}e_p\\w_re_r\end{bmatrix},
\qquad
J_{\text{task}}=\begin{bmatrix}J_p\\w_rJ_r\end{bmatrix}.
\]

`orientation_weight` 默认 0.15 m/rad，用于统一优化尺度。必须同时对旋转误差与旋转 Jacobian 加权；只改其中一项会改变问题含义。

加权误差决定更新方向及线搜索是否改善；IK 是否成功仍分别检查位置误差和姿态角误差，不能用一个合并范数替代两个物理门槛。

## 8. Damped Least Squares

因为 J 非方阵且可能接近奇异，求解：

\[
\min_{\Delta q}\|J\Delta q-e\|^2+\lambda^2\|\Delta q\|^2.
\]

对应：

\[
\Delta q=J^T(JJ^T+\lambda^2I)^{-1}e.
\]

实际代码用 `np.linalg.solve` 解线性方程，不显式构造矩阵逆。阻尼限制接近奇异点时过大的关节增量，但不消除局部最优、不可达、关节限位等问题。

现有求解流程：

1. 从给定初始关节配置计算 FK、误差与 Jacobian。
2. DLS 求 Δq，限制增量范数。
3. 尝试不同步长，并限制在带余量的关节范围内。
4. 重新 FK；只接受加权任务误差下降的候选配置。
5. 达标成功；否则按停滞或迭代上限返回失败原因。

通用默认参数为 damping=0.05、单次增量范数上限 0.08 rad、关节边界余量 0.01 rad。线搜索步长从 1 逐次减半到 0.03125。位姿残差配合几何 Jr 是局部近似，依靠小步迭代和重算 FK 修正。

| 场合 | 数值求解门槛 | 与物理判定的关系 |
| --- | --- | --- |
| 通用 IK 默认 | 位置 2 mm；指定姿态时角误差 0.01 rad | 求解器停止条件 |
| Day12 运动层 | 位置 0.3 mm；姿态 0.005 rad | 路点预检 |
| Day13/Day15 环境 | 位置 0.5 mm；姿态 0.005 rad | 控制目标接纳条件 |
| Panda 实际 Reach | 真实位置 9 mm、姿态 0.035 rad、关节速度低于 0.10 rad/s，持续 0.4 s | 物理任务成功条件 |

早期 Day10 演示曾采用 15 mm 物理到达门槛，不能覆盖后续环境的 9 mm 配置。IK 失败只说明本次起点、迭代和约束下未求出合格解，不自动证明目标数学上绝对不可达。

## 9. 从 IK 解到真实关节运动

普通位置伺服可理解为：

\[
\tau=K_p(q_{\text{ctrl}}-q)-K_d\dot q.
\]

控制器限制的是参考关节命令变化率，实际 qvel 仍由动力学决定。负载与重力可能使实际姿态相对命令存在静差，因此应持续读取真实 TCP 和速度，而非只看参考轨迹结束。

Reach/BC 使用重力前馈：

\[
q_{\text{ctrl}}=\operatorname{clip}
\left(q_{\text{command}}+\tau_g(q)/K_p\right).
\]

`τg` 在独立 MjData、当前姿态、零速度下由偏置力计算得到；没有关闭真实重力，也没有直接写 qpos 或施加“托举 Cube”的外力。该公式依赖项目已验证的单位传动仿射位置执行器，不是通用执行器公式。

Day12 的旧规则运动还使用小范围 TCP 静差估计 `b=p_actual-p_IK,target`，下一段按 `p_IK,target=p_desired-b` 补偿；它与动力学重力前馈是不同方法。

## 10. 平滑轨迹不是单次 IK

`PoseMotionController` 先沿 TCP 路径生成间距约 1 cm 的路点，逐点求六维 IK、复核 FK，全部通过后才提交新轨迹。任一路点失败都不发送半条新路径。

相邻关节路点以 smoothstep 插值：

\[
q(u)=q_0+(3u^2-2u^3)(q_1-q_0),\quad u\in[0,1].
\]

起终点参考速度为零，系数导数最大值 1.5；段时长需要据此满足参考速度限制。路点间是在关节空间插值，因此真实 TCP 轨迹不保证严格直线，也不是碰撞自由路径规划。

## 11. 代码阅读与保留检查

| 文件 | 阅读重点 |
| --- | --- |
| [kinematics.py](../robotics/kinematics.py) | FK、TCP 偏移、DOF 索引、Jacobian |
| [ik_controller.py](../controllers/ik_controller.py) | 姿态误差、加权残差、DLS、线搜索 |
| [joint_controller.py](../controllers/joint_controller.py) | 参考限速与执行器输入 |
| [gravity_compensated_joint_controller.py](../controllers/gravity_compensated_joint_controller.py) | 保持名义命令独立的重力前馈 |
| [pose_motion.py](../controllers/pose_motion.py) | 路点预检与平滑插值 |

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied
PYTHONPATH=. python scripts/30_panda_fk.py
PYTHONPATH=. python scripts/31_panda_jacobian.py
PYTHONPATH=. python scripts/32_verify_jacobian.py
PYTHONPATH=. python scripts/33_test_panda_ik.py
```

34 号例程用于观察 IK 经真实执行器完成 Reach；其参数先看 `PYTHONPATH=. python scripts/34_panda_reach.py --help`。可视化在 macOS 使用 mjpython。

## 12. 自查与来源

应能回答：为什么 TCP 偏移需要乘 R；为何 6×7 Jacobian 没有普通逆；为何姿态误差与 Jr 必须同系；阻尼解决什么、不能解决什么；IK 精度与物理成功精度为何不同；限参考速度为何不是限真实速度。

历史统合来源（原笔记已清理）：Panda 模型、FK/Jacobian、IK/Reach、Day11 知识笔记、Day12 知识笔记、Reach v2。核心公式另与当前运动学和控制器源码核对。
