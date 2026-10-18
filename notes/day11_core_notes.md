# Day 11｜Franka Panda：六维 IK、夹爪、接触与抓取核心复习笔记

> 用途：下次阅读 Day 11 源码时，先用本笔记恢复各个模块的**输入 / 输出 / 关键公式 / 判定逻辑**。
>
> 代码来源：[qiuxiuhao/embodied_intelligence](https://github.com/qiuxiuhao/embodied_intelligence) 的 `main` 分支（核对日期：2026-10-18）。文件路径与函数名根据当天可读取的源码整理。`main` 后续可能变化，若公式或参数不符，以新的实际代码为准。
>
> 本笔记重点是**实现原理的复习**，不代表已在本机重新运行完整 Grasp/Lift 仿真并验证成功率。

## 0. 先看全局：Day 10 → Day 11 多了什么？

Day 10 主要关注：`目标 TCP XYZ → 位置 IK → 七关节目标 → Panda 执行器 → 实际 TCP`。

Day 11 在此基础上加入：

```text
目标抓取位姿 (XYZ + R)
    │
    ▼
PandaIKSolver：位置 / 姿态 IK（七关节目标 q*）
    │
    ▼
Joint Position Controller → 前七个 actuator
    │
    └──────────┐
               ▼
Gripper Controller → actuator8 → 左右手指
               │
               ▼
     MuJoCo 接触与摩擦求解
               │
               ▼
Contact Detector：finger_contact / grasp_candidate / cube_table ...
               │
               ▼
任务层：观察 Cube 高度 + 持续时间 → Lift / Release 判定
```

**必须区分：** `close_command ≠ finger_contact ≠ grasp_candidate ≠ lift_success`。

### 模块速查

| 文件 | 核心内容 | 复习时先看 |
|---|---|---|
| [`assets/franka_panda/scene_cube.xml`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/assets/franka_panda/scene_cube.xml) | Cube、桌面、完整 Keyframe | `freejoint`、Cube 惯量、`day11_home` |
| [`controllers/ik_controller.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/controllers/ik_controller.py) | 位置/姿态 IK、DLS、线搜索 | `rotation_error_world()`、`residual()`、`_compute_dls_step()`、`solve()` |
| [`controllers/gripper_controller.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/controllers/gripper_controller.py) | 名义开度与原生执行器控制量映射 | `__init__()`、`command()`、`feedback()` |
| [`robotics/contacts.py`](https://github.com/qiuxiuhao/embodied_intelligence/blob/main/robotics/contacts.py) | 接触力读取、双侧接触判定 | `CubeContact`、`ContactSnapshot`、`PandaContactDetector.read()` |
| `robotics/kinematics.py`（Day 9） | FK / TCP / Jacobian | `get_tcp_pose()`、`get_tcp_jacobian()` |
| `controllers/joint_controller.py`（Day 10） | 关节目标位置限速 | `reset()`、`step()` |

---

## 1. Cube 场景：为什么它能真的被抓起？

文件：`assets/franka_panda/scene_cube.xml`

- 使用 `<include file="scene.xml"/>` 复用 Panda 原有机械臂和地面。
- Table：`pos="0.50 0 0.10"`、`size="0.18 0.22 0.10"`。MuJoCo 的 `box.size` 是**半尺寸**，所以桌面顶部 `z=0.20 m`。
- Cube：`pos="0.50 0 0.225"`，几何 `size="0.025 0.025 0.025"`，故边长 **0.05 m**；其底面 `0.225-0.025=0.20 m`，初始位于桌面。
- Cube 质量 `m=0.10 kg`，惯量 `I=ma²/6≈4.1667×10⁻⁵ kg·m²`；碰撞 `geom` 配有摩擦，`condim="3"`。
- Cube 包含 `<freejoint name="cube_free"/>`，允许**3 平移 + 3 旋转**，而不是通过焊接等方式跟随夹爪。
- **重要维度变化**：纯 Panda 时 `nq=9,nv=9,nu=8`；加一个自由 Cube 后一般为 `nq=16,nv=15,nu=8`，因为 freejoint 的 `qpos` 占 **7 维**（XYZ + wxyz），`qvel` 占 **6 维**。
- 额外定义 `day11_home`，其 `qpos` 包含 Panda 九个关节状态与 Cube 七个状态。**切勿默认沿用只记录 Panda 的旧 `home` 就能正确重置 Cube。**

**理解重点**：Cube 是否运动，由动力学、摩擦和接触力决定；**不能把强行修改 Cube 的 `qpos` 或让它粘在手上当成抓取成功**。

---

## 2. 六维 IK：为什么 `R_target @ R_current.T`？

文件：`controllers/ik_controller.py` → `rotation_error_world()`

### 2.1 世界系相对旋转

记 `R_current`、`R_target` 为 **TCP 局部坐标轴到世界坐标系**的旋转矩阵。我们要找一个左乘的、以世界系表示的旋转修正 `R_Δ`：

$$
R_{target}=R_{\Delta}R_{current}
$$

两边右乘 `R_current⁻¹`，且旋转矩阵 `R⁻¹=Rᵀ`：

$$
\boxed{R_{\Delta}=R_{target}R_{current}^{T}}
$$

对应：

```python
relative = np.ascontiguousarray(target_rotation @ current_rotation.T)
```

而 `R_current.T @ R_target` 是**当前 TCP 局部坐标系**下的右乘旋转误差，不可直接与世界系 `jacr` 混用。MuJoCo 的几何旋转 Jacobian `Jr` 对应世界系角速度，因此这里使用世界系误差。

### 2.2 旋转矩阵 → 三维误差向量

```python
mujoco.mju_mat2Quat(quaternion, relative.reshape(9))
if quaternion[0] < 0:
    quaternion *= -1
mujoco.mju_quat2Vel(error, quaternion, 1.0)
```

- 转为 Quaternion（MuJoCo 顺序 `wxyz`）。
- `q` 与 `-q` 代表同一个姿态；选择标量部分非负，让旋转误差采用主要/较短的轴角表示。
- `mju_quat2Vel(..., 1.0)` 在此将相对旋转转为三维**旋转向量** `e_r = θu`（数值上单位 rad，`dt=1`）；不是实际测量的关节角速度。

数学速记：

$$
\boxed{e_r = \operatorname{Log}(R_tR_c^T)^\vee \in \mathbb R^3}
$$

这里 `Log` 是 SO(3) 的旋转对数，`∨` 将反对称矩阵转为三维向量。角度接近 180° 时，轴的表示可能有歧义。

---

## 3. `residual()`：为什么需要一个 3 维或 6 维任务误差？

文件：`controllers/ik_controller.py` → `PandaIKSolver.solve()` 内部局部函数 `residual()`。

### 3.1 位置和姿态各自计算

```python
position, _, rotation = get_tcp_pose(self.data)
ep = target_position - position
er = (np.zeros(3) if target_rotation is None
      else rotation_error_world(target_rotation, rotation))
```

- **位置误差**：`e_p = p_target - p_current`，形状 `(3,)`，单位 m。
- **姿态误差**：`e_r = Log(R_target R_currentᵀ)∨`，形状 `(3,)`，单位 rad。
- 没有 `target_rotation`：只解位置，姿态误差视为零。

### 3.2 `orientation_weight` 做尺度统一

```python
task_error = (
    ep if target_rotation is None
    else np.concatenate([ep, orientation_weight * er])
)
```

完整位姿 IK 定义：

$$
\boxed{e_{task}=\begin{bmatrix}e_p\\w_r e_r\end{bmatrix}} 
$$

- `w_r=orientation_weight`，代码默认 **0.15 m/rad**。
- 它把 rad 尺度的旋转误差转换为用于优化的**米量纲等效数值**；是一种任务权重，不表示真正产生了相应位移。
- 不带姿态目标时 `e_task.shape=(3,)`；带姿态目标时 `(6,)`。

**例子**：位置误差 `0.01m`、姿态误差 `0.1rad`，旋转部分加权 `0.15×0.1=0.015m`；合并范数约为 `sqrt(0.01²+0.015²)=0.018m`。

### 3.3 为什么返回三个量？

```python
return task_error, float(np.linalg.norm(ep)), float(np.linalg.norm(er))
```

| 返回项 | 作用 | 单位 |
|---|---|---|
| `task_error` | DLS 更新方向、线搜索总误差 | 任务加权尺度（m） |
| `position_norm` | 独立位置收敛判定 | m |
| `orientation_norm` | 独立姿态收敛判定 | rad |

当前默认门槛：`position_norm < 0.002m`（**2 mm**），`orientation_norm < 0.01rad`（约 **0.57°**，仅姿态 IK 有实质约束）。

**核心区别：** 加权误差的范数用于比较候选关节解是否更好；真正的 IK success 还需**位置与姿态分别过线**。

---

## 4. DLS：把任务误差转为七关节增量

文件：`controllers/ik_controller.py` → `_compute_dls_step(jacobian, position_error)`。

### 4.1 从 Jacobian 一阶近似出发

$$
\Delta x \approx J\Delta q
$$

- 七维关节：`Δq ∈ R⁷`。
- 位置任务：`Jp ∈ R^(3×7)`，`e_p ∈ R³`。
- 位姿任务：`Jtask ∈ R^(6×7)`，`e_task ∈ R⁶`。
- `J` 不是方阵，不能直接求通常意义的 `J⁻¹`。

阻尼最小二乘的优化问题：

$$
\boxed{\min_{\Delta q}\;\|J\Delta q-e\|^2+\lambda^2\|\Delta q\|^2}
$$

前半部分希望减小任务误差，后半部分抑制很大的关节变化；`damping=λ` 默认 `0.05`。

### 4.2 代码对应的 DLS 公式

$$
\boxed{\Delta q=J^T(JJ^T+\lambda^2 I)^{-1}e}
$$

```python
jjt = jacobian @ jacobian.T
damped_matrix = jjt + self.damping ** 2 * np.eye(jacobian.shape[0])
solution = np.linalg.solve(damped_matrix, position_error)
delta_q = jacobian.T @ solution
```

`np.linalg.solve(A,e)` 是解线性方程 `Ay=e`，**不是显式计算矩阵逆**；对于这里的 3/6 维任务通常比构造 7×7 系统更合适。

实际形参叫 `position_error`，但传入的可能是 `task_error`（包括姿态），所以阅读时把它理解成“任务误差”。

### 4.3 位置和姿态必须同步加权

```python
task_jacobian = (
    jacp if target_rotation is None
    else np.vstack([jacp, orientation_weight * jacr])
)
delta_q = self._compute_dls_step(task_jacobian, task_error)
```

对应：

$$
\boxed{J_{task}=\begin{bmatrix}J_p\\w_rJ_r\end{bmatrix},\qquad
 e_{task}=\begin{bmatrix}e_p\\w_re_r\end{bmatrix}}
$$

**不要只加权误差而忘记加权 Jacobian**。对应优化目标是：

$$
\|J_p\Delta q-e_p\|^2+w_r^2\|J_r\Delta q-e_r\|^2+\lambda^2\|\Delta q\|^2
$$

> **近似性**：`Jr` 是世界系瞬时角速度 Jacobian；将 SO(3) 对数误差直接用它线性化是一阶局部近似，大角度误差时并非严格的对数残差 Jacobian。求解器通过小步迭代＋重新 FK＋线搜索缓解偏差。

### 4.4 DLS 输出仍要限制与验证

1. 先限 `||Δq||₂ ≤ max_step_norm`，当前默认 **0.08 rad**。
2. 依次尝试 `α ∈ {1, 0.5, 0.25, 0.125, 0.0625, 0.03125}`。
3. 计算 `q_candidate = clip(q + αΔq, lower_limits, upper_limits)`，同时保留关节边界余量 `joint_margin=0.01 rad`。
4. 对每个候选解调用 FK 和 `residual()`，仅在**加权任务误差下降**时接受。
5. 不改善则 `reason="stalled"`；迭代上限 `reason="maximum_iterations"`；达到阈值才 `success=True`。

**IK 求解成功 ≠ 物理执行成功 ≠ 无碰撞抓取成功。**

---

## 5. Gripper Controller：名义开度（m）怎么对应 actuator8？

文件：`controllers/gripper_controller.py` → `PandaGripperController`。

### 5.1 两个手指、一个控制输入

- `finger_joint1/2`：两个滑动关节，各有真实 `qpos/qvel`。
- `actuator8`：一个基于 `split` fixed tendon 的执行器。
- 代码先验证真实 actuator 配置（固定 gain、affine bias、无激活动态、控制限幅），再推导映射；**不是所有 Panda 实现都适用相同系数**。

名义开度定义：

$$
\boxed{w=q_{finger1}+q_{finger2}}
$$

`tendon split` 的长度：

$$
\boxed{l_{tendon}=0.5 q_{finger1}+0.5 q_{finger2}=w/2}
$$

它是**关节滑动位移之和**，不等于精确测量的指垫接触面间距。

### 5.2 控制公式及来源

当前模型仿射伺服的静态平衡表达：

$$
F=g\,u+b_0+b_1\,(gear\cdot l_{tendon})
$$

设名义目标开度的静态输出 `F=0`，解得：

$$
\boxed{u=-\frac{b_0}{g}-\frac{b_1\,gear}{2g}\,w}
$$

对应源码：

```python
self.ctrl_per_meter = -bias1 * gear * 0.5 / gain
self.ctrl_offset = -bias0 / gain
ctrl = self.ctrl_offset + self.ctrl_per_meter * width
```

`command(data, opening_m)` 对开度和原生 `ctrl` **分别限幅**，最后赋值 `data.ctrl[self.actuator_id]`。

**命令与实际状态不能混淆：** 发 `opening_m=0` 不代表手指立即合拢，更不等于已经抓到 Cube；碰到 Cube 后，接触约束可能使真实开度停在非零值。

### 5.3 `feedback()` 的实际反馈

| 字段 | 含义 |
|---|---|
| `finger_position_m` | 左右手指实际位移 |
| `finger_velocity_m_s` | 左右手指实际速度 |
| `opening_m` | `q_finger1+q_finger2` 的实际值 |
| `tendon_length_m` | MuJoCo 计算的腱长度 |
| `actuator_force_N` | 执行器输出力（不直接等于两侧 Contact 法向力） |

---

## 6. Contact Detector：何时叫“接触”，何时叫“抓住”？

文件：`robotics/contacts.py` → `PandaContactDetector.read(data)`。

### 6.1 如何读取一条有效接触

```python
for i in range(data.ncon):
    c = data.contact[i]
    if c.efc_address < 0:
        continue
    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force)
    if force[0] < min_force_N:
        continue
```

- `data.ncon`：当前 Contact 数；**>0 并不意味着 Cube 被抓住**。
- `c.geom1/geom2`：碰撞接触的两个几何体 ID；通过 `model.geom_bodyid` 识别是否为左/右手指。
- `c.efc_address`：对应约束的地址；小于零则跳过。
- `mj_contactForce`：返回接触坐标系下六维接触力/力矩；`force[0]` 取法向力分量。
- `min_force_N` 默认 **0.05 N**：只是过滤微弱接触，不是稳定抓取所需的理论夹持力。

### 6.2 为什么统一 `normal_on_cube` 方向？

```python
normal = c.frame[:3].copy() * (1 if g2 == cube_geom_id else -1)
```

原始 Contact 有 `geom1/geom2` 顺序；为了让后续判断与该顺序无关，代码把法向统一定义成**作用在 Cube 上的法向方向**。

### 6.3 双侧夹持候选 `grasp_candidate`

定义：

- `axis = hand.xmat.reshape(3,3)[:,1]`：Hand 局部 **Y** 轴映射到世界系后的夹爪闭合轴。
- `center = cube_geom.xpos`：Cube 中心。
- 左、右手指必须分别有超过最小力阈值的有效 Cube 接触。

对于左接触点 `pL` 与右接触点 `pR`，必须满足：

$$
(p_L-c)\cdot axis>0.005
$$

$$
(p_R-c)\cdot axis<-0.005
$$

它们位于 Cube 中心沿夹爪闭合轴的**不同侧**，至少有 5mm 的投影偏移。

同时法向需要朝向 Cube 内部、左右相对：

$$
n_L\cdot axis<-0.5,\quad n_R\cdot axis>0.5,\quad n_L\cdot n_R<-0.5
$$

也就是：**两侧有接触 + 接触点合理 + 接触法向相对**，才是 `grasp_candidate=True`。

### 6.4 `ContactSnapshot` 字段速查

| 字段 | 含义 |
|---|---|
| `finger_contact` | 至少一侧手指对 Cube 有有效受力接触 |
| `grasp_candidate` | 双侧有效接触符合几何/法向筛选（只表示候选） |
| `cube_table` / `cube_floor` | Cube 是否存在与桌面/地面的有效受力接触 |
| `finger_table` | 手指是否与桌面有有效受力接触 |
| `left_force_N` / `right_force_N` | 有效左/右接触法向力的累计值 |
| `cube_contacts` | 当前保存的有效接触记录 |

**警惕**：

- `grasp_candidate=True` 是**瞬时几何和受力证据**，不是抓取稳定性证明。
- `actuator_force_N` 与 `left/right_force_N` 是不同层级的力，不能直接相等。
- `data.contact` 与 `mj_contactForce` 必须与同一求解后的物理状态对应；若手动改 `qpos`，按需要执行完整的 `mj_forward()` 后再读取。

---

## 7. Lift / Release：任务成功应该怎么理解？

以下是**任务层判定原则**，不是 `contacts.py` 单次 `read()` 自动提供的成功结论。

推荐区分四层：

1. **Close Command**：已发送闭合命令，仅表明执行器想闭合。
2. **Finger Contact**：至少一个手指和 Cube 发生有效受力接触。
3. **Grasp Candidate**：两个手指在 Cube 两侧形成合理法向接触。
4. **Stable Lift**：Cube 真实离开支撑面，达到高度阈值并保持足够时间，通常还需合理的持续夹持证据。

简单的任务层检查示意：

$$
\boxed{z_{cube}-z_{initial}>h_{threshold}}
$$

并要求满足持续时间：

$$
\boxed{t_{stable}\ge t_{threshold}}
$$

真实 Release 也不能只检查“已发张开命令”，应观察 Cube 和夹爪分离以及 Cube 受重力运动后的状态。

对这份 Day 11 场景，Cube 重力约为：

$$
mg=0.10\times9.81=0.981\ \mathrm N
$$

仅凭一次 0.05 N 接触无法证明有足够的摩擦支撑其重量。

---

## 8. 复习时最容易犯的 8 个错误

1. 将 `R_target @ R_current.T` 和 `R_current.T @ R_target` 当成完全等价；实际上对应世界系与当前局部系的不同误差定义。
2. 用普通矩阵元素相减表示三维姿态误差；这里应使用 SO(3) 相对旋转的轴角向量。
3. 混淆 `e_p(m)` 与 `e_r(rad)`；代码通过 `orientation_weight(m/rad)` 联合优化。
4. 只给 `e_r` 加权，不给 `Jr` 加权；正确写法两者都加权。
5. 把 DLS 输出视为最终命令；实际还有限范数、关节限位、线搜索、位置控制器和真实动力学。
6. 将 `data.ctrl[7]` 当成夹爪开度（m）；需要由伺服参数映射，实际开度以 `feedback()` 为准。
7. 用 `data.ncon>0`、`finger_contact=True` 或 `grasp_candidate=True` 直接宣告抬升成功。
8. 加入 Cube 的 `freejoint` 后仍假定 `nq=9` 或使用不包含 Cube 状态的旧 Home Keyframe。

---

## 9. 两分钟速记：五个核心公式

1. **世界系旋转误差**：
   \[e_r=\operatorname{Log}(R_{target}R_{current}^T)^\vee\]
2. **任务残差**：
   \[e_{task}=\begin{bmatrix}e_p\\w_re_r\end{bmatrix},\quad e_p=p_t-p_c\]
3. **加权 Jacobian**：
   \[J_{task}=\begin{bmatrix}J_p\\w_rJ_r\end{bmatrix}\]
4. **DLS 关节增量**：
   \[\Delta q=J_{task}^T(J_{task}J_{task}^T+\lambda^2I)^{-1}e_{task}\]
5. **夹爪名义开度控制**：
   \[u=-\frac{b_0}{g}-\frac{b_1\,gear}{2g}w,\quad w=q_{finger1}+q_{finger2}\]

## 10. 下次打开仓库建议的阅读顺序

`scene_cube.xml` → `robotics/kinematics.py` → `controllers/ik_controller.py`（旋转误差、残差、DLS）→ `controllers/joint_controller.py` → `controllers/gripper_controller.py` → `robotics/contacts.py` → Day 11 实验脚本。

每次读到不熟悉的函数，先追问三个问题：**输入什么（坐标系/shape/单位）？输出什么？输出被谁消费？** 这样能把代码、数学公式和真实物理行为重新串起来。
