# Day 10 — Panda IK 与末端位置控制

## 一、学习目标

1. 理解 Inverse Kinematics。
2. 理解 Jacobian Pseudoinverse。
3. 掌握 Damped Least Squares。
4. 理解 IK 迭代与关节限位。
5. 实现 Panda Joint Position Controller。
6. 完成实际 TCP Reach Task。

## 二、Day 9 与 Day 10 的区别

Day 9：

Joint Configuration
→ Forward Kinematics
→ TCP Pose

Day 10：

Target TCP Position
→ Inverse Kinematics
→ Target Joint Configuration
→ Position Controller
→ Actual TCP Position

## 三、IK

目标位置： p_target ∈ R³

当前关节：q ∈ R⁷

当前位置：p_current = FK(q)

位置误差：e = p_target - p_current

目标：||e|| < tolerance

## 四、Damped Least Squares

DLS： Δq = Jᵀ(JJᵀ + λ²I)⁻¹e

其中：
J ∈ R^(3×7)
e ∈ R³
Δq ∈ R⁷

阻尼：
λ = 0.05

位置误差阈值：

2 mm

## 五、IK 迭代

1. 设置当前 q。
2. 使用 FK 计算 TCP 位置。
3. 计算目标位置误差。
4. 计算 Jacobian。
5. 使用 DLS 求 Δq。
6. 限制关节增量和关节范围。
7. 使用线搜索选择新的 q。
8. 重复直到收敛或失败。

## 六、IK 失败

可能原因：

- 目标不可达
- 初始姿态不合适
- 接近奇异配置
- 关节限位
- 迭代次数不足
- 求解器停滞

IK 求解失败不一定意味着目标绝对不可达。

## 七、位置控制

IK 输出： q_target

Position Controller 输出： data.ctrl

MuJoCo 执行： mj_step()

然后使用：get_tcp_pose()

计算实际 TCP 世界位置。

## 八、仿真成功条件

TCP Position Error：

< 15 mm

最大实际关节角速度：

< 0.10 rad/s

持续满足：

0.4 s
## 九、Day 10 总结

应能够解释：

1. FK 与 IK 的区别。
2. 为什么 Jacobian 不能直接求普通逆矩阵。
3. Pseudoinverse 的意义。
4. DLS 为什么需要阻尼。
5. 为什么 IK 需要迭代。
6. 为什么存在多个 IK 解。
7. 为什么需要 Joint Limits。
8. 为什么 IK Success 不等于 Physical Reach Success。
9. Position Command 与 Actual Joint State 的区别。
10. 为什么抓取任务还需要末端姿态约束。