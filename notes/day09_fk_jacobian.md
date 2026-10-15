# Day 9 — Franka Panda FK、Jacobian 与末端位姿

## 一、学习目标

1. 理解 Forward Kinematics。
2. 理解 Hand Frame 与 TCP。
3. 掌握 Position、Quaternion 和 Rotation Matrix。
4. 理解 SE(3) 齐次变换矩阵。
5. 掌握平移 Jacobian 和旋转 Jacobian。
6. 使用 Finite Difference 验证 Jacobian。

## 二、使用模型

Franka Emika Panda

机械臂自由度：7

完整模型：
- nq = 9
- nv = 9
- nu = 8

## 三、Forward Kinematics

输入：

q = [q1, q2, ..., q7]

输出：

TCP Position
 
TCP Orientation

数学：

T_tcp = FK(q)

## 四、Hand Frame 与 TCP

Hand Frame：

机器人 hand Body 的坐标系。

TCP：

工具中心点，通常位于 Hand 坐标系中的某个位置。

Day 9 使用：

TCP_OFFSET_LOCAL = [0, 0, 0.10]

注意：

目前为教学参考点，尚未根据抓取几何校准。

## 五、坐标变换

p_tcp_world =
    p_hand_world
    +
    R_hand_world @ p_tcp_local

## 六、Pose

Position：

[x, y, z]

Quaternion：

[w, x, y, z]

Rotation Matrix：

R ∈ SO(3)

Homogeneous Transform：

T ∈ SE(3)

## 七、Jacobian

平移 Jacobian：

Jp ∈ R^(3×7)

旋转 Jacobian：

Jr ∈ R^(3×7)

完整 Jacobian：

J ∈ R^(6×7)

## 八、Velocity Mapping

Linear Velocity：

v = Jp @ qdot

Angular Velocity：

omega = Jr @ qdot

## 九、Small Displacement

Delta p ≈ Jp @ Delta q

注意：

这是局部一阶近似。

## 十、Finite Difference

中心差分：

df/dq ≈ [f(q+eps)-f(q-eps)] / (2eps)

测试：

Home Configuration

Perturbed Configuration

## 十一、Day 9 总结

我需要能够解释：

1. FK 的输入和输出。
2. Hand 与 TCP 的区别。
3. Local Frame 与 World Frame 的关系。
4. Rotation Matrix 的性质。
5. Jacobian 的物理含义。
6. 为什么 Jacobian 取决于当前关节配置。
7. 为什么 J 的形状是 6×7。
8. 为什么有限差分能够验证 Jacobian。
9. 为什么不能简单对 Quaternion 求差得到角速度。
10. 为什么七自由度机械臂具有运动学冗余。
