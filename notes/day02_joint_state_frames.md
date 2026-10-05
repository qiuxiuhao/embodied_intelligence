# day 2 - Joint、State 与 Coordinate Frame

## 1. Joint 类型

｜ Joint ｜ DOF ｜ qpos | qvel|
|---|---:|---:|---:|
| hinge | 1 | 1 | 1 |
| slide | 1 | 1 | 1 |
| ball | 3 | 4 | 3 |
| free | 6 | 7 | 6 |

## 2.核心理解
- nq = qpos 的维度
- nv = qvel 的维数 / DOF 数量
- nq 不一定等于 nv
- quaternion 使用的4个数表示 3D orientation
- free joint 的 qpos = XYZ + quaternion
- free joint 的 qvel = linear velocity + angular velocity

## 3. Pose
 Pose = Position + Orientation

## 4. Coordinate Frame
body.pos:
相对于parent 的 local position

data.body(...).xpos:
当前 world position

## 5. Kinematics VS Dynamics

Kinematics：
给定 q，求物体的pose

Dynamics：
根据力、质量、速度计算运动变化
