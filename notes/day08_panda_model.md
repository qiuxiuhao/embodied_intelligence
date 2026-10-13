# Day 8 - Franka Panda 多关节机器人模型

## 一、学习目标
本日开始 Stage 01。

主要学习：

1. Franka Panda 机器人模型。
2. 多关节运动学树。
3. Joint State。
4. Actuator 与 Control。
5. Home Keyframe。
6. 多关节机器人可视化。

---
## 二、使用模型

来源：

Google DeepMind MuJoCo Menagerie

模型：

Franka Emika Panda

模型文件：

assets/franka_panda/panda.xml

场景文件：

assets/franka_panda/scene.xml

---
## 三、机器人结构

Panda 机械臂具有 7 个旋转关节：

joint1
joint2
joint3
joint4
joint5
joint6
joint7

夹爪具有 2 个滑动关节：

finger_joint1
finger_joint2

因此：

njnt = 9

nq = 9

nv = 9

---

## 四、Actuator

模型共有 8 个执行器。

前 7 个：

控制机械臂关节的位置目标。

第 8 个：

通过 Tendon 控制两个夹爪手指。

因此：

nu = 8

注意：

Joint DOF 数量不一定等于 Actuator 数量。

---

## 五、机器人状态

qpos：

关节位置与配置。

qvel：

关节速度。

ctrl：

执行器控制输入。

当前 Panda 模型：

qpos.shape = (9,)

qvel.shape = (9,)

ctrl.shape = (8,)

---

## 六、Home Pose

模型提供：

home Keyframe

可以使用：

mj_resetDataKeyframe()

恢复对应的初始状态。

Home 包含：

关节位置和控制目标。

---

## 七、多关节控制

控制流程：

Target Joint Position
    ↓
data.ctrl
    ↓
Position Servo Actuator
    ↓
MuJoCo Dynamics
    ↓
New qpos / qvel

需要区分：

Target Position

Actual Position

---

## 八、Day 8 总结

本日成功：

1. 加载 Franka Panda 模型。
2. 读取全部关节状态。
3. 读取全部执行器信息。
4. 恢复 Home Pose。
5. 实现基础多关节运动。
6. 实现夹爪开合控制。

---