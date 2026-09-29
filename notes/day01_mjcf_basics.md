| 名称 | 理解 | Pendulum 中 |
|---|---|---|
| `body` | 刚体节点/坐标系 | 整个摆杆 |
| `joint` | body 如何相对 parent 运动 | 绕 Y 轴旋转 |
| `geom` | 几何形状、碰撞、质量等 | capsule 摆杆 |
| `worldbody` | 世界根节点 | 整个场景 |
| `MjModel` | 模型/物理规则 | XML 编译后的世界 |
| `MjData` | 当前运行状态 | 当前角度、速度、时间 |
| `qpos` | generalized position | 摆杆角度 |
| `qvel` | generalized velocity | 摆杆角速度 |