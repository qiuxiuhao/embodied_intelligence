# Vision-based Robotic Manipulation Agent：基于 MuJoCo 的视觉机械臂抓取与放置 Agent
## 0. 在mac上部署仿真环境
### 0.1 创建embodied环境
```bash
conda create -n embodied python=3.12 -y
conda activate embodied
python -m pip install -r requirements.txt
```
### 0.2 学习笔记：见notes

## 计划
| 阶段 | 学什么 | 做什么 | 最终产物 | 算力 |
|---|---|---|---|---|
| 00 | 具身智能基本概念 | 搭 MuJoCo 环境 | 可交互机械臂 Demo | Mac |
| 01 | 机器人状态、动作、控制 | 手动控制机械臂抓物体 | Rule-based Agent | Mac |
| 02 | RL | PPO/SAC 学抓取 | State-based RL baseline | Mac / 4090 |
| 03 | Vision Policy | RGB → Action | 视觉抓取 Agent | 4090 |
| 04 | Imitation Learning | BC / ACT | 专家数据模仿 | 4090 |
| 05 | 泛化与完整评估 | 随机物体、位置、光照 | 完整简历项目 | 4090 |