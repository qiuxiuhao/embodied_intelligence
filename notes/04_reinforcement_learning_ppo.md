# 具身智能强化学习：从手写 PPO 到 Panda Reach

统合日期：2026-10-25。整合 Day5–7 的手写 PPO 与 Day14 的 Reach 实验。本项目 PPO 的成果是单关节控制和机械臂 Reach；完整抓放的学习成果属于 [BC](03_imitation_learning_bc.md)，不能把两者成绩混写。

## 1. 强化学习与 BC 的信息来源不同

| 项目 | BC | PPO |
| --- | --- | --- |
| 更新来源 | 固定专家观测—动作对 | 当前策略与环境交互的 rollout |
| 更新目标 | 模仿专家动作 | 提高预期折扣回报 |
| 关键估计 | 监督损失 | 价值、Advantage、策略概率比 |
| 本项目用途 | 94D/4D 抓放 | 单关节、Reach v2 |

PPO 是 on-policy 方法：一批当前策略数据可做多轮有限更新，随后需要重新采样。不能无限重复旧 rollout 而仍称为同样的 PPO 过程。BC 用了 SB3 的 PPO 容器，也不意味着它执行了强化学习。

## 2. Transition、回报、Actor 与 Critic

一条 transition 至少包含：

\[
(o_t,a_t,r_t,o_{t+1},terminated_t,truncated_t).
\]

折扣回报：

\[
G_t=\sum_{k\ge0}\gamma^k r_{t+k}.
\]

Actor 表示 `πθ(a|o)`，Critic 估计 `Vφ(o)`。Advantage 描述某动作相对当前状态平均表现的好坏，不能直接用未折扣回报或即时奖励替代。

单关节观测是角度、角速度、目标角；随机、PD、学习策略可以在同初态种子上做比较。到 Panda 时，策略仍输出控制命令，IK、执行器和动力学负责真正运动。

## 3. TD 误差与 GAE：两种边界掩码

\[
\delta_t=r_t+\gamma(1-terminated_t)V(o_{t+1})-V(o_t).
\]

\[
A_t=\delta_t+\gamma\lambda(1-done_t)A_{t+1},
\quad done_t=terminated_t\lor truncated_t.
\]

\[
R_t^{target}=A_t+V(o_t).
\]

真正终止不自举；外部时间截断通常可以对真实末状态自举，但 GAE 递推不能穿过回合边界。必须分别保存 terminated/truncated，不能因为采样器自动 reset 就把下一回合初态拿来估计上一回合末状态价值。

本项目的两种任务语义：

- 单关节与旧 Reach/Day15 外部截断：按对应实现处理末状态 bootstrap。
- Reach v2：10 s 是任务定义内的有限期限，时间进入观测，超时 terminated，不 bootstrap。

`γ=0.99`、`λ=0.95` 时 γλ=0.9405，只是优势递推的衰减尺度，不是硬截成固定步数。

## 4. 概率比与 PPO clip

\[
\rho_t(\theta)=\exp\left(\log\pi_\theta(a_t|o_t)
-\log\pi_{\theta_{old}}(a_t|o_t)\right).
\]

\[
L^{clip}=
\mathbb E\left[\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right].
\]

训练最小化负的 clipped objective，并加入价值损失和熵项。Advantage 正负决定概率应被鼓励增加还是减少，因此必须使用 min 的完整表达，不能只把 ratio 裁剪后随意乘 A。

一个常见总损失表示为：

\[
\mathcal L=-L^{clip}
+c_v\mathcal L_V-c_eH.
\]

不同实现对 value loss 是否内含 1/2 不同；本项目手写 update.py 内部先算 0.5×均方误差，再乘 value_coef，比较参数时需同时看实现。

`clip_range`、梯度范数裁剪与 target_kl 是不同层面的保护。clip 不保证新旧策略 KL 严格受限，KL 早停也不回滚已经执行的梯度更新。

## 5. 本项目两套动作分布不能混同

### 5.1 手写单关节 PPO

Actor 与 Critic 分别使用两层 64 单元 Tanh MLP；高斯原始动作经 tanh 与仿射变换映射到合法动作范围：

\[
a=b+s\tanh(z),\qquad z\sim\mathcal N(\mu,\sigma).
\]

log probability 包含变量变换修正：

\[
\log\pi(a)=\sum_j\left[
\log\mathcal N(z_j;\mu_j,\sigma_j)
-\log\left(s_j(1-\tanh^2z_j)+\epsilon\right)\right].
\]

重新评估动作时使用逆 tanh；不能用原始高斯密度直接当有界动作密度。代码中的 entropy 使用原始高斯熵作为近似诊断，不是变换后动作分布的精确熵。

### 5.2 SB3 Reach PPO

当前使用普通对角高斯，送入 Box 环境前裁剪动作；rollout buffer 保留原始动作及其 log_prob。不能把它描述成与手写 tanh 策略相同。

环境内记录的越界率还可能遗漏 SB3 上游裁剪，因此回调额外测量 raw XYZ 动作越界。只看环境收到的动作均在范围内，不能证明策略没有饱和。

## 6. 手写实现的阅读顺序与历史结果

```text
ppo/model.py → collector.py → gae.py → update.py
            → 23 数学检查 → 24 完整训练 → 25 统一比较
```

- collector 保存动作、旧 log_prob、价值、末状态价值和独立结束标志。
- update 对优势标准化，打乱 minibatch，更新 actor/critic 并记录 KL、clip fraction 等。
- 教学 collector 每次调用起始会 reset，不是完整保存 MuJoCo 半条回合的连续训练器。
- 24 号保留独立验证选模；旧 21/22 简版入口已删除。

Day07 的历史同批评估为：Random 14%，PD 100%，三个手写 PPO seed 均 100%，SB3 100%。这只说明当时单关节分布下的表现，不能推出机械臂抓放也已由 PPO 解决。

当前 17、24、25 号基础脚本仍有搬迁后的产物根路径问题：`parents[1]` 会指向 scripts 而非仓库根目录。Day05/06/07 产物已按用户要求删除，17 号和 25 号还分别缺少原轨迹和模型。本文保留其学习价值，不将它们列为当前无修改可运行的完整训练命令；Day07 数字仅作为历史笔记记载。

## 7. Reach v2 为什么使用专用环境

旧 Reach 使用公共 94D/4D 观测动作，包含夹爪和多任务信息；v2 缩为 46D/3D，夹爪固定张开。

| 46D 字段 | 维数 | 归一化 |
| --- | ---: | --- |
| 臂关节位置、速度 | 14 | 是 |
| TCP 世界位置 | 3 | 是 |
| target−TCP、command−TCP | 6 | 是 |
| 世界系姿态误差 | 3 | 是 |
| 当前关节命令、IK 目标 | 14 | 是 |
| 上一 XYZ 动作 | 3 | 否 |
| Reach 计时、剩余预算、IK 连败比例 | 3 | 否 |

40D 物理量缩放，6D 有界量保持原值。剩余时间与任务记忆不应因为稀有事件方差极小而被放大成突变输入。

XYZ 每轴每控制步最多 5 mm；控制周期 0.04 s；关节命令限速 0.5 rad/s；重力补偿开启。真实 Reach 条件仍为位置误差 <9 mm、姿态 <0.035 rad、最大关节速度 <0.10 rad/s，连续 0.4 s。

v2 在物理子步完成保持后锁存成功事件；本步如有物理故障或无效控制，失败/有效性判断仍优先。

## 8. 当前 Reach 奖励

令 d 为 TCP 误差，v 为最大绝对臂关节速度，a 为裁剪后 XYZ 动作，Δt 为实际仿真时间：

\[
\begin{aligned}
r={}&-2\min(d/0.20,5)\Delta t\\
&+0.5\exp(-(d/0.012)^2)\exp(-(v/0.10)^2)\Delta t\\
&-0.01\,mean(a^2)\Delta t
-0.001\,mean((a-a_{prev})^2)\\
&+10I_{\text{success}}-10I_{\text{failure}}
-I_{\text{IK failure}}.
\end{aligned}
\]

近且静项仅在靠近目标且低速时明显，不鼓励在远处原地静止。动作变化成本按控制步计，未乘 Δt；修改控制频率后不能沿用其数值解释。

无效 IK/物理失败步禁止正奖励，数值失败明确终止。普通有限期限超时不额外制造物理失败惩罚。v2 不是旧势函数奖励的小幅调权，新旧原始 Return 不应直接比较。

## 9. 当前训练配置

| 参数 | Reach v2 |
| --- | --- |
| seed、设备、线程 | 42、CPU、1 |
| 环境数、n_steps | 2、每环境 1024；每 rollout 2048 步 |
| Actor / Critic | 各 [128,128] Tanh |
| learning_rate | 1e-4 |
| batch_size、n_epochs | 256、最多 5 |
| gamma、gae_lambda | 0.99、0.95 |
| clip_range、target_kl | 0.2、0.01 |
| ent_coef、vf_coef、max_grad_norm | 0、0.5、0.5 |
| 初始 std | 0.3，后续可学习 |
| normalization | physical obs；不归一化 reward |
| 验证集 | 100000–100031，共 32 |
| 最大累计预算 | 1000000；按完整 rollout 实际为 999424 |

完整更新名义为 2048/256×5=40 个 optimizer step；KL 早停时会减少。不能把 SB3 的 epoch 计数直接当实际 optimizer step 次数。

## 10. 历史失败教会了什么

旧版存在“曾到达位置范围，但无法低速保持”的现象：只看最终距离或曾进入目标球不足以判断策略会停稳。归一化稀有计时、弱距离成本、较大更新幅度和位置伺服重力静差都值得分开分析。

旧五组全历史验证最佳在同一批 100 回合测试上的成功数为 Original 0、A 13、B 4、C 4、D 5。它们分别尝试选择性归一化、低初始噪声、近 Home 课程或更强距离成本；没有某个单独变更可靠解决旧 Reach。

历史控制校准、Jacobian/IK 检查支持修复控制精度而非优先怀疑运动学；更新消融支持降低学习率并增加 KL 保护。但 v2 同时调整了环境、控制、观测与奖励，因此其提升不能全部归功于某一个参数。

旧诊断工具与部分训练链日志已清理，以上是保留的历史结论，不表示本地还能完整重建原消融。训练设备、验证频次、预算和目标分布也不能混作同一个因果对照。

势函数整形的正确望远镜关系是：

\[
\sum_{t=0}^{T-1}\gamma^t
[\gamma\Phi(s_{t+1})-\Phi(s_t)]
=-\Phi(s_0)+\gamma^T\Phi(s_T).
\]

终止边界正确处理时，不能仅由某一步势奖励为负就断言策略应该远离目标；需要区分任务目标、奖励尺度与学习可行性。

## 11. 结项时保留的正式结果

训练共 999424 步；最佳模型由验证规则提前固定为 688128 步。

| 模型 | 原 holdout 100 | 新增 1000 | 新增集平均末态误差 |
| --- | ---: | ---: | ---: |
| 最佳 | 100/100 | 1000/1000 | 2.965 mm |
| 最终 | 100/100 | 1000/1000 | 3.950 mm |

同原 100 回合 Random/Zero 均 0/100。正式成功包含 0.4 s 低速保持，不能替换成“TCP 曾进入距离范围”。1000/1000 是既定任务分布和训练 seed 下的结果，不代表任意场景必定成功或已完成多训练 seed 验证。

主运行：[模型与训练记录](../outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe)，[正式 assessment](../outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/assessment_20261020/assessment_summary.json)。

## 12. 加载与历史复现

模型需要 model.zip、vecnormalize.pkl、manifest.json 配套。推理时冻结统计、关闭奖励归一化；rng.pt 用于恢复算法随机数状态，但没有保存半条物理轨迹，恢复后从新回合采样，不能承诺逐位续训。

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 既有模型的小规模推理检查；输出必须是新文件
python -m training.evaluate_ppo_reach \
  --bundle outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/best_model/step_000688128 \
  --episodes 3 --seed-start 160000 \
  --output /tmp/embodied_ppo_reach_review.json

# 保留下来的手写 PPO 数学检查
PYTHONPATH=. python scripts/stage0_learn_base/23_test_ppo_math.py
```

macOS 演示使用 `PYTHONPATH=. mjpython scripts/42_view_ppo_reach.py`，追加同样模型、回合与新输出参数。

历史配置的从零复现入口为：

```bash
python -m training.train_ppo_reach \
  --config configs/day14_reach_v2.json \
  --steps 1000000 --total-step-limit 1000000 --device cpu \
  --label reproduction --output-root outputs/reach_reproduction_new
```

项目已结项，上述不是继续训练计划；本文整理没有执行训练。旧 v1 模型不能作为 v2 从零训练的权重/归一化替代品。

## 13. 自查与来源

应能解释：两个 GAE 掩码为何不同；为什么要存 old log_prob；tanh 与裁剪高斯为何不同；clip/KL/梯度裁剪各管什么；为何奖励、Return、value loss 都不能替代成功率；为什么 Reach 成绩不等于抓放成绩。

代码：[手写模型](../ppo/model.py)、[采样](../ppo/collector.py)、[GAE](../ppo/gae.py)、[更新](../ppo/update.py)、[Reach 训练](../training/train_ppo_reach.py)、[回调](../training/ppo_callbacks.py)、[共用工具](../training/reach_common.py)。

历史统合来源（原笔记已清理）：回报、PPO 概念、单关节实验、Day14 完整记录，并核对当前 PPO 实现、Reach v2 配置与正式评测。
