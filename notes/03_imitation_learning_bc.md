# 具身智能模仿学习：Behavior Cloning

统合日期：2026-10-25。本文整合 Day15 的专家、数据、监督训练、选模和独立测试。当前结项路线为纯 BC，输入为仿真状态，不是图像；不包含 ACT、DAgger 或在线 PPO 微调。

物理环境见 [仿真环境](01_simulation_environment.md)，抓放动作与成功条件见 [Pick & Place](05_pick_and_place.md)，与 PPO 的区别见 [强化学习](04_reinforcement_learning_ppo.md)。

## 1. BC 学什么

专家提供观测—动作样本：

\[
\mathcal D=\{(o_t,a_t^{E})\},\qquad \hat a_t=\pi_\theta(o_t).
\]

BC 用监督损失使策略动作接近专家动作。它不需要用环境奖励估计 Advantage，也不靠试错采样更新 actor。本项目训练时使用存好的示范；闭环验证时再让网络独立控制真实仿真。

```text
专家物理验收 → 原始回合采集与哈希审计 → train-only 归一化
            → actor 监督训练 → validation 闭环选模 → 固定模型后 test
```

离线动作误差小不保证闭环成功。网络稍有偏差就可能进入专家数据较少覆盖的状态；随后误差可能累积。该机制解释 BC 的一般局限，不能不做实验就认定为本项目所有失败的唯一原因。

## 2. 当前专家和策略使用同一个接口

| 项目 | 当前实现 |
| --- | --- |
| 专家 | PickPlaceExpert |
| 输入 | 未归一化的 94D 状态 |
| 输出 | 四维 float32：XYZ 命令增量和夹爪开度动作 |
| 物理推进 | 仅通过 env.step |
| 环境 | PickPlaceLearningEnv，重力补偿开启 |
| 动作尺度 | 每轴每个控制步最多 0.005 m |
| 控制周期 | 0.04 s |
| 回合上限 | 750 步 / 30 s，外部截断 |

专家不持有 env/model/MjData，不直接写执行器或物体状态。八个阶段为 PRE_GRASP、APPROACH、GRASP、LIFT、TRANSPORT、PLACE、RELEASE、RETREAT；阶段仅作数据标签和诊断，不额外输入 BC 网络。

运输/接近/放置的专家命令速度为 0.10/0.06/0.04 m/s；预抓取和抬升等距离约 0.12 m。专家阶段完成不能替代环境最终 SUCCESS。

## 3. 先验收专家，再采示范

当前正式专家验收共 135 回合：

| 类型 | 回合数 | 当前结果 |
| --- | ---: | ---: |
| 固定 Home | 1 | 1/1 |
| 固定预抓取起点 | 1 | 1/1 |
| 替代放置目标 | 1 | 1/1 |
| Cube/目标角点与偏航组合 | 32 | 32/32 |
| 独立随机 Home 场景 | 100 | 100/100 |

固定/边界组要求全部成功，随机组门槛为 95%。正式报告为 135/135，见 [数据集内验收副本](../outputs/day15_demos_1000_200/acceptance.json)。独立专家验收目录已按用户要求删除，数据协议依赖的内嵌副本保留。这是脚本专家成绩，不是网络成绩。

任务分布是单个 50 mm、0.1 kg Cube、水平桌面、当前摩擦和小范围 XY/偏航随机化；不是任意物体、任意初始关节姿态或实机泛化。

采集固定计划种子，每个尝试只执行一次，不挑成功 seed 补齐数量。失败也保存，只有真实 SUCCESS 才进入 BC 可用名单；专家主动中止时记录 abort，不能伪造环境终止标志。

## 4. 当前数据划分

| 集合 | 种子 | 回合数 | transition 数 | 用途 |
| --- | --- | ---: | ---: | --- |
| train | 400000–400999 | 1000 | 349396 | 拟合归一化、监督更新 |
| validation | 410000–410199 | 200 | 69819 | 离线误差、闭环选模 |
| test | 900000–900199 | 200 | 评估时产生 | 固定模型后的最终测试 |

当前 1200 条示范均成功并通过完整审计。划分单位是完整回合，不能把同一轨迹相邻帧随机拆进训练和验证，否则会低估误差。

test 不需要专家示范，不参与归一化、BC 更新或选 epoch。旧 holdout 500000–500099 已使用；当前 test 也已公开，不能据其继续调参再宣称它仍是全新的最终测试。

旧数据是 500 train/100 validation，共 174892/34892 个 transition；保存在独立旧目录。其 600 个 NPZ 与新目录对应文件逐字节相同，但协议、索引和源码快照独立，本次没有去重或删除。

## 5. 每份数据保存什么

| 文件/字段 | 内容与作用 |
| --- | --- |
| protocol.json | 展开配置、观测/动作语义、来源和网格哈希 |
| acceptance.json | 采集对应的专家验收副本 |
| episodes.jsonl | 全部尝试的索引、种子、结果、阶段历史和文件哈希 |
| manifest.json | 完成标记、各划分汇总与可用文件列表 |
| verification.json | 全量读回审计结果 |
| source/ | 采集时 Python/XML 快照，不是当前训练入口 |
| observations | T×94，动作前原始观测 |
| actions/applied_actions | T×4，提交动作与环境实际接纳动作 |
| next_observations | T×94，动作后的观测，包含真正终态 |
| rewards、reward_terms | 原始环境奖励与分项，用于核验和分析 |
| terminated/truncated | 独立结束标志 |
| phase_ids | 八阶段标签，用于采样和分阶段误差分析 |

同一回合非末步满足 `observations[t+1] == next_observations[t]`。不能把 VecEnv 自动 reset 后的新回合首帧误当成上一回合终态。NPZ 使用 allow_pickle=False 读取。

`verify_dataset()` 检查索引/协议/来源快照/每个回合哈希、种子隔离、形状、相邻状态对齐、奖励与物理成功证据。因此 collect_pick_place_demos.py 不只是采集脚本，也是 BC 加载依赖。

## 6. 只使用 train 拟合归一化

对于选中的物理量：

\[
\tilde o_i=\operatorname{clip}
\left(\frac{o_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},-10,10\right).
\]

`PhysicalVecNormalize` 仅缩放物理连续量；姿态 6D、标志、动作、计时比例等保持原值。统计量只由 train 的动作前 observations 拟合，随后训练、验证和推理均冻结。validation/test 不更新统计，也不做 reward normalization。

如果模型训练于一种归一化，部署时却使用另一种统计，等于改变模型输入分布。必须成套保存策略和 vecnormalize.pkl。

## 7. 网络、损失与阶段均衡

策略使用 SB3 MlpPolicy 的 actor 结构：94 → 256 Tanh → 256 Tanh → 四维动作均值。

这里使用 PPO 类仅为了复用网络结构、分布与存档接口。实际冻结 critic 与 log_std，只更新 actor；不调用 PPO.learn，不采 PPO rollout，PPO num_timesteps 保持零。

每个样本损失为：

\[
\ell_i=w_{xyz}\frac{1}{3}\sum_{j=1}^3(\hat a_{ij}-a^E_{ij})^2
+w_g(\hat a_{ig}-a^E_{ig})^2.
\]

当前两个权重均为 1。XYZ 三维先求均值，再与一个夹爪误差相加，不是无解释地把全部四维求平均。

长等待阶段占据大量帧，短促的松爪阶段容易被总体 MSE 淹没。令 N 为总 transition 数，K 为阶段数，阶段 k 有 n_k 个样本：

\[
P(i)=\frac{1-\beta}{N}+\frac{\beta}{K n_{\text{phase}(i)}},
\qquad \beta=0.5.
\]

每个 epoch 有放回抽 N 个样本，一半按普通帧分布、一半按阶段均衡。它不是让所有阶段帧完全等量，也不是增加独立示范回合。离线报告同时保留 XYZ MSE、夹爪 MSE、符号准确率及每阶段误差。

| 参数 | 当前值 |
| --- | --- |
| seed / device / torch_threads | 42 / CPU / 1 |
| epochs / batch_size / learning_rate | 500 / 512 / 3e-4 |
| hidden_sizes | [256,256]，Tanh |
| phase_balance / max_grad_norm | 0.5 / 1.0 |
| 验证 | epoch 0、每 10 轮、最后一轮；全部 validation |
| initial_std | [0.10,0.10,0.10,0.05]，冻结；确定性评估不采噪声 |

## 8. 离线指标与闭环选模

每次评估：

1. 计算 train/validation 离线动作误差和阶段均衡损失。
2. 在全部 200 个 validation seed 上，由网络独立运行完整抓放。
3. 优先选完整抓放成功率更高的模型；相同则选 validation 阶段均衡损失更小者。
4. 单独保存最后模型，不能假定最后一轮就是最佳。

500 轮、每 10 轮验证共 51 次，即 10200 个闭环验证回合。这个开销改善选模依据，不增加监督更新量，也不保证策略本身更好。

闭环时只做“原始观测→冻结归一化→确定性网络动作→env.step”，不调用专家、不修正策略动作。分别统计抓取率、抬升率和最终成功率；三者不能混为一个分数。

## 9. 固定下来的正式结果

| 模型 | validation | 独立 test |
| --- | ---: | ---: |
| 当前主用 epoch 250 | 155/200，77.5% | 162/200，81.0% |
| epoch 500 最终模型 | 117/200，58.5% | 123/200，61.5% |

epoch 250 在测试前由 validation 选定。最终模型低 19.5 个百分点说明“训练更久”不保证闭环更好；不能反过来用 test 重新挑 epoch。

每 10/50 轮验证的两次运行，最佳都为 epoch 250，对应策略张量、归一化及逐回合测试完全一致；这不是不同训练 seed 的独立重复实验。重复模型实体已去重，四个历史目录别名仍有效。

旧 500 示范 epoch 60 在扩展 validation 上为 165/200（82.5%）。它是历史验证成绩，不是当前预绑定 test，不能直接拿它与 81.0% test 证明新旧谁更强。

证据：[正式 test 汇总](../outputs/day15_bc_fullval_test_20261023/summary.json)、[完整报告](../outputs/day15_bc_fullval_test_20261023/report.md)。

## 10. 当前可用入口

下面是已有实现的使用说明，不是继续训练计划。所有新输出路径必须不存在，避免覆盖历史证据。

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 只读重审已有示范
python -m training.collect_pick_place_demos \
  --verify outputs/day15_demos_1000_200

# 读取已完成主运行，复核 validation；不是重新选模
python -m training.evaluate_bc_pick_place \
  --run outputs/day15_bc_1000_fullval_seed42_500epochs \
  --split validation --output /tmp/embodied_bc_validation_review.json
```

如需展示网络实际抓放，可用 `PYTHONPATH=. mjpython -m training.evaluate_bc_pick_place`，追加相同 --run、--split validation、--episodes 1、--human 和新的 --output。这不是已删除的原始动作回放脚本。

历史方法的复现命令如下，使用新的运行目录，显式指定当前数据集，避免默认读旧 500/100：

```bash
python -m training.train_bc_pick_place \
  --dataset outputs/day15_demos_1000_200 \
  --config configs/day15_bc.json \
  --evaluation-config configs/day15_bc_evaluation.json \
  --output outputs/bc_reproduction_new \
  --epochs 500 --eval-interval 10 --eval-episodes all --device cpu
```

重新采集并非当前训练所必需。相关入口仍为 accept_pick_place_expert 与 collect_pick_place_demos，采集需 --config、--output、--acceptance-report，且验收配置/源码必须匹配。

## 11. 文件、边界与复习

核心代码：[BC 共用工具](../training/bc_pick_place_common.py)、[训练](../training/train_bc_pick_place.py)、[评估](../training/evaluate_bc_pick_place.py)、[选模/测试协议](../training/bc_pick_place_protocol.py)、[数据校验](../training/collect_pick_place_demos.py)。

主模型在 [epoch 250 模型包](../outputs/day15_bc_1000_fullval_seed42_500epochs/best_model/epoch_0250/manifest.json)。model.zip、vecnormalize.pkl、manifest.json 应配套保留。

旧回归测试及 45 号示范回放入口已按用户选择删除；本文不提供其运行命令。采集时源码快照仍在数据目录，物理/mesh 兼容检查通过；历史训练工具的变化不等于原始数据损坏。

应能解释：为什么专家阶段不是网络输入；为什么按回合划分；为什么只用 train 归一化；为什么均衡阶段；为什么 MSE 和闭环成功率不同；为什么只能用 validation 选模。

历史统合来源（原笔记已清理）：专家与数据、旧 BC 实现、1000/200 数据、完整验证与 test、历史模型清理，并核对当前配置、数据 verification 与正式 test JSON。
