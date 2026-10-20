# Day14：PPO Reach 训练、失败分析与 Reach v2

更新日期：2026-10-20。

本日使用 Stable-Baselines3 PPO 训练 Panda Reach。正式方案为 Reach v2：CPU 从零训练
999,424 步，训练 seed42。验证集选出的688,128步最佳模型和999,424步最终模型，
分别在100个原测试种子、1000个新增种子上全部成功。

推荐使用688,128步最佳模型：新增1000回合平均末端误差2.965 mm，最终模型为3.950 mm。
这是固定Home起点、当前目标随机范围、单训练seed的结果，不代表任意起点、整个工作空间或真实机器人均可靠。

本文合并当前实现、正式评测、旧版实验和审查中仍有用的内容。旧设备部署、批处理、PID、耗时预测和预算变更流水账已移除。

## 1. 任务、成功标准与版本

只训练Reach，不包含grasp/lift/pick_place。Day12脚本Agent和外部目标控制探针没有作为训练数据。
网络使用状态向量，并非视觉输入；项目早期自写的`ppo/`模块不在此训练路径上。

- Home起点；目标范围X=[0.46,0.60]、Y=[-0.08,0.08]、Z=[0.42,0.55] m。
- 工具目标姿态随初始Cube的偏航确定，偏航范围±0.2 rad。
- 物理步长2 ms，frame_skip=20，控制步长40 ms；每回合最多250控制步/10 s。
- 成功要求位置误差<9 mm、姿态误差<0.035 rad、七关节最大绝对速度<0.1 rad/s，连续保持0.4 s。
- 辅助的20 mm距离比例、较高Return或短暂进入9 mm范围，都不能代替成功判定。

| 项目 | 旧版Reach v1 | 正式Reach v2 |
| --- | --- | --- |
| 环境 | PickPlaceEnv(task_mode=reach) | ReachEnv，继承公共物理环境 |
| 观测/动作 | 94D / 4D，包含无关夹爪和多任务信息 | 46D / 3D XYZ，夹爪固定张开 |
| 观测归一化 | Original全维；A–D选择性57D物理量 | 40D物理量归一化，6D有界量保持原值 |
| 相对状态 | 网络需组合已有位置字段 | 显式target−TCP、command−TCP及姿态误差 |
| 底层控制 | 原位置伺服 | 原位置执行器上叠加重力前馈 |
| 奖励 | 折扣距离势函数、弱距离成本、稀疏成功 | 明确距离成本、近且静奖励、动作成本、成功奖励 |
| 超时 | truncated，SB3对终态自举 | 有限任务截止，terminated，不自举 |
| 成功事件 | 控制步末读取保持计时 | 物理子步完成保持即锁存，控制步末返回 |

v2没有放宽物理成功阈值，但动作、控制器、奖励和超时/成功事件记录协议发生了变化。
新旧Return不能直接比较；整套v2有效不等于已经分离各项修改的因果贡献。

## 2. 文件与数据流

| 文件 | 作用 |
| --- | --- |
| `configs/day14_reach_v2.json` | 正式环境、奖励、PPO、网络、验证/测试种子与预算配置 |
| `envs/reach_env.py` | Reach专用动作、观测、奖励、有限时域与成功事件锁存 |
| `envs/pick_place_env.py` | 公共动力学、IK、接触、物理条件和异常处理 |
| `controllers/gravity_compensated_joint_controller.py` | 保持名义命令独立的重力前馈控制器 |
| `training/train_ppo_reach.py` | 从零训练、续训、预算检查、版本/实际设备记录、源码SHA256 |
| `training/reach_common.py` | 配置校验、环境工厂、成套保存加载、独立物理评估 |
| `training/selective_normalize.py` | 根据字段构造归一化掩码；序列化兼容旧94D和新46D |
| `training/ppo_callbacks.py` | 更新后评估、最佳模型选择、保存、KL和动作/奖励诊断 |
| `training/evaluate_ppo_reach.py` | 使用存档配置执行确定性推理与独立测试 |
| `training/compare_ppo_reach.py` | 核对训练步数、配置和初态后比较模型；协议变化需显式标注 |
| `training/plot_ppo_reach.py` | 由真实日志绘制训练曲线 |
| `scripts/42_view_ppo_reach.py` | Mac Viewer入口，复用正式加载/推理代码 |
| `tests/41_test_ppo_reach.py` | 旧版训练/恢复、独立性、归一化等回归测试 |
| `tests/43_test_reach_v2.py` | 新版接口、奖励、控制、预算、归一化和保存恢复测试 |
| `tests/44_calibrate_reach_controller.py` | 重力补偿与物理可达性校准 |
| `tests/diagnose_ppo_reach.py` | 旧版计时屏蔽、冻结动作等归因工具 |
| `tests/audit_day14_training.py` | 旧版物理/IK探针和单轮优化器消融 |
| `tests/summarize_day14_audit.py` | 旧审查汇总；完整重建需要已清理的历史训练链数据 |
| `tests/test.py` | 旧环境外部目标控制器探针，直接执行才运行仿真 |

训练数据流：PPO→VecNormalize→VecEnv→Monitor→ReachEnv→IK/关节/夹爪控制→MuJoCo。
Monitor在归一化内层记录原始环境奖励。每个评估使用独立MjData，加载或复制匹配的归一化统计，
设置training=False、norm_reward=False。评估不会更新训练统计。

Day13基类只修改了数值失败指标的一致性、异常观测形状以及可覆写成功接口。
场景XML、IK求解器、Stable-Baselines3源码没有为v2改写。

## 3. Reach v2观测、控制与奖励

动作是裁剪到[-1,1]的XYZ增量，每轴每个控制步最多5 mm，加到上一条合法TCP命令，
并非加到当前真实TCP。零动作表示不再移动命令，不保证实际机械臂立即停止。
0.5 rad/s限制的是关节目标命令变化率，不是对真实qvel的硬限制。

46D观测组成：

| 字段 | 维数 | 归一化 |
| --- | ---: | --- |
| 关节位置/速度 | 7+7 | 是 |
| TCP世界位置 | 3 | 是 |
| target−TCP、command−TCP | 3+3 | 是 |
| 世界坐标姿态误差 | 3 | 是 |
| 名义关节命令、IK关节目标 | 7+7 | 是 |
| 上一XYZ动作 | 3 | 否，[-1,1] |
| Reach保持比例、剩余时间比例、连续IK失败比例 | 1+1+1 | 否，[0,1] |

内部命令、上一动作及计时提供已发生的控制状态，不是专家动作或未来信息。
PhysicalVecNormalize仍维护完整统计，但仅对40D物理量应用缩放；稀有计时不再被极小方差放大。

令d为TCP误差(m)，v为最大绝对关节速度(rad/s)，a为裁剪后的XYZ动作，dt为实际仿真时间：

```text
r = -2.0 * min(d / 0.20, 5) * dt
    +0.5 * exp(-(d / 0.012)^2) * exp(-(v / 0.10)^2) * dt
    -0.01 * mean(a^2) * dt
    -0.001 * mean((a - previous_a)^2)
    +10.0 * success
    -10.0 * physical_failure
    -1.0 * ik_failure
```

无效IK/物理失败步去掉正奖励；数值失败返回−10并终止。动作变化项按控制步计，不乘dt，改变控制频率须重新校准。
近且静项最大0.02/完整控制步，仅在接近目标时鼓励停稳，避免远处全程速度惩罚诱发原地不动。

10 s是v2任务本身的期限，超时返回terminated=True、truncated=False，不追加物理失败惩罚，
观测中包含剩余时间。物理子步完成0.4 s保持后锁存事件；物理失败和无效IK优先于成功。

### 重力补偿

对已检查的单位传动仿射位置执行器：

```text
force = kp * (ctrl - q) - kd * qvel
ctrl = clip(command_q + gravity_torque / kp, actuator_ctrl_low, actuator_ctrl_high)
```

使用单独MjData，在当前真实姿态、零速度下计算重力偏置。
名义command_q仍用于限速和控制记忆；真实重力、qpos/qvel、外力及执行器力限制保持原样。
代码检查传动、增益与偏置形式，不适配时直接报错。

历史20目标配对校准摘要：原控制器尾段平均误差6.050 mm，补偿后0.106 mm；
最差目标的尾段平均误差由6.837降至0.198 mm。外部目标控制器两侧均20/20成功，平均29.05步。
这是控制精度/可达性证据，不是PPO成功率；校准原始输出已按本次清理要求删除，可用测试工具重新生成。

## 4. PPO配置与参数含义

实际配置来源是JSON及存档manifest，不使用旧文档参数覆盖新版配置。

| PPO参数 | Reach v2实际值/行为 |
| --- | --- |
| policy | MlpPolicy / ActorCriticPolicy |
| env、n_envs | Monitor包装，2个spawn采样环境；独立评估始终1个环境 |
| learning_rate | 恒定1e-4 |
| n_steps | 每环境1024，总rollout2048 |
| batch_size | 256，整除总rollout |
| n_epochs | 上限5，完整更新名义40次optimizer.step |
| gamma、gae_lambda | 0.99、0.95 |
| clip_range、clip_range_vf | 0.2、None |
| normalize_advantage | True |
| ent_coef、vf_coef | 0.0、0.5 |
| max_grad_norm | 0.5 |
| target_kl | 0.01，按SB3近似KL提前结束当前更新 |
| use_sde、sde_sample_freq | False、-1；当前后者不发挥作用 |
| rollout_buffer_class、rollout_buffer_kwargs | 默认标准RolloutBuffer、默认参数 |
| stats_window_size | 默认100，仅影响日志窗口 |
| tensorboard_log | 构造默认值，随后由set_logger配置每run的TensorBoard |
| policy_kwargs | 独立pi/vf各[128,128]、Tanh、ortho_init=True、log_std_init=ln(0.3) |
| verbose、seed、device | 默认输出由callback补充；42；CPU |
| _init_setup_model | 默认True |

策略使用普通对角高斯，std会学习；初值0.3不是固定探索噪声。
SB3在送入Box环境前裁剪动作，buffer保留原始动作及对应log_prob。
因此环境日志看不到全部上游裁剪，需额外记录raw XYZ越界率。

策略其余设置：FlattenExtractor、默认共享无参数特征提取器；这不表示Actor/Critic共用可训练MLP。
Adam沿用默认betas及eps=1e-5，无自定义权重衰减。full_std/use_expln主要关联gSDE，
normalize_images对当前向量状态不发挥作用；不能只把squash_output改True就假定得到合法的普通高斯PPO。

观测归一化默认clip_obs=10、epsilon=1e-8；norm_reward=False，clip_reward不裁剪训练成功奖励。
torch_threads=1；网络、优化器、每环境物理状态和回报相互隔离。

关键计算与解释：

```text
delta_t = r_t + gamma * V(s_next) - V(s_t)
GAE_t = sum_l (gamma * lambda)^l * delta_(t+l)
rho_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
L_clip = E[min(rho_t*A_t, clip(rho_t,1-eps,1+eps)*A_t)]
```

真实终止不自举；旧版TimeLimit按终态观测自举，v2有限期限终止不自举。
gamma×lambda=0.9405，指数权重尺度约16.8步/0.67 s，不是硬截断。
较高explained_variance、较低value_loss、逐步降低熵都不等于物理任务成功。

target_kl是软保护：SB3检查minibatch近似KL，已执行更新不回滚；
精确高斯KL仍可能超过0.01。clip_range和max_grad_norm也不是KL硬上限。
`_n_updates`记录epoch计数，KL早停时不能当作完整梯度更新次数。

## 5. 旧版实验：保留什么结论

旧Original为94D/4D、全维归一化、lr3e-4、n_steps2048、batch256、epochs10、target_kl=None，
其余主要gamma/GAE/网络规模与v2相同。后续实验变量：

| 方案 | 改动 |
| --- | --- |
| Original | 原环境、原奖励、全维归一化、初始std1 |
| A | 仅57D物理量归一化，37D有界量保持原值 |
| B | A基础上初始std改为0.3 |
| C | B基础上，最早20480步固定近Home目标，后切回完整随机任务 |
| D | B基础上，距离成本率0.2改为2.0 |

20480步短跑不能替代98304步或百万步结论。低初始噪声、距离成本增大、课程或选择性归一化，
在这些单seed实验中都没有单独解决Reach。C的近目标课程本身未可靠学会，不能据此否定所有课程方法。

| 方案 | 98304步同预算测试成功/30 | 999424步同预算测试成功/30 | 百万步最终误差mm |
| --- | ---: | ---: | ---: |
| Original | 0 | 0 | 23.088 |
| A | 1 | 1 | 38.908 |
| B | 0 | 0 | 54.027 |
| C | 2 | 0 | 56.687 |
| D | 0 | 2 | 33.191 |

两列来自不同固定测试集，不能把差异完全归因于训练时长。98304步逐回合原始对照已清理，
表中该列是原实验文字摘要；百万步同预算逐回合数据仍保留。

沿每组完整训练链、根据原验证集选出的最佳模型，再用83000–83099测试：

| 方案 | 全历史成功优先最佳步数 | 测试成功/100 | 平均误差mm |
| --- | ---: | ---: | ---: |
| Original | 61440 | 0 | 31.260 |
| A | 618496 | 13 | 22.178 |
| B | 155648 | 4 | 71.599 |
| C | 86016 | 4 | 48.271 |
| D | 241664 | 5 | 36.125 |

A的Return优先最佳为557056步，同批测试9/100；不能按测试集重新挑选模型。
Original/C的整链最佳不同于百万步续训段内最佳。保留索引分别记录lineage_best和run_best，
同时保留两种原有选模规则，避免把局部恢复段的最佳冒充整链最佳。

历史最佳训练步数不同，不能放进完全同预算主表；验证仅10个固定seed且反复选模，存在选择偏差。
不同训练设备、验证频次和并发负载混杂，不能用这些结果证明硬件导致收敛差别。
v2与旧版协议不同，也不能仅凭新旧表格把收益全部归给某个参数。

## 6. 失败分析与修复依据

### 到不了，与到了停不住

旧98304步最终模型在30回合诊断中，位置合格物理时间仅6.872%，低速合格47.666%，
三项同时合格2.415%。15/30回合曾在物理子步进入9 mm，原PPO仍0/30成功。
已开始的计时中断主要由速度超限造成。首次进入9 mm后外部冻结XYZ能挽救9/30回合；
只屏蔽计时输入为2/30；外部目标保持参考30/30成功。
这些是干预证据，不能报告为原策略成功率。

原训练链成功样本极少，密集奖励没有明确指导近目标低速保持。Original timer原为[0,1]比例，
再按稀有事件方差标准化，约19 ms保持即可裁剪到+10；输入会在接近成功时剧变。
A–D规避该问题仍低成功，说明归一化不是唯一原因。

### 奖励解释

旧势项为`gamma*Phi(next)-Phi(current)`，`Phi=2/(1+d/0.2)`，原距离成本弱。
接近目标时单步势项可能更负，但不能由单步或未折扣Return证明最优策略应远离目标：

```text
sum_t gamma^t * (gamma*Phi(s_next)-Phi(s_t))
= -Phi(s_initial) + gamma^T * Phi(s_terminal)
```

正确处理终止和自举边界后，势函数整形不改变基础任务最优策略。
真正需要实验的是稀疏成功、早期价值估计和这种信号的可学习性。
D只改距离成本，旧奖励可以逐项还原；v2奖励改变更多，不能作同样换算。

### PPO更新幅度

旧后期模型使用lr3e-4、10 epochs、无KL保护，每rollout80个优化步。
两批固定数据、相同初始模型/Adam/minibatch随机序列的独立单轮消融得到：

| 改动 | 第一批精确KL | 第二批精确KL |
| --- | ---: | ---: |
| 原配置 | 0.0431 | 0.0724 |
| 仅target_kl=0.01 | 0.0249 | 0.0267 |
| 仅lr=1e-4 | 0.0173 | 0.0238 |
| 仅epochs=3 | 0.0432 | 0.0639 |
| batch64 | 1.1649 | 0.8304 |
| batch512 | 0.0318 | 0.0362 |
| clip0.1 | 0.0244 | 0.0414 |
| lr1e-4、epochs5、target_kl0.01 | 0.0158 | 0.0136 |

batch64同时增加优化步数，不能解释为纯batch噪声效应；一轮KL较小也不能直接推出长期成功率较高。
v2使用组合稳定配置，持续记录更新后的精确高斯KL、真实优化器步数与优势标准差。

### 控制、IK与其他问题

- 原位置伺服静态误差约6.5 mm；去重力诊断降至约0.118 mm，确认重力造成显著静差。正式修复是重力前馈，不是去掉物理重力。
- 81个目标/偏航网格IK全部成功，最多12次迭代，最大位置残差0.400 mm、姿态残差0.00490 rad；Jacobian有限差分通过，没有证据优先归咎IK坐标系或无解。
- 旧原始动作越界大部分来自无关夹爪，XYZ也有越界。不能把四维总越界率当成XYZ饱和率，也不能因环境裁剪日志为零忽略SB3上游裁剪。
- 原数值失败路径漏了Monitor要求的姿态字段，隔离注入NaN会触发KeyError；已修复。历史未观察该路径，不能用它解释既有低成功率。
- 恢复后最佳分数曾局部重置，现已持久化同协议最佳指标与路径。
- 超时语义与倒计时需一致；v2明确采用有限任务截止，性能贡献没有单独消融。
- 网络容量、GPU选择、扩大回合上限都没有足够证据成为主要修复方向。

本节为历史审查摘要。对应`day14_audit_20261020`和`day14_reach_v2_validation`原始输出已按清理要求删除，
不再提供失效链接，也不能声称本地仍有全部原始轨迹。诊断工具可重新运行，但完整旧训练链日志已精简，不能重建原完整训练统计。

## 7. 正式v2百万步评测

训练从零开始，不继承先前中断分支。CPU用时30.11分钟；999424步共488份完整rollout。
训练验证集100000–100031，原独立测试集110000–110099，新增测试集130000–130999。
最佳模型在测试前由原验证规则选定；两个模型的各组测试初态完全相同，评估冻结统计。

| 模型/数据集 | 成功 | 平均误差mm | P95误差mm | 最大误差mm | 平均仿真用时s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最佳/原测试100 | 100/100 | 3.146 | 5.079 | 6.071 | 1.194 |
| 最终/原测试100 | 100/100 | 4.028 | 6.059 | 7.082 | 1.159 |
| 最佳/新增1000 | 1000/1000 | 2.965 | 4.875 | 6.826 | 1.170 |
| 最终/新增1000 | 1000/1000 | 3.950 | 6.057 | 7.699 | 1.137 |

所有成功回合满足0.4 s保持；没有超时、物理失败或IK失败。
最佳模型新增1000回合成功率的Wilson 95%区间约99.62%–100%，不是任意情况必定成功的保证。
平均位置误差比最终模型低24.9%，最终模型平均快约0.034 s。

同原测试100回合的random和zero均0/100，平均误差分别114.21与86.71 mm。
它们来自保留的`seed42_holdout.json`；正式复测完全复现已有最佳模型汇总，再复用同初态基线。

学习过程：98304步验证仍0/32，131072步3/32，196608步19/32，262144步27/32；
294912步首次32/32，之后所有已记录验证点维持32/32。
训练总结束22144回合、成功21135；最后约10万步为3501/3501。

精确KL全程均值0.0102、最大0.0184，140/488次更新未执行完整40个优化步。
末期原始XYZ任一维越界比例约32.0%，平均命令跟踪间隙11.81 mm。
这仍是平滑性优化方向，但未造成当前测试分布失败。
继续到100万步没有优于验证最佳模型的定位精度，当前无需只为成功率继续堆训练步数。

评测当时核对了训练源码快照；之后测试目录重组和历史工具移除使部分旧路径不再存在。
清理前后保留文件哈希逐一核对，模型和评测内容未改写。
多训练seed复验、任意初始姿态、扩大目标分布及真实机器人验证尚未完成。

## 8. 当前保留的模型与证据

当前主实验：

`outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe`

目录名字含100k是启动标签，实际模型为999424步，不按目录名推断训练量。

- [推荐最佳模型](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/best_model/step_000688128/manifest.json)：688128步。
- [最终模型](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/final_model/manifest.json)：999424步。
- [正式评测汇总](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/assessment_20261020/assessment_summary.json)：模型hash、协议、四组逐回合文件索引及诊断。
- [训练曲线](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe/assessment_20261020/training_curves.png)。
- [旧五组保留模型索引](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_ppo_reach/retained_models.json)：final、续训段best与整链best分开记录，两种选模规则去重后共15套。
- [旧版百万步同预算与段内最佳评测](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_ppo_reach/local_mixed_1m_20261020_094526_d4800d/protocol.json)。
- [旧版全历史最佳评测](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_ppo_reach/best_weights_20261020_reeval_83000/history_comparison.json)。
- [目录迁移后回归与推理验收](/Users/qiu_star/Desktop/me_github/embodied_intelligence/outputs/day14_tests_reorganization_20261020/legacy_acceptance.json)：旧版8项、新版10项通过，另有4回合推理Smoke。

单套目录保留model.zip、vecnormalize.pkl、manifest.json及rng.pt。
模型ZIP包括优化器状态；观测归一化必须与模型配对。只保存model.zip不足以按原输入协议推理。
恢复从新物理回合采样，不恢复半条rollout或MuJoCo状态，不能称为逐位续训。

本次清理删除了三篇设备/调度笔记、其他独立v2运行、所有周期及未选中权重、旧零散实验和两组审查/验收输出。
保留配置、选模评估、最终评测、必要训练指标和当前逐回合训练记录；旧完整训练链原始回合日志已移除。
归档manifest中的resume_from是历史来源记录，父存档未必仍在；从当前保留模型加载推理不依赖该父存档。

## 9. 使用命令

在项目根目录、已有embodied环境中执行。所有结果JSON拒绝覆盖，重复运行时换新输出文件名。

```bash
cd /Users/qiu_star/Desktop/me_github/embodied_intelligence
conda activate embodied

# 新的从零训练；实际999424步，不超过100万累计上限。
python -m training.train_ppo_reach \
  --config configs/day14_reach_v2.json \
  --steps 1000000 --total-step-limit 1000000 --device cpu \
  --label reach_v2_seed42_1m --output-root outputs/day14_reach_v2

DAY14_RUN=outputs/day14_reach_v2/20261020_115304_reach_v2_seed42_100k_db4ebe

# 存档内100个holdout种子；换新输出名以免与已有结果冲突。
python -m training.evaluate_ppo_reach \
  --bundle "$DAY14_RUN/best_model/step_000688128" \
  --policies ppo random zero \
  --output "$DAY14_RUN/manual_holdout_100.json"

# 新种子扩展评测；显式指定数量与起始seed。
python -m training.evaluate_ppo_reach \
  --bundle "$DAY14_RUN/best_model/step_000688128" \
  --episodes 1000 --seed-start 150000 \
  --output "$DAY14_RUN/manual_fresh1000.json"

# Mac Viewer定性演示不替代正式评测。
PYTHONPATH=. mjpython scripts/42_view_ppo_reach.py \
  --bundle "$DAY14_RUN/best_model/step_000688128" \
  --episodes 3 --seed-start 160000 --output "$DAY14_RUN/manual_viewer.json"

# 训练曲线可由保留的episodes、metrics和evaluations重新生成。
python -m training.plot_ppo_reach "$DAY14_RUN" \
  --combined-output "$DAY14_RUN/manual_curves.png"

# 明确选择带编号的回归测试模块。
PYTHONPATH=. python -m unittest tests.41_test_ppo_reach tests.43_test_reach_v2 -v
```

v2预算向下取整到完整2048步rollout。旧存档续训默认保留其旧上限；显式给出
`--total-step-limit`才允许在当前100万硬上限内扩展。配置和观测版本不兼容时拒绝加载；
不能继承旧v1权重/归一化来充当从零训练的v2。

复现环境以保留config中的versions及source_sha256为依据。requirements.txt含TensorBoard，
现有requirements-lock不是完整可靠的训练环境锁文件，后续仍需完善依赖锁定。

参考：[SB3 PPO参数](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters)、
[VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)、
[MuJoCo执行器模型](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model)。
