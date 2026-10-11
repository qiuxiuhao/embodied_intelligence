from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO

from envs.single_joint_env import SingleJointEnv


# ============================================================
# 1. 实验配置
# ============================================================

SEED = 42

TOTAL_TIMESTEPS = 102400

LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

N_STEPS = 1024

BATCH_SIZE = 256

N_EPOCHS = 10


# ============================================================
# 2. 设置随机种子
# ============================================================

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# 3. 输出目录
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "day07"
    / "sb3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. 创建 Environment
# ============================================================

env = SingleJointEnv()


# ============================================================
# 5. 定义 Policy 网络结构
# ============================================================

policy_kwargs = {

    "activation_fn": torch.nn.Tanh,

    "net_arch": {
        "pi": [64, 64],
        "vf": [64, 64],
    },
}


# ============================================================
# 6. 创建 SB3 PPO
# ============================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

    policy_kwargs=policy_kwargs,

    learning_rate=LEARNING_RATE,

    n_steps=N_STEPS,

    batch_size=BATCH_SIZE,

    n_epochs=N_EPOCHS,

    gamma=GAMMA,

    gae_lambda=GAE_LAMBDA,

    clip_range=CLIP_RANGE,

    ent_coef=0.0,

    seed=SEED,

    device="cpu",

    verbose=1,
)


# ============================================================
# 7. 正式训练
# ============================================================

model.learn(
    total_timesteps=TOTAL_TIMESTEPS
)


# ============================================================
# 8. 保存模型
# ============================================================

save_path = (
    OUTPUT_DIR
    / "ppo_sb3"
)

model.save(
    str(save_path)
)


# ============================================================
# 9. 关闭环境
# ============================================================

env.close()


print("\nSB3 Training Finished.")

print(
    "Checkpoint:",
    save_path.with_suffix(".zip"),
)