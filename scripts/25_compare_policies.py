from pathlib import Path
import csv

import numpy as np
import torch

from agents.policies import random_policy, pd_policy
from envs.single_joint_env import SingleJointEnv
from ppo.model import ActorCritic


# ============================================================
# 1. Evaluation 配置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "day07"

EVAL_DIR = OUTPUT_DIR / "evaluation"

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_SEEDS = [42, 43, 44]

TEST_SEEDS = list(
    range(20000, 20100)
)

DEVICE = torch.device("cpu")


# ============================================================
# 2. Random Policy Adapter
# ============================================================

def random_adapter(obs, env):

    return random_policy(
        obs,
        env.action_space,
    )


# ============================================================
# 3. PD Policy Adapter
# ============================================================

def pd_adapter(obs, env):

    return pd_policy(
        obs,
        env.action_space,
        kp=8.0,
        kd=1.0,
    )


# ============================================================
# 4. 加载手写 PPO 模型
# ============================================================

def load_ppo_policy(checkpoint_path):

    env = SingleJointEnv()

    try:

        model = ActorCritic(
            obs_dim=env.observation_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        ).to(DEVICE)

    finally:

        env.close()

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ============================================================
# 5. 把 PPO 包装成统一接口
# ============================================================

def make_ppo_adapter(model):

    def policy(obs, env):

        del env

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():

            action_tensor, _, _ = model.act(
                obs_tensor,
                deterministic=True,
            )

        return (
            action_tensor
            .cpu()
            .numpy()
        )

    return policy


# ============================================================
# 6. 单个 Policy 的 Evaluation
# ============================================================

def evaluate_policy(
    policy_name,
    policy_fn,
    seeds,
):

    env = SingleJointEnv()

    records = []

    try:

        for seed in seeds:

            # ----------------------------------------
            # Reset
            # ----------------------------------------

            obs, info = env.reset(
                seed=int(seed)
            )

            # Random Policy 使用可复现的动作采样序列
            env.action_space.seed(
                int(seed) + 50000
            )

            episode_return = 0.0
            episode_length = 0

            # ----------------------------------------
            # Episode
            # ----------------------------------------

            while True:

                action = policy_fn(
                    obs,
                    env,
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break

            # ----------------------------------------
            # 保存 Episode 记录
            # ----------------------------------------

            records.append({

                "policy": policy_name,

                "seed": int(seed),

                "success": int(
                    info["success"]
                ),

                "episode_return": float(
                    episode_return
                ),

                "episode_length": int(
                    episode_length
                ),

                "target_deg": float(
                    info["target_deg"]
                ),

                "final_angle_deg": float(
                    info["q_deg"]
                ),

                "final_error_deg": float(
                    abs(info["error_deg"])
                ),

                "final_velocity_rad_s": float(
                    info["qvel"]
                ),

                "terminated": int(terminated),

                "truncated": int(truncated),
            })

    finally:

        env.close()

    return records


# ============================================================
# 7. 计算 Evaluation 指标
# ============================================================

def summarize_policy(
    policy_name,
    records,
):

    success = np.array(
        [r["success"] for r in records],
        dtype=np.float64,
    )

    returns = np.array(
        [r["episode_return"] for r in records],
        dtype=np.float64,
    )

    lengths = np.array(
        [r["episode_length"] for r in records],
        dtype=np.float64,
    )

    errors = np.array(
        [r["final_error_deg"] for r in records],
        dtype=np.float64,
    )

    return {

        "policy": policy_name,

        "episodes": len(records),

        "success_rate": float(
            success.mean()
        ),

        "return_mean": float(
            returns.mean()
        ),

        "return_std": float(
            returns.std()
        ),

        "length_mean": float(
            lengths.mean()
        ),

        "final_error_mean_deg": float(
            errors.mean()
        ),
    }


# ============================================================
# 8. CSV 保存函数
# ============================================================

def save_csv(path, records):

    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)


# ============================================================
# 9. 主程序
# ============================================================

def main():

    # ----------------------------------------
    # 9.1 准备 Policy
    # ----------------------------------------

    policies = {

        "Random": random_adapter,

        "PD": pd_adapter,
    }

    # ----------------------------------------
    # 9.2 添加手写 PPO
    # ----------------------------------------

    for seed in TRAIN_SEEDS:

        checkpoint = (
            OUTPUT_DIR
            / f"seed_{seed}"
            / "ppo_best.pt"
        )

        if not checkpoint.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        model = load_ppo_policy(
            checkpoint
        )

        policies[
            f"PPO_seed_{seed}"
        ] = make_ppo_adapter(model)

    # ----------------------------------------
    # 9.3 可选添加 SB3 PPO
    # ----------------------------------------

    sb3_checkpoint = (
        OUTPUT_DIR
        / "sb3"
        / "ppo_sb3.zip"
    )

    if sb3_checkpoint.exists():

        from stable_baselines3 import PPO

        sb3_model = PPO.load(
            str(sb3_checkpoint),
            device="cpu",
        )

        def sb3_adapter(obs, env):

            del env

            action, _ = sb3_model.predict(
                obs,
                deterministic=True,
            )

            return action

        policies["SB3_final"] = sb3_adapter

    # ----------------------------------------
    # 9.4 正式 Evaluation
    # ----------------------------------------

    all_episode_records = []
    all_summary_records = []

    for name, policy_fn in policies.items():

        print(f"\nEvaluating {name} ...")

        records = evaluate_policy(
            policy_name=name,
            policy_fn=policy_fn,
            seeds=TEST_SEEDS,
        )

        summary = summarize_policy(
            policy_name=name,
            records=records,
        )

        all_episode_records.extend(
            records
        )

        all_summary_records.append(
            summary
        )

        print(
            f"Success Rate: "
            f"{summary['success_rate']:.3f}"
        )

        print(
            f"Mean Return: "
            f"{summary['return_mean']:.3f}"
        )

        print(
            f"Return Std: "
            f"{summary['return_std']:.3f}"
        )

        print(
            f"Mean Episode Length: "
            f"{summary['length_mean']:.2f}"
        )

    # ----------------------------------------
    # 9.5 保存逐 Episode 结果
    # ----------------------------------------

    save_csv(
        EVAL_DIR / "episodes.csv",
        all_episode_records,
    )

    # ----------------------------------------
    # 9.6 保存汇总结果
    # ----------------------------------------

    save_csv(
        EVAL_DIR / "summary.csv",
        all_summary_records,
    )

    print("\nEvaluation finished.")

    print(
        "Episode results:",
        EVAL_DIR / "episodes.csv",
    )

    print(
        "Summary:",
        EVAL_DIR / "summary.csv",
    )


if __name__ == "__main__":

    main()