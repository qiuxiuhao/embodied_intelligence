from pathlib import Path
import argparse
import csv
import json

import numpy as np
import torch

from envs.single_joint_env import SingleJointEnv
from ppo.model import ActorCritic
from ppo.collector import collect_rollout
from ppo.gae import compute_gae
from ppo.update import ppo_update


# ============================================================
# 1. 默认训练配置
# ============================================================

DEFAULT_UPDATES = 100
DEFAULT_ROLLOUT_STEPS = 1024

GAMMA = 0.99
GAE_LAMBDA = 0.95

LEARNING_RATE = 3e-4

CLIP_COEF = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.0

UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256

VALIDATION_START_SEED = 10000


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Day 7: Audited PPO Training"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--updates",
        type=int,
        default=DEFAULT_UPDATES,
    )

    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--val-episodes",
        type=int,
        default=30,
    )

    return parser.parse_args()


# ============================================================
# 3. 统一随机种子
# ============================================================

def set_seed(seed):

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# 4. 创建 Actor-Critic
# ============================================================

def create_model(env, device):

    model = ActorCritic(
        obs_dim=env.observation_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )

    return model.to(device)


# ============================================================
# 5. 独立验证当前 Policy
# ============================================================

def evaluate_model(
    model,
    device,
    seeds,
):

    env = SingleJointEnv()

    episode_returns = []
    episode_lengths = []
    episode_successes = []

    # 保存进入验证前的模型状态
    was_training = model.training

    model.eval()

    try:

        for seed in seeds:

            obs, info = env.reset(
                seed=int(seed)
            )

            total_reward = 0.0
            episode_length = 0

            while True:

                # ------------------------------------
                # Observation -> Tensor
                # ------------------------------------

                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device,
                )

                # ------------------------------------
                # Deterministic Policy
                # ------------------------------------

                with torch.no_grad():

                    action_tensor, _, _ = model.act(
                        obs_tensor,
                        deterministic=True,
                    )

                # ------------------------------------
                # Tensor -> NumPy Action
                # ------------------------------------

                action = (
                    action_tensor
                    .cpu()
                    .numpy()
                )

                # ------------------------------------
                # Environment Step
                # ------------------------------------

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                total_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            # ----------------------------------------
            # 保存一个 Episode 的评估结果
            # ----------------------------------------

            episode_returns.append(
                float(total_reward)
            )

            episode_lengths.append(
                int(episode_length)
            )

            episode_successes.append(
                float(info["success"])
            )

    finally:

        env.close()

        # 恢复验证前的训练状态
        model.train(was_training)

    # ========================================================
    # 汇总结果
    # ========================================================

    results = {

        "success_rate": float(
            np.mean(episode_successes)
        ),

        "return_mean": float(
            np.mean(episode_returns)
        ),

        "return_std": float(
            np.std(episode_returns)
        ),

        "length_mean": float(
            np.mean(episode_lengths)
        ),
    }

    return results


# ============================================================
# 6. 保存训练记录
# ============================================================

def save_history(history, output_dir):

    if not history:
        return

    csv_path = output_dir / "metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 7. 保存实验配置
# ============================================================

def save_config(config, output_dir):

    config_path = output_dir / "config.json"

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# 8. PPO 训练主函数
# ============================================================

def train(args):

    # --------------------------------------------------------
    # 8.1 参数检查
    # --------------------------------------------------------

    if args.seed < 0:
        raise ValueError("seed must be >= 0")

    if args.updates <= 0:
        raise ValueError("updates must be > 0")

    if args.rollout_steps <= 0:
        raise ValueError("rollout-steps must be > 0")

    if args.eval_interval <= 0:
        raise ValueError("eval-interval must be > 0")

    if args.val_episodes <= 0:
        raise ValueError("val-episodes must be > 0")

    if args.rollout_steps < 2:
        raise ValueError(
            "rollout-steps must be >= 2 "
            "for advantage normalization"
        )

    # --------------------------------------------------------
    # 8.2 随机种子
    # --------------------------------------------------------

    set_seed(args.seed)

    # Day 7 使用 CPU
    device = torch.device("cpu")

    # --------------------------------------------------------
    # 8.3 创建 Environment
    # --------------------------------------------------------

    env = SingleJointEnv()

    # --------------------------------------------------------
    # 8.4 创建 Actor-Critic
    # --------------------------------------------------------

    model = create_model(
        env=env,
        device=device,
    )

    # --------------------------------------------------------
    # 8.5 Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # 8.6 输出目录
    # --------------------------------------------------------

    root = Path(__file__).resolve().parents[1]

    output_dir = (
        root
        / "outputs"
        / "day07"
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 8.7 Validation Seeds
    # --------------------------------------------------------

    validation_seeds = list(
        range(
            VALIDATION_START_SEED,
            VALIDATION_START_SEED + args.val_episodes,
        )
    )

    # --------------------------------------------------------
    # 8.8 保存实验配置
    # --------------------------------------------------------

    config = {

        "seed": args.seed,

        "num_updates": args.updates,

        "rollout_steps": args.rollout_steps,

        "total_environment_steps": (
            args.updates * args.rollout_steps
        ),

        "gamma": GAMMA,

        "gae_lambda": GAE_LAMBDA,

        "learning_rate": LEARNING_RATE,

        "clip_coef": CLIP_COEF,

        "value_coef": VALUE_COEF,

        "entropy_coef": ENTROPY_COEF,

        "update_epochs": UPDATE_EPOCHS,

        "minibatch_size": MINIBATCH_SIZE,

        "eval_interval": args.eval_interval,

        "validation_seeds": validation_seeds,

        "device": str(device),

        "physics_timestep": float(
            env.model.opt.timestep
        ),

        "frame_skip": env.frame_skip,

        "control_dt": float(env.dt),

    }

    save_config(
        config=config,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # 8.9 初始化训练记录
    # --------------------------------------------------------

    history = []

    best_score = (
        float("-inf"),
        float("-inf"),
    )

    best_update = None

    print("=" * 60)
    print("Day 7 PPO Training")
    print("=" * 60)

    print("Seed:", args.seed)
    print("Updates:", args.updates)
    print("Rollout steps:", args.rollout_steps)
    print("Validation interval:", args.eval_interval)
    print("Validation episodes:", args.val_episodes)
    print("Device:", device)
    print("Output:", output_dir)

    # ========================================================
    # 9. 训练循环
    # ========================================================

    try:

        for update in range(
            1,
            args.updates + 1,
        ):

            # =================================================
            # 9.1 使用当前 Policy 采集 Rollout
            # =================================================

            batch, stats = collect_rollout(
                env=env,
                model=model,
                num_steps=args.rollout_steps,
                device=device,
                seed=args.seed + update,
            )

            # =================================================
            # 9.2 计算 GAE 和 Returns
            # =================================================

            advantages, returns = compute_gae(
                rewards=batch["rewards"],
                values=batch["values"],
                next_values=batch["next_values"],
                terminated=batch["terminated"],
                truncated=batch["truncated"],
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
            )

            # =================================================
            # 9.3 PPO Update
            # =================================================

            metrics = ppo_update(
                model=model,
                optimizer=optimizer,
                batch=batch,
                advantages=advantages,
                returns=returns,
                clip_coef=CLIP_COEF,
                value_coef=VALUE_COEF,
                entropy_coef=ENTROPY_COEF,
                update_epochs=UPDATE_EPOCHS,
                minibatch_size=MINIBATCH_SIZE,
            )

            # =================================================
            # 9.4 训练 Rollout 的统计信息
            # =================================================

            episode_returns = stats[
                "episode_returns"
            ]

            episode_successes = stats[
                "episode_successes"
            ]

            if len(episode_returns) > 0:

                rollout_return = float(
                    np.mean(episode_returns)
                )

                rollout_success = float(
                    np.mean(episode_successes)
                )

            else:

                rollout_return = float("nan")
                rollout_success = float("nan")

            # =================================================
            # 9.5 Validation 默认值
            # =================================================

            val_success = float("nan")
            val_return = float("nan")
            val_return_std = float("nan")
            val_length = float("nan")

            # =================================================
            # 9.6 每隔一定轮数进行 Validation
            # =================================================

            should_evaluate = (
                update % args.eval_interval == 0
                or update == args.updates
            )

            if should_evaluate:

                val_results = evaluate_model(
                    model=model,
                    device=device,
                    seeds=validation_seeds,
                )

                val_success = val_results[
                    "success_rate"
                ]

                val_return = val_results[
                    "return_mean"
                ]

                val_return_std = val_results[
                    "return_std"
                ]

                val_length = val_results[
                    "length_mean"
                ]

                # =============================================
                # 9.7 Best Checkpoint Selection
                # =============================================

                current_score = (
                    val_success,
                    val_return,
                )

                if current_score > best_score:

                    best_score = current_score
                    best_update = update

                    torch.save(
                        model.state_dict(),
                        output_dir / "ppo_best.pt",
                    )

            # =================================================
            # 9.8 检查训练数值是否合法
            # =================================================

            for name, value in metrics.items():

                if not np.isfinite(value):

                    raise RuntimeError(
                        f"Non-finite metric: "
                        f"{name}={value}, "
                        f"update={update}"
                    )

            # =================================================
            # 9.9 保存当前 Update 的数据
            # =================================================

            record = {

                "update": update,

                "total_steps": (
                    update * args.rollout_steps
                ),

                "rollout_return": rollout_return,

                "rollout_success": rollout_success,

                "policy_loss": metrics[
                    "policy_loss"
                ],

                "value_loss": metrics[
                    "value_loss"
                ],

                "entropy": metrics[
                    "entropy"
                ],

                "approx_kl": metrics[
                    "approx_kl"
                ],

                "clip_fraction": metrics[
                    "clip_fraction"
                ],

                "val_success": val_success,

                "val_return": val_return,

                "val_return_std": val_return_std,

                "val_length": val_length,
            }

            history.append(record)

            # 每一轮更新日志，防止意外中断后全部丢失
            save_history(
                history=history,
                output_dir=output_dir,
            )

            # =================================================
            # 9.10 打印训练结果
            # =================================================

            print(

                f"update={update:03d}  "

                f"steps={record['total_steps']:7d}  "

                f"rollout_return="
                f"{rollout_return:+9.3f}  "

                f"rollout_success="
                f"{rollout_success:.3f}  "

                f"val_success="
                f"{val_success:.3f}  "

                f"val_return="
                f"{val_return:+9.3f}  "

                f"pi_loss="
                f"{metrics['policy_loss']:+.4f}  "

                f"v_loss="
                f"{metrics['value_loss']:.4f}  "

                f"kl="
                f"{metrics['approx_kl']:.5f}  "

                f"clip="
                f"{metrics['clip_fraction']:.3f}"

            )

        # ====================================================
        # 10. 保存 Final Checkpoint
        # ====================================================

        torch.save(
            model.state_dict(),
            output_dir / "ppo_final.pt",
        )

        # ====================================================
        # 11. 保存最佳模型信息
        # ====================================================

        best_info = {

            "best_update": best_update,

            "best_validation_success": (
                best_score[0]
            ),

            "best_validation_return": (
                best_score[1]
            ),
        }

        with open(
            output_dir / "best_info.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                best_info,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print("\nTraining finished.")

        print(
            "Best update:",
            best_update,
        )

        print(
            "Best validation success:",
            best_score[0],
        )

        print(
            "Best validation return:",
            best_score[1],
        )

        print(
            "Final checkpoint:",
            output_dir / "ppo_final.pt",
        )

    finally:

        env.close()


# ============================================================
# 12. 程序入口
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    train(args)