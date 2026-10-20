"""SB3 PPO Reach 训练入口；每次新建实验，支持成套存档继续训练。"""

import argparse
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from training.ppo_callbacks import ReachTrainingCallback
from training.reach_common import (default_config, device_name, load_bundle, make_vector,
                                   restore_rng, retraining_config, save_bundle, versions, write_json,
                                   environment_version, observation_schema, validate_config,
                                   MAX_REACH_V2_STEPS)


def load_config(path):
    """Expand a JSON experiment profile into the full immutable run configuration."""
    values = json.loads(Path(path).read_text())
    config = default_config()
    allowed = set(config) | {"environment_version", "experiment_version", "reward_version",
        "timeout_mode", "gravity_compensation", "max_run_steps", "normalization_mode",
        "log_std_init", "policy_kwargs", "reach_rewards", "device"}
    if set(values)-allowed:
        raise ValueError(f"未知配置字段：{sorted(set(values)-allowed)}")
    for key, value in values.items():
        if key == "env":
            env_values = dict(value)
            rewards = env_values.pop("rewards", {})
            config["env"].update(env_values)
            config["env"]["rewards"].update(rewards)
        elif key == "ppo":
            config["ppo"].update(value)
        else:
            config[key] = value
    config["observation_schema"] = observation_schema(config)
    validate_config(config)
    policy_arguments(config)
    return config


def policy_arguments(config):
    """Resolve a small explicit class-name registry, never eval executable JSON."""
    values = {"net_arch": config["net_arch"], "log_std_init": config.get("log_std_init", 0.)}
    extra = dict(config.get("policy_kwargs", {}))
    if "net_arch" in extra or "log_std_init" in extra:
        raise ValueError("net_arch/log_std_init 应放在配置顶层")
    if "activation_fn" in extra:
        activations = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU, "ELU": torch.nn.ELU}
        if extra["activation_fn"] not in activations:
            raise ValueError("不支持的activation_fn")
        extra["activation_fn"] = activations[extra["activation_fn"]]
    return {**values, **extra}


class Tee:
    """终端与 train.log 同时写入，异常 traceback 也保留在该实验目录。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def planned_budget(config, budget, resume=None, total_step_limit=None):
    """区分新增步数与累计上限；完整rollout向下取整，禁止SB3隐式超预算。

    未显式提供累计上限时保留原100000步保护。跨设备续训仍计入源模型步数，
    不能把恢复后的新增预算误当成一个新的累计训练预算。
    """
    initial = 0
    if resume:
        initial = json.loads((Path(resume)/"manifest.json").read_text())["timesteps"]
    block = config["ppo"]["n_steps"]*config["n_envs"]
    if total_step_limit is not None:
        if total_step_limit <= initial:
            raise ValueError("累计上限必须大于源模型步数")
        remaining = total_step_limit-initial
        budget = remaining if budget is None else budget
        if budget > remaining:
            raise ValueError("新增预算将超过显式累计上限")
    else:
        budget = 100000 if budget is None else budget
        if budget > 100000:
            raise ValueError("超过100000新增步数时必须显式提供total_step_limit")
    actual = (budget//block)*block
    if actual <= 0:
        raise ValueError("预算不足一个完整rollout")
    return budget, actual, initial


def resolve_step_limit(config, total_step_limit, resume=None):
    """Explicit resume limits may extend an old saved budget within the current ceiling."""
    if environment_version(config) != "reach_v2":
        return total_step_limit
    cap = config.get("max_run_steps", 250000)
    if total_step_limit is not None and total_step_limit > cap:
        if not resume or total_step_limit > MAX_REACH_V2_STEPS:
            raise ValueError(f"Reach v2 累计上限超过配置预算；续训显式扩展最多支持 {MAX_REACH_V2_STEPS} 步")
        config["previous_max_run_steps"] = cap
        config["max_run_steps"] = cap = total_step_limit
    return cap if total_step_limit is None else total_step_limit


def train(config, budget, output_root, device="auto", resume=None, label="main", total_step_limit=None,
          fork_hyperparameters=False):
    """budget为新增环境步；total_step_limit为显式授权的累计上限。"""
    config = deepcopy(config)
    validate_config(config)
    config["observation_schema"] = observation_schema(config)
    v2 = environment_version(config) == "reach_v2"
    total_step_limit = resolve_step_limit(config, total_step_limit, resume)
    if not v2 and config["ppo"]["gamma"] != config["env"]["rewards"]["shaping_discount"]:
        raise ValueError("PPO gamma 必须与折扣势函数 gamma 匹配")
    if config["n_envs"] not in (1, 2):
        raise ValueError("已验收的环境数仅为1或2；不能隐式扩展并发规模")
    if fork_hyperparameters and not resume:
        raise ValueError("超参数分支必须指定来源存档")
    if fork_hyperparameters:
        saved = json.loads((Path(resume)/"manifest.json").read_text())["config"]
        allowed = {"n_steps", "batch_size", "learning_rate", "target_kl"}
        changed = {name for name in config["ppo"] if saved["ppo"].get(name) != config["ppo"][name]}
        if changed-allowed:
            raise ValueError(f"当前候选分支不允许修改这些PPO参数：{changed-allowed}")
        if config["ppo"]["n_steps"]*config["n_envs"] != saved["ppo"]["n_steps"]*saved["n_envs"]:
            raise ValueError("候选分支必须保持每次更新的总采样数不变")
        config = {**config, "hyperparameter_fork": {
            "source": str(Path(resume).resolve()),
            "changes": {name: {"old": saved["ppo"][name], "new": config["ppo"][name]} for name in sorted(changed)},
            "source_n_envs": saved["n_envs"], "new_n_envs": config["n_envs"],
            "weights_optimizer_and_normalizer_inherited": True}}
    budget, actual_budget, expected_initial = planned_budget(config, budget, resume, total_step_limit)
    if resume:
        saved_config = json.loads((Path(resume)/"manifest.json").read_text())["config"]
        if environment_version(saved_config) != environment_version(config):
            raise ValueError("不同环境版本必须从零训练，不能复用旧权重或归一化")
    device = device_name(device)
    torch.set_num_threads(config["torch_threads"])
    run = Path(output_root) / (time.strftime("%Y%m%d_%H%M%S")+"_"+label+"_"+uuid.uuid4().hex[:6])
    run.mkdir(parents=True, exist_ok=False)
    config = {**config, "device": device, "requested_steps": budget, "scheduled_steps": actual_budget,
              "resume_from": str(Path(resume).resolve()) if resume else None, "versions": versions()}
    config["backend_availability"] = {"mps_built": torch.backends.mps.is_built(),
                                       "mps_available": torch.backends.mps.is_available(),
                                       "cuda_available": torch.cuda.is_available()}
    if total_step_limit is not None:
        config["total_step_limit"] = total_step_limit
        config["scheduled_final_timesteps"] = expected_initial+actual_budget
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    config["git_revision"] = revision.stdout.strip()
    config["git_status"] = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
    root = Path(__file__).resolve().parents[1]
    source_paths = [p for folder in ("envs", "controllers", "robotics", "training")
                    for p in (root/folder).glob("*.py")]
    source_paths += list((root/"assets/franka_panda").glob("*.xml"))
    config["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(source_paths)}
    write_json(run / "config.json", config)
    started = time.perf_counter()
    with (run / "train.log").open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        vector = None
        try:
            if resume:
                model, vector, saved = load_bundle(resume, device, training=True, monitor_path=run / "monitor",
                                                    environment_config=config,
                                                    model_overrides=config["ppo"] if fork_hyperparameters else None)
                for name in ("env", "ppo", "norm_obs", "norm_reward", "net_arch"):
                    if name == "ppo" and fork_hyperparameters:
                        continue
                    if saved[name] != config[name]:
                        raise ValueError(f"恢复配置不一致：{name}；请建立单独实验版本")
                if not fork_hyperparameters and saved["n_envs"] != config["n_envs"]:
                    raise ValueError("普通恢复不能修改环境数；请使用显式超参数分支")
                for name, default in (("normalization_mode", "all"), ("log_std_init", 0.)):
                    if saved.get(name, default) != config.get(name, default):
                        raise ValueError(f"恢复配置不一致：{name}")
                if saved.get("policy_kwargs", {}) != config.get("policy_kwargs", {}):
                    raise ValueError("恢复配置不一致：policy_kwargs")
                if saved.get("training_reset_options") != config.get("training_reset_options"):
                    if not (config.get("curriculum_transition") == "to_full_random"
                            and not config.get("training_reset_options")):
                        raise ValueError("reset分布变化必须显式标记课程切换")
                restore_rng(resume)
            else:
                vector = make_vector(config, monitor_path=run / "monitor")
                vector.seed(config["seed"])
                model = PPO("MlpPolicy", vector, **config["ppo"],
                    policy_kwargs=policy_arguments(config),
                    seed=config["seed"], device=device)
            model.set_logger(configure(str(run / "tensorboard"), ["tensorboard"]))
            callback = ReachTrainingCallback(run, config)
            before = torch.cat([p.detach().cpu().flatten() for p in model.policy.parameters()]).clone()
            initial_steps = model.num_timesteps
            if initial_steps != expected_initial:
                raise ValueError("源模型内部步数与manifest不一致")
            if next(model.policy.parameters()).device.type != device:
                raise RuntimeError("策略实际设备与指定设备不一致，拒绝静默回退")
            print(f"RUN {run.resolve()} device={device} requested={budget} actual={actual_budget}", flush=True)
            model.learn(total_timesteps=actual_budget, callback=callback, reset_num_timesteps=False, log_interval=1)
            save_bundle(model, vector, run / "final_model", config, "final")
            after = torch.cat([p.detach().cpu().flatten() for p in model.policy.parameters()])
            result = {"status": "completed", "run_dir": str(run.resolve()), "device": device,
                "policy_parameter_device": str(next(model.policy.parameters()).device),
                "initial_timesteps": initial_steps, "training_timesteps": model.num_timesteps,
                "additional_timesteps": model.num_timesteps-initial_steps,
                "wall_time_s": time.perf_counter()-started,
                "parameter_l2_change": float(torch.linalg.norm(after-before)),
                "updates": model._n_updates, "rollout_buffer_full": model.rollout_buffer.full,
                "rollout_observation_shape": list(model.rollout_buffer.observations.shape),
                "rollout_action_shape": list(model.rollout_buffer.actions.shape),
                "runtime_ppo": {"n_envs": model.n_envs, "n_steps": model.n_steps,
                                "batch_size": model.batch_size, "learning_rate": model.learning_rate,
                                "target_kl": model.target_kl},
                "best_models": callback.best_paths}
            if total_step_limit is not None and model.num_timesteps > total_step_limit:
                raise RuntimeError("实际步数超过累计上限")
            result["end_to_end_fps"] = result["additional_timesteps"]/result["wall_time_s"]
            write_json(run / "result.json", result)
            print(json.dumps(result, indent=2), flush=True)
            return result
        except BaseException:
            import traceback
            traceback.print_exc()
            # SIGINT 或异常保留最后可用模型；未保存的物理回合/rollout 从新回合继续。
            if vector is not None and "model" in locals():
                save_bundle(model, vector, run / "interrupted_model", config, "interrupted")
            raise
        finally:
            if vector is not None:
                vector.close()


def main():
    """恢复读取原存档配置，新实验才接收环境与 PPO 初值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, help="新增环境步；不指定时为100000或累计上限剩余预算")
    parser.add_argument("--config", type=Path, help="新实验JSON配置（不能与resume/profile混用）")
    parser.add_argument("--total-step-limit", type=int, help="累计训练上限；完整rollout向下取整")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-root", default="outputs/day14_ppo_reach")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--label", default="main")
    parser.add_argument("--profile", choices=("original", "selective", "selective_low_noise", "selective_low_noise_distance"))
    parser.add_argument("--curriculum", choices=("fixed_near_home",))
    parser.add_argument("--release-curriculum", action="store_true")
    args = parser.parse_args()
    if args.config and (args.resume or any(value is not None for value in (
            args.horizon, args.action_scale, args.profile, args.curriculum)) or args.release_curriculum):
        parser.error("--config 不能与恢复、旧profile或环境覆盖参数混用")
    if args.resume:
        if any(value is not None for value in (args.seed, args.horizon, args.action_scale, args.profile, args.curriculum)):
            parser.error("恢复训练沿用存档seed/环境配置；更改这些参数请创建新实验")
        config = json.loads((args.resume / "manifest.json").read_text())["config"]
        if args.release_curriculum:
            if not config.get("training_reset_options"):
                parser.error("源存档没有课程可解除")
            config["previous_curriculum_options"] = config.pop("training_reset_options")
            config["curriculum_transition"] = "to_full_random"
            config["experiment_version"] += "_full_random"
    elif args.config:
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
    else:
        if args.release_curriculum:
            parser.error("release-curriculum要求resume存档")
        factory = default_config if args.profile in (None, "original") else lambda *a: retraining_config(args.profile, *a)
        config = factory(42 if args.seed is None else args.seed,
                                250 if args.horizon is None else args.horizon,
                                .005 if args.action_scale is None else args.action_scale)
        if args.curriculum:
            config["training_reset_options"] = {"target_position": [.53, 0., .50], "cube_yaw": 0.}
            config["experiment_version"] = config.get("experiment_version", "original")+"_fixed_near_home"
    train(config, args.steps, args.output_root, args.device or config.get("device", "auto"), args.resume, args.label,
          total_step_limit=args.total_step_limit)


if __name__ == "__main__":
    main()
