"""Day 14 共用配置、独立环境、配对存档和确定性评估；不改变 Day 13 任务。"""

from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

from envs.pick_place_env import PickPlaceEnv, PickPlaceEnvConfig, RewardWeights
from envs.pick_place_env import OBSERVATION_FIELDS
from envs.reach_env import ReachEnv, REACH_OBSERVATION_FIELDS, ReachRewardConfig
from training.selective_normalize import PhysicalVecNormalize


SELECTION_SEEDS = list(range(10000, 10010))
HOLDOUT_SEEDS = list(range(20000, 20030))
MAX_REACH_V2_STEPS = 1000000


def environment_version(config):
    version_name = config.get("environment_version", "reach_v1")
    if version_name not in ("reach_v1", "reach_v2"):
        raise ValueError(f"未知环境版本：{version_name}")
    return version_name


def observation_schema(config):
    fields = REACH_OBSERVATION_FIELDS if environment_version(config) == "reach_v2" else OBSERVATION_FIELDS
    return [list(field) for field in fields]


def validate_config(config):
    """Validate serialized task identity before allocating workers or a run directory."""
    version_name = environment_version(config)
    env_config(config["env"])
    if config.get("observation_schema", observation_schema(config)) != observation_schema(config):
        raise ValueError("观测字段版本与当前环境不一致")
    if version_name == "reach_v2":
        if config.get("reward_version") != "reach_dense_v2":
            raise ValueError("Reach v2 requires reward_version=reach_dense_v2")
        if config.get("timeout_mode") != "finite_horizon":
            raise ValueError("Reach v2 requires finite_horizon timeouts")
        if config.get("normalization_mode") != "physical":
            raise ValueError("Reach v2 requires schema-based physical normalization")
        if not isinstance(config.get("gravity_compensation"), bool):
            raise ValueError("gravity_compensation must be explicitly boolean")
        ReachRewardConfig(**config["reach_rewards"])
        if config.get("training_reset_options"):
            raise ValueError("Reach v2 baseline uses the full random target distribution")
        limit = config.get("max_run_steps", 250000)
        if type(limit) is not int or not 0 < limit <= MAX_REACH_V2_STEPS:
            raise ValueError(f"Reach v2 max_run_steps must be an integer in 1..{MAX_REACH_V2_STEPS}")
    ppo = config["ppo"]
    for key in ("n_steps", "batch_size", "n_epochs"):
        if type(ppo[key]) is not int or ppo[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if type(config["n_envs"]) is not int or config["n_envs"] not in (1, 2):
        raise ValueError("环境数必须为1或2")
    size = ppo["n_steps"]*config["n_envs"]
    if ppo["batch_size"] <= 1 or size % ppo["batch_size"]:
        raise ValueError("batch_size 必须大于1且整除总rollout大小")
    for key in ("eval_interval", "checkpoint_interval", "torch_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if not config["selection_seeds"] or not config["holdout_seeds"]:
        raise ValueError("验证与测试种子不能为空")
    if set(config["selection_seeds"]) & set(config["holdout_seeds"]):
        raise ValueError("验证与测试种子必须分离")


def write_json(path, value):
    """只新建 JSON，拒绝覆盖实验文件，并拒绝把 NaN 当作合法 JSON 写入。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def env_config(values):
    """从实验 JSON 重建冻结配置，包括嵌套奖励参数。"""
    values = dict(values)
    values["rewards"] = RewardWeights(**values["rewards"])
    return PickPlaceEnvConfig(**values)


def default_config(seed=42, horizon=250, action_scale=.005):
    """初始实验用完整随机目标、Home 起点；原环境成功阈值保持不变。"""
    config = PickPlaceEnvConfig(action_scale_m=action_scale, max_episode_steps=horizon,
                                max_simulation_time_s=horizon*.04)
    return {"seed": seed, "env": asdict(config), "norm_obs": True, "norm_reward": False,
            "ppo": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256,
                    "n_epochs": 10, "gamma": .99, "gae_lambda": .95, "clip_range": .2,
                    "ent_coef": 0., "vf_coef": .5, "max_grad_norm": .5,
                    "normalize_advantage": True, "use_sde": False, "target_kl": None},
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            "selection_seeds": SELECTION_SEEDS, "holdout_seeds": HOLDOUT_SEEDS,
            "eval_interval": 10000, "checkpoint_interval": 10000,
            "torch_threads": 1, "n_envs": 1}


def retraining_config(profile, seed=42, horizon=250, action_scale=.005):
    """修复实验共享协议；A改归一化，B再改探索，D单独提高距离成本，不改物理。"""
    if profile not in ("selective", "selective_low_noise", "selective_low_noise_distance"):
        raise ValueError("未知重训profile")
    config = default_config(seed, horizon, action_scale)
    config.update(experiment_version=profile, normalization_mode="physical",
                  log_std_init=0. if profile == "selective" else float(np.log(.3)),
                  eval_interval=4096, checkpoint_interval=8192,
                  holdout_seeds=list(range(40000, 40030)))
    if profile == "selective_low_noise_distance":
        config["env"]["rewards"]["reach_distance_rate"] = 2.0
        config["reference_reach_distance_rate"] = .2
    return config


def reference_original_reward(reward, terms, config):
    """D组只改距离成本，因此可逐项精确还原旧奖励口径，避免跨版本比较Return。"""
    if environment_version(config) == "reach_v2":
        return None  # New shaping has no exact conversion to the old reward.
    if "reference_reach_distance_rate" not in config:
        return float(reward)
    distance = terms.get("distance_cost", 0.)
    ratio = config["reference_reach_distance_rate"]/config["env"]["rewards"]["reach_distance_rate"]
    return float(reward-distance+ratio*distance)


def device_name(requested):
    """auto 优先 MPS，其次 CUDA/CPU；显式设备不可用时明确报错。"""
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前进程 MPS 不可用，可使用 --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用")
    return requested


def versions():
    """记录运行时依赖，避免把旧 lock 文件误认为本次真实安装版本。"""
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in ("torch", "numpy", "mujoco", "gymnasium",
                                                "stable-baselines3", "tensorboard")},
            "mps_available": torch.backends.mps.is_available()}


class TrainingResetOptions(gym.Wrapper):
    """课程只在reset设置目标/初始偏航，不提供动作，也不改step或成功判定。"""

    def __init__(self, env, defaults):
        super().__init__(env)
        self.defaults = dict(defaults)

    def reset(self, *, seed=None, options=None):
        """显式reset选项优先，便于检查课程与原随机任务的差别。"""
        return self.env.reset(seed=seed, options={**self.defaults, **(options or {})})


def make_vector(config, monitor_path=None, stats=None, training=True, human=False):
    """训练可用spawn子进程采样；评估始终单环境，冻结统计且不共享物理状态。

    CUDA初始化后不能使用fork复制进程；spawn在Linux/Mac使用一致的独立初始化。
    保留单环境DummyVecEnv默认行为，每个并行环境分别写Monitor文件。
    """
    validate_config(config)
    count = int(config.get("n_envs", 1)) if training else 1
    if count < 1 or (human and count != 1):
        raise ValueError("环境数必须为正，human只支持单环境")
    def factory(rank):
        def build():
            if environment_version(config) == "reach_v2":
                env = ReachEnv(config=env_config(config["env"]), render_mode="human" if human else None,
                               reach_rewards=config["reach_rewards"],
                               gravity_compensation=config["gravity_compensation"])
            else:
                env = PickPlaceEnv(task_mode="reach", config=env_config(config["env"]),
                                   render_mode="human" if human else None)
            if training and config.get("training_reset_options"):
                env = TrainingResetOptions(env, config["training_reset_options"])
            filename = None
            if monitor_path:
                filename = str(monitor_path)+(f"_{rank}" if count > 1 else "")
            return Monitor(env, filename=filename,
                           info_keywords=("is_success", "tcp_position_error", "tcp_orientation_error_rad"))
        return build
    factories = [factory(rank) for rank in range(count)]
    vector = (DummyVecEnv(factories) if count == 1
              else SubprocVecEnv(factories, start_method="spawn"))
    mode = config.get("normalization_mode", "all")
    if mode not in ("all", "physical"):
        vector.close()
        raise ValueError("未知normalization_mode")
    if stats is None:
        cls = PhysicalVecNormalize if mode == "physical" else VecNormalize
        kwargs = {"observation_fields": observation_schema(config)} if mode == "physical" else {}
        normalizer = cls(vector, **kwargs, norm_obs=config["norm_obs"],
                                  norm_reward=config["norm_reward"], gamma=config["ppo"]["gamma"])
    else:
        normalizer = VecNormalize.load(str(stats), vector)
        if isinstance(normalizer, PhysicalVecNormalize) != (mode == "physical"):
            normalizer.close()
            raise ValueError("归一化存档类型与配置不匹配")
        if mode == "physical":
            saved_fields = getattr(normalizer, "observation_fields", OBSERVATION_FIELDS)
            if [list(field) for field in saved_fields] != observation_schema(config):
                normalizer.close()
                raise ValueError("归一化存档的观测字段与配置不匹配")
    normalizer.training = training
    normalizer.norm_reward = bool(config["norm_reward"] and training)
    return normalizer


class ReachEvaluationEvidence:
    """只在独立评估环境观察原始物理判定，不插入任何控制或提前终止。"""

    def __init__(self, env):
        self.env = env
        self.original = env._update_timers
        env._update_timers = self._observe
        self.reset()

    def reset(self):
        """每回合清空诊断统计；不触碰MuJoCo状态或环境计时。"""
        self.count = 0
        self.passed = np.zeros(4, dtype=int)
        self.max_hold_s = 0.
        self.near_speed_sum = 0.
        self.near_count = 0

    def _observe(self, state):
        """先执行原计时，再采集2 ms物理证据，避免控制步末采样漏掉短暂中断。"""
        self.original(state)
        config = self.env.config
        flags = (state["tcp_error"] < config.tcp_position_tolerance_m,
                 state["orientation_error"] < config.tcp_orientation_tolerance_rad,
                 state["arm_speed"] < config.arm_velocity_tolerance_rad_s)
        self.count += 1
        self.passed += (*flags, all(flags))
        self.max_hold_s = max(self.max_hold_s, float(self.env.timers[0]))
        if flags[0]:
            self.near_count += 1
            self.near_speed_sum += state["arm_speed"]

    def result(self):
        """输出可JSON化的独立条件占比与真实最长保持时间。"""
        return {"physics_condition_fractions": {name: float(value/max(1, self.count))
                    for name, value in zip(("position", "orientation", "quiet_arm", "all"), self.passed)},
                "max_reach_stable_time_s": self.max_hold_s,
                "near_goal_mean_arm_speed_rad_s": self.near_speed_sum/self.near_count if self.near_count else None}


def save_bundle(model, normalizer, directory, config, role):
    """同一个目录保存策略、优化器、统计和 RNG；manifest 最后写入表示配对完成。

    不保存半条 rollout 或 MuJoCo 动态状态。恢复时开始新回合，因此不是逐位续跑。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model.zip")
    normalizer.save(directory / "vecnormalize.pkl")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, directory / "rng.pt")
    write_json(directory / "manifest.json", {"timesteps": model.num_timesteps, "role": role,
                "environment_version": environment_version(config), "observation_schema": observation_schema(config),
                "action_shape": list(model.action_space.shape),
                "config": config, "resume_starts_new_episode": True})
    return directory


def load_bundle(directory, device="cpu", training=False, monitor_path=None, human=False, environment_config=None,
                model_overrides=None):
    """要求 manifest、模型与 VecNormalize 成套加载，避免输入分布与策略不匹配。"""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    validate_config(config)
    if environment_config is not None:
        for key, default in (("environment_version", "reach_v1"), ("reach_rewards", None),
                             ("gravity_compensation", None), ("reward_version", None), ("timeout_mode", None)):
            if environment_config.get(key, default) != config.get(key, default):
                raise ValueError(f"存档与目标环境不兼容：{key}")
    if manifest.get("observation_schema", observation_schema(config)) != observation_schema(config):
        raise ValueError("存档观测字段版本不匹配")
    # 显式课程切换可替换reset分布；源配置仍返回给调用者核对控制/归一化兼容性。
    vector = make_vector(environment_config or config, monitor_path,
                         directory / "vecnormalize.pkl", training, human)
    try:
        # 显式超参数分支需在_setup_model前替换rollout长度，不能只修改config.json。
        model = PPO.load(directory / "model.zip", env=vector, device=device, force_reset=True,
                         **(model_overrides or {}))
    except BaseException:
        vector.close()
        raise
    return model, vector, config


def restore_rng(directory):
    """仅对本项目可信本地存档启用 pickle；恢复算法 RNG，不承诺复现未保存的回合。"""
    rng = torch.load(Path(directory) / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def reach_markers(env):
    """Viewer 绿球为真实目标、红球为累积命令；仅绘图，不修改物理或任务状态。"""
    import mujoco

    if not env.viewer.is_running():
        raise RuntimeError("Viewer 已关闭，停止可视化评估")
    with env.viewer.lock():
        scene = env.viewer.user_scn
        if scene.maxgeom < 2:
            return
        for index, (position, color) in enumerate(((env.target_position, [.1, 1., .1, .6]),
                                                  (env.tcp_command, [1., .1, .1, .6]))):
            mujoco.mjv_initGeom(scene.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.008, 0., 0.]), position,
                               np.eye(3).ravel(), np.asarray(color))
        scene.ngeom = 2
    env.render()


def evaluate(config, seeds, model=None, normalizer=None, policy="ppo", stats=None, human=False, reset_options=None):
    """独立物理评估；原始奖励、真实终态及控制诊断按 seed 保存。

    VecEnv 会自动 reset，结束指标必须取 terminal info，而非已复位的物理 data。
    固定评估种子与训练 RNG 分离；确定性 predict 不消耗训练策略采样 RNG。
    """
    if policy not in ("ppo", "random", "zero") or (policy == "ppo" and model is None):
        raise ValueError("需要有效策略，以及 PPO 对应的模型")
    vector = make_vector(config, stats=stats, training=False, human=human)
    try:
        if normalizer is not None:
            sync_envs_normalization(normalizer, vector)
        vector.training, vector.norm_reward = False, False
        episodes, started = [], time.perf_counter()
        evidence = ReachEvaluationEvidence(vector.venv.envs[0].unwrapped)
        for seed in seeds:
            evidence.reset()
            vector.seed(int(seed))
            if reset_options:
                vector.set_options(reset_options)
            observation = vector.reset()
            raw = vector.venv.envs[0].unwrapped
            initial = {"target_position_m": raw.target_position.tolist(),
                       "cube_position_m": raw.data.body("cube").xpos.tolist(),
                       "target_rotation": raw.target_rotation.tolist()}
            rng = np.random.default_rng(int(seed)+1000000)
            total, reference_total, ik_failures, action_norm, lag, max_lag = 0., 0., 0, 0., 0., 0.
            if environment_version(config) == "reach_v2":
                reference_total = None
            costs, minimum_error = Counter(), float("inf")
            max_reach_stable_s, within_20mm_steps = 0., 0
            episode_started = time.perf_counter()
            for step in range(config["env"]["max_episode_steps"]):
                tick = time.perf_counter()
                if human:
                    reach_markers(raw)
                if policy == "ppo":
                    action, _ = model.predict(observation, deterministic=True)
                elif policy == "random":
                    action = rng.uniform(-1, 1, (1, *vector.action_space.shape)).astype(np.float32)
                else:
                    action = np.zeros((1, *vector.action_space.shape), dtype=np.float32)
                observation, reward, done, infos = vector.step(action)
                info = infos[0]
                total += float(reward[0])
                if reference_total is not None:
                    reference_total += reference_original_reward(float(reward[0]), info["reward_terms"], config)
                ik_failures += int(not info.get("ik_success", False))
                action_norm += float(np.linalg.norm(np.clip(action[0, :3], -1, 1)))
                if info["tcp_position_error"] is not None:
                    minimum_error = min(minimum_error, info["tcp_position_error"])
                    within_20mm_steps += int(info["tcp_position_error"] < .02)
                    max_reach_stable_s = max(max_reach_stable_s, info["stable_times_s"][0])
                    gap = float(np.linalg.norm(np.array(info["tcp_command_m"])-info["tcp_position_m"]))
                    lag += gap
                    max_lag = max(max_lag, gap)
                costs.update(info["reward_terms"])
                if human:
                    time.sleep(max(0., raw.dt-(time.perf_counter()-tick)))
                if done[0]:
                    break
            else:
                raise RuntimeError("评估回合超过环境上限却未结束")
            episodes.append({"seed": int(seed), "initial": initial, "success": bool(info["is_success"]),
                "return": total, "reference_original_return": reference_total,
                "steps": step+1, "final_tcp_error_m": info["tcp_position_error"],
                "final_orientation_error_rad": info.get("tcp_orientation_error_rad"),
                "minimum_tcp_error_m": minimum_error if np.isfinite(minimum_error) else None,
                "max_reach_stable_time_s": max_reach_stable_s,
                "auxiliary_within_20mm_fraction": within_20mm_steps/(step+1),
                "ik_failure_steps": ik_failures, "mean_xyz_action_norm": action_norm/(step+1),
                "mean_command_tracking_gap_m": lag/(step+1), "max_command_tracking_gap_m": max_lag,
                "reward_terms_total": dict(costs), "termination_reason": info["termination_reason"],
                "simulation_time_s": info["simulation_time"],
                "wall_time_s": time.perf_counter()-episode_started})
            episodes[-1].update(evidence.result())
        summary = summarize(episodes)
        return {"policy": policy, "deterministic": policy == "ppo", "seeds": list(seeds),
                "environment_version": environment_version(config),
                "reward_version": config.get("reward_version", "legacy"),
                "timeout_mode": config.get("timeout_mode", "external_truncation"),
                "gravity_compensation": config.get("gravity_compensation", False),
                "training_timesteps": int(model.num_timesteps) if model is not None else 0,
                "wall_time_s": time.perf_counter()-started, "summary": summary, "episodes": episodes}
    finally:
        vector.close()


def summarize(episodes):
    """跨回合汇总，失败分类不把 TIME_LIMIT 当作物理失败。"""
    steps = sum(e["steps"] for e in episodes)
    reasons = Counter(e["termination_reason"] for e in episodes)
    def mean(key):
        values = [e[key] for e in episodes if e.get(key) is not None]
        return float(np.mean(values)) if values else None
    summary = {"episodes": len(episodes), "success_rate": mean("success"),
        "mean_return": mean("return"), "mean_episode_length": mean("steps"),
        "mean_reference_original_return": mean("reference_original_return"),
        "mean_final_tcp_error_m": mean("final_tcp_error_m"),
        "mean_distance_to_goal_m": mean("final_tcp_error_m"),
        "mean_final_orientation_error_rad": mean("final_orientation_error_rad"),
        "ik_failure_frequency": sum(e["ik_failure_steps"] for e in episodes)/steps,
        "time_limit_fraction": reasons.get("TIME_LIMIT", 0)/len(episodes),
        "termination_distribution": dict(reasons),
        "physical_failures": {k: v for k, v in reasons.items() if k not in ("SUCCESS", "TIME_LIMIT")}}
    if "physics_condition_fractions" in episodes[0]:
        summary["mean_max_reach_stable_time_s"] = mean("max_reach_stable_time_s")
        for condition in ("position", "orientation", "quiet_arm", "all"):
            summary["physics_"+condition+"_fraction"] = float(np.mean([
                episode["physics_condition_fractions"][condition] for episode in episodes]))
    return summary
