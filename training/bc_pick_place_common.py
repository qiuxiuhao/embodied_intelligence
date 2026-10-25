"""BC data, frozen train-only normalization and SB3-compatible policy bundles."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.pick_place_expert import PHASES
from envs.pick_place_env import OBSERVATION_FIELDS, OBSERVATION_SIZE
from training.collect_pick_place_demos import verify_dataset
from training.pick_place_common import ROOT, IDENTITY, make_env, sha256, write_json
from training.selective_normalize import PhysicalVecNormalize


DEFAULT_BC_CONFIG = ROOT / "configs/day15_bc.json"


def load_bc_config(path=DEFAULT_BC_CONFIG):
    config = json.loads(Path(path).read_text())
    validate_bc_config(config)
    return config


def validate_bc_config(c):
    expected = {"seed", "device", "torch_threads", "epochs", "batch_size", "learning_rate",
                "hidden_sizes", "xyz_loss_weight", "gripper_loss_weight", "phase_balance",
                "max_grad_norm", "initial_std", "eval_interval", "eval_episodes"}
    if set(c) != expected:
        raise ValueError(f"Unknown/missing BC configuration fields: {set(c) ^ expected}")
    for name in ("torch_threads", "epochs", "batch_size", "eval_interval"):
        if type(c[name]) is not int or c[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if c["eval_episodes"] != "all" and (
            type(c["eval_episodes"]) is not int or c["eval_episodes"] <= 0):
        raise ValueError("eval_episodes must be 'all' or a positive integer")
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if c["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu/cuda/mps")
    for name in ("learning_rate", "xyz_loss_weight", "gripper_loss_weight", "max_grad_norm"):
        if not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(c["phase_balance"]) or not 0 <= c["phase_balance"] <= 1:
        raise ValueError("phase_balance must be in [0,1]")
    if not c["hidden_sizes"] or any(type(v) is not int or v <= 0 for v in c["hidden_sizes"]):
        raise ValueError("hidden_sizes must contain positive integers")
    std = np.asarray(c["initial_std"])
    if std.shape != (4,) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("initial_std must contain four positive finite values")


def check_task_compatibility(protocol):
    """Compare recorded physics/schema, allowing newly added BC tooling.

    The dataset's complete source snapshot is separately checked by its auditor.
    Adding a training module must not invalidate unchanged physical trajectories.
    """
    for key, value in IDENTITY.items():
        if protocol["config"][key] != value:
            raise ValueError(f"Dataset protocol mismatch: {key}")
    if protocol["observation_fields"] != [list(f) for f in OBSERVATION_FIELDS]:
        raise ValueError("Observation schema mismatch")
    if protocol["phase_names"] != list(PHASES):
        raise ValueError("Expert phase label schema mismatch")
    for name, digest in protocol["source_sha256"].items():
        if name.split("/")[0] in ("envs", "controllers", "robotics", "assets"):
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Physics sources differ from dataset: {name}")
    for name, digest in protocol["mesh_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Mesh differs from dataset: {name}")


def load_demonstrations(directory):
    directory = Path(directory)
    audit = verify_dataset(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    check_task_compatibility(protocol)
    manifest = json.loads((directory / "manifest.json").read_text())
    splits = {}
    for split in ("train", "validation"):
        parts = {key: [] for key in ("observations", "actions", "phase_ids")}
        eligible = manifest["bc_eligible_files"][split]
        if not eligible:
            raise ValueError(f"No successful demonstrations in {split}")
        for relative in eligible:
            with np.load(directory / relative, allow_pickle=False) as episode:
                for key in parts:
                    parts[key].append(episode[key].copy())
        splits[split] = {key: np.concatenate(value) for key, value in parts.items()}
        counts = np.bincount(splits[split]["phase_ids"], minlength=len(PHASES))
        if np.any(counts == 0):
            raise ValueError(f"{split} lacks complete expert phase coverage")
    return protocol, splits, audit


def make_normalizer(task_config, train_observations=None, stats=None):
    vector = DummyVecEnv([lambda: make_env(task_config)])
    try:
        if stats is not None:
            normalizer = VecNormalize.load(str(stats), vector)
            if (not isinstance(normalizer, PhysicalVecNormalize)
                    or list(map(list, normalizer.observation_fields)) != list(map(list, OBSERVATION_FIELDS))
                    or not normalizer.norm_obs):
                raise ValueError("Invalid BC normalizer schema")
        else:
            normalizer = PhysicalVecNormalize(vector, observation_fields=OBSERVATION_FIELDS,
                                               norm_obs=True, norm_reward=False, training=False)
            if train_observations is None or not len(train_observations):
                raise ValueError("Training observations are required to fit normalization")
            # Only pre-action observations from train; neither validation nor
            # online evaluation can update these moments.
            for start in range(0, len(train_observations), 8192):
                normalizer.obs_rms.update(train_observations[start:start+8192])
        normalizer.training = False
        normalizer.norm_reward = False
        return normalizer
    except BaseException:
        vector.close()
        raise


def build_model(normalizer, config):
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select --device cpu")
    if config["device"] == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; select --device cpu")
    # The PPO container supplies its exact actor and save/load protocol, but no
    # PPO rollout or update is executed during BC. num_timesteps stays zero.
    model = PPO("MlpPolicy", normalizer, device=config["device"], seed=config["seed"],
                learning_rate=config["learning_rate"], n_steps=1024, batch_size=256,
                policy_kwargs={"net_arch": {"pi": config["hidden_sizes"], "vf": config["hidden_sizes"]},
                               "activation_fn": torch.nn.Tanh})
    with torch.no_grad():
        model.policy.log_std.copy_(torch.as_tensor(
            np.log(config["initial_std"]), dtype=torch.float32, device=model.device))
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    # FlattenExtractor has no trainable weights. Critic and log_std stay frozen.
    for module in (model.policy.mlp_extractor.policy_net, model.policy.action_net):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model


def actor_mean(policy, observations):
    return policy.get_distribution(observations).distribution.mean


def action_losses(prediction, target, config):
    squared = (prediction-target).square()
    xyz = squared[:, :3].mean(dim=1)
    grip = squared[:, 3]
    return config["xyz_loss_weight"]*xyz + config["gripper_loss_weight"]*grip, xyz, grip


def sampling_probabilities(phase_ids, balance):
    counts = np.bincount(phase_ids, minlength=len(PHASES))
    active = np.count_nonzero(counts)
    weights = (1-balance)/len(phase_ids) + balance/(active*counts[phase_ids])
    return weights/weights.sum()


@torch.no_grad()
def offline_metrics(model, observations, actions, phase_ids, config):
    model.policy.set_training_mode(False)
    rows, sign_correct = [], []
    for start in range(0, len(observations), 4096):
        obs = torch.as_tensor(observations[start:start+4096], device=model.device)
        target = torch.as_tensor(actions[start:start+4096], device=model.device)
        prediction = actor_mean(model.policy, obs)
        total, xyz, grip = action_losses(prediction, target, config)
        rows.append(torch.stack((total, xyz, grip), dim=1).cpu().numpy())
        sign_correct.append(((prediction[:, 3] >= 0) == (target[:, 3] >= 0)).cpu().numpy())
    losses = np.concatenate(rows)
    by_phase = {}
    for index, name in enumerate(PHASES):
        selected = losses[phase_ids == index]
        by_phase[name] = {"count": len(selected), "loss": float(selected[:, 0].mean()),
                          "xyz_mse": float(selected[:, 1].mean()), "gripper_mse": float(selected[:, 2].mean())}
    return {"loss": float(losses[:, 0].mean()), "xyz_mse": float(losses[:, 1].mean()),
            "gripper_mse": float(losses[:, 2].mean()),
            "gripper_sign_accuracy": float(np.concatenate(sign_correct).mean()),
            "phase_balanced_loss": float(np.mean([v["loss"] for v in by_phase.values()])),
            "by_phase": by_phase}


def evaluate_policy(model, normalizer, task_config, seeds, human=False, progress=None):
    """Closed-loop learned actions only, independent physics, frozen statistics."""
    if not seeds:
        raise ValueError("Evaluation seed list cannot be empty")
    env = make_env(task_config, render_mode="human" if human else None)
    episodes = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed), options={"start_pose": "home"})
            total, grasped, lifted = 0., False, False
            for step in range(env.config.max_episode_steps):
                tick = time.perf_counter()
                action, _ = model.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                timers = info.get("stable_times_s", [0., 0., 0., 0.])
                grasped |= timers[1] >= env.config.grasp_hold_s
                lifted |= info.get("has_lifted", False)
                if human:
                    if not env.viewer.is_running():
                        raise RuntimeError("Viewer closed")
                    time.sleep(max(0., env.dt-(time.perf_counter()-tick)))
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Evaluation exceeded the task horizon")
            episodes.append({"seed": int(seed), "success": info["is_success"], "steps": step+1,
                             "return": total, "grasped": bool(grasped), "lifted": bool(lifted),
                             "final_cube_error_m": info["cube_position_error"],
                             "reason": info["termination_reason"]})
            if progress is not None:
                progress(len(episodes), len(seeds), sum(e["success"] for e in episodes))
    finally:
        env.close()
    errors = [e["final_cube_error_m"] for e in episodes if e["final_cube_error_m"] is not None]
    return {"episodes": episodes, "summary": {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "grasp_rate": float(np.mean([e["grasped"] for e in episodes])),
        "lift_rate": float(np.mean([e["lifted"] for e in episodes])),
        "mean_cube_error_m": float(np.mean(errors)) if errors else None,
        "mean_steps": float(np.mean([e["steps"] for e in episodes]))}}


def save_bundle(directory, model, normalizer, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model.zip")
    normalizer.save(directory / "vecnormalize.pkl")
    write_json(directory / "manifest.json", {
        **metadata, "format": "day15_bc_sb3_v1", "ppo_timesteps": model.num_timesteps,
        "critic_trained": False, "log_std_trained": False, "normalization_frozen": True,
        "sha256": {name: sha256(directory / name) for name in ("model.zip", "vecnormalize.pkl")}})


def load_bundle(directory, device="cpu"):
    """Load only trusted local SB3/pickle bundles created by this project."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["format"] != "day15_bc_sb3_v1":
        raise ValueError("Not a Day15 BC bundle")
    for name, digest in metadata["sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Bundle hash mismatch: {name}")
    check_task_compatibility(metadata["dataset_protocol"])
    normalizer = make_normalizer(metadata["dataset_protocol"]["config"],
                                 stats=directory / "vecnormalize.pkl")
    try:
        model = PPO.load(directory / "model.zip", env=normalizer, device=device)
        if model.observation_space.shape != (OBSERVATION_SIZE,) or model.action_space.shape != (4,):
            raise ValueError("BC policy spaces do not match the dataset")
        return model, normalizer, metadata
    except BaseException:
        normalizer.close()
        raise
