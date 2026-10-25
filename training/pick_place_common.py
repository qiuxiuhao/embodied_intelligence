"""Day15 versioned task protocol, raw transitions and physical evidence.

This module deliberately does not import the Reach-specific training factory,
normalizer or PPO. Demonstration files remain unnormalized and pickle-free.
"""

from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from agents.pick_place_expert import EXPERT_VERSION, PHASES, ExpertConfig, ExpertFailure, PickPlaceExpert
from envs.pick_place_env import OBSERVATION_FIELDS, OBSERVATION_SIZE, OBSERVATION_SLICES, PickPlaceEnvConfig, RewardWeights
from envs.pick_place_learning_env import PickPlaceLearningEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/day15_pick_place_bc_ppo.json"
IDENTITY = {
    "environment_version": PickPlaceLearningEnv.environment_version,
    "reward_version": PickPlaceLearningEnv.reward_version,
    "timeout_mode": PickPlaceLearningEnv.timeout_mode,
    "observation_version": "day13_raw_94_v1",
    "dataset_version": "pick_place_transitions_v1",
    "gravity_compensation": True,
}


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_range(spec):
    if (type(spec["seed_start"]) is not int or spec["seed_start"] < 0
            or type(spec["episodes"]) is not int or spec["episodes"] <= 0):
        raise ValueError("Seed ranges require a nonnegative start and positive episode count")
    return list(range(spec["seed_start"], spec["seed_start"] + spec["episodes"]))


def load_config(path=DEFAULT_CONFIG):
    raw = json.loads(Path(path).read_text())
    required = set(IDENTITY) | {"env", "expert", "acceptance", "demonstrations", "reserved_holdout"}
    if set(raw) != required:
        raise ValueError(f"Configuration fields differ: {set(raw) ^ required}")
    for key, value in IDENTITY.items():
        if raw[key] != value or (key == "gravity_compensation" and raw[key] is not True):
            raise ValueError(f"Incompatible protocol: {key}")
    env = asdict(PickPlaceEnvConfig(action_scale_m=.005, max_episode_steps=750,
                                   max_simulation_time_s=30.))
    changes = dict(raw["env"])
    rewards = changes.pop("rewards", {})
    env.update(changes)
    env["rewards"].update(rewards)
    env["rewards"] = RewardWeights(**env["rewards"])
    raw["env"] = asdict(PickPlaceEnvConfig(**env))
    raw["expert"] = asdict(ExpertConfig(**raw["expert"]))
    if set(raw["demonstrations"]) != {"train", "validation"}:
        raise ValueError("Dataset must contain train and validation episode splits")
    acceptance = raw["acceptance"]
    if not 0 < acceptance["minimum_success_rate"] <= 1:
        raise ValueError("Invalid expert acceptance threshold")
    groups = [seed_range(v) for v in raw["demonstrations"].values()]
    groups += [seed_range(raw["reserved_holdout"]), seed_range({
        "seed_start": acceptance["random_seed_start"], "episodes": acceptance["random_episodes"]})]
    # Fixed/corner acceptance cases use their own reserved seed block.
    groups += [list(range(300000, 300100))]
    seen = set()
    for group in groups:
        if seen.intersection(group):
            raise ValueError("Acceptance, train, validation and reserved holdout seeds must be disjoint")
        seen.update(group)
    # Dataclass defaults contain tuples; persist/reload the same canonical JSON
    # representation so a valid acceptance report can be compared exactly.
    return json.loads(json.dumps(raw))


def make_env(config, render_mode=None):
    values = dict(config["env"])
    values["rewards"] = RewardWeights(**values["rewards"])
    return PickPlaceLearningEnv(PickPlaceEnvConfig(**values), render_mode=render_mode)


def source_hashes():
    paths = [ROOT / "agents/pick_place_expert.py"]
    for folder in ("envs", "controllers", "robotics"):
        paths.extend((ROOT / folder).glob("*.py"))
    paths.extend((ROOT / "training").glob("*pick_place*.py"))
    paths.extend((ROOT / "assets/franka_panda").glob("*.xml"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def provenance(config):
    return {
        "config": config, "expert_version": EXPERT_VERSION,
        "observation_fields": [list(field) for field in OBSERVATION_FIELDS],
        "action_fields": ["tcp_command_dx", "tcp_command_dy", "tcp_command_dz", "gripper"],
        "action_semantics": "XYZ normalized increments on last legal TCP command; width=(gripper+1)*0.04m",
        "raw_observations": True, "normalization_fitted": False,
        "phase_names": list(PHASES),
        "source_sha256": source_hashes(),
        "mesh_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(
            (ROOT / "assets/franka_panda/assets").iterdir()) if p.suffix in (".obj", ".stl")},
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "mujoco", "gymnasium")},
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout,
    }


def rollout_expert(env, config, seed, options=None):
    """Record (obs_t, action_t, next_obs_t) without auto-reset or hidden control.

    On expert abort, the final transition flags remain honest (not fabricated
    terminal flags); the episode is explicitly ineligible for BC.
    """
    started = time.perf_counter()
    obs, initial_info = env.reset(seed=int(seed), options=options)
    expert = PickPlaceExpert(env.config.action_scale_m, env.dt, ExpertConfig(**config["expert"]))
    expert.reset(obs)
    data = {name: [] for name in ("observations", "actions", "applied_actions", "next_observations",
                                  "rewards", "terminated", "truncated", "phase_ids", "reward_terms")}
    info, abort, term_names = initial_info, None, None
    ik_failures = workspace_clips = 0
    max_lift = 0.
    for _ in range(env.config.max_episode_steps):
        try:
            action = expert.act(obs)
        except ExpertFailure as error:
            abort = str(error)
            break
        if not env.action_space.contains(action):
            raise ValueError("Expert emitted an invalid action")
        before = obs.copy()
        obs, reward, terminated, truncated, info = env.step(action)
        names = sorted(info["reward_terms"])
        if term_names is None:
            term_names = names
        elif names != term_names:
            # Numerical failure has a reduced reward dictionary; keep a common
            # schema by filling absent terms with zero, including past rows.
            union = sorted(set(term_names) | set(names))
            if union != term_names:
                data["reward_terms"] = [[dict(zip(term_names, row)).get(k, 0.) for k in union]
                                        for row in data["reward_terms"]]
                term_names = union
        for name, value in {
            "observations": before, "actions": action.copy(),
            "applied_actions": info.get("applied_action", action.tolist()),
            "next_observations": obs.copy(), "rewards": reward,
            "terminated": terminated, "truncated": truncated,
            "phase_ids": PHASES.index(expert.phase),
            "reward_terms": [info["reward_terms"].get(k, 0.) for k in term_names],
        }.items():
            data[name].append(value)
        ik_failures += int(not info.get("ik_success", False))
        workspace_clips += int(info.get("workspace_clipped", False))
        max_lift = max(max_lift, info.get("cube_bottom_clearance") or 0.)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("Environment did not end at its configured horizon")
    arrays = {}
    for key, value in data.items():
        dtype = (np.bool_ if key in ("terminated", "truncated") else
                 np.int16 if key == "phase_ids" else
                 np.float64 if key in ("rewards", "reward_terms") else np.float32)
        arrays[key] = np.asarray(value, dtype=dtype)
    n = len(arrays["actions"])
    for key in ("observations", "next_observations"):
        arrays[key] = arrays[key].reshape(n, OBSERVATION_SIZE)
    for key in ("actions", "applied_actions"):
        arrays[key] = arrays[key].reshape(n, 4)
    arrays["reward_terms"] = arrays["reward_terms"].reshape(n, len(term_names or []))
    success = bool(abort is None and info["is_success"])
    metadata = {
        "seed": int(seed), "reset_options": options or {}, "success": success,
        "eligible_for_bc": success, "expert_abort": abort, "steps": n,
        "return": float(arrays["rewards"].sum()),
        "reason": abort or info["termination_reason"],
        "initial_info": initial_info, "final_info": info,
        "phase_history": expert.history, "reward_term_names": term_names or [],
        "ik_failure_steps": ik_failures, "workspace_clipped_steps": workspace_clips,
        "max_cube_clearance_m": max_lift, "wall_time_s": time.perf_counter()-started,
        "physical_thresholds": {key: getattr(env.config, key) for key in (
            "placement_xy_tolerance_m", "placement_z_tolerance_m",
            "cube_linear_velocity_tolerance_m_s", "cube_angular_velocity_tolerance_rad_s",
            "release_separation_m", "open_feedback_min_m", "place_hold_s")},
    }
    validate_episode(arrays, metadata)
    return arrays, metadata


def validate_episode(arrays, metadata):
    expected = {"observations", "actions", "applied_actions", "next_observations",
                "rewards", "terminated", "truncated", "phase_ids", "reward_terms"}
    if set(arrays) != expected:
        raise ValueError("Transition fields do not match dataset schema")
    n = metadata["steps"]
    for name, values in arrays.items():
        if len(values) != n or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid transition array {name}")
    for key in ("observations", "next_observations"):
        if arrays[key].shape != (n, OBSERVATION_SIZE) or arrays[key].dtype != np.float32:
            raise ValueError("Invalid observation shape")
    for key in ("actions", "applied_actions"):
        if arrays[key].shape != (n, 4) or np.any(np.abs(arrays[key]) > 1):
            raise ValueError("Invalid bounded action labels")
    if arrays["reward_terms"].shape != (n, len(metadata["reward_term_names"])):
        raise ValueError("Reward schema mismatch")
    for key in ("terminated", "truncated"):
        if arrays[key].shape != (n,) or arrays[key].dtype != np.bool_:
            raise ValueError("Invalid termination flags")
    if arrays["rewards"].shape != (n,) or arrays["phase_ids"].shape != (n,):
        raise ValueError("Invalid scalar transition fields")
    np.testing.assert_array_equal(arrays["applied_actions"], arrays["actions"])
    np.testing.assert_array_equal(arrays["observations"][1:], arrays["next_observations"][:-1])
    np.testing.assert_allclose(arrays["reward_terms"].sum(axis=1), arrays["rewards"], atol=1e-12)
    np.testing.assert_allclose(arrays["rewards"].sum(), metadata["return"], atol=1e-10)
    if np.any(arrays["terminated"] & arrays["truncated"]):
        raise ValueError("Both termination flags set")
    if np.any(arrays["terminated"][:-1] | arrays["truncated"][:-1]):
        raise ValueError("Episode contains transitions after a terminal")
    if np.any(arrays["phase_ids"] < 0) or np.any(arrays["phase_ids"] >= len(PHASES)):
        raise ValueError("Invalid expert phase labels")
    if metadata["success"]:
        info = metadata["final_info"]
        c = metadata["physical_thresholds"]
        if (n == 0 or not arrays["terminated"][-1] or arrays["truncated"][-1]
                or metadata["expert_abort"] is not None or info["termination_reason"] != "SUCCESS"
                or not info["is_success"] or not info["has_lifted"] or not info["cube_table_contact"]
                or info["grasp_candidate"] or info["gripper_opening"] < .07):
            raise ValueError("Missing physical success evidence")
        final = arrays["next_observations"][-1]
        if (np.any(final[OBSERVATION_SLICES["finger_contacts"]][:2])
                or np.linalg.norm(final[OBSERVATION_SLICES["tcp_to_cube"]]) < c["release_separation_m"]
                or final[OBSERVATION_SLICES["stable_timers"]][3] < 1.):
            raise ValueError("Release/retreat/stability evidence is missing")
        delta = np.asarray(info["cube_position_m"]) - info["target_position_m"]
        if (np.linalg.norm(delta[:2]) >= c["placement_xy_tolerance_m"]
                or abs(delta[2]) >= c["placement_z_tolerance_m"]
                or abs(info["cube_bottom_clearance"]) >= c["placement_z_tolerance_m"]
                or info["cube_linear_speed_m_s"] >= c["cube_linear_velocity_tolerance_m_s"]
                or info["cube_angular_speed_rad_s"] >= c["cube_angular_velocity_tolerance_rad_s"]
                or info["stable_times_s"][3] < c["place_hold_s"]
                or info["gripper_opening"] < c["open_feedback_min_m"]):
            raise ValueError("Physical placement tolerances were not satisfied")
    elif not metadata["expert_abort"] and (n == 0 or not (arrays["terminated"][-1] or arrays["truncated"][-1])):
        raise ValueError("Unfinished episode without an explicit expert abort")
    if metadata["eligible_for_bc"] != metadata["success"]:
        raise ValueError("BC eligibility mismatch")


def summarize(records):
    successful = [r for r in records if r["success"]]
    errors = [r["final_info"]["cube_position_error"] for r in successful]
    return {
        "episodes": len(records), "successes": len(successful),
        "success_rate": len(successful)/len(records) if records else 0.,
        "transitions": sum(r["steps"] for r in records),
        "mean_steps": float(np.mean([r["steps"] for r in records])) if records else None,
        "reason_counts": dict(Counter(r["reason"] for r in records)),
        "success_mean_cube_error_m": float(np.mean(errors)) if errors else None,
        "success_max_cube_error_m": float(np.max(errors)) if errors else None,
        "ik_failure_steps": sum(r["ik_failure_steps"] for r in records),
        "workspace_clipped_steps": sum(r["workspace_clipped_steps"] for r in records),
    }
