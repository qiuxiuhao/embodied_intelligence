"""Reach v2: XYZ actions, relative observations and dense near/still reward.

The base environment still owns physics, IK, contacts and the original success
thresholds. This version defines the configured time budget as a finite task:
timeout is terminal (not a bootstrap truncation), with time remaining observed.
"""

from dataclasses import asdict, dataclass

from gymnasium import spaces
import numpy as np

from controllers.gravity_compensated_joint_controller import GravityCompensatedJointController
from controllers.ik_controller import rotation_error_world
from envs.pick_place_env import PickPlaceEnv, PickPlaceEnvConfig
from robotics.kinematics import get_arm_qpos, get_arm_qvel


REACH_OBSERVATION_FIELDS = (
    ("arm_qpos", 7, "rad"), ("arm_qvel", 7, "rad/s"),
    ("tcp_position", 3, "m"), ("target_minus_tcp", 3, "m"),
    ("command_minus_tcp", 3, "m"), ("orientation_error", 3, "rad"),
    ("joint_command", 7, "rad"), ("joint_target", 7, "rad"),
    ("previous_action", 3, "[-1,1]"), ("reach_timer", 1, "[0,1]"),
    ("remaining_fraction", 1, "[0,1]"), ("ik_failure_fraction", 1, "[0,1]"),
)
REACH_OBSERVATION_SIZE = sum(size for _, size, _ in REACH_OBSERVATION_FIELDS)


@dataclass(frozen=True)
class ReachRewardConfig:
    distance_rate: float = 2.
    near_still_rate: float = .5
    near_distance_m: float = .012
    quiet_speed_rad_s: float = .10
    action_rate: float = .01
    action_change: float = .001
    success_bonus: float = 10.
    failure_penalty: float = 10.
    ik_failure: float = 1.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.near_distance_m <= 0 or self.quiet_speed_rad_s <= 0:
            raise ValueError("Reward length and velocity scales must be positive")


class ReachEnv(PickPlaceEnv):
    environment_version = "reach_v2"
    observation_fields = REACH_OBSERVATION_FIELDS

    def __init__(self, config=None, render_mode=None, reach_rewards=None, gravity_compensation=True):
        self.reach_rewards = ReachRewardConfig(**(reach_rewards or {}))
        if not isinstance(gravity_compensation, bool):
            raise ValueError("gravity_compensation must be boolean")
        self.gravity_compensation = gravity_compensation
        if config is None:
            config = PickPlaceEnvConfig(action_scale_m=.005, max_episode_steps=250,
                                        max_simulation_time_s=10.)
        super().__init__(task_mode="reach", config=config, render_mode=render_mode)
        if gravity_compensation:
            self.arm = GravityCompensatedJointController(
                self.model, self.config.max_joint_speed_rad_s)
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        low, high = [], []
        for _, size, unit in self.observation_fields:
            lower, upper = (-1., 1.) if unit == "[-1,1]" else (
                (0., 1.) if unit == "[0,1]" else (-np.inf, np.inf))
            low.extend([lower]*size)
            high.extend([upper]*size)
        self.observation_space = spaces.Box(np.asarray(low, np.float32), np.asarray(high, np.float32))
        self._reach_achieved = False

    def reset(self, *, seed=None, options=None):
        if options and options.get("task_mode", "reach") != "reach":
            raise ValueError("Reach v2 only supports task_mode=reach")
        self._reach_achieved = False
        return super().reset(seed=seed, options=options)

    def step(self, action):
        raw = np.asarray(action)
        if raw.shape != (3,) or np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
            raise ValueError("Reach v2 action must be a finite real XYZ vector")
        observation, reward, terminated, truncated, info = super().step(np.r_[raw, 1.])
        if truncated:
            # Finite-horizon task. No success or physical-failure penalty on timeout.
            terminated, truncated = True, False
            info["timeout_is_terminal"] = True
        if "applied_action" in info:
            info["applied_action"] = info["applied_action"][:3]
        info["fixed_gripper_action"] = 1.
        return observation, reward, terminated, truncated, info

    def _update_timers(self, state):
        super()._update_timers(state)
        self._reach_achieved |= bool(self.timers[0] >= self.config.reach_hold_s)

    def _task_success(self, valid, failure):
        return bool(valid and failure is None and self._reach_achieved)

    def _potentials(self, state):
        # Reset may call this shared hook; v2 intentionally has no potential term.
        return {}

    def _reward(self, state, action, ik_valid, clipped, success, failure, elapsed_s):
        w = self.reach_rewards
        xyz = action[:3]
        terms = {
            "distance_cost": -w.distance_rate*min(state["tcp_error"]/self.config.distance_scale_m, 5)*elapsed_s,
            "near_still_reward": w.near_still_rate*elapsed_s*float(
                np.exp(-(state["tcp_error"]/w.near_distance_m)**2)
                * np.exp(-(state["arm_speed"]/w.quiet_speed_rad_s)**2)),
            "action_cost": -w.action_rate*float(np.mean(xyz**2))*elapsed_s,
            "action_change_cost": -w.action_change*float(np.mean((xyz-self.previous_action[:3])**2)),
            "ik_failure_cost": -w.ik_failure*float(not ik_valid),
            "success_bonus": w.success_bonus*float(success and ik_valid and failure is None),
            "failure_penalty": -w.failure_penalty*float(failure is not None),
        }
        if not ik_valid or failure is not None:
            terms = {key: min(0., value) for key, value in terms.items()}
        return terms

    def _observation(self, state):
        c = self.config
        remaining = max(0., 1-max(self.episode_step/c.max_episode_steps,
                                  self.data.time/c.max_simulation_time_s))
        values = (
            get_arm_qpos(self.data), get_arm_qvel(self.data), state["tcp"],
            self.target_position-state["tcp"], self.tcp_command-state["tcp"],
            rotation_error_world(self.target_rotation, state["rotation"]),
            self.arm.command_q, self.joint_target, self.previous_action[:3],
            [min(1., self.timers[0]/c.reach_hold_s)], [remaining],
            [min(1., self.ik_failures/c.max_ik_failures)],
        )
        observation = np.concatenate(values).astype(np.float32)
        if not self.observation_space.contains(observation) or not np.all(np.isfinite(observation)):
            raise FloatingPointError("Invalid Reach v2 observation")
        return observation

    def _info(self, state, terms, success, failure, reason, **extra):
        info = super()._info(state, terms, success, failure, reason, **extra)
        info.update(environment_version=self.environment_version,
                    reward_version="reach_dense_v2", arm_speed_rad_s=state["arm_speed"],
                    command_tracking_gap_m=float(np.linalg.norm(self.tcp_command-state["tcp"])),
                    reach_hold_completed=self._reach_achieved,
                    gravity_compensation=self.gravity_compensation,
                    compensation_clipped=bool(getattr(self.arm, "compensation_clipped", False)))
        return info

    def _numerical_failure(self, detail, physics_steps=0):
        observation, _, terminated, truncated, info = super()._numerical_failure(detail, physics_steps)
        reward = -self.reach_rewards.failure_penalty
        self.episode_return += reward + self.config.rewards.failure_penalty
        info["reward_terms"] = {"failure_penalty": reward}
        info["episode"]["r"] = self.episode_return
        info.update(environment_version=self.environment_version, reward_version="reach_dense_v2")
        return observation, reward, terminated, truncated, info
