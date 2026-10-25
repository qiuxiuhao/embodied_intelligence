"""Day15 state-based pick/place: Day13 task with Day14 gravity feedforward.

The initial data protocol deliberately retains the 94D observation, reward and
external time-limit semantics. Changing any of these requires a new dataset
version. The environment contains no expert stages or action overrides.
"""

from controllers.gravity_compensated_joint_controller import GravityCompensatedJointController
from envs.pick_place_env import OBSERVATION_FIELDS, PickPlaceEnv, PickPlaceEnvConfig


class PickPlaceLearningEnv(PickPlaceEnv):
    environment_version = "pick_place_gc_v1"
    reward_version = "day13_potential_v1"
    timeout_mode = "external_truncation"
    observation_fields = OBSERVATION_FIELDS

    def __init__(self, config=None, render_mode=None):
        config = config or PickPlaceEnvConfig(
            action_scale_m=.005, max_episode_steps=750, max_simulation_time_s=30.)
        super().__init__(task_mode="pick_place", config=config, render_mode=render_mode)
        self.arm = GravityCompensatedJointController(
            self.model, self.config.max_joint_speed_rad_s)

    def reset(self, *, seed=None, options=None):
        if options and options.get("task_mode", "pick_place") != "pick_place":
            raise ValueError("Day15 only supports pick_place")
        return super().reset(seed=seed, options=options)

    def _info(self, state, terms, success, failure, reason, **extra):
        info = super()._info(state, terms, success, failure, reason, **extra)
        info.update(environment_version=self.environment_version,
                    reward_version=self.reward_version, timeout_mode=self.timeout_mode,
                    gravity_compensation=True,
                    compensation_clipped=self.arm.compensation_clipped)
        return info

