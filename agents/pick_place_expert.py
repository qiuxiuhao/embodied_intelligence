"""Feedback expert using only the policy's 94D observation and bounded 4D actions.

No MjData, actuator writes, physics steps or resets are accessible to this agent.
Stages are diagnostics/dataset labels, never additional policy observations.
"""

from dataclasses import dataclass

import numpy as np

from controllers.ik_controller import rotation_error_world
from envs.pick_place_env import OBSERVATION_SIZE, OBSERVATION_SLICES


PHASES = ("PRE_GRASP", "APPROACH", "GRASP", "LIFT", "TRANSPORT",
          "PLACE", "RELEASE", "RETREAT")
EXPERT_VERSION = "observation_waypoint_v1"


@dataclass(frozen=True)
class ExpertConfig:
    hover_m: float = .12
    grasp_above_center_m: float = .004
    place_preload_m: float = .0005
    transit_speed_m_s: float = .10
    approach_speed_m_s: float = .06
    place_speed_m_s: float = .04
    reach_tolerance_m: float = .004
    orientation_tolerance_rad: float = .035
    quiet_arm_rad_s: float = .10
    settle_s: float = .12
    support_hold_s: float = .16
    lost_grasp_grace_s: float = .16
    max_slip_m: float = .018
    phase_timeout_s: float = 10.

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Expert parameters must be finite and positive")


class ExpertFailure(RuntimeError):
    """An explicit collection failure, never an environment success."""


class PickPlaceExpert:
    def __init__(self, action_scale_m=.005, control_dt_s=.04, config=None):
        self.config = config or ExpertConfig()
        if not np.isfinite(action_scale_m) or action_scale_m <= 0:
            raise ValueError("action_scale_m must be positive")
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("control_dt_s must be positive")
        self.scale, self.dt = float(action_scale_m), float(control_dt_s)
        self.started = False

    @staticmethod
    def _read(observation):
        obs = np.asarray(observation)
        if obs.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(obs)):
            raise ValueError("Expert requires a finite raw 94D observation")
        return {key: obs[value].astype(float) for key, value in OBSERVATION_SLICES.items()}

    def reset(self, observation):
        s = self._read(observation)
        if not np.array_equal(s["task_one_hot"], [0, 0, 0, 1]):
            raise ValueError("Expert requires the pick_place task")
        self.phase = PHASES[0]
        self.goal = s["cube_position"] + [0, 0, self.config.hover_m]
        self.target = s["target_position"].copy()
        self.offset = np.zeros(3)
        self.steps = self.enter_step = 0
        self.stable_s = self.lost_s = 0.
        self.history = [{"phase": self.phase, "step": 0}]
        self.started = True

    def _stable(self, condition, duration):
        self.stable_s = self.stable_s + self.dt if condition else 0.
        return self.stable_s + 1e-10 >= duration

    def _enter(self, phase, s):
        c = self.config
        self.phase, self.enter_step = phase, self.steps
        self.stable_s = self.lost_s = 0.
        tcp, cube = s["tcp_position"], s["cube_position"]
        if phase == "APPROACH":
            self.goal = cube + [0, 0, c.grasp_above_center_m]
        elif phase == "LIFT":
            self.offset = cube - tcp
            self.goal = tcp + [0, 0, c.hover_m]
        elif phase in ("TRANSPORT", "PLACE"):
            self.offset = cube - tcp
            dz = c.hover_m if phase == "TRANSPORT" else -c.place_preload_m
            self.goal = self.target + [0, 0, dz] - self.offset
        elif phase == "RETREAT":
            self.goal = tcp + [0, 0, c.hover_m]
        self.history.append({"phase": phase, "step": self.steps})

    def act(self, observation):
        if not self.started:
            raise RuntimeError("Call expert.reset(observation) first")
        s, c = self._read(observation), self.config
        tcp, cube = s["tcp_position"], s["cube_position"]
        quiet = np.max(np.abs(s["arm_qvel"])) < c.quiet_arm_rad_s
        rotation = s["tcp_rotation_6d"].reshape(2, 3).T
        target_rotation = s["target_rotation_6d"].reshape(2, 3).T
        rotation = np.column_stack((rotation, np.cross(rotation[:, 0], rotation[:, 1])))
        target_rotation = np.column_stack((
            target_rotation, np.cross(target_rotation[:, 0], target_rotation[:, 1])))
        oriented = np.linalg.norm(rotation_error_world(target_rotation, rotation)) < c.orientation_tolerance_rad
        reached = (np.linalg.norm(tcp-self.goal) < c.reach_tolerance_m and quiet and oriented
                   and np.linalg.norm(s["tcp_command"]-self.goal) < .001)
        grasp = bool(s["finger_contacts"][2])
        table = bool(s["cube_table_contact"][0])
        still = np.linalg.norm(s["cube_velocity"][:3]) < .02 and np.linalg.norm(s["cube_velocity"][3:]) < .20
        open_released = s["gripper_opening"][0] >= .07 and not np.any(s["finger_contacts"][:2])
        if self.phase in ("LIFT", "TRANSPORT") or (self.phase == "PLACE" and not table):
            slipped = not grasp or np.linalg.norm(cube-tcp-self.offset) > c.max_slip_m
            self.lost_s = self.lost_s + self.dt if slipped else 0.
            if self.lost_s > c.lost_grasp_grace_s:
                raise ExpertFailure(f"OBJECT_SLIPPED:{self.phase}")
        if self.phase == "PRE_GRASP":
            if self._stable(reached and open_released and still, c.settle_s):
                self._enter("APPROACH", s)
        elif self.phase == "APPROACH":
            if self._stable(reached and open_released, c.settle_s):
                self._enter("GRASP", s)
        elif self.phase == "GRASP":
            if s["stable_timers"][1] >= 1.:
                self._enter("LIFT", s)
        elif self.phase == "LIFT":
            if reached and bool(s["has_lifted"][0]):
                self._enter("TRANSPORT", s)
        elif self.phase == "TRANSPORT":
            if self._stable(reached and grasp and still
                            and np.linalg.norm(cube[:2]-self.target[:2]) < .006, c.settle_s):
                self._enter("PLACE", s)
        elif self.phase == "PLACE":
            supported = (table and still and quiet and oriented
                         and np.linalg.norm(cube[:2]-self.target[:2]) < .008
                         and abs(cube[2]-self.target[2]) < .003)
            if self._stable(supported, c.support_hold_s):
                self._enter("RELEASE", s)
        elif self.phase == "RELEASE":
            if self._stable(open_released and table and still, c.settle_s):
                self._enter("RETREAT", s)
        # Final success always comes from env.step(), never from the expert.
        if (self.steps-self.enter_step)*self.dt > c.phase_timeout_s:
            raise ExpertFailure(f"PHASE_TIMEOUT:{self.phase}")
        grip = 1. if self.phase in ("PRE_GRASP", "APPROACH", "RELEASE", "RETREAT") else -1.
        speed = (c.place_speed_m_s if self.phase == "PLACE" else
                 c.approach_speed_m_s if self.phase in ("APPROACH", "LIFT", "RETREAT")
                 else c.transit_speed_m_s)
        delta = self.goal - s["tcp_command"]
        delta *= min(1., speed*self.dt / max(np.linalg.norm(delta), 1e-12))
        action = np.r_[np.clip(delta/self.scale, -1., 1.), grip].astype(np.float32)
        self.steps += 1
        return action
