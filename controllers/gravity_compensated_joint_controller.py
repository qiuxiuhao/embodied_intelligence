"""Gravity feedforward through Panda's existing, force-limited position actuators."""

import mujoco
import numpy as np

from controllers.joint_controller import PandaJointPositionController
from robotics.kinematics import get_arm_dof_indices


class GravityCompensatedJointController(PandaJointPositionController):
    """Keep nominal command_q separate from the gravity-offset actuator input.

    For the verified unit-gear affine actuator, force = kp*(ctrl-q)-kd*qvel.
    ctrl = command_q + gravity/kp therefore adds gravity feedforward while
    retaining MuJoCo's original control and actuator force limits.
    Gravity is computed on separate MjData at the current pose and zero velocity;
    no real qpos/qvel, applied forces, model gravity or body gravcomp are changed.
    """

    def __init__(self, model, max_joint_speed=0.5):
        super().__init__(model, max_joint_speed)
        ids = self.actuator_ids
        self.kp = model.actuator_gainprm[ids, 0].copy()
        if (np.any(self.kp <= 0)
                or np.any(model.actuator_trntype[ids] != mujoco.mjtTrn.mjTRN_JOINT)
                or np.any(model.actuator_gaintype[ids] != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(model.actuator_biastype[ids] != mujoco.mjtBias.mjBIAS_AFFINE)
                or not np.allclose(model.actuator_gear[ids], [1, 0, 0, 0, 0, 0])
                or not np.allclose(model.actuator_biasprm[ids, 0], 0)
                or not np.allclose(model.actuator_biasprm[ids, 1], -self.kp)):
            raise ValueError("Gravity feedforward requires unit-gear affine Panda position actuators")
        self.dofs = get_arm_dof_indices(model)
        self.gravity_data = mujoco.MjData(model)
        self.gravity_torque = np.zeros(7)
        self.applied_ctrl = np.zeros(7)
        self.compensation_clipped = False

    def _apply_gravity(self, data):
        self.gravity_data.qpos[:] = data.qpos
        self.gravity_data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.gravity_data)
        self.gravity_torque = self.gravity_data.qfrc_bias[self.dofs].copy()
        requested = self.command_q + self.gravity_torque/self.kp
        self.applied_ctrl = np.clip(requested, self.ctrl_low, self.ctrl_high)
        self.compensation_clipped = bool(np.any(self.applied_ctrl != requested))
        data.ctrl[self.actuator_ids] = self.applied_ctrl

    def reset(self, data):
        nominal = super().reset(data)
        self._apply_gravity(data)
        return nominal

    def step(self, data, target_q, dt):
        nominal = super().step(data, target_q, dt)
        self._apply_gravity(data)
        return nominal
