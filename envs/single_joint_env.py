from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

import numpy as np

class SingleJointEnv(gym.Env):
    """
    一个基于Mujoco的单关节目标角度控制环境
    Observation:
        [joint_angle, joint_velocity,target_angle]
    Action:
        [motor_torque]
    Task:
        控制单关节机械臂角度到达目标角度，并且稳定下来。
    """

    metadata = {"render_modes":["human"],}

    def __init__(self, render_mode=None,frame_skip=10,max_episode_steps=250):
        super().__init__()

        #=========================
        # 1. 路径
        #=========================

        root = Path(__file__).resolve().parents[1]
        xml_path = root / "assets" / "actuated_joint.xml"

        #=========================
        # 2. Mujoco模型和数据
        #=========================
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        #=========================
        # 3. 环境参数
        #=========================

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self.viewer = None

        # physics timestep
        self.physics_timestep = self.model.opt.timestep

        # 一个Agent action对应多少真实仿真时间
        self.dt = (self.physics_timestep * self.frame_skip)

        #=========================
        # 4. 找到关节和电机
        #=========================
        self.joint_name = "hinge_joint"
        self.actuator_name = "joint_motor"

        self.joint_id = self.model.joint(self.joint_name).id
        self.actuator_id = self.model.actuator(self.actuator_name).id

        #=========================
        # 5. joint range
        #=========================
        self.joint_range = self.model.jnt_range[self.joint_id].copy()  # [min, max]

        #=========================
        # 6. action space
        #=========================
        ctrl_range = self.model.actuator_ctrlrange[self.actuator_id].copy()  # [min, max]
        self.action_space = spaces.Box(
            low = np.array([ctrl_range[0]],dtype=np.float32),
            high = np.array([ctrl_range[1]],dtype=np.float32),
            dtype=np.float32,
        )

        #=========================
        # 7. observation space
        #=========================

        #observation = [joint_angle, joint_velocity, target_angle]
        self.observation_space = spaces.Box(
            low = np.array([
                self.joint_range[0],  # joint_angle min 
                -np.inf,              # joint_velocity min
                self.joint_range[0],  # target_angle min
            ],dtype=np.float32),
            high = np.array([
                self.joint_range[1],  # joint_angle max
                np.inf,               # joint_velocity max
                self.joint_range[1],  # target_angle max
            ],dtype=np.float32),
            dtype=np.float32,
        )

        #=========================
        # 8. target angle range
        #=========================
        self.target_angle = 0.0

    def _get_obs(self):
        q = self.data.joint(self.joint_name).qpos[0]
        qvel = self.data.joint(self.joint_name).qvel[0]
        observation = np.array([q, qvel, self.target_angle],dtype=np.float32)
        return observation

    def _get_error(self):
        q = self.data.joint(self.joint_name).qpos[0]
        error = self.target_angle - q
        error = np.arctan2(np.sin(error), np.cos(error))  # wrap to [-pi, pi]
        return float(error)

    def _get_info(self):
        q = self.data.joint(self.joint_name).qpos[0]
        qvel = self.data.joint(self.joint_name).qvel[0]
        error = self._get_error()

        success = (
            abs(error) < np.deg2rad(5.0) and
            abs(qvel) < np.deg2rad(5.0)
        )

        return {
            "q_deg":float(np.rad2deg(q)),
            "target_deg":float(np.rad2deg(self.target_angle)),
            "error_deg":float(np.rad2deg(error)),
            "qvel":float(qvel),
            "success":bool(success),
            "control_dt":float(self.dt),
        }

    def reset(self,* ,seed=None, options=None):
        super().reset(seed=seed)

        #=========================
        # 1. 重置Mujoco
        #=========================
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        #=========================
        # 2. 随机目标角度
        #========================
        initial_angle = self.np_random.uniform(low=np.deg2rad(-15.0), high=np.deg2rad(15.0))
        self.data.joint(self.joint_name).qpos[0] = initial_angle
        self.data.joint(self.joint_name).qvel[0] = 0.0

        #=========================
        # 3. Target
        #=========================
        if (options is not None) and ("target_angle" in options):
            self.target_angle = float(options["target_angle"])
        else:
            self.target_angle = self.np_random.uniform(low=np.deg2rad(-120.0), high=np.deg2rad(120.0))

        #=========================
        # 4. Forward
        #=========================
        mujoco.mj_forward(self.model, self.data)

        #=========================
        # 5. Observation
        #=========================
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        #=========================
        # 1. Action 格式
        #=========================
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)

        #=========================
        # 2. Control Saturation
        #=========================
        action = np.clip(action, self.action_space.low, self.action_space.high)

        control =float(action[0])

        #=========================
        # 3. action -> mujoco control
        #=========================
        self.data.ctrl[self.actuator_id] = control

        #=========================
        # 4. Physics
        #=========================
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        #=========================
        # 5. Observation
        #=========================
        observation = self._get_obs()

        qvel = self.data.joint(self.joint_name).qvel[0]
        error = self._get_error()

        #=========================
        # 6. Reward
        #=========================
        reward = (-(error ** 2) - 0.01 * (qvel ** 2) - 0.001 * (control ** 2))

        #=========================
        # 7.Termination
        #=========================
        success = (
            abs(error) < np.deg2rad(5.0) and
            abs(qvel) < np.deg2rad(5.0)
        )

        terminated = bool(success)

        #=========================
        # 8. Truncation
        #=========================
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)

        #=========================
        # 9. Info
        #=========================
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return (observation, float(reward), terminated, truncated, info)

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None