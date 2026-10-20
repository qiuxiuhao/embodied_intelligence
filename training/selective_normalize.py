"""仅标准化SI物理量；有界姿态、动作、标志与计时比例保留原值。"""

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from envs.pick_place_env import OBSERVATION_FIELDS


class PhysicalVecNormalize(VecNormalize):
    """保留SB3的统计/终态观测/保存加载机制，只选择性应用观测变换。

    底层记录完整观测统计，便于诊断与使用sync_envs_normalization；
    有界特征不使用这些统计缩放，避免稀有计时输入被放大。
    """

    def __init__(self, venv, observation_fields=None, **kwargs):
        """根据观测字段单位构造可序列化掩码，不依赖某次训练的方差大小。"""
        super().__init__(venv, **kwargs)
        fields = observation_fields if observation_fields is not None else OBSERVATION_FIELDS
        self.observation_fields = tuple(tuple(field) for field in fields)
        size = sum(size for _, size, _ in fields)
        if self.observation_space.shape != (size,):
            raise ValueError("观测字段与环境空间不匹配")
        self.physical_mask = np.zeros(size, dtype=bool)
        self.normalized_fields = []
        offset = 0
        for name, width, unit in fields:
            if unit in ("m", "rad", "rad/s", "m/s,rad/s", "m/s"):
                self.physical_mask[offset:offset+width] = True
                self.normalized_fields.append(name)
            offset += width

    def normalize_obs(self, obs):
        """所有物理量沿用SB3缩放；其余维度直接恢复为本次原始观测，不写入源数组。"""
        normalized = super().normalize_obs(obs)
        normalized[..., ~self.physical_mask] = obs[..., ~self.physical_mask]
        return normalized

    def unnormalize_obs(self, obs):
        """逆变换采用相同掩码；被clip的物理量本身无法精确逆变换。"""
        original = super().unnormalize_obs(obs)
        original[..., ~self.physical_mask] = obs[..., ~self.physical_mask]
        return original
