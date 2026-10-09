import numpy as np

def random_policy(obs,action_space):
    """
    Random Pilicy: 不使用 observation，直接从 action space 随机采样
    """
    del obs
    return action_space.sample()

def pd_policy(obs, action_space, kp=8.0, kd=1.0):
    """
    PD Policy: 
        Observation: [q, qvel, target]
        Action: [motor_control]
    """
    q = float(obs[0])
    qvel = float(obs[1])
    target = float(obs[2])

    delta = target - q
    error = np.arctan2(np.sin(delta), np.cos(delta),)
    action_value = (kp*error - kd*qvel)
    action = np.array([action_value], dtype=np.float32,)
    action = np.clip(action,action_space.low,action_space.high,)

    return action