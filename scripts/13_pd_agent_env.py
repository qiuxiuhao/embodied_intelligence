import numpy as np
from envs.single_joint_env import SingleJointEnv

env = SingleJointEnv(render_mode="human",)

obs ,info = env.reset(seed=0,options={"target_deg":60.0,},)

# =============================
# PD Agent
# =============================
kp  = 8.0
kd = 1.0

total_reward = 0.0

while True:
    # ===========================
    # Observation
    # ===========================

    q = obs[0]

    qvel = obs[1]

    target = obs[2]

    # ===========================
    # policy
    # ===========================
    delta = target - q
    error = np.arctan2(np.sin(delta),np.cos(delta))
    action_value = (kp*error - kd*qvel)

    action = np.array([action_value],dtype=np.float32,)

    action = np.clip(action,env.action_space.low,env.action_space.high,)

    # ===========================
    # environmrnt
    # ===========================
    (obs,reward,terminated,truncated,info) = env.step(action)
    total_reward += reward

    print(
        f"q={info['q_deg']:+7.2f} "
        f"target={info['target_deg']:+7.2f} "
        f"error={info['error_deg']:+7.2f} "
        f"action={action[0]:+6.2f} "
        f"total_reward={total_reward:+7.4f}"
    )


    if terminated or truncated:

        break


print()

print(
    "success:",
    info["success"],
)

print(
    "total reward:",
    total_reward,
)


env.close()