from envs.single_joint_env import SingleJointEnv

env = SingleJointEnv()

obs, info = env.reset(seed=0)

total_reward = 0.0

for step in range(250):
    # ============================
    # policy
    # ============================

    action = env.action_space.sample()

    #=============================
    # environment
    #=============================

    (obs,reward,terminated,truncated,info,) = env.step(action)

    total_reward += reward

    if step % 25 == 0:
        print(
            f"step={step:3d} "
            f"q={info['q_deg']:+7.2f} "
            f"target={info['target_deg']:+7.2f} "
            f"error={info['error_deg']:+7.2f} "
            f"reward={reward:+.4f}"
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