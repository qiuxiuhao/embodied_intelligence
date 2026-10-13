from envs.single_joint_env import SingleJointEnv

env = SingleJointEnv()

print("==============SPACE============")

print("observation space:",env.observation_space)

print("action space:",env.action_space)

print("\n=============================")

obs,info = env.reset(seed=42,options={"target_deg":60.0,},)

print("observation:")
print(obs)

print("info:")
print(info)


# obs1, _ = env.reset(seed=123)

# obs2, _ = env.reset(seed=123)


# print("obs1:", obs1)

# print("obs2:", obs2)

# print(
#     "same:",
#     (obs1 == obs2).all(),
# )


print("\n========== STEP ==========")

action = [1.0]


for i in range(5):

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)


    print(
        f"step={i} "
        f"obs={obs} "
        f"reward={reward:.4f} "
        f"terminated={terminated} "
        f"truncated={truncated}"
    )


env.close()