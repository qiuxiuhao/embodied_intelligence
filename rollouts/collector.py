import numpy as np

def collect_episode(env, policy_fn, seed=None, reset_options=None, ):
    # =======================
    # Reset
    # =======================

    obs, info = env.reset(seed=seed, options = reset_options)

    # =======================
    # Storage
    # =======================
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated_flags = []
    truncated_flags = []

    # =======================
    # Rollout
    # =======================
    while True:
        # -------------------------
        # Policy
        # -------------------------
        action = policy_fn(obs, env.action_space)

        #  保存step前 observation
        current_obs = obs.copy()

        # -------------------------
        # environment
        # -------------------------
        (next_obs, reward, terminated, truncated, info) = env.step(action)

        # -------------------------
        # Save transition
        # -------------------------
        observations.append(current_obs)
        actions.append(np.asarray(action, dtype=np.float32,).copy())
        rewards.append(float(reward))
        next_observations.append(next_obs.copy())
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        # -------------------------
        # Update state
        # -------------------------
        obs = next_obs

        if terminated or truncated:
            break

    # ==============================================
    # Convert to numpy
    # ==============================================
    trajectory = {
        "observations":np.asarray(observations, dtype=np.float32,),
        "actions": np.asarray(actions, dtype=np.float32,),
        "rewards": np.asarray(rewards, dtype=np.float32,),
        "next_observations": np.asarray(next_observations, dtype=np.float32,),
        "terminated": np.asarray(terminated_flags, dtype=np.bool_,),
        "truncated": np.asarray(truncated_flags,dtype=np.bool_,),
        "success": bool(info["success"]),
    }
    return trajectory