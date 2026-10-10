import numpy as np
import torch

def collect_rollout(env, model, num_steps, device, seed,):
    
    obs, info = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    old_log_probs = []
    values = []
    next_values = []
    episode_returns = []
    episode_lengths = []
    episode_successes = []
    current_episode_return = 0.0
    current_episode_length = 0

    for _ in range(num_steps):
        # ==========================================
        # Current observation
        # ==========================================
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

        # ==========================================
        # Policy
        # ==========================================
        with torch.no_grad():
            (action_tensor, log_prob_tensor, value_tensor,) = model.act(obs_tensor, deterministic=False)
        
        action = (action_tensor.cpu().numpy())

        # ==========================================
        # Environment
        # ==========================================
        (next_obs, reward, terminated, truncated, info, ) = env.step(action)

        # ==========================================
        # V(s_{t+1})
        # ==========================================
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device,)

        with torch.no_grad():
            next_value_tensor = (model.value(next_obs_tensor))

        # ==========================================
        # Save
        # ==========================================

        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))
        old_log_probs.append(float(log_prob_tensor.cpu()))
        values.append(float(value_tensor.cpu()))
        next_values.append(float(next_value_tensor.cpu()))

        # ==========================================
        # Episode statistics
        # ==========================================
        current_episode_return += (reward)
        current_episode_length += 1
        done = (terminated or truncated)

        if done:
            episode_returns.append(current_episode_return)
            episode_lengths.append(current_episode_length)
            episode_successes.append(float(info["success"]))

            obs,info = env.reset()
            current_episode_return = 0.0
            current_episode_length = 0

        else:
            obs = next_obs

    batch = {

        "observations": torch.as_tensor( np.asarray( observations, dtype=np.float32, ), device=device, ),
        "actions": torch.as_tensor( np.asarray( actions, dtype=np.float32, ), device=device, ),
        "rewards": torch.as_tensor( rewards, dtype=torch.float32, device=device, ),
        "terminated": torch.as_tensor( terminated_flags, dtype=torch.bool, device=device, ),
        "truncated": torch.as_tensor( truncated_flags, dtype=torch.bool, device=device, ),
        "old_log_probs": torch.as_tensor( old_log_probs, dtype=torch.float32, device=device, ),
        "values": torch.as_tensor( values, dtype=torch.float32, device=device, ),
        "next_values": torch.as_tensor( next_values, dtype=torch.float32, device=device, ),
    }

    stats = {
        "episode_returns":episode_returns,
        "episode_lengths":episode_lengths,
        "episode_successes":episode_successes
    }

    return batch, stats
        



