import torch

def compute_gae(rewards, values, next_values, terminated,
                truncated, gamma=0.99, gae_lambda=0.95, ):
    
    advantages = torch.zeros_like(rewards)

    gae = torch.tensor(0.0, device=rewards.device, )

    for t in reversed( range(rewards.shape[0]) ):

        # ==========================================
        # terminated:
        # 真正 terminal
        # 不 bootstrap
        # ==========================================
        boolstrap_mask = (1.0 - terminated[t].float())

        delta = ( rewards[t] + gamma*boolstrap_mask*next_values[t]-values[t])

        # ==========================================
        # 不允许 GAE 穿过 episode boundary
        # ==========================================
        done = (terminated[t] | truncated[t] )

        continuation_mask = (1.0 - done.float() )
        gae = (delta + gamma*gae_lambda*continuation_mask*gae )

        advantages[t] = gae

    returns = (advantages+values)
    return (advantages, returns, )

