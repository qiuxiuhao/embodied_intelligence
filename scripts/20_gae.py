import torch
from envs.single_joint_env import (SingleJointEnv)
from ppo.model import ActorCritic
from ppo.collector import (collect_rollout)
from ppo.gae import compute_gae

device = torch.device("cpu")

env = SingleJointEnv()

model = ActorCritic(
    obs_dim=env.observation_space.shape[0],
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device)

batch, stats = collect_rollout(env=env, model=model, num_steps=256,device=device, seed=0,)

advantages, returns = compute_gae(
    rewards=batch["rewards"],
    values=batch["values"],
    next_values=batch["next_values"],
    terminated=batch["terminated"],
    truncated=batch["truncated"],
    gamma=0.99,
    gae_lambda=0.95,
)

print( "advantages shape:", advantages.shape,)
print( "returns shape:", returns.shape, )
print( "\nfirst advantages:" )
print( advantages[:10])
print( "\nfirst returns:")
print( returns[:10])

env.close()