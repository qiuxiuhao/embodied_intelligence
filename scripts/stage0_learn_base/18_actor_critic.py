import torch
from envs.single_joint_env import (SingleJointEnv)

from ppo.model import ActorCritic

env = SingleJointEnv()

obs, info = env.reset( seed=0, options={"target_deg":60.0 ,}, )
obs_tensor = torch.as_tensor(obs, dtype=torch.float32,)

model = ActorCritic(
    obs_dim= env.observation_space.shape[0],
    action_low= env.action_space.low,
    action_high= env.action_space.high,
    )

print("observation:", obs_tensor,)

for i in range(5):
    
    action, log_prob, value = (model.act(obs_tensor))
    
    print(f"\nsample {i}")
    print("action:", action,)
    print("log_prob:", log_prob,)
    print("value:", value,)

env.close()