import torch
from envs.single_joint_env import (SingleJointEnv)
from ppo.model import ActorCritic
from ppo.collector import (collect_rollout,)

device = torch.device("cpu")

env = SingleJointEnv()

model =ActorCritic(
    obs_dim= env.observation_space.shape[0],
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device)

batch, stats = collect_rollout(env=env, model=model, num_steps=128, device=device, seed=0, )

for key, value in batch.items():
    print(key, value.shape,)

print()
print("first obs:",batch["observations"][0],)
print("first action:",batch["actions"][0], )
print("first reward:", batch["rewards"][0], )
print("first log_prob:", batch["old_log_probs"][0],)
print("first value:", batch["values"][0],)


env.close()
