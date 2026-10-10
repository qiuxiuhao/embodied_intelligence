from pathlib import Path
import numpy as np
import torch
from envs.single_joint_env import (SingleJointEnv)
from ppo.model import ActorCritic

NUM_EPISODES = 100
device = torch.device( "cpu")
env = SingleJointEnv()

model = ActorCritic(
    obs_dim=env.observation_space.shape[0],
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device)

ROOT = Path(__file__).resolve().parents[1]
checkpoint = ( ROOT / "outputs" / "day06" / "ppo_best.pt" )
state_dict = torch.load( checkpoint, map_location=device,)
model.load_state_dict(state_dict)

model.eval()
returns = []
lengths = []
successes = []

for seed in range( NUM_EPISODES ):

    obs, info = env.reset( seed=1000 + seed )
    episode_return = 0.0
    episode_length = 0

    while True:

        obs_tensor = torch.as_tensor( obs, dtype=torch.float32, device=device, )
        
        with torch.no_grad():
            action_tensor, _, _ = (model.act(obs_tensor,deterministic=True,))

        action = (action_tensor.cpu().numpy())
        (obs, reward, terminated, truncated, info, ) = env.step(action)

        episode_return += ( reward )
        episode_length += 1

        if ( terminated or truncated ):
            break

    returns.append( episode_return )
    lengths.append( episode_length )
    successes.append(float( info["success"] ) )

env.close()

print(
    "========== PPO EVALUATION =========="
)


print(
    "episodes:",
    NUM_EPISODES
)


print(
    "success rate:",
    np.mean(successes)
)


print(
    "return mean:",
    np.mean(returns)
)


print(
    "return std:",
    np.std(returns)
)


print(
    "episode length mean:",
    np.mean(lengths)
)
 


