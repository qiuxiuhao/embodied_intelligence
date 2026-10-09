from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


path = (
    ROOT
    / "outputs"
    / "day05"
    / "pd_trajectory_seed0.npz"
)


data = np.load(path)


rewards = data["rewards"]

def discounted_returns(rewards,gamma):
    returns = np.zeros_like(rewards, dtype=np.float64)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = (rewards[t] + gamma*running_return)
    return returns

for gamma in [0.0,0.9,0.99,1.0]:
    returns = discounted_returns(rewards, gamma)
    print(
        f"\ngamma = {gamma}"
    )

    print(
        "G_0 =",
        returns[0],
    )

    print(
        "first 5 returns ="
    )

    print(
        returns[:5]
    )