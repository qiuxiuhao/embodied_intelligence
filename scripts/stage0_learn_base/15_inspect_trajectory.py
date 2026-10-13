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


print("========== KEYS ==========")

print(data.files)


print("\n========== SHAPES ==========")

for key in data.files:

    print(
        key,
        data[key].shape,
        data[key].dtype,
    )


print("\n========== FIRST TRANSITION ==========")


print(
    "obs:"
)

print(
    data["observations"][0]
)


print(
    "action:"
)

print(
    data["actions"][0]
)


print(
    "reward:"
)

print(
    data["rewards"][0]
)


print(
    "next_obs:"
)

print(
    data["next_observations"][0]
)


print(
    "terminated:"
)

print(
    data["terminated"][0]
)


print(
    "truncated:"
)

print(
    data["truncated"][0]
)


print("\n========== LAST TRANSITION ==========")


last = len(
    data["rewards"]
) - 1


print(
    "obs:",
    data["observations"][last],
)

print(
    "action:",
    data["actions"][last],
)

print(
    "reward:",
    data["rewards"][last],
)

print(
    "next_obs:",
    data["next_observations"][last],
)

print(
    "terminated:",
    data["terminated"][last],
)

print(
    "truncated:",
    data["truncated"][last],
)


for t in range(
    min(5, len(data["rewards"]))
):

    print(
        f"\nTransition {t}"
    )

    print(
        "obs      =",
        data["observations"][t],
    )

    print(
        "action   =",
        data["actions"][t],
    )

    print(
        "reward   =",
        data["rewards"][t],
    )

    print(
        "next_obs =",
        data["next_observations"][t],
    )