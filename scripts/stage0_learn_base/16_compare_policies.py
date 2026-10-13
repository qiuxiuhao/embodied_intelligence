import numpy as np

from agents.policies import (
    random_policy,
    pd_policy,
)

from envs.single_joint_env import (
    SingleJointEnv,
)

from rollouts.collector import (
    collect_episode,
)


NUM_EPISODES = 50


def evaluate_policy(
    name,
    policy_fn,
):

    env = SingleJointEnv()


    returns = []

    lengths = []

    successes = []


    for seed in range(
        NUM_EPISODES
    ):

        trajectory = collect_episode(

            env=env,

            policy_fn=policy_fn,

            seed=seed,

            reset_options=None,
        )


        episode_return = float(
            trajectory[
                "rewards"
            ].sum()
        )


        episode_length = len(
            trajectory[
                "rewards"
            ]
        )


        success = bool(
            trajectory[
                "success"
            ]
        )


        returns.append(
            episode_return
        )

        lengths.append(
            episode_length
        )

        successes.append(
            success
        )


    env.close()


    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    lengths = np.asarray(
        lengths,
        dtype=np.float64,
    )

    successes = np.asarray(
        successes,
        dtype=np.float64,
    )


    print(
        f"\n========== {name} =========="
    )

    print(
        "episodes:",
        NUM_EPISODES,
    )

    print(
        "success rate:",
        successes.mean(),
    )

    print(
        "return mean:",
        returns.mean(),
    )

    print(
        "return std:",
        returns.std(),
    )

    print(
        "episode length mean:",
        lengths.mean(),
    )


evaluate_policy(
    "Random",
    random_policy,
)


evaluate_policy(
    "PD",
    pd_policy,
)