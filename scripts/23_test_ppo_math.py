import numpy as np
import torch

from ppo.gae import compute_gae


def test_gae():

    rewards = torch.tensor(
        [1.0, 2.0]
    )

    values = torch.tensor(
        [0.0, 0.0]
    )

    next_values = torch.tensor(
        [0.0, 0.0]
    )

    terminated = torch.tensor(
        [False, True]
    )

    truncated = torch.tensor(
        [False, False]
    )

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.9,
        gae_lambda=1.0,
    )

    expected = torch.tensor(
        [2.8, 2.0]
    )

    assert torch.allclose(
        advantages,
        expected,
        atol=1e-6,
    )

    assert torch.allclose(
        returns,
        expected,
        atol=1e-6,
    )

    print("GAE test passed")

def test_truncation():

    rewards = torch.tensor(
        [1.0, 2.0]
    )

    values = torch.tensor(
        [0.0, 0.0]
    )

    next_values = torch.tensor(
        [5.0, 0.0]
    )

    terminated = torch.tensor(
        [False, True]
    )

    truncated = torch.tensor(
        [True, False]
    )

    advantages, _ = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.9,
        gae_lambda=1.0,
    )

    expected = torch.tensor(
        [5.5, 2.0]
    )

    assert torch.allclose(
        advantages,
        expected,
        atol=1e-6,
    )

    print("Truncation test passed")

from ppo.model import ActorCritic


def test_policy_logprob():

    torch.manual_seed(42)

    model = ActorCritic(
        obs_dim=3,
        action_low=np.array(
            [-3.0],
            dtype=np.float32,
        ),
        action_high=np.array(
            [3.0],
            dtype=np.float32,
        ),
    )

    obs = torch.tensor(
        [0.1, 0.0, 1.0],
        dtype=torch.float32,
    )

    action, old_log_prob, value = model.act(
        obs
    )

    new_log_prob, entropy, new_value = (
        model.evaluate_actions(
            obs,
            action,
        )
    )

    ratio = torch.exp(
        new_log_prob - old_log_prob
    )

    print("action:", action)
    print("old log_prob:", old_log_prob)
    print("new log_prob:", new_log_prob)
    print("ratio:", ratio)

    assert torch.isfinite(ratio)

    assert torch.allclose(
        ratio,
        torch.ones_like(ratio),
        atol=1e-3,
    )

    print("Policy log_prob test passed")

if __name__ == "__main__":

    test_gae()

    test_truncation()

    test_policy_logprob()

    print("\nAll PPO math tests passed.")
    