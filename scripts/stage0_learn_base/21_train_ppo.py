from pathlib import Path
import numpy as np
import torch
from envs.single_joint_env import (SingleJointEnv,)
from ppo.model import ActorCritic
from ppo.collector import (collect_rollout,)
from ppo.gae import compute_gae
from ppo.update import ppo_update

# ==============================================
# Config
# ==============================================
SEED = 42
NUM_UPDATES = 100
ROLLOUT_STEPS = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_COEF = 0.2
VALUE_COEF = 0.5
UPDATE_EPOCHS = 10
MINIBATCH_SIZE = 256

# ==============================================
# Seed
# ==============================================
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

# ==============================================
# Environment
# ==============================================
env =SingleJointEnv()

# ==============================================
# Model
# ==============================================
model = ActorCritic(
    obs_dim=env.observation_space.shape[0],
    action_low=env.action_space.low,
    action_high=env.action_space.high,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# ==============================================
# Output
# ==============================================
ROOT = Path(__file__).resolve().parents[1]
output_dir = (ROOT / "outputs" / "day06" )
output_dir.mkdir( parents=True, exist_ok=True,)

best_success_rate = -1.0

# ==============================================
# Training
# ==============================================
for update in range(1,NUM_UPDATES+1,):
    # ------------------------------------------
    # 1. Rollout
    # ------------------------------------------
    batch, stats = collect_rollout(env = env, model=model, num_steps=ROLLOUT_STEPS, device=device, seed= SEED+update)

    # ------------------------------------------
    # 2. GAE
    # ------------------------------------------
    advantages, returns = compute_gae( rewards=batch["rewards"], values=batch["values"], next_values=batch[ "next_values"],
        terminated=batch["terminated"], truncated=batch["truncated"], gamma=GAMMA, gae_lambda=GAE_LAMBDA, )

    # ------------------------------------------
    # 3. PPO Update
    # ------------------------------------------
    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        advantages=advantages,
        returns=returns,
        clip_coef=CLIP_COEF,
        value_coef=VALUE_COEF,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
    )

    # ------------------------------------------
    # 4. Rollout stats
    # ------------------------------------------
    episode_returns = ( stats["episode_returns"])
    episode_successes = ( stats["episode_successes"] )

    if len(episode_returns)>0:
        mean_return = float( np.mean( episode_returns ) )
        success_rate = float( np.mean( episode_successes ) )
    else:
        mean_return = float( "nan")
        success_rate = float( "nan")
    print(
        f"update={update:03d}  "
        f"return={mean_return:+8.3f}  "
        f"success={success_rate:5.2f}  "
        f"pi_loss={metrics['policy_loss']:+.4f}  "
        f"v_loss={metrics['value_loss']:.4f}  "
        f"kl={metrics['approx_kl']:.5f}  "
        f"clip={metrics['clip_fraction']:.3f}"
    )

    # ------------------------------------------
    # 5. Save best observed rollout policy
    # ------------------------------------------
    if (np.isfinite(success_rate) and success_rate > best_success_rate):
        best_success_rate = (success_rate)
        torch.save(model.state_dict(), output_dir/"ppo_best.pt")
    
# ==============================================
# Save final
# ==============================================

torch.save( model.state_dict(), output_dir / "ppo_final.pt",)
env.close()

