from pathlib import Path
import numpy as np
from agents.policies import pd_policy
from envs.single_joint_env import SingleJointEnv
from rollouts.collector import collect_episode

ROOT = Path(__file__).resolve().parents[1]
output_dir = ROOT / "outputs" / "day05"
output_dir.mkdir(parents=True,exist_ok=True,)

env = SingleJointEnv()

trajectory = collect_episode(env=env, policy_fn=pd_policy, seed=0, reset_options={"target_deg":60.0,},)

output_path = (output_dir/"pd_trajectory_seed0.npz")

np.savez_compressed(output_path, **trajectory, )

print( "saved:", output_path,)
print( "trajectory length:", len( trajectory["rewards"] ),)
print( "success:", trajectory["success"],)
print( "episode return:", trajectory["rewards"].sum(), )

env.close()