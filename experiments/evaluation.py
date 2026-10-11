from envs.single_joint_env import SingleJointEnv
import numpy as np

def evaluate_policy( policy_fn, seeds, frame_skip=10,):

    env = SingleJointEnv( frame_skip=frame_skip )
    records = []

    try:

        for seed in seeds:
            # 环境随机初始条件
            obs, info = env.reset( seed=int(seed) )
            
            # 控制 Random Policy 的采样随机性
            env.action_space.seed( int(seed) + 50000 )

            episode_return = 0.0
            episode_length = 0

            while True:

                # ==========================
                # Policy
                # ==========================
                action = policy_fn( obs, env, )

                # ==========================
                # Environment
                # ==========================
                ( obs, reward, terminated,truncated, info,) = env.step(action)

                episode_return += reward
                episode_length += 1

                if terminated or truncated:
                    break
            records.append({
                "seed": int(seed),
                "success": int( info["success"] ),
                "episode_return": float( episode_return),
                "episode_length": int( episode_length ),
                "final_error_deg": float( abs(info["error_deg"])),
            })

    finally:

        env.close()

    return records
    
def summarize_records(records):

    return {

        "success_rate": float(
            np.mean([
                r["success"]
                for r in records
            ])
        ),

        "return_mean": float(
            np.mean([
                r["episode_return"]
                for r in records
            ])
        ),

        "return_std": float(
            np.std([
                r["episode_return"]
                for r in records
            ])
        ),

        "length_mean": float(
            np.mean([
                r["episode_length"]
                for r in records
            ])
        ),

        "final_error_mean_deg": float(
            np.mean([
                r["final_error_deg"]
                for r in records
            ])
        ),

    }

