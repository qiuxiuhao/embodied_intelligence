from gymnasium.utils.env_checker import check_env

from envs.single_joint_env import SingleJointEnv

env = SingleJointEnv()

check_env(env,skip_render_check = True)

print("Environment check passed.")

env.close()