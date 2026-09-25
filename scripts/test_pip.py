import sys
import platform
import mujoco
import gymnasium as gym

# print(sys.version)
# print(platform.machine())
# print(mujoco.__version__)
# print(gymnasium.__version__)

env = gym.make("Reacher-v5")
obs , info = env.reset()
print(type(obs))
print(obs.shape)

print(env.observation_space)
print(env.action_space)
env.close()

