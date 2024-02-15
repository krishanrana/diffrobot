import gymnasium as gym
import diffrobot.panda_mujoco_gym
import pytest


def run_env(env):
    env.reset()
    while True:
        action = env.action_space.sample()
        print(action)
        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()
    # check that it allows to be closed multiple times
    env.close()


env = gym.make("FrankaPushSparse-v0", render_mode="human")
run_env(env)
