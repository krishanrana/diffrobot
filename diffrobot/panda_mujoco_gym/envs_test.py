import gymnasium as gym
import diffrobot.panda_mujoco_gym
import pytest
import pdb

def run_env(env):
    obs = env.reset()[0]
    print(obs)
    while True:

        curr_pos = obs['achieved_goal']
        goal_pos = obs['desired_goal']

        # Write a P controller to move the robot to the goal

        #diff = curr_pos - goal_pos
        #action = diff * 10

        action = env.action_space.sample()
        print(action)
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()
    # check that it allows to be closed multiple times
    env.close()


env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
run_env(env)
