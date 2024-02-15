import gymnasium as gym
import diffrobot.panda_mujoco_gym
import pdb
from diffrobot.panda_mujoco_gym.scripts.pick_place_controller import get_pick_and_place_control

def run_env(env):
    obs = env.reset()[0]
    print(obs)


    while True:

        # Write a P controller to move the robot to the goal

        #diff = curr_pos - goal_pos
        #action = diff * 10

        action = get_pick_and_place_control(obs)

        # action = env.action_space.sample()
        # print(action)
        action = action[0]
        # action = [0.0, 0.0, 0.0, 1.0]
        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated:
            env.reset()
    env.close()
    # check that it allows to be closed multiple times
    env.close()


env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
run_env(env)
