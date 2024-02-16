import gymnasium as gym
import diffrobot.panda_mujoco_gym
import pdb
from diffrobot.panda_mujoco_gym.scripts.pick_place_controller import get_pick_and_place_control
import time
import cv2


def get_control_action(obs):
    action = get_pick_and_place_control(obs, gain=2.0)[0]
    pos_ctrl = action[:3]
    pos_ctrl *= 0.1
    pos_ctrl += obs['ee_position']
    action[:3] = pos_ctrl
    return action

max_steps = 100

def run_env(env):
    

    while True:
        obs = env.reset()[0]    
        for i in range(max_steps):
            # run while loop at 10Hz
            action = get_control_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            out = env.render()
            # bgr to rgb
            out = out[:, :, ::-1]
            cv2.imshow("render", out)
            cv2.waitKey(1)
        
            if terminated:
                break

            time.sleep(0.1)

        print(terminated)
            
        print(i)


env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
run_env(env)
