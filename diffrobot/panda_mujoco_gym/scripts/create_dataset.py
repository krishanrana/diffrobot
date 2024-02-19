import gymnasium as gym
import diffrobot.panda_mujoco_gym
import pdb
from diffrobot.panda_mujoco_gym.scripts.pick_place_controller import get_control_action
import time
import cv2
import argparse
import os
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--num_demos", type=int)
args = parser.parse_args()



class DemoCollector:
    def __init__(self, dataset_name, num_demos):
        self.dataset_name = dataset_name
        self.num_demos = num_demos
        self.max_steps = 100
        self.env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="rgb_array")
        path_to_datasets = "/home/krishan/work/2024/datasets/franka_pusht_sim"
        self.dataset_path = os.path.join(path_to_datasets, self.dataset_name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, "episodes"), exist_ok=True)
        self.demos_done = 0

    def collect_demos(self):

        while self.demos_done < self.num_demos:
            ep_path = os.path.join(self.dataset_path, "episodes", f"{self.demos_done}")
            os.makedirs(ep_path, exist_ok=True)
            im_path = os.path.join(ep_path, "images")
            os.makedirs(im_path, exist_ok=True)
            states = []
            actions = []
            ims = []

            obs = self.env.reset()[0]
            im = self.env.render()
            im = im[:, :, ::-1]
            
            for j in range(self.max_steps):
                # run while loop at 10Hz
                action = get_control_action(obs, gain=2.0)

                states.append(obs)
                actions.append(action)
                ims.append(im)

                # print(action)
                obs, _, done, truncated, _ = self.env.step(action)
                im = self.env.render()
                # bgr to rgb
                im = im[:, :, ::-1]
                cv2.imshow("render", im)
                cv2.waitKey(1)
            
                if done:
                    break
                # time.sleep(0.1)
            
            if done and len(states) > 16:
                # save states and actions
                states = np.array(states)
                actions = np.array(actions)
                np.save(os.path.join(ep_path, "states.npy"), states)
                np.save(os.path.join(ep_path, "actions.npy"), actions)

                # save images
                for i, im in enumerate(ims):
                    cv2.imwrite(os.path.join(im_path, f"{i}.png"), im)

                self.demos_done += 1
                # ensure print overwrites
                print(f"Demo {self.demos_done}/{self.num_demos}", end="\r")
        





if __name__ == "__main__":
    collector = DemoCollector(args.dataset_name, args.num_demos)
    collector.collect_demos()
  
    print("Done!")


