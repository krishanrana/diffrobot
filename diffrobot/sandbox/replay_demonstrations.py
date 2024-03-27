from diffrobot.robot.visualizer import RobotViz
import os
import json
import pdb
import time
import spatialmath as sm
import numpy as np

dataset_path = "/home/krishan/work/2024/datasets/door_open_v2.0"
episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
env = RobotViz()

X_FE = np.array([[0.70710678, 0.70710678, 0.0, 0.0], 
                [-0.70710678, 0.70710678, 0, 0], 
                [0.0, 0.0, 1.0, 0.1], 
                [0.0, 0.0, 0.0, 1.0]])

X_FE = sm.SE3(X_FE, check=False).norm()

for episode in episodes:
    episode_path = os.path.join(dataset_path, "episodes", episode, "state.json")
    with open(episode_path, "r") as f:
        data = json.load(f)
    
    for state in data:
        env.ee_pose.T = sm.SE3(state["X_BE"], check=False).norm()
        target_pose = env.robot.fkine(state["gello_q"], "panda_link8") * X_FE
        env.policy_pose.T = target_pose 
        env.step(state["robot_q"])
        time.sleep(0.1)

    
    pdb.set_trace()
    
    
    

   