from diffrobot.robot.visualizer import RobotViz
import os
import json
import pdb
import time
import spatialmath as sm
import numpy as np

dataset_path = "/home/krishan/work/2024/datasets/cup_rotate"
episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
env = RobotViz()

# read camera calibration json
camera_calibration_path = os.path.join(dataset_path, "calibration", "hand_eye.json")
with open(camera_calibration_path, "r") as f:
    data = json.load(f)
    X_FC = np.array(data["X_FC"])


X_FE = np.array([[0.70710678, 0.70710678, 0.0, 0.0], 
                [-0.70710678, 0.70710678, 0, 0], 
                [0.0, 0.0, 1.0, 0.2], 
                [0.0, 0.0, 0.0, 1.0]])

X_FE = sm.SE3(X_FE, check=False).norm()

for episode in episodes:
    episode_path = os.path.join(dataset_path, "episodes", episode, "state.json")
    with open(episode_path, "r") as f:
        data = json.load(f)
    
    object_frame_path = os.path.join(dataset_path, "episodes", episode, "object_frames.json")
    with open(object_frame_path, "r") as f:
        object_data = json.load(f)
    
    for idx, state in enumerate(data):
        X_BE = np.array(state["X_BE"])
        X_BF = np.dot(X_BE, np.linalg.inv(X_FE))
        X_BC = np.dot(X_BF, X_FC)


        if object_data[idx]["X_BO"] is not None:
            if idx == 0:
                X_BO = np.array(object_data[idx]["X_BO"])
            else:
                temp = np.array(object_data[idx]["X_BO"])
                dist = np.dot(np.array([0,0,1]), temp[:3,2])
                if dist > 0.98:
                    X_BO = temp


        env.ee_pose.T = sm.SE3(X_BE, check=False).norm()
        target_pose = env.robot.fkine(state["gello_q"], "panda_link8") * X_FE
        # env.policy_pose.T = target_pose 
        env.object_pose.T = sm.SE3(X_BO, check=False).norm()
        env.step(state["robot_q"])
        time.sleep(0.1)

    pdb.set_trace()

    
    
    
    

   