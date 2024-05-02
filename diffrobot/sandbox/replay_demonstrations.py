from diffrobot.robot.visualizer import RobotViz
import os
import json
import pdb
import time
import spatialmath as sm
import numpy as np
from scipy.spatial.transform import Rotation as R

from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils


dataset_path = "/home/krishan/work/2024/datasets/cup_saucer"
dutils = DatasetUtils(dataset_path)
rlds = dutils.create_rlds()
env = RobotViz()

# read camera calibration json
camera_calibration_path = os.path.join(dataset_path, "calibration", "hand_eye.json")
with open(camera_calibration_path, "r") as f:
    data = json.load(f)
    X_EC_b = np.array(data["X_EC_b"])
    X_EC_f = np.array(data["X_EC_f"])
    

for episode in rlds:
    ep_data = rlds[episode]

    for phase in ep_data:
        phase_data = ep_data[phase]
        len_phase = len(phase_data['ee_poses'])

        for idx in range(len_phase):
            X_BE = np.array(phase_data['ee_poses'][idx])
            X_BO = np.array(phase_data['dynamic_object_poses'][idx])
            X_BOO = np.array(phase_data['oriented_dynamic_object_poses'][idx])

            X_BO_saucer = np.array(phase_data['static_object_pose'])
            X_BOO_saucer = np.array(phase_data['oriented_static_object_pose'])

            print('Progress: ', phase_data['progress'][idx]*100, '%')

            env.object_pose.T = sm.SE3(X_BO, check=False).norm()
            env.policy_pose.T = sm.SE3(X_BO_saucer, check=False).norm()
            env.cup_handle.T = sm.SE3(X_BOO_saucer, check=False).norm()
            env.step(phase_data['robot_q'][idx])
            time.sleep(0.1)
    
    
    # for idx, state in enumerate(data):

    #     X_BO = np.array(object_data[idx]["X_BO"])

    #     X_BE = np.array(state["X_BE"])
    #     # X_BF = np.dot(X_BE, np.linalg.inv(X_FE))
    #     X_BC_b = np.dot(X_BE, X_EC_b)
    #     X_BC_f = np.dot(X_BE, X_EC_f)

    #     phase = state["phase"]
    #     print("Phase: ", phase)

    #     # print angle between Z axis on X_BO and vector [0,0,1]
    #     angle = np.arccos(np.dot(X_BO[:3,2], np.array([0,0,1]) / (np.linalg.norm(X_BO[:3,2]) * np.linalg.norm(np.array([0,0,1])))))
    #     # print(np.rad2deg(angle))

    #     X_BO = dutils.adjust_orientation_to_z_up(X_BO)
    #     X_BOO = dutils.compute_oriented_affordance_frame(X_BO)

    #     X_BO_saucer = dutils.adjust_orientation_to_z_up(X_BO_saucer)
    #     X_BOO_saucer = dutils.compute_oriented_affordance_frame(X_BO_saucer, base_frame=X_BO)
        

    #     # env.ee_pose.T = sm.SE3(X_BE, check=False).norm()
    #     target_pose = env.robot.fkine(state["gello_q"], "panda_link8") * X_FE
    #     # env.policy_pose.T = target_pose
    #     env.policy_pose.T = sm.SE3(X_BO_saucer, check=False).norm()


    #     env.object_pose.T = sm.SE3(X_BO, check=False).norm()
    #     # cup_handle_pose = env.object_pose.T * sm.SE3(0.0, 0.083, 0.0)
    #     env.cup_handle.T = X_BOO_saucer
    #     # env.cup_handle.T = cup_handle_pose
    #     env.step(state["robot_q"])
    #     time.sleep(0.01)

    #     # pdb.set_trace()


    # # pdb.set_trace()





    
    
    
    

   