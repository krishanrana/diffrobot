from diffrobot.robot.visualizer import RobotViz
import os
import json
import pdb
import time
import spatialmath as sm
import numpy as np
from scipy.spatial.transform import Rotation as R

from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils

from diffrobot.diffusion_policy.diffusion_policy import DiffusionPolicy

import collections


# dataset_path = "/home/krishan/work/2024/datasets/cup_10_demos_again"
dataset_path = "/home/krishan/work/2024/datasets/saucer_video_demo_push_down"
dutils = DatasetUtils(dataset_path)
rlds, stats = dutils.create_rlds(num_noisy_variations=0, transformed_affordance=False, transformed_ee=False)
env = RobotViz()

# policy_name = 'twilight-dawn-164_state'
# policy = DiffusionPolicy(mode='infer',
#                         policy_type='state',
#                         config_file=f'../diffusion_policy/runs/{policy_name}/config_state_pretrain',
#                         finetune=False,
#                         saved_run_name=policy_name)


# read camera calibration json
camera_calibration_path = os.path.join(dataset_path, "transforms", "hand_eye.json")
with open(camera_calibration_path, "r") as f:
    data = json.load(f)
    X_EC_b = np.array(data["X_EC_b"])
    X_EC_f = np.array(data["X_EC_f"])



# X_OA_ee_path = os.path.join(dataset_path, "transforms", "ee_transform.json")
# X_OA_ee = json.load(open(X_OA_ee_path, "r"))['X_OA']





for episode in rlds:
    ep_data = rlds[episode]

    # if episode % 10 != 0:
    #     continue

    # X_B_O2_path = os.path.join(dataset_path, "episodes", str(episode), "secondary_affordance_frames.json")
    # secondary_object_data = json.load(open(X_B_O2_path, "r"))


    # obs_deque = collections.deque(maxlen=3)

    X_AE = None

    for phase in ep_data:
        phase_data = ep_data[phase]
        len_phase = len(phase_data['X_BE_follower'])

        for idx in range(len_phase):
            # if idx == 1:
            #     pdb.set_trace()

            if idx % 2 == 0:
                continue

            X_BE = np.array(phase_data['X_BE_follower'][idx])
            X_BE_leader = np.array(phase_data['X_BE_leader'][idx])
            X_B_O1 = np.array(phase_data['X_B_O1'][idx])
            X_B_OO1 = np.array(phase_data['X_B_OO1'][idx])
            q = phase_data['gello_q'][idx]

            X_BE_gello = env.robot.fkine(np.array(q), "panda_link8") * env.X_FE


            # recover X_BE
            # X_OA = X_OA_ee
            # X_B_O2 = np.array(secondary_object_data[idx]['X_BO'])
            # X_BA = X_B_O2 @ X_OA
            # if X_AE is None:
            #     X_AE = np.linalg.inv(X_BA) @ X_BE_gello.A
            # X_BE_recovered = X_BE @ X_AE


            # obs = {"X_BE": X_BE, 
            # "X_BO": X_B_O1,
            # "X_B_OO": sm.SE3(X_B_OO1, check=False).norm(),
            # "X_OO_O": np.array(phase_data['X_OO1_O1'][idx]), 
            # "gripper_state": np.array(phase_data['gripper_state'][idx]),
            # "progress": np.array(phase_data['progress'][idx]),
            # "phase": np.array(phase_data['phase'][idx]),}

            # obs_deque.append(obs)

            # if len(obs_deque) < 3:
            #     continue

            # out = policy.infer_action(obs_deque)
            # X_BE_policy = out['action'][0]

            
            

            print('Progress: ', phase_data['progress'][idx]*100, '%')
            env.object_pose.T = sm.SE3(X_B_O1, check=False).norm()
            env.policy_pose.T = sm.SE3(X_BE_leader, check=False).norm()
            env.cup_handle.T = sm.SE3(X_BE, check=False).norm()
            env.step(phase_data['gello_q'][idx])
            # env.step(phase_data['gello_q'][idx], robot_q_recovered[0])
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





    
    
    
    

   