
import numpy as np
import torch
from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils
import pdb
import yaml
import json



class DiffusionStateDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 stage: str,
                 transform,
                 freq_divisor: int,
                 symmetric: bool,
                 transformed_affordance: bool,
                 transformed_ee: bool,
                 action_frame: str):

        self.action_frame = action_frame
        self.dataset_path = dataset_path
        self.dutils = DatasetUtils(dataset_path)
        self.all_data, self.stats = self.dutils.create_rlds(transformed_affordance=transformed_affordance, transformed_ee=transformed_ee)
        self.stage = stage
        self.symmetric = symmetric

        self.freq_divisor = freq_divisor

        # indices = self.dutils.create_sample_indices(self.all_data)
        # WIP
        indices = self.dutils.create_sample_indices(self.all_data, sequence_length=pred_horizon*self.freq_divisor)
        
        if self.action_frame == 'global':
            pass

        if self.action_frame == 'object_centric':
            pass

        # shuffle indices
        np.random.seed(0)
        np.random.shuffle(indices)
        
        self.index_order = indices.copy()
        
        # split into train and val
        if self.stage == 'train':
            indices = indices[:int(0.9*len(indices))]
        elif self.stage == 'val':
            indices = indices[int(0.9*len(indices)):]

        self.indices = indices
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        # Precompute ee_centric transformations if needed
        if self.action_frame == 'ee_centric':
            self.precompute_ee_centric_transformations()



    def precompute_ee_centric_transformations(self):
        self.precomputed_data = {}
        for episode, phase, start_idx, end_idx in self.indices:
            phase = str(phase)
            if episode not in self.precomputed_data:
                self.precomputed_data[episode] = {}
            if phase not in self.precomputed_data[episode]:
                self.precomputed_data[episode][phase] = {}
            
            if (start_idx, end_idx) not in self.precomputed_data[episode][phase]:
                X_BE_follower = self.all_data[episode][phase]['X_BE_follower'][start_idx:end_idx:self.freq_divisor]
                X_OO_E_follower = self.all_data[episode][phase]['X_OO_E_follower'][start_idx:end_idx:self.freq_divisor]
                X_E_OO_follower = [np.linalg.inv(x_oo_e) for x_oo_e in X_OO_E_follower]
                
                X_B_O1 = self.all_data[episode][phase]['X_B_O1'][start_idx:end_idx:self.freq_divisor]
                X_BS_follower = np.array(X_BE_follower[0])
                X_SE_follower = [np.linalg.inv(X_BS_follower) @ x_be for x_be in X_BE_follower]
                X_SO = [np.linalg.inv(X_BS_follower) @ x_bo for x_bo in X_B_O1]

                X_S_OO = [x_se @ x_e_oo for x_se, x_e_oo in zip(X_SE_follower, X_E_OO_follower)]
                goal_pos, goal_orien = self.dutils.extract_robot_pos_orien(X_S_OO)

                pos_follower, orien_follower = self.dutils.extract_robot_pos_orien(X_SE_follower)
                pos_object, orien_object = self.dutils.extract_robot_pos_orien(X_SO)

                # Normalize
                pos_follower = self.dutils.normalize_data(pos_follower, self.stats['ee_centric'])
                pos_object = self.dutils.normalize_data(pos_object, self.stats['ee_centric'])

                # Precompute action data
                X_BE_leader = self.all_data[episode][phase]['X_BE_leader'][start_idx:end_idx:self.freq_divisor]
                X_BS_leader = np.array(X_BE_leader[0])
                X_SE_leader = [np.linalg.inv(X_BS_leader) @ x_be for x_be in X_BE_leader]
                pos_leader, orien_leader = self.dutils.extract_robot_pos_orien(X_SE_leader)
                
                # Normalize
                pos_leader = self.dutils.normalize_data(pos_leader, self.stats['ee_centric'])
                
                self.precomputed_data[episode][phase][(start_idx, end_idx)] = {
                    'pos_follower': pos_follower,
                    'orien_follower': orien_follower,
                    'pos_object': pos_object,
                    'orien_object': orien_object,
                    'pos_leader': pos_leader,
                    'orien_leader': orien_leader,
                    'goal_pos': goal_pos,
                    'goal_orien': goal_orien
                }

    def sample_sequence(self, episode, phase, start_idx, end_idx):

        # episode = str(episode)
        phase = str(phase)

        # state data
        pos_follower = self.all_data[episode][phase]['pos_follower'][start_idx:end_idx:self.freq_divisor]
        orien_follower = self.all_data[episode][phase]['orien_follower'][start_idx:end_idx:self.freq_divisor]
        progress = self.all_data[episode][phase]['progress'][start_idx:end_idx:self.freq_divisor]
        orien_object = self.all_data[episode][phase]['orien_object'][start_idx:end_idx:self.freq_divisor]
        gripper_state = self.all_data[episode][phase]['gripper_state'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)
        progress = self.all_data[episode][phase]['progress'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)

        pos_follower_global = self.all_data[episode][phase]['pos_follower_global'][start_idx:end_idx:self.freq_divisor]
        orien_follower_global = self.all_data[episode][phase]['orien_follower_global'][start_idx:end_idx:self.freq_divisor]
        orien_object_global = self.all_data[episode][phase]['orien_object_global'][start_idx:end_idx:self.freq_divisor]
        pos_object_global = self.all_data[episode][phase]['pos_object_global'][start_idx:end_idx:self.freq_divisor]


        if self.action_frame == 'ee_centric':
            precomputed = self.precomputed_data[episode][phase][(start_idx, end_idx)]
            pos_follower = precomputed['pos_follower']
            orien_follower = precomputed['orien_follower']
            pos_object = precomputed['pos_object']
            orien_object_ee = precomputed['orien_object'] # using the orien wrt to reference frame 
            goal_orien = precomputed['goal_orien']

            if not self.symmetric:
                robot_state = np.concatenate([pos_follower, orien_follower, goal_orien, orien_object_ee, pos_object, gripper_state], axis=-1)
            else:
                robot_state = np.concatenate([pos_follower, orien_follower, pos_object, orien_object_ee, gripper_state], axis=-1)



        # if self.action_frame == 'ee_centric':
        #     X_BE_follower = self.all_data[episode][phase]['X_BE_follower'][start_idx:end_idx:self.freq_divisor]
        #     X_BS_follower = np.array(X_BE_follower[0])
        #     X_SE_follower = [np.linalg.inv(X_BS_follower) @ x_be for x_be in X_BE_follower]

        #     X_B_O1 = self.all_data[episode][phase]['X_B_O1'][start_idx:end_idx:self.freq_divisor]
        #     X_SO = [np.linalg.inv(X_BS_follower) @ x_bo for x_bo in X_B_O1]

        #     pos_follower, orien_follower = self.dutils.extract_robot_pos_orien(X_SE_follower)
        #     pos_object, orien_object = self.dutils.extract_robot_pos_orien(X_SO)

        #     # normalize
        #     pos_follower = self.dutils.normalize_data(pos_follower, self.stats['ee_centric'])
        #     pos_object = self.dutils.normalize_data(pos_object, self.stats['ee_centric'])

        #     if not self.symmetric:
        #         robot_state = np.concatenate([pos_follower, orien_follower, orien_object, pos_object ,gripper_state], axis=-1)
        #     else:
        #         robot_state = np.concatenate([pos_follower, orien_follower, pos_object, gripper_state], axis=-1)


        if self.action_frame == 'object_centric':
            if not self.symmetric:
                robot_state = np.concatenate([pos_follower, orien_follower, orien_object, gripper_state], axis=-1)
            else:
                robot_state = np.concatenate([pos_follower, orien_follower, gripper_state], axis=-1)

        if self.action_frame == 'global':
            if not self.symmetric:
                robot_state = np.concatenate([pos_follower_global, orien_follower_global, orien_object_global, pos_object_global, gripper_state], axis=-1)
            else:
                robot_state = np.concatenate([pos_follower_global, orien_follower_global, pos_object_global, gripper_state], axis=-1)


        # action data
        if self.action_frame == 'object_centric':
            pos_leader = self.all_data[episode][phase]['pos_leader'][start_idx:end_idx:self.freq_divisor]
            orien_leader = self.all_data[episode][phase]['orien_leader'][start_idx:end_idx:self.freq_divisor]
            gripper_action = self.all_data[episode][phase]['gripper_action'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)
            robot_action = np.concatenate([pos_leader, orien_leader, gripper_action, progress], axis=-1)

        if self.action_frame == 'global':
            pos_leader_global = self.all_data[episode][phase]['pos_leader_global'][start_idx:end_idx:self.freq_divisor]
            gripper_action = self.all_data[episode][phase]['gripper_action'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)
            orien_leader_global = self.all_data[episode][phase]['orien_leader_global'][start_idx:end_idx:self.freq_divisor]
            robot_action = np.concatenate([pos_leader_global, orien_leader_global, gripper_action, progress], axis=-1)

        # if self.action_frame == 'ee_centric':
        #     X_BE_leader = self.all_data[episode][phase]['X_BE_leader'][start_idx:end_idx:self.freq_divisor]
        #     X_BS_leader = np.array(X_BE_leader[0])
        #     X_SE_leader = [np.linalg.inv(X_BS_leader) @ x_be for x_be in X_BE_leader]
        #     pos_leader, orien_leader = self.dutils.extract_robot_pos_orien(X_SE_leader)

        #     gripper_action = self.all_data[episode][phase]['gripper_action'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)

        #     # normalize
        #     pos_leader = self.dutils.normalize_data(pos_leader, self.stats['ee_centric'])
        #     robot_action = np.concatenate([pos_leader, orien_leader, gripper_action, progress], axis=-1)

        if self.action_frame == 'ee_centric':
            precomputed = self.precomputed_data[episode][phase][(start_idx, end_idx)]
            pos_leader = precomputed['pos_leader']
            orien_leader = precomputed['orien_leader']
            gripper_action = self.all_data[episode][phase]['gripper_action'][start_idx:end_idx:self.freq_divisor].reshape(-1, 1)
            
            robot_action = np.concatenate([pos_leader, orien_leader, gripper_action, progress], axis=-1)

        return {'state': robot_state,
                'action': robot_action}
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        episode, phase, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = self.sample_sequence(
            episode, phase, sample_start_idx, sample_end_idx
        )

        # normalize data
        nsample['state'] = nsample['state'][:self.obs_horizon,:]

        return nsample
    

# # # # # test
# fpath = "/home/krishan/work/2024/datasets/cup_10_demo"
# dataset = DiffusionStateDataset(
#     dataset_path=fpath,
#     pred_horizon=16,
#     obs_horizon=2,
#     action_horizon=8,
#     stage='train',
#     transform=None,
#     symmetric=False,
#     action_frame='object_centric')
# pdb.set_trace()

# dataset.__getitem__(0)


