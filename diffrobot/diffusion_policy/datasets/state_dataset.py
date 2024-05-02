
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
                 action_frame: str):

        self.action_frame = action_frame
        self.dataset_path = dataset_path
        self.dutils = DatasetUtils(dataset_path)
        self.all_data, self.stats = self.dutils.create_rlds()
        self.stage = stage

        indices = self.dutils.create_sample_indices(self.all_data)
        
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


    def sample_sequence(self, episode, phase, start_idx, end_idx):

        episode = str(episode)
        phase = str(phase)

        # state data
        pos_follower = self.all_data[episode][phase]['pos_follower'][start_idx:end_idx]
        orien_follower = self.all_data[episode][phase]['orien_follower'][start_idx:end_idx]
        progress = self.all_data[episode][phase]['progress'][start_idx:end_idx]
        orien_object = self.all_data[episode][phase]['orien_object'][start_idx:end_idx]
        gripper_state = self.all_data[episode][phase]['gripper_state'][start_idx:end_idx].reshape(-1, 1)
        ep_phase = self.all_data[episode][phase]['phase'][start_idx:end_idx].reshape(-1, 1)    

        robot_state = np.concatenate([pos_follower, orien_follower, orien_object, gripper_state, ep_phase], axis=-1)

        # action data
        pos_leader = self.all_data[episode][phase]['pos_leader'][start_idx:end_idx]
        orien_leader = self.all_data[episode][phase]['orien_leader'][start_idx:end_idx]
        gripper_action = self.all_data[episode][phase]['gripper_action'][start_idx:end_idx].reshape(-1, 1)
        progress = self.all_data[episode][phase]['progress'][start_idx:end_idx].reshape(-1, 1)

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
# fpath = "/home/krishan/work/2024/datasets/cup_saucer"
# dataset = DiffusionStateDataset(
#     dataset_path=fpath,
#     pred_horizon=16,
#     obs_horizon=2,
#     action_horizon=8,
#     stage='train',
#     action_frame='object_centric')
# pdb.set_trace()

# dataset.__getitem__(0)


