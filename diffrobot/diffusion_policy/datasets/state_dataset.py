
import numpy as np
import torch
from diffrobot.diffusion_policy.utils.dataset_utils import create_sample_indices, get_data_stats, normalize_data, \
    parse_dataset, extract_goal_poses, extract_robot_pos_orien, extract_goal_positions
import pdb
import yaml
import json



class DiffusionStateDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 phase: str,
                 action_frame: str,
                 transform):
        

        self.action_frame = action_frame
        self.dataset_path = dataset_path
        self.all_data = parse_dataset(dataset_path)
        # self.all_goal_poses = extract_goal_poses(dataset_path)
        self.phase = phase
        self.learn_progress = True


        indices = create_sample_indices(sequence_length=pred_horizon,
                                        dataset_path=dataset_path)


        # with open(f'{dataset_path}/calibration/transforms.json', 'r') as f:
        #     trans = json.load(f)

        # self.X_BO = np.array(trans['X_BO'])
        # self.X_EC = np.array(trans['X_EC'])

        if self.action_frame == 'global':
            self.all_ee_poses = self.all_data['ee_poses']
            self.all_gello_poses = self.all_data['gello_poses']

        # transform all robot state poses to the object(goal) frame
        if self.action_frame == 'object_centric':
            # X_BE: Robot end effector wrt base frame
            # X_CO: Object frame wrt camera frame
            # X_BC: Camera frame wrt base frame
            # X_BO: Object frame wrt base frame
            self.object_centric_states = []
            self.object_centric_actions = []
            self.object_state  = []
            self.all_ee_poses = self.all_data['ee_poses']
            self.all_gello_poses = self.all_data['gello_poses']
            self.all_object_poses = self.all_data['object_poses']
            self.all_oriented_object_poses = self.all_data['oriented_object_poses']

            for i in range(len(self.all_ee_poses)): # for each episode
                temp_ee = []
                temp_ee_gello = []
                temp_object_state = []

                X_BO = self.all_object_poses[i][0]
                X_BOO = self.all_oriented_object_poses[i][0]

                for j in range(len(self.all_ee_poses[i])): # for each state in the episode
                   
                    # temp_X_BO = self.all_object_poses[i][j]

                    # if j == 0:
                    #     X_BO = np.array(temp_X_BO)
                    # else:
                    #     if temp_X_BO.shape != ():
                    #         temp = np.array(temp_X_BO)
                    #         dist = np.dot(np.array([0,0,1]), temp[:3,2])
                    #         if dist > 0.98: # filter out bad object poses
                    #             X_BO = temp
                    
                    # temp_object_state.append(X_BO)

                    X_BE = self.all_ee_poses[i][j]
                    # transform all ee poses to oriented object frame
                    X_OE = np.dot(np.linalg.inv(X_BOO), X_BE)
                    temp_ee.append(X_OE)

                    X_BE_gello = self.all_gello_poses[i][j]
                    X_OE_gello = np.dot(np.linalg.inv(X_BOO), X_BE_gello)
                    temp_ee_gello.append(X_OE_gello)

                self.object_centric_states.append(temp_ee)
                self.object_centric_actions.append(temp_ee_gello)
                self.object_state.append([X_BO])
            
            self.all_ee_poses = self.object_centric_states
            self.all_gello_poses = self.object_centric_actions
            self.all_object_poses = self.object_state


        # Accumulate all the data 
        self.all_ee_pos, self.all_ee_orien = extract_robot_pos_orien(self.all_ee_poses)
        self.all_gello_pos, self.all_gello_orien = extract_robot_pos_orien(self.all_gello_poses)
        self.all_object_pos, self.all_object_orien = extract_robot_pos_orien(self.all_object_poses)
        # self.all_tactile_data = self.all_data['tactile_data']
        self.all_joint_torques = self.all_data['joint_torques']
        self.all_ee_forces = self.all_data['ee_forces']
        self.all_progress = [np.linspace(0, 1, len(states)) for states in self.all_ee_pos]
        # self.all_tactile_0 = [np.array(data)[:,0,:,:,:] for data in self.all_tactile_data]
        # self.all_tactile_1 = [np.array(data)[:,1,:,:,:] for data in self.all_tactile_data]

        # place channels first
        # self.all_tactile_0 = [np.transpose(data, (0, 3, 1, 2)) for data in self.all_tactile_0]
        # self.all_tactile_1 = [np.transpose(data, (0, 3, 1, 2)) for data in self.all_tactile_1]

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # stats["goals"] = get_data_stats(self.all_goals)
        stats["ee_positions"] = get_data_stats(self.all_ee_pos)
        stats["joint_torques"] = get_data_stats(self.all_joint_torques)
        stats["ee_forces"] = get_data_stats(self.all_ee_forces)
        stats["progress"] = get_data_stats(self.all_progress)
        stats["ee_positions_gello"] = get_data_stats(self.all_gello_pos)

        stats["tactile_data"] = {
            'min': 0.0,
            'max': 9.5
        }

        stats["images"] = {
            'min': 0,
            'max': 255
        }

        normalized_train_data = dict()
        normalized_train_data['ee_positions'] = [normalize_data(data, stats['ee_positions']) for data in self.all_ee_pos]
        normalized_train_data['ee_positions_gello'] = [normalize_data(data, stats['ee_positions_gello']) for data in self.all_gello_pos]
        normalized_train_data['joint_torques'] = [normalize_data(data, stats['joint_torques']) for data in self.all_joint_torques]
        normalized_train_data['ee_forces'] = [normalize_data(data, stats['ee_forces']) for data in self.all_ee_forces]
        normalized_train_data['progress'] = [normalize_data(data, stats['progress']) for data in self.all_progress]
        # normalized_train_data['tactile_0'] = [normalize_data(data, stats['tactile_data']) for data in self.all_tactile_0]
        # normalized_train_data['tactile_1'] = [normalize_data(data, stats['tactile_data']) for data in self.all_tactile_1]

        # shuffle indices
        np.random.seed(0)
        np.random.shuffle(indices)
        
        self.index_order = indices.copy()
        
        # split into train and val
        if self.phase == 'train':
            indices = indices[:int(0.9*len(indices))]
        elif self.phase == 'val':
            indices = indices[int(0.9*len(indices)):]

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


    def sample_sequence(self, episode, start_idx, end_idx):
        # state data
        ee_pos = self.normalized_train_data['ee_positions'][episode][start_idx:end_idx]
        ee_orien = self.all_ee_orien[episode][start_idx:end_idx] # we dont normalize orientations
        # joint_torques = self.normalized_train_data['joint_torques'][episode][start_idx:end_idx]
        # ee_forces = self.normalized_train_data['ee_forces'][episode][start_idx:end_idx]
        progress = self.normalized_train_data['progress'][episode][start_idx:end_idx].reshape(-1, 1)

        object_orien = self.all_object_orien[episode][0] # orientation of object in the base frame
        
        # create a state tensor - exclude tactile data for now
        # robot_state = np.concatenate([ee_pos, ee_orien, joint_torques, ee_forces, progress], axis=-1)
        robot_state = np.concatenate([ee_pos, ee_orien], axis=-1)

        # action data 
        action_pos = self.normalized_train_data['ee_positions_gello'][episode][start_idx:end_idx] 
        action_orien = self.all_gello_orien[episode][start_idx:end_idx] # we dont normalize orientations

        robot_action = np.concatenate([action_pos, action_orien, progress], axis=-1)
    
        return {'state': robot_state,
                'object_orientation': object_orien,
                'action': robot_action}
 
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        episode, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = self.sample_sequence(
            episode, sample_start_idx, sample_end_idx
        )

        # normalize data
        nsample['state'] = nsample['state'][:self.obs_horizon,:]

        return nsample
    

# # # # # test
# fpath = "/home/krishan/work/2024/datasets/cup_rotate_X"
# dataset = DiffusionStateDataset(
#     dataset_path=fpath,
#     pred_horizon=16,
#     obs_horizon=2,
#     action_horizon=8,
#     phase='train',
#     action_frame='object_centric',
#     transform=None)

# dataset.__getitem__(0)


