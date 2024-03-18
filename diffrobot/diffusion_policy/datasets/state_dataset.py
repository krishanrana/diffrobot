
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
        

        # with open(f'{dataset_path}/cameras.yaml', 'r') as stream:
        #         cameras = yaml.safe_load(stream)
        # X_BC = np.array(cameras['side']['X_BC'])
        # self.X_BC = X_BC

        with open(f'{dataset_path}/calibration/transforms.json', 'r') as f:
            trans = json.load(f)
        
        self.X_BO = np.array(trans['X_BO'])
        self.X_EC = np.array(trans['X_EC'])

        if self.action_frame == 'global':
            self.all_ee_poses = self.all_data['ee_poses']

        # transform all robot state poses to the object(goal) frame
        if self.action_frame == 'object_centric':
            # X_BE: Robot end effector wrt base frame
            # X_CO: Object frame wrt camera frame
            # X_BC: Camera frame wrt base frame
            # X_BO: Object frame wrt base frame
            self.object_centric_states = []
            self.all_ee_poses = self.all_data['ee_poses']

            for i in range(len(self.all_ee_poses)):
                temp = []
                for X_BE in self.all_ee_poses[i]:
                    #X_CO = self.all_goal_poses[i]
                    # get the pose of the robot end effector in the object frame
                    #X_BO = np.dot(X_BC, X_CO)
                    X_OE = np.dot(np.linalg.inv(self.X_BO), X_BE)
                    temp.append(X_OE)

                self.object_centric_states.append(temp)
            
            self.all_ee_poses = self.object_centric_states

        # Accumulate all the data 
        self.all_ee_pos, self.all_ee_orien = extract_robot_pos_orien(self.all_ee_poses)
        self.all_tactile_data = self.all_data['tactile_data']
        self.all_joint_torques = self.all_data['joint_torques']
        self.all_ee_forces = self.all_data['ee_forces']
        self.all_progress = [np.linspace(0, 1, len(states)) for states in self.all_ee_pos]
        self.all_tactile_0 = [np.array(data)[:,0,:,:,:] for data in self.all_tactile_data]
        self.all_tactile_1 = [np.array(data)[:,1,:,:,:] for data in self.all_tactile_data]

        # place channels first
        self.all_tactile_0 = [np.transpose(data, (0, 3, 1, 2)) for data in self.all_tactile_0]
        self.all_tactile_1 = [np.transpose(data, (0, 3, 1, 2)) for data in self.all_tactile_1]

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # stats["goals"] = get_data_stats(self.all_goals)
        stats["ee_positions"] = get_data_stats(self.all_ee_pos)
        stats["ee_orientations"] = get_data_stats(self.all_ee_orien)
        stats["joint_torques"] = get_data_stats(self.all_joint_torques)
        stats["ee_forces"] = get_data_stats(self.all_ee_forces)
        stats["progress"] = get_data_stats(self.all_progress)

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
        normalized_train_data['joint_torques'] = [normalize_data(data, stats['joint_torques']) for data in self.all_joint_torques]
        normalized_train_data['ee_forces'] = [normalize_data(data, stats['ee_forces']) for data in self.all_ee_forces]
        normalized_train_data['progress'] = [normalize_data(data, stats['progress']) for data in self.all_progress]
        normalized_train_data['tactile_0'] = [normalize_data(data, stats['tactile_data']) for data in self.all_tactile_0]
        normalized_train_data['tactile_1'] = [normalize_data(data, stats['tactile_data']) for data in self.all_tactile_1]

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
        tactile_0 = self.normalized_train_data['tactile_0'][episode][start_idx:end_idx]
        tactile_1 = self.normalized_train_data['tactile_1'][episode][start_idx:end_idx]
        joint_torques = self.normalized_train_data['joint_torques'][episode][start_idx:end_idx]
        ee_forces = self.normalized_train_data['ee_forces'][episode][start_idx:end_idx]
        progress = self.normalized_train_data['progress'][episode][start_idx:end_idx].reshape(-1, 1)

        # create a state tensor - exclude tactile data for now
        robot_state = np.concatenate([ee_pos, ee_orien, joint_torques, ee_forces, progress], axis=-1)

        # action data
        action_pos = self.normalized_train_data['ee_positions'][episode][start_idx+1:end_idx+1]
        action_orien = self.all_ee_orien[episode][start_idx+1:end_idx+1] # we dont normalize orientations

        robot_action = np.concatenate([action_pos, action_orien, progress], axis=-1)
    
        return {'state': robot_state,
                'action': robot_action,
                'tactile_0': tactile_0,
                'tactile_1': tactile_1}
 
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
        nsample['tactile_0'] = nsample['tactile_0'][:self.obs_horizon,:]
        nsample['tactile_1'] = nsample['tactile_1'][:self.obs_horizon,:]

        return nsample
    

# # # # test
fpath = "/home/krishan/work/2024/datasets/door_open"
dataset = DiffusionStateDataset(
    dataset_path=fpath,
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    phase='train',
    action_frame='object_centric',
    transform=None)

dataset.__getitem__(0)


