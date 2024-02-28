
import numpy as np
import torch
from diffrobot.diffusion_policy.utils.dataset_utils import create_sample_indices, sample_sequence_states, get_data_stats, normalize_data, \
    extract_robot_poses, extract_goal_poses, extract_robot_positions, extract_goal_positions
import pdb
import yaml

class DiffusionStateDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 phase: str,
                 use_object_centric: bool,
                 transform):
        

        self.use_object_centric = use_object_centric
        self.dataset_path = dataset_path
        self.all_state_poses = extract_robot_poses(dataset_path)
        self.all_goal_poses = extract_goal_poses(dataset_path)

        # transform all robot state poses to the object(goal) frame
        if self.use_object_centric:
            # X_BE: Robot end effector wrt base frame
            # X_CO: Object frame wrt camera frame
            # X_BC: Camera frame wrt base frame

            with open(f'{dataset_path}/cameras.yaml', 'r') as stream:
                cameras = yaml.safe_load(stream)

            X_BC = np.array(cameras['side']['X_BC'])

            self.object_centric_states = []

            for i in range(len(self.all_state_poses)):
                temp = []
                for X_BE in self.all_state_poses[i]:
                    X_CO = self.all_goal_poses[i]

                    # get the pose of the robot end effector in the object frame
                    X_BO = np.dot(X_BC, X_CO)
                    X_OE = np.dot(np.linalg.inv(X_BO), X_BE)
                    temp.append(X_OE)

                self.object_centric_states.append(temp)
            
            self.all_state_poses = self.object_centric_states


        # get xyz from the object centric states
        self.all_states = extract_robot_positions(self.all_state_poses)
        self.all_goals = extract_goal_positions(self.all_goal_poses)

        
        # self.all_states = np.load(f'{dataset_path}/all_states.pkl', allow_pickle=True)
        # self.all_goals = np.load(f'{dataset_path}/all_goals.pkl', allow_pickle=True)

        self.phase = phase

        indices = create_sample_indices(
            sequence_length=pred_horizon,
            dataset_path=dataset_path)
        
        # shuffle indices
        np.random.seed(0)
        np.random.shuffle(indices)
        
        self.index_order = indices.copy()
        
        # split into train and val
        if self.phase == 'train':
            indices = indices[:int(0.9*len(indices))]
        elif self.phase == 'val':
            indices = indices[int(0.9*len(indices)):]

        
        # compute statistics and normalized data to [-1,1]
        stats = dict()
        # normalized_train_data = dict()
        stats["goals"] = get_data_stats(self.all_goals)
        stats["states"] = get_data_stats(self.all_states)
        stats["actions"] = get_data_stats(self.all_states)
        stats["images"] = {
            'min': 0,
            'max': 255
        }

        # normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        # normalized_train_data['image'] = train_image_data

        self.indices = indices
        self.stats = stats
        # self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        episode, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence_states(
            dataset_path=self.dataset_path,
            states=self.all_states[episode],
            episode=episode,
            goals=self.all_goals,
            start_idx=sample_start_idx,
            end_idx=sample_end_idx)
        
        # normalize data
        nsample['robot_state'] = normalize_data(nsample['robot_state'], self.stats['states'])
        nsample['action'] = normalize_data(nsample['action'], self.stats['actions'])
        nsample['goal'] = normalize_data(nsample['goal'], self.stats['goals'])
        nsample['robot_state'] = nsample['robot_state'][:self.obs_horizon,:]

        return nsample
    

# # test
# fpath = "/home/krishan/work/2024/datasets/franka_3D_reacher"
# dataset = DiffusionStateDataset(
#     dataset_path=fpath,
#     pred_horizon=16,
#     obs_horizon=2,
#     action_horizon=8,
#     phase='train',
#     transform=None)

