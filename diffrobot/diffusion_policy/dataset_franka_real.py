#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

import numpy as np
import torch
import pdb
import os
import json
import tqdm
import pickle
from torchvision.io import read_image


def create_sample_indices(sequence_length:int,
                          dataset_path:str):
    
    # iterate through all the episode folders
    indices = list()

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        state = json.load(open(os.path.join(dataset_path, "episodes", episode, "state.json")))

        # get the length of the episode
        episode_length = len(state)
        # iterate through the episode
        range_idx = episode_length - (sequence_length + 2)
        for idx in range(range_idx):
            # get the start and end index of the sequence
            buffer_start_idx = idx
            buffer_end_idx = idx + sequence_length
            indices.append([int(episode), buffer_start_idx, buffer_end_idx])
            assert buffer_end_idx - buffer_start_idx == sequence_length
        
    indices = np.array(indices)
    return indices

def sample_sequence_states(dataset_path: str, states: list, goals: list, episode: int, start_idx: int, end_idx: int):

    data = {
        # 'image_top': f_top,
        'goal': goals[episode],
        'robot_state': states[start_idx:end_idx],
        'action': states[start_idx+1:end_idx+1]
    }

    return data


def flatten_2d_lists(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def get_data_stats(data: list):
    data = np.array(flatten_2d_lists(data))
    stats = {
       'min': np.min(data, axis=0),
       'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data



# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 phase: str,
                 transform):
        
        self.dataset_path = dataset_path
        self.all_states = np.load(f'{dataset_path}/all_states.pkl', allow_pickle=True)
        self.transform = transform
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
        stats["states"] = get_data_stats(self.all_states)
        stats["actions"] = get_data_stats(self.all_states)
        stats["images"] = {
            'min': 0,
            'max': 255
        }
    
        self.indices = indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def sample_sequence_images(self, dataset_path: str, states: list, episode: int, start_idx: int, end_idx: int):

        #Paths to images
        img_dir_front = os.path.join(dataset_path, "episodes", str(episode), "images", "front")
        img_dir_hand = os.path.join(dataset_path, "episodes", str(episode), "images", "hand")

        # Initialize lists to store slices
        f_front = []
        f_hand = []

        start = start_idx+1
        end = start + self.obs_horizon

        for idx in range(start, end):
            img_front = read_image(f'{img_dir_front}/{idx}.png')
            img_hand = read_image(f'{img_dir_hand}/{idx}.png')

            f_hand.append(img_hand)
            f_front.append(img_front)

        f_hand = torch.stack(f_hand, dim=0)
        f_front = torch.stack(f_front, dim=0)

        data = {
            'image_hand': f_hand,
            'image_front': f_front,
            'robot_state': states[start_idx:end_idx],
            'action': states[start_idx+1:end_idx+1]
        }

        return data



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        episode, sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = self.sample_sequence_images(
            dataset_path=self.dataset_path,
            states=self.all_states[episode],
            episode=episode,
            start_idx=sample_start_idx,
            end_idx=sample_end_idx)
        
        # normalize data
        nsample['robot_state'] = normalize_data(nsample['robot_state'], self.stats['states'])
        nsample['action'] = normalize_data(nsample['action'], self.stats['actions'])
        nsample['image_hand'] = nsample['image_hand'] / 255.0
        nsample['image_front'] = nsample['image_front'] / 255.0

        nsample['image_hand'] = nsample['image_hand'][:self.obs_horizon,:]
        nsample['image_front'] = nsample['image_front'][:self.obs_horizon,:]
        nsample['robot_state'] = nsample['robot_state'][:self.obs_horizon,:]

        nsample['image_hand'] = torch.stack([self.transform(img) for img in nsample['image_hand']])
        nsample['image_front'] = torch.stack([self.transform(img) for img in nsample['image_front']])

        return nsample



class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 phase: str,
                 transform):
        
        self.dataset_path = dataset_path
        self.all_states = np.load(f'{dataset_path}/all_states.pkl', allow_pickle=True)
        self.all_goals = np.load(f'{dataset_path}/all_goals.pkl', allow_pickle=True)
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
    
