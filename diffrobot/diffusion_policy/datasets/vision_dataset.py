
import numpy as np
import torch
import pdb
import os
from torchvision.io import read_image
from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils



# dataset
class DiffusionImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 phase: str,
                 use_object_centric: bool,
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




