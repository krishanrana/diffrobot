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


def sample_sequence_images(dataset_path: str, states: list, episode: int, start_idx: int, end_idx: int):

    #Paths to images
    img_dir_front = os.path.join(dataset_path, "episodes", str(episode), "images", "front")
    # img_dir_left = os.path.join(dataset_path, "episodes", str(ep isode), "images", "left")

    # Initialize lists to store slices
    f_front = []
    # f_left = []

    for idx in range(start_idx+1, end_idx+1):
        img_front = read_image(f'{img_dir_front}/{idx}.png')
        # img_left = read_image(f'{img_dir_left}/{idx}.png')

        # f_top.append(img_top)
        f_front.append(img_front)

    # f_top = torch.stack(f_top, dim=0)
    f_front = torch.stack(f_front, dim=0)

    data = {
        # 'image_top': f_top,
        'image_front': f_front,
        'robot_state': states[start_idx:end_idx],
        'action': states[start_idx+1:end_idx+1]
    }

    return data



def sample_sequence(dataset_path: str, states: list, episode: int, start_idx: int, end_idx: int):
    
    # Paths to the video files
    vp_top = os.path.join(dataset_path, "episodes", str(episode), "video", "1.mp4")
    vp_left = os.path.join(dataset_path, "episodes", str(episode), "video", "2.mp4")

    # Initialize lists to store slices
    f_top = []
    f_left = []

    # Open both video containers
    container_top = av.open(vp_top)
    container_left = av.open(vp_left)
    
    stream_top = container_top.streams.video[0]
    stream_left = container_left.streams.video[0]

    # Calculate the timestamp to seek to
    time_base_top = stream_top.time_base
    time_base_left = stream_left.time_base
    fps_top = stream_top.average_rate
    fps_left = stream_left.average_rate
    timestamp_top = (start_idx / fps_top) / time_base_top
    timestamp_left = (start_idx / fps_left) / time_base_left
    
    # Seek to the nearest keyframe before start_idx
    container_top.seek(int(timestamp_top), stream=stream_top)
    container_left.seek(int(timestamp_left), stream=stream_left)

    frame_idx = start_idx  # Assume seeking gets us to the start_idx or slightly before
    while True:
        try:
            frame_top = next(container_top.decode(stream_top))
            frame_left = next(container_left.decode(stream_left))
        except StopIteration:
            break  # End of video
        
        if frame_idx >= end_idx:
            break
        
        # Process top video frame
        img_top = frame_top.to_image()
        # img_top.save(f'top_frame_{frame_idx}.png')  # Save if needed
        img_top = np.array(img_top)[:, :, :3]  # Convert to numpy array and select RGB

        f_top.append(torch.from_numpy(img_top))
        
        # Process left video frame
        img_left = frame_left.to_image()
        # img_left.save(f'left_frame_{frame_idx}.png')  # Save if needed
        img_left = np.array(img_left)[:, :, :3]  # Convert to numpy array and select RGB
        f_left.append(torch.from_numpy(img_left))
        
        frame_idx += 1  # Increment frame index
    
    # Close containers
    container_top.close()
    container_left.close()

    # Stack frames along a new dimension
    f_top = torch.stack(f_top, dim=0)
    f_left = torch.stack(f_left, dim=0)


    data = {
        'image_top': f_top,
        'image_left': f_left,
        'robot_state': states[start_idx:end_idx],
        'action': states[start_idx+1:end_idx+1]
    }

    return data


# sample_sequence("/home/krishan/work/2024/datasets/franka_pusht", [], 0, 5, 21)


def create_xy_state_dataset(dataset_path:str):
    state = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        raw_data = json.load(open(os.path.join(dataset_path, "episodes", episode ,"state.json")))
        temp = []
        for idx in range(len(raw_data)):
            pose = np.array(raw_data[idx]["X_BE"])[:2,3]
            temp.append(pose)
        
        state.append(temp)

    # save file as pickle
    with open(f'{dataset_path}/all_states.pkl', 'wb') as f:
        pickle.dump(state, f)
    print("Done saving state data")
    return

fpath = "/home/krishan/work/2024/datasets/franka_reacher"
create_xy_state_dataset(fpath)


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

        # # read from zarr dataset
        # dataset_root = zarr.open(dataset_path, 'r')


        # # float32, [0,1], (N,96,96,3)
        # train_image_data = dataset_root['data']['img'][:]
        # train_image_data = np.moveaxis(train_image_data, -1,1)
        # # (N,3,96,96)

        # # (N, D)
        # train_data = {
        #     # first two dims of state vector are agent (i.e. gripper) locations
        #     'agent_pos': dataset_root['data']['state'][:,:2],
        #     'action': dataset_root['data']['action'][:]
        # }
        # episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
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

        
        stats["states"] = get_data_stats(self.all_states)
        stats["actions"] = get_data_stats(self.all_states)
        stats["images"] = {
            'min': 0,
            'max': 255
        }

        # save stats
        with open(f'saved_weights/stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
        

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
        nsample = sample_sequence_images(
            dataset_path=self.dataset_path,
            states=self.all_states[episode],
            episode=episode,
            start_idx=sample_start_idx,
            end_idx=sample_end_idx)
        
        # normalize data
        nsample['robot_state'] = normalize_data(nsample['robot_state'], self.stats['states'])
        nsample['action'] = normalize_data(nsample['action'], self.stats['actions'])
        # nsample['image_top'] = nsample['image_top'] / 255.0
        nsample['image_front'] = nsample['image_front'] / 255.0


        # discard unused observations
        # apply transform


        # nsample['image_top'] = nsample['image_top'][:self.obs_horizon,:]
        nsample['image_front'] = nsample['image_front'][:self.obs_horizon,:]
        nsample['robot_state'] = nsample['robot_state'][:self.obs_horizon,:]

        # nsample['image_top'] = torch.stack([self.transform(img) for img in nsample['image_top']])
        nsample['image_front'] = torch.stack([self.transform(img) for img in nsample['image_front']])

        return nsample
    
