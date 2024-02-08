import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from vision_encoder import get_resnet, replace_bn_with_gn
from network import ConditionalUnet1D
from dataset import PushTImageDataset, normalize_data, unnormalize_data
import wandb
import os
from vision_encoder import get_resnet, replace_bn_with_gn
import collections
from pushT_env import PushTImageEnv
import cv2
import pdb
import pickle as pkl

# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = 2*(vision_feature_dim) + lowdim_obs_dim
action_dim = 2

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)


device = torch.device('cuda')


vision_encoder_top = get_resnet('resnet18')
vision_encoder_left = get_resnet('resnet18')

vision_encoder_top = replace_bn_with_gn(vision_encoder_top)
vision_encoder_left = replace_bn_with_gn(vision_encoder_left)


noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

ckpt_path = "saved_weights/nets.ckpt"
state_dict = torch.load(ckpt_path, map_location='cuda')

nets = nn.ModuleDict({
    'vision_encoder_top': vision_encoder_top,
    'vision_encoder_left': vision_encoder_left,
    'noise_pred_net': noise_pred_net
})

_ = nets.to(device)

ema_nets = nets
ema_nets.load_state_dict(state_dict)
print('Pretrained weights loaded.')

#read stats pkl
stats_path = "saved_weights/stats.pkl"
with open(stats_path, 'rb') as f:
    stats = pkl.load(f)


#@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination
max_steps = 2000


def get_obs():
    # read images from realsense and scale down
    # read position of the robot ee
    # save as dict
    pass


while True:
    # get first observation
    obs = get_obs()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0


    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        image_top = np.stack([x['image_top'] for x in obs_deque])
        image_left = np.stack([x['image_left'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
        nimage_top = image_top / 255.0
        nimage_left = image_left / 255.0

        # device transfer
        nimage_top = torch.from_numpy(nimage_top).to(device, dtype=torch.float32)
        nimage_left = torch.from_numpy(nimage_left).to(device, dtype=torch.float32)
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # get image features
            image_features_top = ema_nets['vision_encoder_top'](nimage_top)
            image_features_left = ema_nets['vision_encoder_left'](nimage_left)


            image_features = torch.cat([image_features_top, image_features_left], dim=-1)
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning


        for i in range(len(action)):
            # stepping env
            # execute on robot - use waypoint frankx - remove for loop
            obs, reward, done, _, info = env.step(action[i])

            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            im = env.render(mode='rgb_array')
            # show image cv2
            cv2.imshow('image', im)
            cv2.waitKey(1)

            step_idx += 1
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))

# # visualize
# from IPython.display import Video
# vwrite('vis.mp4', imgs)
# Video('vis.mp4', embed=True, width=256, height=256)