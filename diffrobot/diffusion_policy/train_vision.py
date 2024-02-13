import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from vision_encoder import get_resnet, replace_bn_with_gn
from network import ConditionalUnet1D
from dataset_franka_real import PushTImageDataset
import wandb
import pdb
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

wandb.init(project="diffusion_experiments")

################################################################## Parameters ##################################################################


# ResNet18 has output dim of 512
vision_feature_dim = 512
# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = vision_feature_dim + vision_feature_dim + lowdim_obs_dim
action_dim = 2

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset_path = "/home/krishan/work/2024/datasets/franka_pusht"


################################################################## Networks ##################################################################

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder_top = get_resnet('resnet18')
vision_encoder_left = get_resnet('resnet18')
# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder_top = replace_bn_with_gn(vision_encoder_top)
vision_encoder_left = replace_bn_with_gn(vision_encoder_left)


# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder_top': vision_encoder_top,
    'vision_encoder_left': vision_encoder_left,
    'noise_pred_net': noise_pred_net
})

device = torch.device('cuda')
_ = nets.to(device)

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

################################################################## Dataloader ##################################################################
transform = Compose([Resize((256,256))])


dataset = PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    transform=transform
)


# save training data statistics (min, max) for each dim
stats = dataset.stats
# np.save('saved_weights/stats.npy', stats)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    num_workers=11,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

################################################################## Training ##################################################################

num_epochs = 1000

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nimage_top = nbatch['image_top'][:,:obs_horizon].to(device)
                nimage_left = nbatch['image_left'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['robot_state'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device).to(dtype=torch.float32)
                B = nagent_pos.shape[0]

                # encoder vision features

                # combine images along batch dim
                # nimage_combined = torch.cat([nimage_top, nimage_left], dim=1)

                image_features_top = nets['vision_encoder_top'](
                    nimage_top.flatten(end_dim=1))
                image_features_top = image_features_top.reshape(
                    *nimage_top.shape[:2],-1)
                
                image_features_left = nets['vision_encoder_left'](
                    nimage_left.flatten(end_dim=1))
                image_features_left = image_features_left.reshape(
                    *nimage_left.shape[:2],-1)
                
                image_features = torch.cat([image_features_top, image_features_left], dim=-1)
                

                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1).to(dtype=torch.float32)
                #
                # to float32


                # (B, obs_horizon * obs_dim)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)


                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
                


                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                wandb.log({"loss": loss})

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

            # save nets state dict ckpt every 1000 steps
            torch.save(nets.state_dict(), 'saved_weights/nets.ckpt')





        tglobal.set_postfix(loss=np.mean(epoch_loss))

