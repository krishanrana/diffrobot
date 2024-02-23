import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from vision_encoder import get_resnet, replace_bn_with_gn
from network import ConditionalUnet1D
from dataset_franka_real import PushTStateDataset
import wandb
import pdb
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage, RandomCrop

wandb.init(project="diffusion_experiments")

################################################################## Parameters ##################################################################


# agent_pos is 2 dimensional
obs_dim = 2
goal_dim = 2
action_dim = 2

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8

dataset_path = "/home/krishan/work/2024/datasets/franka_reacher"


################################################################## Networks ##################################################################

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=(obs_dim*obs_horizon) + goal_dim,
)


device = torch.device('cuda')
_ = noise_pred_net.to(device)

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

train_dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    phase='train')

val_dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    phase='val')


assert np.all(train_dataset.index_order == val_dataset.index_order)

# save training data statistics (min, max) for each dim
stats = train_dataset.stats
# np.save('saved_weights/stats.npy', stats)

# create dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    # num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    # persistent_workers=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    # num_workers=1,
    shuffle=False,
    pin_memory=True,
    # persistent_workers=True
)   



################################################################## Training ##################################################################

# ############
# # Load pretrained weights
# ckpt_path = "saved_weights/nets_reacher.ckpt"
# state_dict = torch.load(ckpt_path, map_location='cuda')
# nets.load_state_dict(state_dict)
# print('Pretrained weights loaded.')
# #############

num_epochs = 5000
last_best_loss = 10000


# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                # nimage_top = nbatch['image_top'][:,:obs_horizon].to(device)
                ngoal = nbatch['goal'].to(device).to(dtype=torch.float32)
                nagent_pos = nbatch['robot_state'][:,:obs_horizon].to(device).to(dtype=torch.float32)
                naction = nbatch['action'].to(device).to(dtype=torch.float32)
                B = nagent_pos.shape[0]

                obs_cond = nagent_pos.flatten(start_dim=1)
                obs_cond =torch.cat([ngoal, obs_cond], dim=-1)

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

                wandb.log({"train_loss": loss})

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

            # save nets state dict ckpt every 1000 steps
            torch.save(noise_pred_net.state_dict(), 'saved_weights/nets_reacher_state.ckpt')

        tglobal.set_postfix(loss=np.mean(epoch_loss))


        if epoch_idx % 5 == 0:
            # validate
            with torch.no_grad():
                val_loss = list()
                for nbatch in val_dataloader:
                    ngoal = nbatch['goal'].to(device).to(dtype=torch.float32)
                    nagent_pos = nbatch['robot_state'][:,:obs_horizon].to(device).to(dtype=torch.float32)
                    naction = nbatch['action'].to(device).to(dtype=torch.float32)
                    B = nagent_pos.shape[0]

                    obs_cond = nagent_pos.flatten(start_dim=1)
                    obs_cond =torch.cat([ngoal, obs_cond], dim=-1)

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
                    val_loss.append(loss.item())

                wandb.log({"val_loss": np.mean(val_loss)})

                if np.mean(val_loss) < last_best_loss:
                    torch.save(noise_pred_net.state_dict(), 'saved_weights/nets_reacher_state_best.ckpt')
                    last_best_loss = np.mean(val_loss)
                    print('Saved best model!')






