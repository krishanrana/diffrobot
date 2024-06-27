import torch

dataset_path = "/home/krishan/work/2024/datasets/cup_video_demo_3" # TODO: DID YOU UPDATE THE SYMETRIC FLAG BELOW?
action_frame = 'object_centric' # 'global' or 'object_centric' or 'ee_centric'


symmetric = False # this should actually be if there is a goal orientation or not
transformed_affordance = False
transformed_ee = False
oriented_frame = True
z_up = True

down_dims = [128,256,256]
diffusion_step_embed_dim = 128

if action_frame == 'object_centric':
    if symmetric:
        low_dim = 10
    else:   
        low_dim = 16 #17 #23
elif action_frame == 'global':
    if symmetric:
        low_dim = 13
    else:
        low_dim = 19
elif action_frame == 'ee_centric':
    if symmetric:
        low_dim = 19
    else:
        low_dim = 25

elif action_frame == 'e2e':
    low_dim = 46


tactile_dim = 16
obs_dim = low_dim 

action_dim = 11 # 3 for position, 1 for progress
pred_horizon = 16
obs_horizon = 3
action_horizon = 16
global_cond_dim = (obs_dim*obs_horizon)

num_diffusion_iters = 50

batch_size = 256 #128
num_workers = 11
num_epochs = 8500

lr = 1e-4
lr_scheduler_profile = 'cosine'
weight_decay = 1e-6
num_warmup_steps = 500

freq_divisor = 2



