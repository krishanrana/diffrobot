
dataset_path = "/home/krishan/work/2024/datasets/cup_10_demo_clean"

symmetric = False

if symmetric:
    low_dim = 4
else:   
    low_dim = 16 #17 #23

tactile_dim = 16
obs_dim = low_dim 

action_dim = 11 # 3 for position, 1 for progress
pred_horizon = 16
obs_horizon = 3
action_horizon = 8
global_cond_dim = (obs_dim*obs_horizon)

num_diffusion_iters = 100 # use 16 during inference

batch_size = 256 #128
num_workers = 11
num_epochs = 4500

lr = 1e-4
lr_scheduler_profile = 'cosine'
weight_decay = 1e-6
num_warmup_steps = 500

freq_divisor = 2

action_frame = 'object_centric' # 'global' or 'object_centric'


