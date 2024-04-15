

low_dim = 9 #23
tactile_dim = 16
object_rotation_dim = 6
obs_dim = low_dim 

action_dim = 10 # 3 for position, 1 for progress
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
global_cond_dim = (obs_dim*obs_horizon) + object_rotation_dim

num_diffusion_iters = 100 # use 16 during inference

batch_size = 128
num_workers = 11
num_epochs = 4500

lr = 1e-4
lr_scheduler_profile = 'cosine'
weight_decay = 1e-6
num_warmup_steps = 500

action_frame = 'object_centric' # 'global' or 'object_centric'

dataset_path = "/home/krishan/work/2024/datasets/cup_rotate_X"