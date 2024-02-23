

im_height = 240
im_width = 320
crop_height = 216
crop_width = 288

vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = 514
action_dim = 2
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
global_cond_dim = obs_dim*obs_horizon

num_diffusion_iters = 100

batch_size = 64
num_workers = 8
num_epochs = 500

lr = 6.6e-6
lr_scheduler_profile = 'linear'
weight_decay = 1e-6
num_warmup_steps = 50

dataset_path = "/home/krishan/work/2024/datasets/franka_reacher"