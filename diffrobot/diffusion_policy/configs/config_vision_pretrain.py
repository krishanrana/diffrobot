

im_height = 240
im_width = 320
crop_height = 216
crop_width = 288

num_cameras = 2
vision_feature_dim = 512
lowdim_obs_dim = 3
obs_dim = (num_cameras*vision_feature_dim) + lowdim_obs_dim
action_dim = 3
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
global_cond_dim = obs_dim*obs_horizon

num_diffusion_iters = 100

batch_size = 128
num_workers = 11
num_epochs = 3500

lr = 1e-4
lr_scheduler_profile = 'cosine'
weight_decay = 1e-6
num_warmup_steps = 500

dataset_path = "/home/krishan/work/2024/datasets/franka_3D_reacher"