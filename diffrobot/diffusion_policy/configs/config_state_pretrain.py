
goal_dim = 3
obs_dim =  3
action_dim = 3
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
global_cond_dim = (obs_dim*obs_horizon)

num_diffusion_iters = 70 #100

batch_size = 128
num_workers = 11
num_epochs = 4500

lr = 1e-4
lr_scheduler_profile = 'cosine'
weight_decay = 1e-6
num_warmup_steps = 500

action_frame = 'object_centric' # 'absolute', 'end-effector', 'deltas'

dataset_path = "/home/krishan/work/2024/datasets/franka_3D_reacher"