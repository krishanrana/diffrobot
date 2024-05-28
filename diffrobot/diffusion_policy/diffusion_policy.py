import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffrobot.diffusion_policy.models.vision_encoder import get_resnet, replace_bn_with_gn
from diffrobot.diffusion_policy.models.unet import ConditionalUnet1D
from diffrobot.diffusion_policy.models.tactile_encoder import TactileEncoder
from diffrobot.diffusion_policy.datasets.vision_dataset import DiffusionImageDataset
from diffrobot.diffusion_policy.datasets.state_dataset import DiffusionStateDataset
import wandb
import pdb
from torchvision.transforms import Compose, Resize, RandomCrop
from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils
import yaml
import json
import copy
import os
import pickle as pkl
import numpy as np
from diffrobot.diffusion_policy.utils.im_utils import FixedCropTransform
from diffrobot.diffusion_policy.utils.config_utils import get_config
from diffrobot.diffusion_policy.utils.rotation_transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


# torch.backends.cuda.matmul.allow_tf32 = True


class DiffusionPolicy():
    def __init__(self, config_file='config_vision', 
                 finetune=False, 
                 saved_run_name=None,
                 mode='train',
                 policy_type='vision'):
        
        self.params = get_config(config_file, mode=mode)
        self.policy_type = policy_type
        self.mode = mode

        if mode == 'train':
            self.precision = torch.float32
        elif mode == 'infer':
            self.precision = torch.float16

        if self.mode == 'train':
<<<<<<< HEAD
            self.dutils = DatasetUtils(self.params.dataset_path)
        elif self.mode == 'infer':
            self.params.dataset_path = "/home/bumblebee/work/diffrobot/diffrobot/diffusion_policy/runs/"
            self.dutils = DatasetUtils(self.params.dataset_path + saved_run_name)
=======
            self.dutils = DatasetUtils(self.params.dataset_path, 
                                       transformed_affordance=self.params.transformed_affordance,
                                       transformed_ee=self.params.transformed_ee)
        elif self.mode == 'infer':
            self.params.dataset_path = "/home/bumblebee/work/diffrobot/diffrobot/diffusion_policy/runs/"
            self.dutils = DatasetUtils(self.params.dataset_path + saved_run_name, 
                                       transformed_affordance=self.params.transformed_affordance,
                                       transformed_ee=self.params.transformed_ee)
>>>>>>> origin/main

        print('Using {} action frame'.format(self.params.action_frame))

        # create network object
        self.noise_pred_net = ConditionalUnet1D(
                        input_dim=self.params.action_dim,
                        global_cond_dim=self.params.global_cond_dim,
                        diffusion_step_embed_dim=self.params.diffusion_step_embed_dim,
                        down_dims=self.params.down_dims,
                        )
        
        # create tactile encoder
        # self.tactile_encoder = TactileEncoder(
        #     out_dim=self.params.tactile_dim,
        # )

        # the final arch has 2 parts
        self.nets = nn.ModuleDict({
        # 'tactile_encoder': self.tactile_encoder,
        'noise_pred_net': self.noise_pred_net
        })

        if policy_type == 'vision':
            self.vision_encoder_front = get_resnet('resnet18')
            self.vision_encoder_front = replace_bn_with_gn(self.vision_encoder_front)

            self.vision_encoder_hand = get_resnet('resnet18')
            self.vision_encoder_hand = replace_bn_with_gn(self.vision_encoder_hand)

            self.nets['vision_encoder_hand'] = self.vision_encoder_hand
            self.nets['vision_encoder_front'] = self.vision_encoder_front
        
        self.device = torch.device('cuda')
        _ = self.nets.to(self.device)

        # half precision
        if self.precision == torch.float16:
            self.nets.half()
        
        self.ema = EMAModel(
            parameters=self.nets.parameters(),
            power=0.75)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.params.num_diffusion_iters,
            # the choice of beta schedule has big impact on performance
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # network predicts noise (instead of denoised action)
            prediction_type='epsilon'
            )

        # self.noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=self.params.num_diffusion_iters,
        #     # the choice of beta schedule has big impact on performance
        #     beta_schedule='squaredcos_cap_v2',
        #     # clip output to [-1,1] to improve stability
        #     clip_sample=True,
        #     # network predicts noise (instead of denoised action)
        #     prediction_type='epsilon'
        #     )

        # self.noise_scheduler = DPMSolverMultistepScheduler(
        #     num_train_timesteps=self.params.num_diffusion_iters,
        #     beta_start=0.0001,
        #     beta_end=0.02,
        #     beta_schedule="squaredcos_cap_v2",
        #     trained_betas=None,
        #     solver_order=2,
        #     prediction_type="epsilon",
        #     thresholding=False,
        #     dynamic_thresholding_ratio=0.995,
        #     sample_max_value=1.0,
        #     algorithm_type="sde-dpmsolver++",
        #     solver_type="midpoint",
        #     lower_order_final=True,
        #     use_karras_sigmas=True,
        #     lambda_min_clipped= -float("inf"),
        #     variance_type=None,
        #     timestep_spacing="linspace",
        #     steps_offset=0,
        #     )
        
        
        if mode == 'train':

            wandb.init(project="diffusion_policy_runs")
            run_name = wandb.run.name

            if finetune is True:
                self.load_weights(saved_run_name)
                wandb.run.name = saved_run_name + '_' + '_finetune'
            else:
                wandb.run.name = run_name + '_' + policy_type
    
            self.fpath = 'runs/' + wandb.run.name
            os.makedirs(self.fpath, exist_ok=True)
            os.makedirs(self.fpath + '/saved_weights', exist_ok=True)
            os.makedirs(self.fpath + '/saved_weights/best', exist_ok=True)
            # copy params file to run folder
            os.system('cp ' + 'configs/' + config_file + '.py' + ' ' + self.fpath)

            # copy transforms folder to run folder
            os.system('cp -r ' + self.params.dataset_path + '/transforms ' + self.fpath)

            self.last_best_loss = 10000


            if policy_type == 'vision':
                # get transform values from self.params yaml
                self.transform = Compose([Resize((self.params.im_height, self.params.im_width)), 
                                        RandomCrop((self.params.crop_height, self.params.crop_width))])

            # DatasetClass = PushTImageDataset if policy_type == 'vision' else PushTStateDataset
            DatasetClass = DiffusionImageDataset if policy_type == 'vision' else DiffusionStateDataset
            
            dataset_params = {"dataset_path": self.params.dataset_path,
                            "pred_horizon": self.params.pred_horizon,
                            "obs_horizon": self.params.obs_horizon,
                            "action_horizon": self.params.action_horizon,
                            "symmetric": self.params.symmetric,
                            "freq_divisor": self.params.freq_divisor,
<<<<<<< HEAD
                            "transformed_ee": self.params.transformed_ee,
                            "transformed_affordance": self.params.transformed_affordance,
=======
>>>>>>> origin/main
                            "action_frame": self.params.action_frame,}
            
            self.train_dataset = DatasetClass(stage='train', **dataset_params, 
                                              transform=self.transform if policy_type == 'vision' else None)
            
            self.val_dataset = DatasetClass(stage='val', **dataset_params, 
                                            transform=self.transform if policy_type == 'vision' else None)

            self.stats = self.train_dataset.stats
            np.save(self.fpath + '/stats.npy', self.stats)

            # create dataloader
            self.train_dataloader = torch.utils.data.DataLoader(
                                    self.train_dataset,
                                    batch_size=self.params.batch_size,
                                    num_workers=self.params.num_workers,
                                    shuffle=True,
                                    pin_memory=True,
                                    persistent_workers=True
                                    )

            self.val_dataloader = torch.utils.data.DataLoader(
                                    self.val_dataset,
                                    batch_size=self.params.batch_size,
                                    num_workers=self.params.num_workers,
                                    shuffle=False,
                                    pin_memory=True,
                                    persistent_workers=True
                                    )   
            
            # Standard ADAM optimizer
            # Note that EMA parametesr are not optimized
            self.optimizer = torch.optim.AdamW(
                params=self.nets.parameters(),
                lr=self.params.lr, weight_decay=self.params.weight_decay)

            # Cosine LR schedule with linear warmup
            self.lr_scheduler = get_scheduler(
                name=self.params.lr_scheduler_profile,
                optimizer=self.optimizer,
                num_warmup_steps=self.params.num_warmup_steps,
                num_training_steps=len(self.train_dataloader) * self.params.num_epochs
                )
            
        if mode == 'infer':
            self.load_weights(saved_run_name)
            stats_path = os.path.join("/mnt/droplet/", saved_run_name, 'stats.npy')
            # stats_path = os.path.join("../diffusion_policy/runs/", saved_run_name, 'stats.npy')
            self.stats = np.load(stats_path, allow_pickle=True).item()
            if policy_type == 'vision':
                self.transform = Compose([Resize((self.params.im_width, self.params.im_height)), FixedCropTransform(10, 10, 288, 216)])

            print('Inference Mode.')
                    
    def load_weights(self, saved_run_name, load_best=True):
        self.ema_nets = copy.deepcopy(self.nets)

        if load_best:
            fpath_ema = os.path.join("/mnt/droplet/", saved_run_name, "saved_weights", 'best', 'ema.ckpt')
            fpath_nets = os.path.join("/mnt/droplet/", saved_run_name, "saved_weights", 'best', 'net.ckpt')
        else:
            fpath_ema = os.path.join("/mnt/droplet/", saved_run_name, "saved_weights", 'ema.ckpt')
            fpath_nets = os.path.join("/mnt/droplet/", saved_run_name, "saved_weights", 'net.ckpt')

        # if load_best:
        #     fpath_ema = os.path.join("../diffusion_policy/runs/", saved_run_name, "saved_weights", 'best', 'ema.ckpt')
        #     fpath_nets = os.path.join("../diffusion_policy/runs/", saved_run_name, "saved_weights", 'best', 'net.ckpt')
        # else:
        #     fpath_ema = os.path.join("/runs/", saved_run_name, "saved_weights", 'ema.ckpt')
        #     fpath_nets = os.path.join("/runs/", saved_run_name, "saved_weights", 'net.ckpt')

            

        state_dict_nets = torch.load(fpath_nets, map_location='cuda')
        self.nets.load_state_dict(state_dict_nets)
        state_dict_ema = torch.load(fpath_ema, map_location='cuda')
        self.ema_nets.load_state_dict(state_dict_ema)

        if self.precision == torch.float16:
            self.nets.half()
            self.ema_nets.half()

        self.ema = EMAModel(parameters=self.ema_nets.parameters(), power=0.75)

        print('Pretrained weights loaded.')


    def process_batch_state(self, nbatch):

        nstate = nbatch['state'].to(self.device, dtype=self.precision)
        naction = nbatch['action'].to(self.device, dtype=self.precision)
        # ntactile_0 = nbatch['tactile_0'].to(self.device, dtype=self.precision)
        # ntactile_1 = nbatch['tactile_1'].to(self.device, dtype=self.precision)

        # process tactile data
        # tactile_features = self.nets['tactile_encoder'](ntactile_1.flatten(end_dim=1)).reshape(*ntactile_1.shape[:2], -1)

        # obs_cond = torch.cat([nstate, tactile_features], dim=-1)
        obs_cond = nstate
        obs_cond = obs_cond.flatten(start_dim=1)
        
        return obs_cond, naction
    
    
    def process_batch_vision(self, nbatch):

        nimage_hand = nbatch['image_hand'].to(self.device)
        nimage_front = nbatch['image_front'].to(self.device)
        nagent_pos = nbatch['robot_state'].to(self.device)
        naction = nbatch['action'].to(self.device, dtype=self.precision)

        image_features_hand = self.nets['vision_encoder_hand'](nimage_hand.flatten(end_dim=1)).reshape(*nimage_hand.shape[:2], -1)
        image_features_front = self.nets['vision_encoder_front'](nimage_front.flatten(end_dim=1)).reshape(*nimage_front.shape[:2], -1)

        image_features = torch.cat([image_features_hand, image_features_front], dim=-1)
        obs_features = torch.cat([image_features, nagent_pos], dim=-1)
        obs_cond = obs_features.flatten(start_dim=1).to(dtype=self.precision)

        return obs_cond, naction


    def train_policy(self):
        # train loop
        for epoch in tqdm(range(self.params.num_epochs), desc='Epoch'):
            for nbatch in self.train_dataloader:
                
                if self.policy_type == 'vision':
                    obs_cond, naction = self.process_batch_vision(nbatch)
                elif self.policy_type == 'state':
                    obs_cond, naction = self.process_batch_state(nbatch)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=self.device, dtype=self.precision)

                # sample a diffusion iteration for each data point
                B = naction.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                # add noise to the clean actions according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(
                    naction, noise, timesteps)
                                                
                # predict the noise residual
                noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                wandb.log({"train_loss": loss})

                # optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # step lr scheduler every batch
                self.lr_scheduler.step()
                # update Exponential Moving Average of the model weights
                self.ema.step(self.nets.parameters())

                # log learning rate
                wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr']})

            # save weights
            self.ema_nets = copy.deepcopy(self.nets)
            self.ema.copy_to(self.ema_nets.parameters())
            torch.save(self.nets.state_dict(), f'{self.fpath}/saved_weights/net.ckpt')
            torch.save(self.ema_nets.state_dict(), f'{self.fpath}/saved_weights/ema.ckpt')

            if epoch % 10 == 0:
                self.validate_policy()
    

    def validate_policy(self):
    
        with torch.no_grad():
            val_loss = list()
            for nbatch in self.val_dataloader:
            
                if self.policy_type == 'vision':
                    obs_cond, naction = self.process_batch_vision(nbatch)
                elif self.policy_type == 'state':
                    obs_cond, naction = self.process_batch_state(nbatch)

                noise = torch.randn(naction.shape, device=self.device, dtype=self.precision)
                B = naction.shape[0]

                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                noisy_actions = self.noise_scheduler.add_noise(
                    naction, noise, timesteps)
                
                noise_pred = self.ema_nets['noise_pred_net'](
                    noisy_actions, timesteps, global_cond=obs_cond)

                loss = nn.functional.mse_loss(noise_pred, noise)
                val_loss.append(loss.item())

            wandb.log({"val_loss": np.mean(val_loss)})

            if np.mean(val_loss) < self.last_best_loss:
                # save weights
                self.ema_nets = copy.deepcopy(self.nets)
                self.ema.copy_to(self.ema_nets.parameters())
                torch.save(self.nets.state_dict(), f'{self.fpath}/saved_weights/best/net.ckpt')
                torch.save(self.ema_nets.state_dict(), f'{self.fpath}/saved_weights/best/ema.ckpt')
                self.last_best_loss = np.mean(val_loss)
                print('Saved best model!')



    def process_inference_state(self, obs_deque):


        # X_OE = [np.dot(np.linalg.inv(o['X_BO']), o['X_BE']) for o in obs_deque]

        X_BO = obs_deque[0]['X_BO']
        X_B_OO = obs_deque[0]['X_B_OO']
        

        gripper_state = [o['gripper_state'] for o in obs_deque]
        phase = [o['phase'] for o in obs_deque]
        progress = [o['progress'] for o in obs_deque]

        if self.params.action_frame == 'object_centric':
            ee_pose = [np.dot(np.linalg.inv(X_B_OO), o['X_BE']) for o in obs_deque] # X_OO_E
            # ee_pose = [np.dot(np.linalg.inv(X_BO), o['X_BE']) for o in obs_deque] # X_OE
        elif self.params.action_frame == 'global':
            ee_pose = [o['X_BE'] for o in obs_deque]

        ee_pos = [x[:3,3] for x in ee_pose]
        ee_orien = [matrix_to_rotation_6d(x[:3,:3]) for x in ee_pose]

        object_pose = [o['X_OO_O'] for o in obs_deque] # object pose in oriented frame
        object_orien = [matrix_to_rotation_6d(x[:3,:3]) for x in object_pose]

        # normalize data
        nee_pos = self.dutils.normalize_data(ee_pos, stats=self.stats['pos_follower'])
        ngripper_state = self.dutils.normalize_data(gripper_state, stats=self.stats['gripper_state']).reshape(-1, 1)
        nprogress = self.dutils.normalize_data(progress, stats=self.stats['progress']).reshape(-1, 1)

        if not self.params.symmetric:
            robot_state = torch.from_numpy(np.concatenate([nee_pos, ee_orien, object_orien, ngripper_state], axis=-1)).to(self.device, dtype=self.precision)
        else:
            robot_state = torch.from_numpy(np.concatenate([nee_pos, ee_orien, ngripper_state], axis=-1)).to(self.device, dtype=self.precision)
        
        obs_cond = robot_state
        obs_cond = obs_cond.flatten(start_dim=0).unsqueeze(0)
                
        return obs_cond
    
    
    def process_inference_vision(self, obs_deque):
        image_front = np.stack([x['image_front'] for x in obs_deque])
        image_hand = np.stack([x['image_hand'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

        nagent_poses = self.dutils.normalize_data(agent_poses, stats=self.stats['states'])
        nimage_front = image_front / 255.0
        nimage_hand = image_hand / 255.0

        nagent_poses = torch.from_numpy(nagent_poses).to(self.device, dtype=self.precision)
        nimage_front = torch.from_numpy(nimage_front).to(self.device, dtype=self.precision)
        nimage_hand = torch.from_numpy(nimage_hand).to(self.device, dtype=self.precision)

        nimage_front = nimage_front.permute(0,3,1,2)
        nimage_front = torch.stack([self.transform(img) for img in nimage_front])

        nimage_hand = nimage_hand.permute(0,3,1,2)
        nimage_hand = torch.stack([self.transform(img) for img in nimage_hand])

        image_features_front = self.ema_nets['vision_encoder_front'](nimage_front)
        image_features_hand = self.ema_nets['vision_encoder_hand'](nimage_hand)

        image_features = torch.cat([image_features_hand, image_features_front], dim=-1)
        obs_features = torch.cat([image_features, nagent_poses], dim=-1)
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        
        return obs_cond



    def infer_action(self, obs_deque):

        with torch.no_grad():

            if self.policy_type == 'vision':
                obs_cond = self.process_inference_vision(obs_deque)
            elif self.policy_type == 'state':
                obs_cond = self.process_inference_state(obs_deque)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.params.pred_horizon, self.params.action_dim), device=self.device, dtype=self.precision)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.params.num_diffusion_iters)
            # self.noise_scheduler.set_timesteps(20)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
        
        naction = naction.detach().to('cpu').numpy()[0] # X_OE

        # extract components
        action_pos = naction[:,:3]
        action_orien = naction[:,3:9]
        action_gripper = naction[:,9]
        action_progress = naction[:,10]

        # unnormalize action
        action_pos = self.dutils.unnormalize_data(action_pos, stats=self.stats['pos_leader'])
        action_progress = self.dutils.unnormalize_data(action_progress, stats=self.stats['progress'])
        action_gripper = self.dutils.unnormalize_data(action_gripper, stats=self.stats['gripper_action'])

        # convert orientation to rotation matrix
        action_orien = [rotation_6d_to_matrix(torch.FloatTensor(x)) for x in action_orien]

        # create a 4x4 transform matrix from action_pos (3 dim) and action_orien (3x3 dim)

        # create a list if 4x4 identity matrices
        X_OE_mtx = [np.eye(4) for _ in range(self.params.pred_horizon)]
        for i in range(self.params.pred_horizon):
            X_OE_mtx[i][:3, :3] = action_orien[i]
            X_OE_mtx[i][:3, 3] = action_pos[i]

        
        # only take action_horizon number of actions
        start = self.params.obs_horizon - 1
        end = start + self.params.action_horizon
        action = X_OE_mtx[start:end]
        progress = action_progress[start:end]
        
        if self.params.action_frame == 'object_centric':
            # The action is a series of points in the object frame
            # Convert each one to a [x,y,z] point in robot frame

            X_BO = obs_deque[0]['X_BO']
            X_B_OO = obs_deque[0]['X_B_OO'].A
            # X_BE = np.array([np.dot(X_BO, X_OE) for X_OE in action])


            X_BE = [X_B_OO @ X_OOE for X_OOE in action]
            # X_BE = [X_BO @ X_OE for X_OE in action]

            # X_BE = action

        elif self.params.action_frame == 'global':
            X_BE = action

        return {'action': X_BE, 
                'action_gripper': action_gripper,
                'progress': progress}



if __name__ == '__main__':

    run_info = input('Whats special about this run? ')
    policy = DiffusionPolicy(mode='train', 
                             policy_type='state', 
                             config_file='config_state_pretrain', 
                             finetune=False, 
                             saved_run_name=None)
    
    wandb.run.notes = run_info
    
    policy.train_policy()
    # policy.infer_action(obs_deque)
