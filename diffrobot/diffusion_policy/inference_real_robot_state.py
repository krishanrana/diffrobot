import numpy as np
import torch
import torch.nn as nn
from diffrobot.realsense.multi_realsense import MultiRealsense
from diffrobot.robot.robot import Robot, to_affine, pos_orn_to_matrix
from frankx import Affine, JointMotion, Waypoint, WaypointMotion, PathMotion


from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from diffrobot.diffusion_policy.vision_encoder import get_resnet, replace_bn_with_gn
from diffrobot.diffusion_policy.network import ConditionalUnet1D
from diffrobot.diffusion_policy.dataset_franka_real import normalize_data, unnormalize_data
# import wandb
import os
import collections
import cv2
import pdb
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib
import time
from torchvision.transforms import Compose, Resize
from torchvision.transforms.functional import crop
from diffrobot.realsense.multi_camera_visualizer import MultiCameraVisualizer

import reactivex as rx
from reactivex import operators as ops

from multiprocessing.managers import SharedMemoryManager
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco
from diffrobot.realsense.single_realsense import SingleRealsense




class FixedCropTransform:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return crop(img, self.top, self.left, self.height, self.width)



# agent_pos is 2 dimensional
lowdim_obs_dim = 2
# observation feature has 514 dims in total per step
obs_dim = 6
action_dim = 2

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 10

# num_diffusion_iters = 100
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon')

device = torch.device('cuda')


noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=6
)

ckpt_path = "saved_weights/nets_reacher_state_best.ckpt"
state_dict = torch.load(ckpt_path, map_location='cuda')


_ = noise_pred_net.to(device)

ema_nets = noise_pred_net
ema_nets.load_state_dict(state_dict)
print('Pretrained weights loaded.')

#read stats pkl
stats_path = "saved_weights/stats.pkl"
with open(stats_path, 'rb') as f:
    stats = pkl.load(f)


#@markdown ### **Inference**

# limit enviornment interaction to 200 steps before termination
max_steps = 2000


################################################################################


# Setup cameras
record_fps = 30
sh = SharedMemoryManager()
sh.start()
cam = SingleRealsense(sh, "035122250692")
cam.start()
marker_detector = ArucoDetector(cam, 0.05, aruco.DICT_4X4_50, 8, visualize=False)

cam.set_exposure(exposure=100, gain=60)


time.sleep(1.0)

# vis = MultiCameraVisualizer(cams, row=2, col=1)
# vis.start()
# time.sleep(1.0)


print("Camera setup complete")
# Setup robot
panda = Robot("172.16.0.2")
panda.set_dynamic_rel(0.4, accel_rel=0.005, jerk_rel=0.005)

panda.move_to_joints([-1.56832675,  0.39303148,  0.02632776, -1.98690212, -0.00319773,  2.35042797, 0.94667396])

pose = panda.get_tcp_pose()
trans = pose[:3, 3]
z_height = trans[2]
orien = panda.get_orientation()
# motion = panda.start_cartesian_controller()

################################################################################



def get_obs():
    # print("Getting observation")
    pose = panda.get_tcp_pose()
    goal = marker_detector.estimate_pose()
    # read images from realsense and scale down
    # read position of the robot ee
    # save as dict
    return {'goal': goal[:2, 3],
            'agent_pos': pose[:2, 3]}

    
# Discard first frames
get_obs()


obs_deque = collections.deque(maxlen=obs_horizon)
obs_deque.append(get_obs())
time.sleep(0.2)
obs_deque.append(get_obs())
obs_stream = rx.interval(1.0/10.0, scheduler=rx.scheduler.NewThreadScheduler()) \
    .pipe(ops.map(lambda _: get_obs())) \
    .pipe(ops.share()) \
    .subscribe(lambda x: obs_deque.append(x))
    

while True:
    # get first observation
    # obs = get_obs()

    # keep a queue of last 2 steps of observations
    # obs_deque = collections.deque(
    #     [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0

    while not done:
        B = 1
        # stack the last obs_horizon number of observations
        # image_top = np.stack([x['image_top'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
        goal = obs_deque[-1]['goal']
        # normalize observation
        nagent_poses = normalize_data(agent_poses, stats=stats['states'])
        ngoal = normalize_data(goal, stats=stats['goals'])
 
        # device transfer
        nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
        ngoal = torch.from_numpy(ngoal).to(device, dtype=torch.float32)
  

        # infer action
        with torch.no_grad():

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nagent_poses.unsqueeze(0).flatten(start_dim=1)

            # import pdb; pdb.set_trace() 
            obs_cond =torch.cat([ngoal.unsqueeze(dim=0), obs_cond], dim=-1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                # noise_pred = ema_nets['noise_pred_net'](
                #     sample=naction,
                #     timestep=k,
                #     global_cond=obs_cond
                # )
                noise_pred = ema_nets(
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
        action_pred = unnormalize_data(naction, stats=stats['actions'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning

        # create waypoints from the action
        # action = [(0.0, -0.4), (0.0, -0.41), (0.0, -0.42), (0.3, -0.43), (0.0, -0.44), (0.0, -0.45)]
        points = []
        for i in action:
            tp = [i[0], i[1], trans[2]]
            points.append(tp)
        points = np.array(points)
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        # check if last point is clsoe to the first point
        too_close = np.linalg.norm(points[0] - points[-1]) < 0.005
        if too_close:
            points = points[:-1]
        else:
            points = points[np.where(dists > 0.005)]

        waypoints = []
        for point in points:
            waypoints.append(to_affine(point, orien))
        # if too_close:
        #     print("Waypoints are close. Using waypoint motion")
        #     panda.recover_from_errors()
        #     panda.waypoints(waypoints)
        # else:
        panda.recover_from_errors()
        if len(waypoints) > 2:
            print("Waypoints are far. Using path motion")
            panda.path(waypoints, blend=0.4)
        elif len(waypoints) >= 1:
            print("Waypoints are close. Using waypoint motion")
            panda.waypoints(waypoints)


        # robot_motion = WaypointMotion([Waypoint(x, 0.0, Waypoint.Absolute) for x in waypoints])

        # obs = get_obs()

        # for i in range(len(action)):
          
        #     tp = [action[i][0], action[i][1], trans[2]]
        #     waypoint = to_affine(tp, orien)
        #     robot_motion = PathMotion([Waypoint(waypoint, 0.0, Waypoint.Absolute)])
        #     panda.frankx.move(robot_motion)
        #     obs = get_obs()
        #     print("Action: ", action[i])

        #     obs_deque.append(obs)

        #     step_idx += 1
        #     if step_idx > max_steps:
        #         done = True
        #     if done:
        #         break

# print out the maximum target coverage
print('Score: ', max(rewards))

# # visualize
# from IPython.display import Video
# vwrite('vis.mp4', imgs)
# Video('vis.mp4', embed=True, width=256, height=256)