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
from diffrobot.diffusion_policy.utils.dataset_utils import normalize_data, unnormalize_data
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
from diffrobot.diffusion_policy.diffusion_policy import DiffusionPolicy


# empty cache
torch.cuda.empty_cache()

policy = DiffusionPolicy(mode='infer', 
                        policy_type='state', 
                        config_file='config_state_pretrain', 
                        finetune=False, 
                        saved_run_name='distinctive-star-4_state')


# Setup cameras
record_fps = 30
sh = SharedMemoryManager()
sh.start()
cam = SingleRealsense(sh, "035122250692")
cam.start()
marker_detector = ArucoDetector(cam, 0.05, aruco.DICT_4X4_50, 8, visualize=False)

cam.set_exposure(exposure=100, gain=60)
time.sleep(1.0)


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

def get_marker():
    res =  marker_detector.estimate_pose()
    if res is None:
        return None
    else:
        return res[:3, 3]
    
def get_obs():
    # print("Getting observation")
    pose = panda.get_tcp_pose()
    # goal = marker_detector.estimate_pose()
    return pose[:3, 3]

    return {'goal': goal[:3, 3],
            'agent_pos': pose[:3, 3]}

marker_stream = rx.interval(1.0/10.0, scheduler=rx.scheduler.NewThreadScheduler()) \
    .pipe(ops.map(lambda _: get_marker())) \
    .pipe(ops.filter(lambda x: x is not None))
    
# Discard first frames
# get_obs()


obs_deque = collections.deque(maxlen=policy.params.obs_horizon)
# obs_deque.append(get_obs())
# time.sleep(0.2)
# obs_deque.append(get_obs())
obs_stream = rx.interval(1.0/10.0, scheduler=rx.scheduler.NewThreadScheduler()) \
    .pipe(ops.map(lambda _: get_obs())) \
    .pipe(ops.share()) 


combined_stream = rx.combine_latest(marker_stream, obs_stream) \
    .pipe(ops.map(lambda x: {"goal": x[0], "agent_pos": x[1]})) \
    .subscribe(lambda x: obs_deque.append(x))
    

while True:
    done = False
    step_idx = 0

    while not done:

        # wait for obs_deque to have len 2
        while len(obs_deque) < 2:
            time.sleep(0.1)
            print("Waiting for observation")

        # print(obs_deque)
        action = policy.infer_action(obs_deque)

        points = []
        for i in action:
            tp = [i[0], i[1], i[2]]
            points.append(tp)
        
        points = np.array(points)
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        # # check if last point is clsoe to the first point
        #threshold = 0.005
        #too_close = np.linalg.norm(points[0] - points[-1]) < threshold
        #if too_close:
        #    points = points[:-1]
        #else:
        #    points = points[np.where(dists > threshold)]

        waypoints = []
        for point in points:
            waypoints.append(to_affine(point, orien))

        panda.recover_from_errors()
        #if len(waypoints) > 2:
        #    print("Waypoints are far. Using path motion")
        #    panda.path(waypoints, blend=1.2)
        #elif len(waypoints) >= 1:
        #    print("Waypoints are close. Using waypoint motion")
        panda.waypoints(waypoints[-1:])

