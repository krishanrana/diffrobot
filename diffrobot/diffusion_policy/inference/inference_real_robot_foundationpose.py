import numpy as np
import time
import json
import collections
import reactivex as rx
from reactivex import operators as ops
from typing import Optional
from dataclasses import dataclass
# from multiprocessing.managers import SharedMemoryManager
from diffrobot.realsense.multi_realsense import MultiRealsense

import torch
import cv2
# Import necessary modules from your libraries
# from diffrobot.realsense.single_realsense import SingleRealsense
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco
from diffrobot.tactile_sensors.xela import SensorSocket
from diffrobot.diffusion_policy.diffusion_policy import DiffusionPolicy
from diffrobot.robot.robot import Robot, to_affine, matrix_to_pos_orn
from diffrobot.diffusion_policy.utils.config_utils import get_config
from diffrobot.robot.visualizer import RobotViz
from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils, adjust_orientation_to_z_up, compute_oriented_affordance_frame
from diffrobot.pose_extraction.FoundationPose.multi_object_tracker import MultiObjectTracker

import pdb
import spatialmath as sm
import gc

import threading
import queue

def clear_cuda_memory():
    # Clear cache
    torch.cuda.empty_cache()
    # Force garbage collection
    gc.collect()


class DataLogger:
    def __init__(self, path: str):
        self.path = path
        self.data_queue = queue.Queue()
        self.rgb = np.zeros((2000, 480, 640, 3), dtype=np.uint8)
        self.depth = np.zeros((2000, 480, 640), dtype=np.uint16)
        self.poses = []
        self.actions = []
        self.stop_logging = threading.Event()
        self.logging_thread = threading.Thread(target=self._save_data_thread)
        self.logging_thread.start()

    def log_data(self, data):
        self.data_queue.put(data)
    
    def _save_data_thread(self):
        while not self.stop_logging.is_set():
            try:
                data = self.data_queue.get(timeout=1)
                self._save_data(data)
            except queue.Empty:
                continue

    def _save_data(self, data):
        frame_idx = len(self.poses)
        self.rgb[frame_idx] = data['rgb']
        self.depth[frame_idx] = data['depth']
        self.poses.append(data['pose'])
        self.actions.append(data['action'])
    
    def save_data(self):
        self.stop_logging.set()
        self.logging_thread.join()
        np.savez(self.path, rgb=self.rgb, depth=self.depth, poses=self.poses, actions=self.actions)


@dataclass
class ManipObject:
    name: str
    aruco_key: int
    X_BO_reference: Optional[np.ndarray] = None
    X_BO_last_seen: Optional[np.ndarray] = None

    def copy_X_BO_to_reference(self):
        self.X_BO_reference = self.X_BO_last_seen

class Task:
    def __init__(self, 
                 name, 
                 affordance_frame: str,
                 oriented_frame_reference: str,
                 secondary_affordance_frame: str = "",
                 policy_name: str = "",
                 progress_threshold: float = 0.98,
                 transform_affordance_frame: bool = False,
                 transform_ee_frame: bool = False):
        
        self.name = name
        self.policy_name = policy_name
        self.objects = {}
        self.max_progress_made = 0.0
        self.progress = 0.0
        self.affordance_frame = affordance_frame
        self.secondary_affordance_frame = secondary_affordance_frame
        self.oriented_frame_reference = oriented_frame_reference
        self.gripper_allowed_to_move = True
        self.progress_threshold = progress_threshold
        self.transform_affordance_frame = transform_affordance_frame
        self.affordance_transform = None
        self.transform_ee_frame = transform_ee_frame
        self.ee_transform = None
        self.X_EA = None

        if self.transform_affordance_frame:
            run_path = f'/mnt/droplet/{self.policy_name}/transforms/affordance_transform.json'
            with open(run_path, 'r') as f:
                self.affordance_transform = json.load(f)['X_OA']
        
        if self.transform_ee_frame:
            run_path = f'/mnt/droplet/{self.policy_name}/transforms/ee_transform.json'
            with open(run_path, 'r') as f:
                self.ee_transform = json.load(f)['X_OA']
    
    def set_progress(self, progress: float):
        self.progress = progress
        self.max_progress_made = max(self.max_progress_made, progress)

    
    def print_progress(self):
        print(f"Task {self.name} Progress: {self.progress:.2f} (Max Progress: {self.max_progress_made:.2f}/{self.progress_threshold:.2f})")

    def is_phase_finished(self, progress: float) -> bool:
        if progress >= self.progress_threshold:
            return True
        else:
            return False
    
    def new_detection_made(self):
        for obj in self.objects.values():
            obj.copy_X_BO_to_reference()



class MakeTeaFullTask(Task):
    def __init__(self, teacup:ManipObject, saucer:ManipObject, teapot:ManipObject, teaspoon:ManipObject, **kwargs):
        super().__init__('make_tea', **kwargs)
        self.objects = {
            'teacup': teacup,
            'sauce': saucer,
            'teapot': teapot,
            'teaspoon': teaspoon,
        }


class TeacupRotate(Task):
    def __init__(self, teacup:ManipObject, **kwargs):
        super().__init__('teacup_rotate', **kwargs)
        self.objects = {
            'teacup': teacup,
        }
    

class PlaceSaucer(Task):
    def __init__(self, saucer:ManipObject, teacup:ManipObject, **kwargs):
        super().__init__('place_saucer', **kwargs)
        self.objects = {
            'saucer': saucer,
            'teacup': teacup,
        }


class TeapotRotate(Task):
    def __init__(self, teacup: ManipObject, teapot: ManipObject, **kwargs):
        super().__init__('teapot_rotate', **kwargs)
        self.objects = {
            'teacup': teacup,
            'teapot': teapot,
        }


class TeapotPour(Task):
    def __init__(self, teacup: ManipObject, **kwargs):
        super().__init__('teapot_pour', **kwargs)
        self.objects = {
            'teacup': teacup,
        }


class TeapotPlace(Task):
    def __init__(self, teacup: ManipObject, teapot: ManipObject, **kwargs):
        super().__init__('teapot_place', **kwargs)
        self.objects = {
            'teacup': teacup,
            'teapot': teapot
        }

class PickTeaspoon(Task):
    def __init__(self, teaspoon: ManipObject, **kwargs):
        super().__init__('pick_teaspoon', **kwargs)
        self.objects = {
            'teaspoon': teaspoon,
        }
        

class StirTeaspoon(Task):
    def __init__(self, teacup: ManipObject, **kwargs):
        super().__init__('stir_teaspoon', **kwargs)
        self.objects = {
            'teacup': teacup,
        }

        
class PerceptionSystem:
    def __init__(self, robot):
        self.X_BC = self.get_camera_pose(robot, load_transform=True)
        objects = ["teacup" ,"saucer", "teapot", "teaspoon"]
        self.tracker = MultiObjectTracker(objects)
    
    def start(self):
        self.tracker.register_objects()
    
    def stop(self):
        self.tracker.pipeline.stop()
        cv2.destroyAllWindows()

    def get_camera_pose(self, robot, load_transform: bool = False):
        # returns the external camera pose in the robot base frame

        if load_transform:
            with open(f'../../calibration/calibration_data/static_front_cam.json', 'r') as f:
                X_BC_f = json.load(f)['X_BC']
                return np.array(X_BC_f)


    

def create_robot(ip:str = "172.16.0.2", dynamic_rel: float=0.4):
    panda = Robot(ip)
    panda.gripper.open()
    panda.set_dynamic_rel(dynamic_rel, accel_rel=0.2, jerk_rel=0.05)
    panda.frankx.set_collision_behavior(
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
    )
    return panda

class MakeTeaTask:
    def __init__(self, perception_system: PerceptionSystem, robot: Robot, vis: bool = False):
        self.perception_system = perception_system
        self.vis = vis
        self.robot = robot
        self.objects = {
            'base': ManipObject(name='base', aruco_key=None, X_BO_last_seen=np.eye(4), X_BO_reference=np.eye(4)), # 'base' is the robot's base frame
            'teacup': ManipObject(name='teacup', aruco_key=3),
            'saucer': ManipObject(name='saucer', aruco_key=10),
            'teapot': ManipObject(name='teapot', aruco_key=4),
            'teaspoon': ManipObject(name='teaspoon', aruco_key=8),
        }       
        self.sub_tasks : list[Task] = [
            TeacupRotate(
                policy_name = 'dry-sea-235_state',
                #policy_name= 'grateful-lion-168_state',  #'colorful-fire-216_state' ,#'dark-night-177_state', #'grateful-lion-168_state',
                oriented_frame_reference='base', 
                progress_threshold= 0.88, #0.94,
                affordance_frame='teacup', 
                teacup=self.objects['teacup']),
            PlaceSaucer(
                policy_name= 'fancy-glade-228_state',
                #policy_name= 'cosmic-universe-169_state', #'cosmic-universe-169_state', #'rich-brook-184_state',#'cosmic-universe-169_state', #'hopeful-tree-173_state',
                oriented_frame_reference='teacup', 
                progress_threshold=0.83,
                affordance_frame='saucer',
                secondary_affordance_frame='teacup',
                transform_ee_frame=False, 
                saucer=self.objects['saucer'],
                teacup=self.objects['teacup']),
            TeapotRotate(
                policy_name= 'dry-sky-157_state', #'morning-moon-217_state' ,#'noble-water-180_state',#'dry-sky-157_state',
                oriented_frame_reference='base', 
                affordance_frame='teapot',
                progress_threshold=0.92, 
                teacup=self.objects['teacup'], 
                teapot=self.objects['teapot']),
            TeapotPour(
                policy_name= 'dainty-bird-158_state', #'comic-pond-207_state', #'genial-night-181_state',#'dainty-bird-158_state',
                oriented_frame_reference='teapot',
                progress_threshold=0.84, 
                affordance_frame='teacup', 
                teacup=self.objects['teacup']),
            TeapotPlace(
                oriented_frame_reference='teapot',
                secondary_affordance_frame='teapot',
                policy_name= 'curious-music-239_state', # 'good-sponge-208_state',#'hopeful-flower-182_state', #'twilight-microwave-178_state', #'pious-water-167_state', #'twilight-dawn-164_state',
                progress_threshold=0.80,
                affordance_frame='teacup', 
                teacup=self.objects['teacup'],
                teapot=self.objects['teapot'],
                transform_ee_frame=False),
            PickTeaspoon(
                oriented_frame_reference='base', 
                # policy_name= 'lively-haze-162_state',#'stilted-aardvark-209_state', #'charmed-tree-186_state',#'lively-haze-162_state',
                policy_name= 'eager-frost-240_state', #'cerulean-donkey-214_state', #'stilted-aardvark-209_state', #'charmed-tree-186_state',#'lively-haze-162_state',
                progress_threshold=0.90,
                affordance_frame='teaspoon', 
                teaspoon=self.objects['teaspoon']),
            # StirTeaspoon(
            #     oriented_frame_reference='teacup',
            #     policy_name= 'usual-snow-165_state', #'vital-moon-212_state', #'helpful-waterfall-210_state',#'jumping-water-185_state' ,#'usual-snow-165_state',
            #     progress_threshold=0.92,
            #     affordance_frame='teacup', 
            #     teacup=self.objects['teacup'],
            #     transform_affordance_frame=True),
        ]

        # FIND LAST POLICY LOADED
        self.last_policy_idx = -1
        for task in self.sub_tasks:
            if task.policy_name != "":
                self.last_policy_idx += 1
            else:
                break
        self.phase = 0

        # initialize last seen objetc poses
        for obj in self.perception_system.tracker.pose_dict.keys():
            X_CO = self.perception_system.tracker.pose_dict[obj]
            self.objects[obj].X_BO_last_seen = self.perception_system.X_BC @ X_CO



    def check_if_is_last_phase(self):
        return self.phase == self.last_policy_idx

    def go_to_next_phase(self):
        if self.phase < self.last_policy_idx:
            self.phase += 1
        else:
            raise Exception("Tried to load a policy that does not exist")

    def current_policy_name(self):
        name =  self.sub_tasks[self.phase].policy_name 
        assert name != "", f"Warning: Policy name for phase {self.phase} not found"
        return name

    def current_task(self):
        return self.sub_tasks[self.phase]
    
    def current_affordance_frame_pose(self):
        obj_name = self.current_task().affordance_frame
        assert obj_name in self.objects, f"Object {obj_name} not found in objects"
        return self.objects[obj_name].X_BO_last_seen
    
    def current_secondary_affordance_frame_pose(self):
        obj_name = self.current_task().secondary_affordance_frame
        assert obj_name in self.objects, f"Object {obj_name} not found in objects"
        return self.objects[obj_name].X_BO_last_seen
    
    def current_oriented_frame_reference(self):
        obj = self.current_task().oriented_frame_reference
        assert obj in self.objects, f"Object {obj} not found in objects"
        X_BO = self.objects[obj].X_BO_reference
        assert X_BO is not None, f"Object {obj} has no reference pose"
        return X_BO

    def detect_objects(self, vis=False):
        robot = self.robot
        p = self.perception_system

        obj_list = self.current_task().objects
        im = p.tracker.track_objects_once(obj_list)
        if vis:
            p.tracker.visualize_poses(im)

        for obj in p.tracker.pose_dict.keys():
            X_CO = p.tracker.pose_dict[obj]
            self.objects[obj].X_BO_last_seen = p.X_BC @ X_CO

        self.current_task().new_detection_made()


    
    def move_to_phase_start(self):
        start_object = self.current_task().affordance_frame
        start_object = self.objects[start_object]
        X_BO = start_object.X_BO_last_seen
        angle = np.arctan2(X_BO[1, 3], X_BO[0, 3])
        robot.move_to_joints(np.deg2rad([np.rad2deg(angle), 0, 0, -90, 0, 90, 45]))

    def get_observation(self):
        s = self.robot.get_state()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T
        self.detect_objects(vis=self.vis)

        if self.current_task().transform_ee_frame:
            X_EA = task.current_task().X_EA
            if X_EA is None:
                temp_X_BO = task.current_secondary_affordance_frame_pose()
                temp_X_BO = adjust_orientation_to_z_up(temp_X_BO)
                temp_X_OA = self.current_task().ee_transform
                temp_X_EO = np.linalg.inv(X_BE) @ temp_X_BO
                temp_X_EA = temp_X_EO @ temp_X_OA
                temp_X_EA = sm.SE3(temp_X_EA[:3,3]).A  # extract translation component only
                task.current_task().X_EA = temp_X_EA
            # X_BE = temp_X_BA
            X_EA = task.current_task().X_EA
            X_BE = X_BE @ X_EA

        X_BO = task.current_affordance_frame_pose()
        X_BO = adjust_orientation_to_z_up(X_BO) 

        if task.current_task().affordance_transform is not None:
            X_BO = X_BO @ task.current_task().affordance_transform

        X_B_OO = compute_oriented_affordance_frame(
            X_BO, 
            task.current_oriented_frame_reference()
            )
        

        X_OO_O = np.dot(np.linalg.inv(X_B_OO), X_BO) 
        # self.robot_visualiser.object_pose.T = self.X_BO
        return {"X_BE": X_BE, 
                "X_BO": X_BO,
                "X_BO_teacup": self.objects['teacup'].X_BO_last_seen,
                "X_BO_saucer": self.objects['saucer'].X_BO_last_seen,
                "X_BO_teapot": self.objects['teapot'].X_BO_last_seen,
                "X_BO_teaspoon": self.objects['teaspoon'].X_BO_last_seen,
                "X_B_OO": X_B_OO,
                "X_OO_O": X_OO_O,
                "gripper_state": robot.gripper.width(),
                "progress": task.current_task().progress,
                "phase": task.phase}

    
    def detected_objects(self):
        return [obj for obj in self.objects.values() if obj.X_BO_last_seen is not None]



class RobotInferenceController:
    def __init__(self, 
                perception_system: PerceptionSystem,
                data_logger: DataLogger,
                robot: Robot,
                task: MakeTeaTask,
                ):
        self.robot = robot
        self.perception_system = perception_system
        self.task = task
        self.robot_visualiser = RobotViz()
        self.setup_diffusion_policy()
        self.data_logger = data_logger


    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()

        self.policies = {}
        for t in self.task.sub_tasks:
            if t.policy_name == "":
                print(f"Policy name not found for task {t.name}")
                continue
            policy = DiffusionPolicy(mode='infer',
                                    policy_type='state',
                                    # config_file=f'../runs/{t.policy_name}/config_state_pretrain',
                                    config_file=f'/mnt/droplet/{t.policy_name}/config_state_pretrain',
                                    finetune=False,
                                    saved_run_name=t.policy_name)
            self.policies[t.name] = policy

        # get first policy
        key = list(self.policies.keys())[0]
        policy = self.policies[key]   
        self.obs_horizon = policy.params.obs_horizon
        self.obs_deque = collections.deque(maxlen=policy.params.obs_horizon)

    
    def start_inference(self):
        task = self.task
        robot = self.robot
        obs_stream = rx.interval(1.0/5.0, scheduler=rx.scheduler.NewThreadScheduler()) \
            .pipe(ops.map(lambda _: self.task.get_observation())) \
            .pipe(ops.filter(lambda x: x["X_BO"] is not None)) \
            .subscribe(lambda x: self.obs_deque.append(x))  
      
        motion = robot.start_impedance_controller(830, 40, 1)

        done = False
        while not done:
            # wait for obs_deque to have len 2
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(0.1)
                # print("Waiting for observation")

            current_task = self.task.current_task()
            task_name = current_task.name
            assert task_name in self.policies, f"Policy {task_name} not found"
            policy = self.policies[task_name]

            out = policy.infer_action(self.obs_deque.copy())
            
            action = out['action']
            action_gripper = out['action_gripper']
            progress = out['progress']

            X_BE = self.obs_deque[-1]["X_BE"]

            # skip every second action 
            action = action[::2]

            for i in range(len(action)):
                current_task.set_progress(progress[i])

                if task.current_task().transform_ee_frame:
                    pass


                trans, orien = matrix_to_pos_orn(action[i])

                # # task is teapot place then do somethin
                #if task.current_task().name != 'teapot_place':
                motion.set_target(to_affine(trans, orien))

                g = action_gripper[i]

                # print('Grasp Action:', g)

                # print(f"Gripper Action: {action_gripper[-1]}, Allowing Gripper to Move: {task.current_task().gripper_allowed_to_move}")
                if task.current_task().gripper_allowed_to_move:
                    res = False
                    if g > 0.5:
                        res = self.robot.close_gripper_if_open()
                    else:
                        res = self.robot.open_gripper_if_closed()
                    if res:
                        task.current_task().gripper_allowed_to_move = False 
                    
                    # check if task if pour
                    if task.current_task().name == 'teapot_pour':
                        res = True
                    
                if progress[i] > current_task.progress_threshold and res == True:
                    if task.check_if_is_last_phase():
                        print("Task Finished")
                        done = True
                        # stop reactivex stream
                        obs_stream.dispose()

                    else: 
                        print(f"Moving to next phase")
                        self.task.go_to_next_phase()
                        self.obs_deque.clear()
                        time.sleep(0.5)
                        input("Press Enter to continue...")
                    break

                robot_state = motion.get_robot_state()
                # X_BE = np.array(robot_state.O_T_EE).reshape(4,4).T
                # X_EA = task.current_task().X_EA
                # self.robot_visualiser.ee_pose.T = sm.SE3(X_BE, check=False).norm()	
                # if X_EA is not None:
                    # self.robot_visualiser.ee_pose.T = X_BE 
                self.robot_visualiser.policy_pose.T = action[i]
                # self.robot_visualiser.policy_pose.T = action[i] 
                # visualize the X_B_OO from deques
                self.robot_visualiser.object_pose.T = self.obs_deque[-1]["X_BO"]
                # self.robot_visualiser.orientation_frame.T = self.obs_deque[-2]["X_BO"]
                self.robot_visualiser.step(robot_state.q)

                # # if task is teapot place, check if the teapot is in the teacup
                # if task.current_task().name == 'teapot_place':
                #     pdb.set_trace()

                current_task.print_progress()
                time.sleep(0.2)

                # Log the data
                rgb = self.perception_system.tracker.current_rgb_frame
                depth = self.perception_system.tracker.current_depth_frame
                pose_dict = self.perception_system.tracker.current_poses
                self.data_logger.log_data({
                    'rgb': rgb,
                    'depth': depth,
                    'pose': pose_dict,
                    'action': action[i],
                    'progress': progress[i]
                })
                
                

if __name__ == '__main__':
    clear_cuda_memory()
    robot = create_robot(dynamic_rel=0.4)
    perception_system = PerceptionSystem(robot)
    perception_system.start()
    task = MakeTeaTask(perception_system=perception_system, robot=robot, vis=False)
    task.move_to_phase_start()
    data_logger = DataLogger('video_data_1.npz')

    controller = RobotInferenceController(
        perception_system=perception_system, 
        robot=robot,
        task=task,
        data_logger=data_logger
        )
        
    controller.start_inference()

    perception_system.stop()
    data_logger.save_data()
