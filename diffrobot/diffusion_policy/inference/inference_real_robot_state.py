import numpy as np
import time
import json
import collections
import reactivex as rx
from reactivex import operators as ops
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
from diffrobot.diffusion_policy.utils.dataset_utils import DatasetUtils
import pdb
import spatialmath as sm

def read_X_BF(s) -> np.ndarray:
    import spatialmath as sm # for poses
    X_BE = np.array(s.O_T_EE).reshape(4, 4).astype(np.float32).T
    X_FE = np.array(s.F_T_EE).reshape(4, 4).astype(np.float32).T
    X_EF = np.linalg.inv(X_FE)
    X_BF = X_BE @ X_EF 
    return X_BF


class RobotInferenceController:
    def __init__(self, saved_run_name, robot_ip, sensor_ip, sensor_port, record_fps=30):
        self.saved_run_name = saved_run_name
        # self.run_dir = f'../runs/{self.saved_run_name}'
        # self.params = get_config(f'{self.run_dir}/config_state_pretrain', mode='infer')
        self.aruco_keys = { 'cup': 3,
                            'saucer': 10,
                            'teapot': 4,
                            'spoon': 8,
                            }
        
        self.object_poses = {key: None for key in self.aruco_keys}
        self.task_objects = {'cup_rotate': 'cup', 
                             'place_saucer': 'saucer', 
                            'teapot_rotate': 'teapot', 
                            'teapot_pour': 'cup',
                            'teapot_place': 'cup',
                            'pick_spoon': 'spoon',
                            'stir_spoon': 'cup',}
        
        self.oriented_frame_reference = {'cup_rotate': 'base',
                                         'place_saucer': 'cup',
                                         'teapot_rotate': 'base',
                                         'teapot_pour': 'teapot',
                                         'teapot_place': 'cup',
                                         'pick_spoon': 'base',
                                         'stir_spoon': 'cup',}
        
        self.stored_reference_frames = {'base': np.eye(4)}

        
        self.task_phases = {
            0: 'cup_rotate',
            1: 'place_saucer',
            2: 'teapot_rotate',
            3: 'teapot_pour',
            4: 'teapot_place',
            5: 'pick_spoon',
            6: 'stir_spoon',
        }

        
        self.phase = 0    
        self.robot_ip = robot_ip
        self.sensor_ip = sensor_ip
        self.sensor_port = sensor_port
        self.record_fps = record_fps
        self.reset_progress = False
        self.robot_visualiser = RobotViz()
        self.setup_diffusion_policy()
        self.setup_cameras_and_sensors()
        self.setup_robot()



    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()

        self.policies = {}

        for key in self.saved_run_name:
            saved_run_name = self.saved_run_name[key]

            run_dir = f'../runs/{saved_run_name}'
            params = get_config(f'{run_dir}/config_state_pretrain', mode='infer')

            policy = DiffusionPolicy(mode='infer', 
                                    policy_type='state',
                                    config_file=f'{run_dir}/config_state_pretrain',
                                    finetune=False, 
                                    saved_run_name=saved_run_name)
            
            self.policies[key] = policy

        # get first policy
        key = list(self.policies.keys())[0]
        policy = self.policies[key]   
        self.dutils = policy.dutils
        self.obs_horizon = policy.params.obs_horizon
        self.obs_deque = collections.deque(maxlen=policy.params.obs_horizon)


    def setup_cameras_and_sensors(self):
        # self.sh = SharedMemoryManager()
        # self.sh.start()
        # self.cam = SingleRealsense(self.sh, "f1230727")

        self.cams = MultiRealsense(
            serial_numbers=['128422271784', '123622270136'],
            resolution=(640,480),
        )

        self.cams.start()
        self.cams.set_exposure(exposure=5000, gain=60)
        self.marker_detector_front = ArucoDetector(self.cams.cameras['123622270136'], 0.025, aruco.DICT_4X4_50, marker_id=None, visualize=False)
        self.marker_detector_back = ArucoDetector(self.cams.cameras['128422271784'], 0.025, aruco.DICT_4X4_50, marker_id=None, visualize=False)
        
        # self.sensor_socket = SensorSocket(self.sensor_ip, self.sensor_port)
        time.sleep(1.0)

    def setup_robot(self):
        self.panda = Robot(self.robot_ip)
        self.gripper = self.panda.gripper

        self.gripper.open()
        self.gripper_state = 0.0

        self.panda.set_dynamic_rel(0.4, accel_rel=0.2, jerk_rel=0.05)
        self.panda.frankx.set_collision_behavior(
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
		)

        self.load_transforms()
        self.search_for_objects()

        
    def detect_objects(self):
        all_detections_front = self.marker_detector_front.detect_objects()
        all_detections_back = self.marker_detector_back.detect_objects()

        for object_name, marker_id in self.aruco_keys.items():
            X_CO_f = self.marker_detector_front.estimate_pose(detected_markers=all_detections_front, marker_id=marker_id)
            X_CO_b = self.marker_detector_back.estimate_pose(detected_markers=all_detections_back, marker_id=marker_id)

            s = self.panda.get_state()
            X_BE = np.array(s.O_T_EE).reshape(4, 4).T

            if X_CO_f is not None:
                X_BC = X_BE @ self.X_EC_f
                self.object_poses[object_name] = X_BC @ X_CO_f

            if X_CO_b is not None:
                X_BC = X_BE @ self.X_EC_b
                self.object_poses[object_name] = X_BC @ X_CO_b

        # only update the stored reference frame for the phase object
        if self.object_poses[self.task_objects[self.task_phases[self.phase]]] is not None:
            self.stored_reference_frames[self.task_objects[self.task_phases[self.phase]]] = self.object_poses[self.task_objects[self.task_phases[self.phase]]]
        

    def search_for_objects(self):
        # This function sweeps the arm across its workspace to search for the initial pose of all objects.
        # These poses will be stored in the base frame of the robot.
        # Once these poses are found, the robot will initiate the policy.

        sweep_angles = np.linspace(-110, 90, 8)
        for sweep_angle in sweep_angles:
            self.panda.move_to_joints(np.deg2rad([sweep_angle, 0, 0, -110, 0, 110, 45]))

            self.detect_objects()

            # all objects are found break
            if all([pose is not None for pose in self.object_poses.values()]):
                break

        # populate the stored reference frames
        for object in self.task_objects.values():
            self.stored_reference_frames[object] = self.object_poses[object]

        # more above the cup
        # compute sweep angle to go above the cup
        cup_pose = self.object_poses['cup']
        angle = np.arctan2(cup_pose[1, 3], cup_pose[0, 3])
        self.panda.move_to_joints(np.deg2rad([np.rad2deg(angle), 0, 0, -110, 0, 110, 45]))

    def load_transforms(self):
        self.X_BO = None
        self.X_OO_O = None
        saved_run_name = self.saved_run_name['cup_rotate']
        with open(f'../runs/{saved_run_name}/transforms/hand_eye.json', 'r') as f:
            trans = json.load(f)
            self.X_EC_b = np.array(trans['X_EC_b'])
            self.X_EC_f = np.array(trans['X_EC_f'])
        
        
    def get_obs(self):
        s = self.panda.get_state()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T
        self.detect_objects()
        X_BO = self.object_poses[self.task_objects[self.task_phases[self.phase]]]
        X_BO = self.dutils.adjust_orientation_to_z_up(X_BO) 
        X_B_OO = self.dutils.compute_oriented_affordance_frame(X_BO, self.stored_reference_frames[self.oriented_frame_reference[self.task_phases[self.phase]]])

        X_OO_O = np.dot(np.linalg.inv(X_B_OO), X_BO) 

        # self.robot_visualiser.object_pose.T = self.X_BO

        return {"X_BE": X_BE, 
                "X_BO": X_BO,
                "X_B_OO": X_B_OO,
                "X_OO_O": X_OO_O,
                "gripper_state": self.gripper.width(),
                "phase": self.phase,}
    

    
    def start_inference(self):
        obs_stream = rx.interval(1.0/5.0, scheduler=rx.scheduler.NewThreadScheduler()) \
            .pipe(ops.map(lambda _: self.get_obs())) \
            .pipe(ops.filter(lambda x: x["X_BO"] is not None)) \
            .subscribe(lambda x: self.obs_deque.append(x))  
      
        # motion = self.panda.start_impedance_controller(200, 30, 5)
        motion = self.panda.start_impedance_controller(830, 40, 1)
        self.latch = False


        while True:
            done = False

            while not done:

                # wait for obs_deque to have len 2
                while len(self.obs_deque) < self.obs_horizon:
                    time.sleep(0.1)
                    # print("Waiting for observation")

                # print(obs_deque)

                # select policy based on phase
                self.policy = self.policies[self.task_phases[self.phase]]


                out = self.policy.infer_action(self.obs_deque.copy())
                self.action = out['action']
                self.action_gripper = out['action_gripper']
                self.progress = out['progress']


                temp_X_BO = self.obs_deque[0]['X_BO']
                X_B_OO = self.obs_deque[0]['X_B_OO']

                waypoints = []
                # self.panda.recover_from_errors()
                # take very 3rd action
                # for i in range(len(self.action)):
                for i in range(len(self.action)):
                    print('Task Progress: ', self.progress[i])
    
                    trans, orien = matrix_to_pos_orn(self.action[i])
                    motion.set_target(to_affine(trans, orien))

                    # keep a rolling window for phases

             
                    print('Gripper Action: ', self.action_gripper[i])
                    if self.action_gripper[i] > 0.5 and self.progress[i] > 0.85:
                        self.gripper.close()
                        self.gripper_state = 1.0
                        print('Closing gripper')
                        # if self.phase == 0:
                        #     self.phase = 1  
                        #     print('Closing gripper')
                        #     time.sleep(0.5)
                        # input('Press Enter to continue')
                        self.phase += 1
                        break

                    elif self.action_gripper[i] < 0.5 and self.progress[i] > 0.85:
                        self.gripper.open()
                        self.gripper_state = 0.0
                        self.phase += 1
                        break


    
                    # robot_q = self.panda.get_joint_positions()
                    # self.robot_visualiser.ee_pose.T = self.panda.get_tcp_pose()
                    # self.robot_visualiser.policy_pose.T = self.action[i]
                    # self.robot_visualiser.orientation_frame.T = X_B_OO    

                    # cup_handle_pose = temp_X_BO
                    # self.robot_visualiser.cup_handle.T = cup_handle_pose

                    # self.robot_visualiser.step(robot_q)
                    # time.sleep(0.2)
                    time.sleep(0.2)

                    # if i > 3:
                    #     break

                    # pdb.set_trace()
                    

                    # waypoints.append(to_affine(trans, orien))



                # self.panda.recover_from_errors()
                # for i, waypoint in enumerate(waypoints):
                #     # motion.set_target(waypoint)

                #     robot_q = self.panda.get_joint_positions()

                #     self.robot_visualiser.ee_pose.T = self.panda.get_tcp_pose()
                #     self.robot_visualiser.policy_pose.T = self.action[i]    
                #     self.robot_visualiser.step(robot_q)
                    

                # self.robot_visualiser.ee_pose.T = self.panda.get_tcp_pose()
                # self.robot_visualiser.policy_pose.T = self.action[-1]  
                # robot_q = self.panda.get_joint_positions()  
                # self.robot_visualiser.step(robot_q)
                # self.panda.waypoints(waypoints[:-1])

                
                



# Example usage
controller = RobotInferenceController(saved_run_name={'cup_rotate': 'golden-grass-127_state', #'fiery-pond-126_state
                                                      'place_saucer': 'laced-cosmos-124_state'}, 
                                      robot_ip='172.16.0.2', 
                                      sensor_ip='131.181.33.191', 
                                      sensor_port=5000)
controller.start_inference()
