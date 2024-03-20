import numpy as np
import time
import json
import collections
import reactivex as rx
from reactivex import operators as ops
from multiprocessing.managers import SharedMemoryManager
import torch
import cv2
# Import necessary modules from your libraries
from diffrobot.realsense.single_realsense import SingleRealsense
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco
from diffrobot.tactile_sensors.xela import SensorSocket
from diffrobot.diffusion_policy.diffusion_policy import DiffusionPolicy
from diffrobot.robot.robot import Robot, to_affine, matrix_to_pos_orn
from diffrobot.diffusion_policy.utils.config_utils import get_config
from diffrobot.robot.visualizer import RobotViz
import pdb


class RobotInferenceController:
    def __init__(self, saved_run_name, robot_ip, sensor_ip, sensor_port, record_fps=30):
        self.saved_run_name = saved_run_name
        self.run_dir = f'../runs/{self.saved_run_name}'
        self.params = get_config(f'{self.run_dir}/config_state_pretrain', mode='infer')
        self.robot_ip = robot_ip
        self.sensor_ip = sensor_ip
        self.sensor_port = sensor_port
        self.record_fps = record_fps
        self.reset_progress = False
        self.robot_visualiser = RobotViz()
        self.setup_diffusion_policy()
        self.setup_cameras_and_sensors()
        self.setup_robot()
        self.obs_deque = collections.deque(maxlen=self.policy.params.obs_horizon)
        self.progress = np.zeros((self.params.action_horizon))
        self.toggle_key = ord(' ')  # ASCII code for space bar
        
        


    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()
        self.policy = DiffusionPolicy(mode='infer', 
                                      policy_type='state',
                                      config_file=f'{self.run_dir}/config_state_pretrain',
                                      finetune=False, 
                                      saved_run_name=self.saved_run_name)

    def setup_cameras_and_sensors(self):
        self.sh = SharedMemoryManager()
        self.sh.start()
        self.cam = SingleRealsense(self.sh, "f1230727")
        self.cam.start()
        self.marker_detector = ArucoDetector(self.cam, 0.05, aruco.DICT_4X4_50, 6, visualize=False)
        self.cam.set_exposure(exposure=100, gain=60)
        self.sensor_socket = SensorSocket(self.sensor_ip, self.sensor_port)
        time.sleep(1.0)

    def setup_robot(self):
        self.panda = Robot(self.robot_ip)
        self.panda.set_dynamic_rel(0.4, accel_rel=0.005, jerk_rel=0.005)
        self.panda.move_to_joints([0.00487537496966383, 0.140028320465115, -0.4990375894491169,
                                   -2.148368699077172, 1.0965412648672856, 1.1600619074643155, 0.0020968194298945724])
        self.load_transforms()

    def load_transforms(self):
        with open(f'../runs/{self.saved_run_name}/transforms.json', 'r') as f:
            trans = json.load(f)
            self.X_EC = np.array(trans['X_EC'])
        self.X_BO = self.estimate_marker_pose()

    def estimate_marker_pose(self):
        X_CO = self.marker_detector.estimate_pose()
        X_BE = self.panda.get_tcp_pose()
        X_BC = np.dot(X_BE, self.X_EC)
        X_BO = np.dot(X_BC, X_CO)
        self.robot_visualiser.object_pose.T = X_BO
        return X_BO

    def get_marker(self):
        res = self.marker_detector.estimate_pose()
        return None if res is None else res

    def get_obs(self):
        X_BE = self.panda.get_tcp_pose()
        tactile_sensor = self.sensor_socket.get_forces()
        joint_torques = self.panda.get_joint_torques()
        ee_forces = self.panda.get_ee_forces()
        if self.reset_progress:
            self.reset_progress = False
            print("Resetting progress")
            progress = self.progress[-1]*0
        else:
            progress = self.progress[-1]

        return {"X_BE": X_BE, 
                "X_BO": self.X_BO, 
                "X_EC": self.X_EC,
                "tactile_sensor": tactile_sensor, 
                "joint_torques": joint_torques, 
                "ee_forces": ee_forces,
                "progress": progress}
    
    def start_inference(self):
        obs_stream = rx.interval(1.0/10.0, scheduler=rx.scheduler.NewThreadScheduler()) \
            .pipe(ops.map(lambda _: self.get_obs())) \
            .subscribe(lambda x: self.obs_deque.append(x))
        
        motion = self.panda.start_impedance_controller(200, 40, 5)

        # cv2.namedWindow('Inference')


        while True:
            done = False

            while not done:

                # wait for obs_deque to have len 2
                while len(self.obs_deque) < 2:
                    time.sleep(0.1)
                    print("Waiting for observation")


                # print(obs_deque)
                out = self.policy.infer_action(self.obs_deque.copy())
                self.action = out['action']
                self.progress = out['progress']

                waypoints = []
                for i in range(len(self.action)):
                    print('Task Progress: ', self.progress[i])
                    # if self.progress[i] >  0.85:
                    #     print('I think im done with the task!')
                    #     input('Should I continue?')
                    #     self.progress = np.zeros((self.params.action_horizon))

                    

                    trans, orien = matrix_to_pos_orn(self.action[i])
                    waypoints.append(to_affine(trans, orien))

                # self.panda.recover_from_errors()
                for waypoint in waypoints:
                    motion.set_target(waypoint)

                    robot_q = self.panda.get_joint_positions()
                    self.robot_visualiser.step(robot_q)
                    
                    # self.key = cv2.waitKey(100) & 0xFF
                    # if self.key == self.toggle_key:
                    #     print('Key has been toggled')
                    #     self.reset_progress = True
                    #     #self.obs_deque.clear()
                        
                    #     break

 
                # self.panda.waypoints(waypoints)

                
                



# Example usage
controller = RobotInferenceController(saved_run_name='royal-aardvark-41_state',
                                      robot_ip='172.16.0.2', 
                                      sensor_ip='131.181.33.191', 
                                      sensor_port=5000)
controller.start_inference()
