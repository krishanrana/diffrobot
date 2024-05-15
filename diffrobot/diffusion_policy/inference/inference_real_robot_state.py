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
        self.phase = 0
        self.dutils = DatasetUtils()

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
        # self.sh = SharedMemoryManager()
        # self.sh.start()
        # self.cam = SingleRealsense(self.sh, "f1230727")

        self.cams = MultiRealsense(
            serial_numbers=['128422271784', '123622270136'],
            resolution=(640,480),
        )

        self.cams.start()
        self.cams.set_exposure(exposure=5000, gain=60)
        self.marker_detector_front = ArucoDetector(self.cams.cameras['123622270136'], 0.025, aruco.DICT_4X4_50, 3, visualize=False)
        self.marker_detector_back = ArucoDetector(self.cams.cameras['128422271784'], 0.025, aruco.DICT_4X4_50, 3, visualize=False)

        self.marker_detector_front_saucer = ArucoDetector(self.cams.cameras['123622270136'], 0.025, aruco.DICT_4X4_50, 8, visualize=False) 
        self.marker_detector_back_saucer = ArucoDetector(self.cams.cameras['128422271784'], 0.025, aruco.DICT_4X4_50, 8, visualize=False)
        
        # self.sensor_socket = SensorSocket(self.sensor_ip, self.sensor_port)
        time.sleep(1.0)

    def setup_robot(self):
        self.panda = Robot(self.robot_ip)
        self.gripper = self.panda.gripper

        self.gripper.open()
        # self.gripper.close()
        self.gripper_state = 0.0

        # self.panda.set_dynamic_rel(1.0, accel_rel=0.2, jerk_rel=0.05)
        # self.panda.set_dynamic_rel(0.4, accel_rel=0.005, jerk_rel=0.05)

        self.panda.set_dynamic_rel(0.4, accel_rel=0.2, jerk_rel=0.05)
        self.panda.frankx.set_collision_behavior(
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
		)

        # self.panda.move_to_joints([0.00487537496966383, 0.140028320465115, -0.4990375894491169,
                                #    -2.148368699077172, 1.0965412648672856, 1.1600619074643155, 0.0020968194298945724])

        # self.panda.move_to_joints(np.deg2rad([0, 0, 0, -90, 0, 90, 45]))
        # self.panda.move_to_joints([-1.5665102330276086, 0.06832033950119581, 0.07771335946889425, -2.02817877543619, -0.005529185095817712, 2.2603790660269496, 0.7730436232405306])


        self.load_transforms()

        self.search_for_objects()
        

    def search_for_objects(self):
        # this function will sweep the arm across its workspace to search for the intial pose of all objects
        # these poses will be stored in the base frame of the robot
        # Once these poses are found the robot will initiate the policy
        
        # create a sweep path for the impedance controller to follow

        found_O1 = False
        found_O2 = False
        saved_angle = -90
        
        for sweep_angle in np.linspace(-90, 90, 5):
            self.panda.move_to_joints(np.deg2rad([sweep_angle, 0, 0, -90, 0, 90, 45]))

            X_C_O1 = self.marker_detector_front.estimate_pose()
            X_C_O2 = self.marker_detector_front_saucer.estimate_pose()

            s = self.panda.get_state()
            X_BE = np.array(s.O_T_EE).reshape(4,4).T
            X_BC = X_BE @ self.X_EC_f
            

            if X_C_O1 is not None:
                found_O1 = True
                self.X_B_O1 = X_BC @ X_C_O1
                saved_angle = sweep_angle
                print('Found the cup!')

            if X_C_O2 is not None:
                found_O2 = True
                self.X_B_O2 = X_BC @ X_C_O2
                print('Found the saucer!')

            if found_O1 and found_O2:
                break
        
        # move and wait above the cup
        self.panda.move_to_joints(np.deg2rad([saved_angle, 0, 0, -90, 0, 90, 45]))



    def load_transforms(self):
        self.X_BO = None
        self.X_OO_O = None
        with open(f'../runs/{self.saved_run_name}/hand_eye.json', 'r') as f:
            trans = json.load(f)
            self.X_EC_b = np.array(trans['X_EC_b'])
            self.X_EC_f = np.array(trans['X_EC_f'])

    def get_marker(self):
        marker_front = self.marker_detector_front.estimate_pose()
        marker_back = self.marker_detector_back.estimate_pose()

        if marker_front is not None:
            self.X_EC = self.X_EC_f
            return marker_front
        elif marker_back is not None:
            self.X_EC = self.X_EC_b
            return marker_back
        else:
            return None
        
    def get_obs(self):
        s = self.panda.get_state()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T
        # X_BF = read_X_BF(s)
        # if self.phase == 0:
        X_CO = self.get_marker()
    

        if X_CO is not None:
            # X_BC = X_BF @ self.X_FC

            if self.phase == 0:
                X_BC = X_BE @ self.X_EC
                X_BO = X_BC @ X_CO
            elif self.phase == 1:
                X_BO = self.X_B_O2

            # dist = np.dot(np.array([0,0,1]), X_BO[:3,2])
            # if dist > 0.99: # filter out bad object poses
            #     self.X_BO = X_BO
            self.X_BO = self.dutils.adjust_orientation_to_z_up(X_BO)

            #TODO WIP delete this
            self.saved_X_B_OO1 = self.dutils.compute_oriented_affordance_frame(self.X_B_O1)

            if self.phase == 0:
                self.X_B_OO = self.dutils.compute_oriented_affordance_frame(self.X_BO)
                self.saved_X_B_OO1 = self.X_B_OO
            elif self.phase == 1:
                self.X_B_OO = self.dutils.compute_oriented_affordance_frame(self.X_BO, self.saved_X_B_OO1)
            
            self.X_OO_O = np.dot(np.linalg.inv(self.X_B_OO), self.X_BO) 
            
        # self.robot_visualiser.object_pose.T = self.X_BO

        return {"X_BE": X_BE, 
                "X_BO": self.X_BO,
                "X_B_OO": self.X_B_OO,
                "X_OO_O": self.X_OO_O,
                "gripper_state": self.gripper.width(),
                "phase": self.phase,}
    
    def start_inference(self):
        obs_stream = rx.interval(1.0/10.0, scheduler=rx.scheduler.NewThreadScheduler()) \
            .pipe(ops.map(lambda _: self.get_obs())) \
            .pipe(ops.filter(lambda x: x["X_BO"] is not None)) \
            .subscribe(lambda x: self.obs_deque.append(x))  
      
        # motion = self.panda.start_impedance_controller(200, 30, 5)
        motion = self.panda.start_impedance_controller(830, 40, 1)


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
                    # if self.progress[i] >  0.85:
                    #     print('I think im done with the task!')
                    #     input('Should I continue?')
                    #     self.progress = np.zeros((self.params.action_horizon))

                    trans, orien = matrix_to_pos_orn(self.action[i])
                    motion.set_target(to_affine(trans, orien))

                    print('Gripper Action: ', self.action_gripper[i])

                    #WIP
             
                    if self.action_gripper[i] > 0.1:
                        self.gripper.close()
                        self.gripper_state = 1.0
                        if self.phase == 0:
                            self.phase = 1  
                            print('Closing gripper')
                            time.sleep(0.5)
                    else:
                        self.gripper.open()
                        self.gripper_state = 0.0



                    # robot ee height
                    
                


                    robot_q = self.panda.get_joint_positions()
                    self.robot_visualiser.ee_pose.T = self.panda.get_tcp_pose()
                    self.robot_visualiser.policy_pose.T = self.action[i]
                    self.robot_visualiser.orientation_frame.T = X_B_OO    

                    cup_handle_pose = temp_X_BO
                    self.robot_visualiser.cup_handle.T = cup_handle_pose

                    self.robot_visualiser.step(robot_q)
                    time.sleep(0.1)

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
controller = RobotInferenceController(saved_run_name='royal-cloud-77_state',
                                      robot_ip='172.16.0.2', 
                                      sensor_ip='131.181.33.191', 
                                      sensor_port=5000)
controller.start_inference()
