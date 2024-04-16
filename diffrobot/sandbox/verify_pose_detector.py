from multiprocessing.managers import SharedMemoryManager
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco
from diffrobot.realsense.single_realsense import SingleRealsense
from diffrobot.robot.robot import *
from diffrobot.robot.visualizer import RobotViz
import time

hostname = "172.16.0.2"
panda = Robot(hostname)
panda.set_dynamic_rel(0.04)
home_q = np.deg2rad([-90, 0, 0, -90, 0, 90, 45]) 
panda.move_to_joints(home_q)

robot_visualiser = RobotViz()
robot_visualiser.step(home_q)

sh = SharedMemoryManager()
sh.start()
cam = SingleRealsense(sh, "f1230727")
cam.start()
time.sleep(1.0)
marker_detector = ArucoDetector(cam, 0.025, aruco.DICT_4X4_50, 3, visualize=True)

X_FC = np.array([[-0.7000802392623863, 0.7135257817028211, -0.027723950650968963, -0.05561641872987054],
[-0.7140575288482107, -0.6993782374237717, 0.031494864870394226, 0.06288103976785503],
[0.0030828703355744516, 0.041845428225742776, 0.9991193402427451, 0.07064798549355376],
[0.0, 0.0, 0.0, 1.0]])



def read_X_BF(s) -> np.ndarray:
    import spatialmath as sm # for poses
    X_BE = np.array(s.O_T_EE).reshape(4, 4).astype(np.float32).T
    X_FE = np.array(s.F_T_EE).reshape(4, 4).astype(np.float32).T
    X_EF = np.linalg.inv(X_FE)
    X_BF = X_BE @ X_EF 
    return X_BF



X_CO = marker_detector.estimate_pose()

s = panda.get_state()
X_BE = np.array(s.O_T_EE).reshape(4,4).T
X_BF = read_X_BF(s)
ee_trans, ee_orien = matrix_to_pos_orn(X_BE)

X_BC = X_BF @ X_FC
X_BO = X_BC @ X_CO

while True:    
    
    trans, orien = matrix_to_pos_orn(X_BO)

    robot_q = panda.get_joint_positions()
    robot_visualiser.ee_pose.T = panda.get_tcp_pose()
    robot_visualiser.policy_pose.T = X_BO
    robot_visualiser.step(robot_q)

    panda.move_to_pose_linear(trans, ee_orien)

    time.sleep(0.1)



