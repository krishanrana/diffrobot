import time
from dataclasses import dataclass
import numpy as np
import tyro
import cv2
from scipy.optimize import least_squares

from diffrobot.robot.robot import Robot, pos_orn_to_matrix, matrix_to_pos_orn
from diffrobot.calibration.cal_utils import quat_to_euler, euler_to_quat
from diffrobot.realsense.single_realsense import SingleRealsense
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco

@dataclass
class Params:
    hostname: str = "172.16.0.2"

def compute_residuals_gripper_cam(x, T_robot_tcp, T_cam_marker):
    m_R = np.array([*x[6:], 1])
    T_cam_tcp = pos_orn_to_matrix(x[3:6], x[:3])

    residuals = []
    for i in range(len(T_cam_marker)):
        m_C_observed = T_cam_marker[i][:3, 3]
        m_C = T_cam_tcp @ np.linalg.inv(T_robot_tcp[i]) @ m_R
        residuals += list(m_C_observed - m_C[:3])
    return residuals

def calibrate_gripper_cam_least_squares(T_robot_tcp, T_cam_marker):
    initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(
        fun=compute_residuals_gripper_cam, x0=initial_guess, method="lm", args=(T_robot_tcp, T_cam_marker)
    )
    trans = result.x[3:6]
    rot = result.x[0:3]
    T_cam_tcp = pos_orn_to_matrix(trans, rot)
    T_tcp_cam = np.linalg.inv(T_cam_tcp)
    return T_tcp_cam

def calculate_error(T_tcp_cam, T_robot_tcp_list, T_cam_marker_list):
    result = []
    for T_robot_tcp, T_cam_marker in zip(T_robot_tcp_list, T_cam_marker_list):
        T_robot_marker = T_robot_tcp @ T_tcp_cam @ T_cam_marker
        pos, orn = matrix_to_pos_orn(T_robot_marker)
        result.append(pos)
    print(np.std(result, axis=0))
    return result

def visualize_calibration_gripper_cam(cam, T_tcp_cam):
    T_cam_tcp = np.linalg.inv(T_tcp_cam)

    left_finger = np.array([0, 0.04, 0, 1])
    right_finger = np.array([0, -0.04, 0, 1])
    tcp = np.array([0, 0, 0, 1])
    x = np.array([0.03, 0, 0, 1])
    y = np.array([0, 0.03, 0, 1])
    z = np.array([0, 0, 0.03, 1])

    left_finger_cam = T_cam_tcp @ left_finger
    right_finger_cam = T_cam_tcp @ right_finger
    tcp_cam = T_cam_tcp @ tcp
    x_cam = T_cam_tcp @ x
    y_cam = T_cam_tcp @ y
    z_cam = T_cam_tcp @ z

    rgb = cam.get()['color']
    cv2.circle(rgb, cam.project(left_finger_cam), radius=4, color=(255, 0, 0), thickness=3)
    cv2.circle(rgb, cam.project(right_finger_cam), radius=4, color=(0, 255, 0), thickness=3)

    cv2.line(rgb, cam.project(tcp_cam), cam.project(x_cam), color=(255, 0, 0), thickness=3)
    cv2.line(rgb, cam.project(tcp_cam), cam.project(y_cam), color=(0, 255, 0), thickness=3)
    cv2.line(rgb, cam.project(tcp_cam), cam.project(z_cam), color=(0, 0, 255), thickness=3)

    cv2.imshow("calibration", rgb[:, :, ::-1])
    cv2.waitKey(0)

class GripperCamPoseSampler:
    """
    Randomly sample end-effector poses for gripper cam calibration.
    Poses are sampled with polar coordinates theta and r around initial_pos, which are then perturbed with a random
    positional and rotational offset

    Args:
        initial_pos: TCP position around which poses are sampled
        initial_orn: TCP orientation around which poses are sampled
        theta_limits: Angle for polar coordinate sampling wrt. X-axis in robot base frame
        r_limits: Radius for plar coordinate sampling
        h_limits: Sampling range for height offset
        trans_limits: Sampling range for lateral offset
        yaw_limits: Sampling range for yaw offset
        pitch_limit: Sampling range for pitch offset
        roll_limit: Sampling range for roll offset
    """

    def __init__(
        self,
        initial_pos,
        initial_orn,
        theta_limits,
        r_limits,
        h_limits,
        trans_limits,
        yaw_limits,
        pitch_limit,
        roll_limit,
    ):
        self.initial_pos = np.array(initial_pos)
        self.initial_orn = quat_to_euler(np.array(initial_orn))
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.yaw_limits = yaw_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit

    def sample_pose(self):
        """
        Sample a random pose
        Returns:
            target_pos: Position (x,y,z)
            target_pos: Orientation quaternion (x,y,z,w)
        """
        theta = np.random.uniform(*self.theta_limits)
        vec = np.array([np.cos(theta), np.sin(theta), 0])
        vec = vec * np.random.uniform(*self.r_limits)
        yaw = np.random.uniform(*self.yaw_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        height = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        trans_final = self.initial_pos + vec + trans + height
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)

        target_pos = np.array(trans_final)
        target_orn = np.array([np.pi + pitch, roll, theta + np.pi + yaw])
        target_orn = euler_to_quat(target_orn)
        return target_pos, target_orn

def record_gripper_cam_trajectory(
        robot: Robot, 
        pose_sampler: GripperCamPoseSampler, 
        marker_detector: ArucoDetector,
        num_poses: int):
    """
    Move robot to randomly generated poses and estimate marker poses.

    Args:
        robot: Robot interface.
        marker_detector: Marker detection library.
        cfg: Hydra config.

    Returns:
        tcp_poses (list): TCP poses as list of 4x4 matrices.
        marker_poses (list): Detected marker poses as list of 4x4 matrices.
    """
    robot.move_to_start()
    tcp_poses = []
    marker_poses = []
    for i in range(num_poses):
        print(f"Moving to pose {i+1}/{num_poses}")
        pos, orn = pose_sampler.sample_pose()
        robot.move_to_pose(pos, orn)
        time.sleep(0.3)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            tcp_poses.append(robot.get_tcp_pose())
            marker_poses.append(marker_pose)
        
    robot.move_to_start()

    return tcp_poses, marker_poses



if __name__ == "__main__":
    from multiprocessing.managers import SharedMemoryManager
    params = tyro.cli(Params)

    # Camera
    sh = SharedMemoryManager()
    sh.start()
    cam = SingleRealsense(sh, "128422271784", resolution=(640, 480))
    cam.start()
    marker_detector = ArucoDetector(cam, 0.05, aruco.DICT_4X4_50, 9)
    # marker_detector = ArucoDetector(cam, 0.1, aruco.DICT_6X6_50, 0)

    # Robot
    robot = Robot(params.hostname)

    robot.set_dynamic_rel(0.2)
    robot.frankx.jerk_rel = 0.01
    robot.frankx.accel_rel = 0.01

    robot.move_to_start()
    current_orientation = robot.get_orientation()
    pose_sampler = GripperCamPoseSampler(
        initial_pos=[0.5, 0, 0.26],
        initial_orn=current_orientation,
        theta_limits=[2.36, 3.93],
        r_limits=[0.05, 0.1],
        h_limits= [-0.05, 0.06],
        trans_limits=[-0.05, 0.05],
        yaw_limits=[-0.087, 0.087],
        pitch_limit=[-0.087, 0.087],
        roll_limit=[-0.087, 0.087])
    
    tcp_poses, marker_poses = record_gripper_cam_trajectory(
        robot, 
        pose_sampler, 
        marker_detector,
        20)

    # tcp_poses.append(robot.get_tcp_pose())
    

    
    # T_tcp_cam = calibrate_gripper_cam_least_squares(tcp_poses, marker_poses)
    X_EC = calibrate_gripper_cam_least_squares(tcp_poses, marker_poses)
    X_CE = np.linalg.inv(X_EC)
    print(X_EC)
    calculate_error(X_EC, tcp_poses, marker_poses)
    X_BE = robot.get_tcp_pose()
    X_BC = X_BE @ X_EC
    # save calibration
    import json
    res = {
        "X_EC": X_EC.tolist(),
    }
    intrinsics = cam.get_intrinsics().tolist()
    res["intrinsics"] = intrinsics
    
    with open("calibration_data/hand_eye.json", "w") as f:
        json.dump(res, f)


    # visualize_calibration_gripper_cam(cam, T_tcp_cam)

    # visualize tcp poses with open3d
    import open3d as o3d
    geometries = []
    for tcp_pose in tcp_poses:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        axis.transform(tcp_pose)
        geometries.append(axis)
    plane = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    geometries.append(plane)
    ground_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.005)
    ground_box.translate([-0.25, -0.25, -0.005])
    geometries.append(ground_box)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    axis.transform(X_BC)
    geometries.append(axis)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    axis.transform(X_BE)
    geometries.append(axis)

    o3d.visualization.draw_geometries(geometries)


    # Clean up
    del cam
    del sh

