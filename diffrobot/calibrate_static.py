import time
from dataclasses import dataclass
import numpy as np
import tyro
import cv2
import json
from scipy.optimize import least_squares
from multiprocessing.managers import SharedMemoryManager

from robot.robot import Robot, pos_orn_to_matrix, matrix_to_pos_orn
from calibration.cal_utils import quat_to_euler, euler_to_quat
from realsense.single_realsense import SingleRealsense
from calibration.aruco_detector import ArucoDetector, aruco
from pathlib import Path

@dataclass
class Params:
    path: Path
    name: str
    serial: str
    marker_id: int = 9

def compute_residuals_static_cam(x, T_robot_tcp, T_cam_marker):
    m_tcp = np.array([*x[6:], 1])
    T_cam_robot = pos_orn_to_matrix(x[3:6], x[:3])

    residuals = []
    for i in range(len(T_cam_marker)):
        m_C_observed = T_cam_marker[i][:3, 3]
        m_C = T_cam_robot @ T_robot_tcp[i] @ m_tcp
        residuals += list(m_C_observed - m_C[:3])
    return residuals

def calibrate_static_cam_least_squares(T_robot_tcp, T_cam_marker):
    initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.1])
    result = least_squares(
        fun=compute_residuals_static_cam, x0=initial_guess, method="lm", args=(T_robot_tcp, T_cam_marker)
    )
    trans = result.x[3:6]
    rot = result.x[0:3]
    T_cam_robot = pos_orn_to_matrix(trans, rot)
    T_robot_cam = np.linalg.inv(T_cam_robot)
    return T_robot_cam

def detect_marker_from_trajectory(robot: Robot, qs, marker_detector:ArucoDetector):
    """
    Move to previously recorded poses and estimate marker poses.

    Args:
        robot: Robot interface
        tcp_poses: Previously saved tcp poses as list of 4x4 matrices.
        marker_detector: Marker detection library.
        cfg: Hydra config.

    Returns:
        valid_tcp_poses (list): TCP poses where a marker has been detected.
        marker_poses (list): The detected marker poses.
    """
    marker_poses = []
    valid_tcp_poses = []

    for i, q in enumerate(qs):
        q = np.array(q)
        robot.move_to_joints(q)
        time.sleep(0.1)
        marker_pose = marker_detector.estimate_pose()
        if marker_pose is not None:
            valid_tcp_poses.append(robot.get_tcp_pose())
            marker_poses.append(marker_pose)

    return valid_tcp_poses, marker_poses

if __name__ == "__main__":
    params = tyro.cli(Params)
    assert params.path.exists() and params.path.is_file() and params.path.suffix == ".json"
    with open(params.path, "r") as f:
        qs = json.load(f)

    # Camera
    sh = SharedMemoryManager()
    sh.start()
    cam = SingleRealsense(sh, params.serial) 
    cam.start()

    # Detector
    marker_detector = ArucoDetector(cam, 0.05, aruco.DICT_4X4_50, params.marker_id)


    # Robot
    robot = Robot()
    robot.set_dynamic_rel(0.3)

    robot.move_to_start()
    tcp_poses, marker_poses = detect_marker_from_trajectory(robot, qs, marker_detector)
    robot.move_to_start()

    X_WV = calibrate_static_cam_least_squares(tcp_poses, marker_poses)
    print(X_WV)

    res = {
        "X_WV": X_WV.tolist(),
    }

    with open(f"data/camera_calibration/{params.name}_static.json", "w") as f:
        json.dump(res, f)

    X_WE = robot.get_tcp_pose()
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
    axis.transform(X_WV)
    geometries.append(axis)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    axis.transform(X_WE)
    geometries.append(axis)
    o3d.visualization.draw_geometries(geometries)

    del cam
