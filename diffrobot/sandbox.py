# Test aruco detection

import numpy as np
import cv2

from realsense.single_realsense import SingleRealsense
from calibration.aruco_detector import ArucoDetector, aruco
from multiprocessing.managers import SharedMemoryManager


from robot.robot import Robot, to_affine, pos_orn_to_matrix
from frankx import Affine, JointMotion, Waypoint, WaypointMotion, PathMotion
import pdb


panda = Robot("172.16.0.2")
panda.set_dynamic_rel(0.4)

panda.move_to_joints([-1.56832675,  0.39303148,  0.02632776, -1.98690212, -0.00319773,  2.35042797, 0.94667396])

pose = panda.get_tcp_pose()
trans = pose[:3, 3]
z_height = trans[2]
orien = panda.get_orientation()

# move in a straight line in the x direction
waypoint_1 = to_affine(trans, orien)
trans[0] += 0.01
waypoint_2 = to_affine(trans, orien)
trans[0] += 0.01
waypoint_3 = to_affine(trans, orien)
trans[0] += 0.01
waypoint_4 = to_affine(trans, orien)

motion_down = PathMotion([
        waypoint_1, 
        waypoint_2, 
        waypoint_3, 
        waypoint_4
    ], blend_max_distance=0.05)

panda.frankx.move(motion_down)