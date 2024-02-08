# Test aruco detection

import numpy as np
import cv2

from realsense.single_realsense import SingleRealsense
from calibration.aruco_detector import ArucoDetector, aruco
from multiprocessing.managers import SharedMemoryManager



sh = SharedMemoryManager()
sh.start()
cam = SingleRealsense(sh, "032522250135")
cam.start()
marker_detector = ArucoDetector(cam, 0.039, aruco.DICT_4X4_50, 37, visualize=False)

while True:
    marker_pose = marker_detector.estimate_pose()
# marker_pose = marker_detector.estimate_pose()
    
    # print(marker_pose)

