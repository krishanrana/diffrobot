from multiprocessing.managers import SharedMemoryManager
from calibration.aruco_detector import ArucoDetector, aruco
from realsense.single_realsense import SingleRealsense

sh = SharedMemoryManager()
sh.start()
cam = SingleRealsense(sh, "035122250692")
cam.start()
marker_detector = ArucoDetector(cam, 0.05, aruco.DICT_4X4_50, 8, visualize=False)
while True:
    print(marker_detector.estimate_pose())