# Test aruco detection

# import numpy as np
# import cv2

# from realsense.single_realsense import SingleRealsense
# from calibration.aruco_detector import ArucoDetector, aruco
# from multiprocessing.managers import SharedMemoryManager


from diffrobot.robot.robot import Robot, to_affine, pos_orn_to_matrix
# from frankx import Affine, JointMotion, Waypoint, WaypointMotion, PathMotion
# import pdb


panda = Robot("172.16.0.2")
panda.set_dynamic_rel(0.4)


# if __name__ == "__main__":
#     from multiprocessing.managers import SharedMemoryManager

#     # Camera
#     sh = SharedMemoryManager()
#     sh.start()
#     cam = SingleRealsense(sh, "f1230727")
#     cam.start()

#     # save camera intrinsics
#     intrinsics = cam.get_intrinsics()
#     print(intrinsics.coeffs)

#     # save camera instrinsics
#    # import json
#    # with open("intrinsics.json", "w") as f:
#    #     json.dump(intrinsics, f)
    



    # save camera instrinsics

import numpy as np

while True:
    print(panda.get_joints())

# panda.move_to_joints([-1.56832675,  0.39303148,  0.02632776, -1.98690212, -0.00319773,  2.35042797, 0.94667396])

# pose = panda.get_tcp_pose()
# trans = pose[:3, 3]
# z_height = trans[2]
# orien = panda.get_orientation()


# # move in a straight line in the x direction
# waypoint_1 = to_affine([0.0, -0.4, 0.2], orien)
# waypoint_2 = to_affine([0.2, -0.3, 0.2], orien)
# waypoint_3 = to_affine([0.4, -0.4, 0.2], orien)

# motion = PathMotion([
#         waypoint_1, 
#         waypoint_2, 
#         waypoint_3, 
#     ], blend_max_distance=0.4)

# panda.frankx.move_async(motion)
# import time
# while True:
#     q = motion.get_robot_state().q
#     print(q)
#     time.sleep(0.2)
