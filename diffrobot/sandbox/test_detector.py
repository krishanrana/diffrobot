from multiprocessing.managers import SharedMemoryManager
from diffrobot.calibration.aruco_detector import ArucoDetector, aruco
from diffrobot.realsense.single_realsense import SingleRealsense
from diffrobot.realsense.multi_realsense import MultiRealsense
from diffrobot.realsense.multi_camera_visualizer import MultiCameraVisualizer
import pdb
import numpy as np
import json

# sh = SharedMemoryManager()
# sh.start()
# cam = SingleRealsense(sh, "f1230727")
serials = [
'123622270136', # front
'128422271784' # back
]
cams = MultiRealsense(serial_numbers=serials,
                      resolution=(640,480))
vis = MultiCameraVisualizer(cams, 2, 1)

# cam = SingleRealsense(sh, "128422271784")
cams.start()
cams.set_exposure(exposure=5000, gain=60)

# back_intrinsics = cams.cameras['128422271784'].get_intrinsics()
# front_intrinsics = cams.cameras['123622270136'].get_intrinsics()

# back_distortion = cams.cameras['128422271784'].get_dist_coeffs()
# front_distortion = cams.cameras['123622270136'].get_dist_coeffs()

# camera_info = {
#     'back': {
#         'intrinsics': back_intrinsics.tolist(),
#         'distortion': back_distortion.tolist()
#     },
#     'front': {
#         'intrinsics': front_intrinsics.tolist(),
#         'distortion': front_distortion.tolist()
#     }
# }

# # save to json
# with open('camera_info.json', 'w') as f:
#     json.dump(camera_info, f, indent=4)



# pdb.set_trace()

# vis.run()
marker_detector_1 = ArucoDetector(cams.cameras['128422271784'], 0.025, aruco.DICT_4X4_50, 10, visualize=True)
# marker_detector_2 = ArucoDetector(cams.cameras['123622270136'], 0.025, aruco.DICT_4X4_50, 3, visualize=True)

while True:
    print(marker_detector_1.estimate_pose())
    # print(marker_detector_2.estimate_pose())




# import pyrealsense2 as rs
# import numpy as np
# import cv2
# import threading

# def stream_camera(serial_number, window_name):
#     # Configure depth and color streams
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_device(serial_number)  # Use the serial number for the camera
#     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#     # Start the pipeline
#     pipeline.start(config)

#     try:
#         while True:
#             # Wait for a coherent pair of frames: depth and color
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             if not color_frame:
#                 continue

#             # Convert images to numpy arrays
#             color_image = np.asanyarray(color_frame.get_data())

#             # Show images
#             cv2.imshow(window_name, color_image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         # Stop streaming
#         pipeline.stop()

# # Camera serial numbers
# camera_1_serial = '123622270136'
# # camera_2_serial = '128422271784'

# # Create threads for each camera
# thread_camera_1 = threading.Thread(target=stream_camera, args=(camera_1_serial, 'Camera 1'))
# # thread_camera_2 = threading.Thread(target=stream_camera, args=(camera_2_serial, 'Camera 2'))

# # Start the threads
# thread_camera_1.start()
# # thread_camera_2.start()

# # Join the threads
# thread_camera_1.join()
# # thread_camera_2.join()

# cv2.destroyAllWindows()






# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_device('123622270136') # front
# # config.enable_device('128422271784') # back

# # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start the pipeline
# pipeline.start(config)

# while True:

#     # Wait for a coherent pair of frames: depth and color
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     # if not color_frame:
#         # continue

#     # Convert images to numpy arrays
#     color_image = np.asanyarray(color_frame.get_data())

#     # Show images
#     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('RealSense', color_image)
#     cv2.waitKey(1)


