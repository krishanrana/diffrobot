import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import pdb

def main():
    # Set up the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)

    # ArUco marker setup
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    marker_id = 3

    # get intrinsics from color stream
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # The K matrix (intrinsic matrix)
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
         [0, intrinsics.fy, intrinsics.ppy],
         [0, 0, 1]])
    

    # get distortion coefficients from depth stream
    depth_stream = profile.get_stream(rs.stream.depth)
    distcoeffs = np.array(depth_stream.as_video_stream_profile().get_intrinsics().coeffs)


    try:
        while True:
            # Wait for a new set of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # ArUco marker detection
            corners, ids, _ = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)
            if ids is not None and marker_id in ids:
                idx = np.where(ids == marker_id)
                corners = np.array(corners)[idx]
                ids = np.array(ids)[idx]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.025, K, distcoeffs)
                # tvecs[:, 0, 1] -= 0.094
                for i in range(len(rvecs)):
                    color_image = cv2.drawFrameAxes(color_image, K, distcoeffs, rvecs[i], tvecs[i], 0.05)

                #show frame
                cv2.imshow("frame", color_image)
                cv2.waitKey(1)

            # Display the image
            cv2.imshow('RealSense ArUco Detection', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
