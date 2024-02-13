from realsense.multi_realsense import MultiRealsense
# from teleop import Teleop
from teleop_cartesian_frankx import Teleop
import time
import json
import numpy as np
from realsense.multi_camera_visualizer import MultiCameraVisualizer
from dataclasses import dataclass
from pathlib import Path
import os
from calibration.aruco_detector import ArucoDetector, aruco

import tyro

@dataclass
class Params:
    name: str
    idx: int = 0



if __name__ == "__main__":
    params = tyro.cli(Params)

    path = Path(f"data/{params.name}")
    path.mkdir(parents=True, exist_ok=True)

    t = Teleop()
    t.home_robot()
    record_fps = 30
    cams = MultiRealsense(
        # resolution=(640, 480),
        record_fps=record_fps,
        serial_numbers=[
            '317222071463', # gripper
            '032522250135', # top
            '035122250388', # side top
            # '825312071857', 
            # '036522071747', 
            # '825312071857', 
            # '035122250388', # side top
            # '032522250135', # top
            ],
        enable_depth=False
    )
    vis = MultiCameraVisualizer(cams, row=3, col=1)
    cams.start()
    cams.set_exposure(exposure=100, gain=60)
    vis.start()
    time.sleep(1.0)
    # cams.start_recording(params.name)

    t.gello_button_stream.subscribe(lambda _: t.toggle_record())

    t.take_control_async()
    # t.take_control()
    time.sleep(5)
    state = []

    idx = params.idx

    cam_side = cams.cameras['035122250388']
    cam_top = cams.cameras['032522250135']
    #marker_detector_top = ArucoDetector(cam_top, 0.039, aruco.DICT_4X4_50, 37, visualize=False)
    #marker_detector_side = ArucoDetector(cam_side, 0.039, aruco.DICT_4X4_50, 37, visualize=False)

    # Save camera intrinsics for cam_side
    cam_side_intrinsics = cam_side.get_intrinsics()
    cam_top_intrinsics = cam_top.get_intrinsics()


    with open(f"data/{params.name}/cam_side_intrinsics.json", "w") as f:
        json.dump(cam_side_intrinsics.tolist(), f, indent=4)
    with open(f"data/{params.name}/cam_top_intrinsics.json", "w") as f:
        json.dump(cam_top_intrinsics.tolist(), f, indent=4)

    while True:

        if t.record_data:
            print("Recording demonstration {}".format(idx))
            # make a new directory for idx
            path = Path(f"data/{params.name}/{idx}/video")
            path.mkdir(parents=True, exist_ok=True)
            cams.start_recording(str(path))
            desired_time = 1.0/record_fps
            while t.record_data: 
                start = time.time()
                state.append({"X_BE" : np.array(t.get_tcp_pose()).tolist(),
                             "q" : np.array(t.get_joint_positions()).tolist(),})
                             #"marker_pose_top" : marker_detector_top.estimate_pose(),
                             #"marker_pose_side" : marker_detector_side.estimate_pose(),})
                cams.record_frame()
                duration = time.time()-start
                sleep_for = max(desired_time - duration, 0)
                time.sleep(sleep_for)
                # print(f"Time: {time.time()-start} - Slept for {sleep_for} - Actual Freq: {1.0/(time.time()-start)} Hz - Reqeuired Freq: {record_fps} Hz")
            
            # make file
            
            cams.stop_recording()
            with open(f"data/{params.name}/{idx}/state.json", "w") as f:
                 json.dump(state, f, indent=4)

            # store as npy
            #np.save(f"data/{params.name}/{idx}/state.npy", state)


            idx+=1

        else:
            print("Resetting...")
            state = []           
            while not t.record_data:
                time.sleep(1/30.0)
            

                # print(marker_pose_top)
                

            

    cams.stop_recording()
    cams.stop(wait=True)
    t.relinquish()
    t.home_robot()
