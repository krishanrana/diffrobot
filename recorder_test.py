from realsense.multi_realsense import MultiRealsense
# from teleop import Teleop
from teleop_cartesian import Teleop
import time
import json
import numpy as np
from realsense.multi_camera_visualizer import MultiCameraVisualizer
from dataclasses import dataclass

import tyro

@dataclass
class Params:
    name: str

if __name__ == "__main__":
    params = tyro.cli(Params)

    t = Teleop()
    t.home_robot()
    cams = MultiRealsense(
        # resolution=(640, 480),
        serial_numbers=[
            '317222071463', # gripper
            '032522250135', # top
            '035122250388', # side top
            '825312071857', 
            # '036522071747', 
            # '825312071857', 
            # '035122250388', # side top
            # '032522250135', # top
            ],
        enable_depth=False
    )
    vis = MultiCameraVisualizer(cams, row=4, col=1)
    cams.start()
    cams.set_exposure(exposure=100, gain=60)
    vis.start()
    time.sleep(1.0)
    # cams.start_recording(params.name)

    # t.take_control_async()
    t.take_control()
    # time.sleep(1)
    # qs = []
    # for i in range(200):
    #     print(i)
    #     qs.append(np.array(t.get_translation()).tolist())
    #     cams.record_frame()
    #     time.sleep(1/30.0)

    # with open(f"{params.name}/qs.json", "w") as f:
    #     json.dump(qs, f, indent=4)

    # cams.stop_recording()
    cams.stop(wait=True)
    t.relinquish()
    t.home_robot()
