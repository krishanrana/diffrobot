from dataclasses import dataclass
from pathlib import Path
import tyro
import json
from robot.robot import Robot
import numpy as np
import time


@dataclass
class Params:
    path: Path 

if __name__ == "__main__":
    params = tyro.cli(Params)
    assert params.path.exists() and params.path.is_file() and params.path.suffix == ".json"
    # read file
    with open(params.path, "r") as f:
        data = json.load(f)
    
    robot = Robot()
    robot.set_dynamic_rel(0.1)
    robot.move_to_start()
    
    for i, q in enumerate(data):
        print(f"Moving to pose {i+1}/{len(data)}")
        q = np.array(q)
        robot.move_to_joints(q)
        time.sleep(0.5)
    print("Rehoming")
    robot.move_to_start()