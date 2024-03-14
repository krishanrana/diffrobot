from teleop import Teleop
from dataclasses import dataclass
from pathlib import Path
import tyro
from shutil import rmtree


@dataclass
class Params:
    path: Path 
    num_poses: int = 10
    overwrite: bool = False

if __name__ == "__main__":
    params = tyro.cli(Params)

    file_path = params.path
    if file_path.exists():
        if params.overwrite:
            print(f"Overwriting {file_path}")
        else:
            raise ValueError(f"{file_path} already exists. Use --overwrite to overwrite.")
    
    assert not file_path.is_dir() and file_path.suffix == ".json"
    file_path.parent.mkdir(parents=True, exist_ok=True)


    teleop = Teleop()
    teleop.home_robot()

    res = [] 
    def on_button_press():
        q = teleop.panda.q
        # q = teleop.robot_pose
        res.append(q.tolist())
        print(f"Saved pose {len(res)}/{params.num_poses}")  
        if len(res) == params.num_poses:
            # save to file
            import json
            print(f"Saving to {file_path}")
            with open(file_path, "w") as f:
                json.dump(res, f)
            teleop.relinquish()

    teleop.gello_button_stream.subscribe(lambda _: on_button_press())
    teleop.take_control()
    teleop.home_robot()