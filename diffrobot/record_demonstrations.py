from diffrobot.realsense.multi_realsense import MultiRealsense
from diffrobot.teleoperation.teleop_cartesian_frankx import Teleop
import time
import json
import numpy as np
from pathlib import Path
from diffrobot.tactile_sensors.xela import SensorSocket
import tyro
from dataclasses import dataclass
import cv2


@dataclass
class Params:
    name: str
    idx: int = 0

class DataRecorder:
    def __init__(self, params):
        self.params = params
        self.t = Teleop()
        self.record_fps = 10
        self.cams = None
        self.sensor_socket = None
        self.idx = params.idx
        
        self.window_name = "Data Recorder"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.demo_state_text = "Resetting..."
        self.demo_number_text = f"Demo Number: {self.idx}"

        self.record_data = False
        self.toggle_key = ord(' ')  # ASCII code for space bar
        self.discard_key = ord('d')  # ASCII code for 'd'
        self.key = None
        
    def setup(self):
        self.t.home_robot()
        self.cams = MultiRealsense(
            record_fps=self.record_fps,
            serial_numbers=['f1230727'],
            enable_depth=True
        )
        self.cams.start()
        self.cams.set_exposure(exposure=100, gain=60)
        self.sensor_socket = SensorSocket("131.181.33.191", 5000)

    def grasp(self, x):
        print(x)
        if x == "open":
            self.t.gripper.open()
            # self.t.constrain_pose = False
        else:
            self.t.gripper.close()
            self.t.saved_trans = self.t.trans
            self.t.saved_orien = self.t.orien
            # self.t.constrain_pose = True

    def toggle_record(self):
        self.record_data = not self.record_data
        if self.record_data:
            self.demo_state_text = "Recording..."
            print("Recording demonstration {}".format(self.idx))
        else:
            self.demo_state_text = "Resetting..."
            print("Recording stopped.")
            print("Resetting...")

    
    def update_window(self):
        # Create a black background
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Set the color based on mode
        color = (0, 0, 255) if self.record_data else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (400, 400), color, -1)
        
        # Add text for demonstration state and number
        cv2.putText(frame, self.demo_state_text, (50, 180), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Demo Number: {self.idx}", (50, 250), self.font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Show the window
        cv2.imshow(self.window_name, frame)


    def start_recording(self):
        cv2.namedWindow(self.window_name)
        cv2.moveWindow(self.window_name, 400, 400)
        while True:
            self.key = cv2.waitKey(1) & 0xFF
            # if key is the space bar or shift key, toggle recording

            if self.key == self.toggle_key:
                self.toggle_record()

            if self.record_data:
                self._record_data()
                # Only increment the index if the recording was not discarded
                if self.key == self.toggle_key:
                    self.idx += 1
                else:
                    print("Discarding demonstration {}".format(self.idx))
            else:
                time.sleep(1/self.record_fps)
            self.update_window()
                

        

    def _record_data(self):
        path = Path(f"data/{self.params.name}/{self.idx}/video")
        path.mkdir(parents=True, exist_ok=True)
        self.cams.start_recording(str(path))
        state = []
        desired_time = 1.0 / self.record_fps

        self.update_window()
        while self.record_data:
            start = time.time()
            state.append({
                "X_BE": np.array(self.t.get_tcp_pose()).tolist(),
                "robot_q": np.array(self.t.get_joint_positions()).tolist(),
                "tactile_sensors": np.array(self.sensor_socket.get_forces()).tolist(),
                "joint_torques": np.array(self.t.get_joint_torques()).tolist(),
                "ee_forces": np.array(self.t.get_ee_forces()).tolist(),
                "gello_q": np.array(self.t.gello.get_joint_state()[:7]).tolist(),
                "gripper_state": self.t.gripper.width()
            })
            self.cams.record_frame()

            self.key = cv2.pollKey() & 0xFF
            if self.key == self.toggle_key or self.key == self.discard_key:
                self.toggle_record()

            duration = time.time() - start
            sleep_for = max(desired_time - duration, 0)
            time.sleep(sleep_for)
            #print(f"Time: {time.time()-start} - Slept for {sleep_for} - Actual Freq: {1.0/(time.time()-start)} Hz - Required Freq: {self.record_fps} Hz")

            

        self.cams.stop_recording()
        with open(f"data/{self.params.name}/{self.idx}/state.json", "w") as f:
            json.dump(state, f, indent=4)
        

    def stop(self):
        self.cams.stop(wait=True)
        self.t.relinquish()
        self.t.home_robot()
        cv2.destroyAllWindows()

    def run(self):
        try:
            self.setup()
            self.t.gello_gripper_stream.subscribe(lambda x: self.grasp(x))
            self.t.take_control_async()
            self.start_recording()
        finally:
            self.stop()

if __name__ == "__main__":
    params = tyro.cli(Params)
    data_recorder = DataRecorder(params)
    data_recorder.run()
