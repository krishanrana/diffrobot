from frankx import Gripper
import time

gripper = Gripper("172.16.0.2", opening_threshold=0.4)
gripper.open(blocking=True)
# gripper.close(blocking=True)
time.sleep(1)
