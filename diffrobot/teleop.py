import panda_py
from dynamixel.robot import DynamixelRobot
import numpy as np
from panda_py import controllers
import reactivex as rx
from reactivex import operators as ops

class Teleop:
	def __init__(self, hostname: str = "172.16.0.2"):
		self.panda = panda_py.Panda(hostname)
		self.gello = create_gello()
		self.home_q = np.deg2rad([-90, 0, 0, -90, 0, 90, 45])
		self.stop_requested = False
		self.create_gello_streams()
	
	def home_robot(self):
		self.panda.move_to_joint_position(self.home_q)
	
	def can_control(self) -> bool:
		gello_q = self.gello.get_joint_state()[:7]
		self.panda.get_robot().read_once()
		return check_joint_discrepency(gello_q, self.panda.q)
	
	def take_control(self):
		assert self.stop_requested == False
		if not self.can_control():
			raise Exception("Gello and Panda are not in the same configuration")
		self.panda.move_to_joint_position(self.gello.get_joint_state()[:7])
		stiffness = [40, 30, 50, 25, 35, 25, 10]
		damping = [4, 6, 5, 5, 3, 2, 1]
		ctrl = controllers.JointPosition(stiffness=stiffness, damping=damping, filter_coeff=0.9)
		self.panda.start_controller(ctrl)
		print("---------YOU ARE IN CONTROL--------")
		with self.panda.create_context(frequency=100) as ctx:
			while ctx.ok() and not self.stop_requested:
				gello_q = self.gello.get_joint_state()
				ctrl.set_control(gello_q[:7])
		self.stop_requested = False
		print("--------RELINQUISHED CONTROL-------")
	
	def relinquish(self):
		self.stop_requested = True
	
	def take_control_async(self):
		from threading import Thread
		self.thread = Thread(target=self.take_control)
		self.thread.start()
		# self.thread.run()
	
	def create_gello_streams(self, frequency=2.0):
		self.gello_joints_stream = rx.interval(1.0/frequency, scheduler=rx.scheduler.NewThreadScheduler()) \
			.pipe(ops.map(lambda _: self.gello.get_joint_state())) \
			.pipe(ops.map(lambda x: np.round(x[-1], 2))) \
			.pipe(ops.distinct_until_changed()) \
			.pipe(ops.share())
		
		threshold = 0.1
		self.gello_gripper_stream = self.gello_joints_stream \
			.pipe(ops.pairwise()) \
			.pipe(ops.filter(lambda x: np.abs(x[0] - x[1]) > threshold)) \
			.pipe(ops.map(lambda x: "open" if x[0] > x[1] else "close")) \
			.pipe(ops.distinct_until_changed()) 
		
		self.gello_button_stream = self.gello_gripper_stream \
			.pipe(ops.filter(lambda x: x == "close")) \
			.pipe(ops.map(lambda _: True)) \


def create_gello() -> DynamixelRobot:
	return DynamixelRobot(
				port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FAFX-if00-port0",
				real=True,
				joint_ids=(1, 2, 3, 4, 5, 6, 7),
				joint_offsets=(
					1 * np.pi / 2,
					2 * np.pi / 2,
					4 * np.pi / 2,
					2 * np.pi / 2,
					2 * np.pi / 2,
					2 * np.pi / 2,
					(2 * np.pi / 2) - np.pi/4,
				),
				joint_signs=(1, 1, 1, 1, 1, -1, 1),
				gripper_config=(8, 195, 153),
			)

def check_joint_discrepency(q1, q2) -> bool:
	abs_deltas = np.abs(q1 - q2)
	id_max_joint_delta = np.argmax(abs_deltas)
	max_joint_delta = 0.8
	res = True
	if abs_deltas[id_max_joint_delta] > max_joint_delta:
		id_mask = abs_deltas > max_joint_delta
		print()
		ids = np.arange(len(id_mask))[id_mask]
		for i, delta, joint, current_j in zip(
			ids,
			abs_deltas[id_mask],
			q1[id_mask],
			q2[id_mask],
		):
			print(f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}")
			res = False
	return res

if __name__ == "__main__":
	teleop = Teleop()
	teleop.home_robot()
	teleop.gello_button_stream.subscribe(lambda _: teleop.relinquish())
	teleop.take_control()