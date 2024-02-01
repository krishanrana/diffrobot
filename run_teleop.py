import numpy as np
from threading import Thread

import reactivex as rx
from reactivex import operators as ops

import panda_py
from panda_py import controllers, libfranka
from dynamixel.robot import DynamixelRobot

hostname = "172.16.0.2"
MAX_OPEN = 0.07754

def main():
	# Panda
	panda = panda_py.Panda(hostname)
	gripper = libfranka.Gripper(hostname)
	home_q = np.deg2rad([0, 0, 0, -90, 0, 90, 45])

	print("Homing gripper and robot")
	panda.move_to_joint_position(home_q)
	gripper.homing()

	# Gello
	gello = create_gello()

	# Check discrepency between Gello and Panda
	gello_q = gello.get_joint_state()[:7]
	assert check_joint_discrepency(gello_q, panda.q), "Gello and Panda are not in the same configuration"
	print(f"Passed joint discrepency check")

	# Syncronize Gello and Panda
	print(f"Syncing Gello and Panda")
	panda.move_to_joint_position(gello_q)

	# Gripper Control
	def gripper_callback_async(x):
		def gripper_callback(x):
			gripper.stop()
			if x == "open":
				gripper.move(width=MAX_OPEN, speed=0.2)
			elif x == "close":
				gripper.move(width=0.01, speed=0.2)
		Thread(target=gripper_callback, args=(x,)).start()

	frequency = 2.0
	threshold = 0.1
	rx.interval(1.0/frequency, scheduler=rx.scheduler.NewThreadScheduler()) \
		.pipe(ops.map(lambda _: gello.get_joint_state())) \
		.pipe(ops.map(lambda x: np.round(x[-1], 2))) \
		.pipe(ops.distinct_until_changed()) \
		.pipe(ops.pairwise()) \
		.pipe(ops.filter(lambda x: np.abs(x[0] - x[1]) > threshold)) \
		.pipe(ops.map(lambda x: "open" if x[0] > x[1] else "close")) \
		.pipe(ops.distinct_until_changed()) \
		.subscribe(lambda x: gripper_callback_async(x))
	

	# Robot Control
	print("-----------------YOU ARE IN CONTROL-----------------")
	stiffness = [40, 30, 50, 25, 35, 25, 10]
	damping = [4, 6, 5, 5, 3, 2, 1]
	ctrl = controllers.JointPosition(stiffness=stiffness, damping=damping, filter_coeff=0.9)
	panda.start_controller(ctrl)
	with panda.create_context(frequency=100) as ctx:
		while ctx.ok():
			gello_q = gello.get_joint_state()
			ctrl.set_control(gello_q[:7])

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

if __name__ == "__main__":
	main()