import panda_py
from diffrobot.dynamixel.robot import DynamixelRobot
import numpy as np
from diffrobot.robot.robot import Robot, to_affine, pos_orn_to_matrix, matrix_to_affine, matrix_to_pos_orn
import reactivex as rx
from reactivex import operators as ops

import time
from frankx import Waypoint
import pdb
from spatialmath import SE3
import open3d as o3d
from diffrobot.robot.visualizer import RobotViz
from diffrobot.tactile_sensors.xela import SensorSocket
import spatialmath as sm

class Teleop:
	def __init__(self, hostname: str = "172.16.0.2"):
		self.panda = Robot(hostname)
		self.gripper = self.panda.gripper
		self.panda.set_dynamic_rel(0.4)
		

		self.gello = create_gello()
		# self.home_q = np.deg2rad([-90, 0, 0, -90, 0, 90, 45]) # left
		self.home_q = np.deg2rad([0, 0, 0, -90, 0, 90, 45]) # front
		self.stop_requested = False
		self.motion = None
		self.create_gello_streams()
		self.robot_pose = None
		self.constrain_pose = False
		self.saved_trans = None
		self.saved_orien = None
		self._callback = None


		self.robot_visualiser = RobotViz()
		self.robot_visualiser.step(self.home_q, self.home_q)

	def set_callback(self, callback):
		self._callback = callback

	def get_translation(self):
		if self.motion:
			return self.motion.current_pose().translation()
		else:
			return self.panda.get_tcp_pose()[:3, 3]
	
	def get_tcp_pose(self):
		if self.motion:
			pose = self.motion.current_pose()
			pos, orn = np.array(pose.translation()), np.array(pose.quaternion())
			orn = np.array([orn[1], orn[2], orn[3], orn[0]]) # xyzw
			return pos_orn_to_matrix(pos, orn)
		else:
			return self.panda.get_tcp_pose()
		
	def get_joint_torques(self):
		return np.array(self.motion.get_robot_state().tau_ext_hat_filtered)
	
	def get_ee_forces(self):
		return np.array(self.motion.get_robot_state().K_F_ext_hat_K)

		
	def get_joint_positions(self):
		return self.motion.get_robot_state().q
		
	def home_robot(self):
		self.panda.move_to_joints(self.home_q)
	
	def can_control(self) -> bool:
		gello_q = self.gello.get_joint_state()[:7]
		q = self.panda.get_joints()
		return check_joint_discrepency(gello_q, q)

	def visualise_poses(self, pose_matrices):
		# Create Open3D visualization window
		visualizer = o3d.visualization.Visualizer()
		visualizer.create_window()

		# Iterate over each pose matrix
		for pose_matrix in pose_matrices:
			# Create a mesh representing the coordinate frame
			frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

			# Apply the pose transformation to the coordinate frame
			frame_mesh.transform(pose_matrix)

			# Add coordinate frame mesh to the visualization window
			visualizer.add_geometry(frame_mesh)

		# Set viewpoint
		visualizer.get_view_control().set_front([0, 0, 1])
		visualizer.get_view_control().set_up([0, -1, 0])
		visualizer.get_view_control().set_lookat([0, 0, 0])
		visualizer.get_view_control().convert_from_pinhole_camera_parameters(
			o3d.camera.PinholeCameraParameters())

		# Run the visualization
		visualizer.run()


	
	def take_control(self):
		assert self.stop_requested == False
		if not self.can_control():
			raise Exception("Gello and Panda are not in the same configuration")
		self.panda.move_to_joints(self.gello.get_joint_state()[:7])
		# q0 = np.array([0.99954873, -0.02642627,  0.01265948, -0.00661308])
		
		# Move/Save to desired height and orientation
		# self.panda.move_to_joints([-1.56832675,  0.39303148,  0.02632776, -1.98690212, -0.00319773,  2.35042797, 0.94667396])
		pose = self.panda.get_tcp_pose()
		trans = pose[:3, 3]
		z_height = trans[2]
		orien = self.panda.get_orientation()
		temp_pose = pose
		

		print("---------YOU ARE IN CONTROL--------")

		#self.panda.set_dynamic_rel(0.7, accel_rel=0.01, jerk_rel=0.01)
		self.panda.set_dynamic_rel(1.0, accel_rel=0.2, jerk_rel=0.05)
		self.panda.frankx.set_collision_behavior(
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
			[30.0, 30.0, 30.0, 30.0, 30.0, 30.0]
		)
		#self.panda.frankx.set_cartesian_impedance([30.0,30.0,30.0,10.0,10.0,10.0])
		# self.motion = self.panda.start_cartesian_controller()
		self.motion = self.panda.start_impedance_controller(200, 40, 5)
		while not self.stop_requested:
			gello_q = self.gello.get_joint_state()
			pose = self.robot_visualiser.robot.fkine(np.array(gello_q[:7]), "panda_link8") * self.robot_visualiser.X_FE
			pose = pose.A
			# pose = panda_py.fk(np.round(gello_q[:7],4)) @ self.
			robot_state = self.motion.get_robot_state()
			gripper_width = self.gripper.width()

			# check if robot_state is close to zeros all the values
			if np.all(np.abs(robot_state.O_T_EE) < 0.01):
				continue	

			if self._callback:
				x= {"robot_state": robot_state,
				"gello_q": gello_q,
				"gripper_width": gripper_width}
				self._callback(x)

			print(robot_state.O_T_EE)
			self.robot_visualiser.ee_pose.T = sm.SE3((np.array(robot_state.O_T_EE)).reshape(4,4).T, check=False).norm()
			print(gello_q)
			# import pdb; pdb.set_trace()
			target_pose = self.robot_visualiser.robot.fkine(np.array(gello_q[:7]), "panda_link8") * self.robot_visualiser.X_FE
				
			self.robot_visualiser.policy_pose.T = target_pose 
			self.robot_visualiser.step(robot_state.q, gello_q[:7])



			# print(gello_q[-1])
			# se3 = SE3(pose)

			# print('Gello: ', gello_q[:7])
			# print('Franka: ', self.panda.get_joints())

			# poses_matrices = [temp_pose, pose]
			# self.visualise_poses(poses_matrices)
			
			self.trans, self.orien = matrix_to_pos_orn(pose)

			if self.constrain_pose:
				self.orien = self.saved_orien
				self.trans[2] = self.saved_trans[2]

			# print(orien)
			# trans[2] = z_height
			#print(np.array(self.motion.get_robot_state().tau_ext_hat_filtered).round(3))
			# print(np.array(self.motion.get_robot_state().K_F_ext_hat_K).round(3))
			# self.motion.set_next_waypoint(Waypoint(to_affine(self.trans, self.orien)))
			self.motion.set_target(to_affine(self.trans, self.orien))
			# self.motion.set_next_waypoint(Waypoint(pose))
			

			time.sleep(1/30.0)
		self.stop_requested = False
		self.motion = None
		print("--------RELINQUISHED CONTROL-------")
	
	def relinquish(self):
		self.stop_requested = True
	

	def take_control_async(self):
		from threading import Thread
		self.thread = Thread(target=self.take_control)
		self.thread.start()
	
	def create_gello_streams(self, frequency=10.0):
		self.gello_joints_stream = rx.interval(1.0/frequency, scheduler=rx.scheduler.NewThreadScheduler()) \
			.pipe(ops.map(lambda _: self.gello.get_joint_state())) \
			.pipe(ops.map(lambda x: np.round(x[-1], 2))) \
			.pipe(ops.distinct_until_changed()) \
			.pipe(ops.share())
		
		threshold = 0.05
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
					0 * np.pi / 2,
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
	q1 = np.array(q1)
	q2 = np.array(q2)
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
	def relinquish_and_home():
		teleop.relinquish()
		teleop.home_robot()
	# teleop.gello_button_stream.subscribe(lambda _: relinquish_and_home())
	def grasp(x):
		print(x)
		if x == "open":
			teleop.gripper.open()
			teleop.constrain_pose = False
		else:
			teleop.gripper.close()
			teleop.saved_trans = teleop.trans
			teleop.saved_orien = teleop.orien
			teleop.constrain_pose = True

	# teleop.gello_gripper_stream_2.subscribe(lambda x: grasp(x))
	# teleop.gello_button_stream.subscribe(lambda x: relinquish_and_home())
	teleop.take_control()
