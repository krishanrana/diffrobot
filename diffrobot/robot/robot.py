from frankx import JointMotion, Kinematics, Affine, LinearMotion, PathMotion
from frankx import Robot as Panda, WaypointMotion, Waypoint
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from panda_py._core import ik
import reactivex as rx
from reactivex import operators as ops

def to_affine(pos, orn):
    """
    pos: [x, y, z]
    orn: [x, y, z, w]
    """
    # convert to wxyz
    orn = np.array([orn[3], orn[0], orn[1], orn[2]])
    return Affine(*pos, *orn)

def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler("xyz", orn).as_matrix()
    mat[:3, 3] = pos
    return mat

def orn_to_matrix(orn):
    mat = np.eye(3)
    if len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler("xyz", orn).as_matrix()
    return mat


def matrix_to_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    return R.from_matrix(mat[:3, :3]).as_quat()


def matrix_to_pos_orn(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    orn = R.from_matrix(mat[:3, :3]).as_quat()
    pos = mat[:3, 3]
    return pos, orn

def matrix_to_affine(mat):
    """
    :param mat: 4x4 homogeneous transformation
    :return: Affine
    """
    pos, orn = matrix_to_pos_orn(mat)
    return to_affine(pos, orn)


class Robot:
    def __init__(self, hostname: str = "172.16.0.2"):
        self.frankx = Panda(hostname, repeat_on_error=True, dynamic_rel=0.1)
        self.frankx.recover_from_errors()
        self.frankx.set_dynamic_rel(0.1)
        self.home_joints = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
        self.motion = None

        # frequency = 10
        # self.pose_stream = rx.interval(1.0/frequency, scheduler=rx.scheduler.NewThreadScheduler()) \
        #     .pipe(ops.map(lambda _: self.get_tcp_pose())) \
		# 	.pipe(ops.share())
    
    def recover_from_errors(self):
        self.frankx.recover_from_errors()
    
    def waypoints(self, affines: list[Affine]):
        waypoints = [Waypoint(affine) for affine in affines]
        self.motion = WaypointMotion(waypoints)
        self.frankx.move(self.motion)
        self.motion = None
    
    def path(self, waypoints: list[Affine], blend: float = 0.3):
        self.motion = PathMotion(waypoints, blend_max_distance=blend)
        self.frankx.move(self.motion)
        self.motion = None
    
    def move_to_joints(self, q):
        self.frankx.move(JointMotion(q))

    def move_to_start(self):
        self.move_to_joints(self.home_joints)

    def move_to_pose_linear(self, pos, orn):
        self.frankx.move(LinearMotion(to_affine(pos, orn))) # w, x, y, z
    
    def move_to_pose(self, pos, orn):
        X = pos_orn_to_matrix(pos, orn)
        q = ik(X)
        self.move_to_joints(q)
    
    def set_dynamic_rel(self, val:float, accel_rel:float=0.01, jerk_rel:float=0.01):
        self.frankx.set_dynamic_rel(val)
        self.frankx.jerk_rel = jerk_rel
        # self.frankx.accel_rel = 0.08
        self.frankx.accel_rel = accel_rel
    
    def get_orientation(self):
        q = self.frankx.current_pose().quaternion()
        q = np.array([q[1], q[2], q[3], q[0]])
        return q
    
    def get_joints(self):
        s = self.frankx.read_once()
        return s.q
    
    def get_tcp_pose(self):
        if self.motion == None:
            pose = self.frankx.current_pose()
            pos, orn = np.array(pose.translation()), np.array(pose.quaternion())
            orn = np.array([orn[1], orn[2], orn[3], orn[0]]) # xyzw
            return pos_orn_to_matrix(pos, orn)
        else:
            pose = np.array(self.motion.get_robot_state().O_T_EE).reshape(4, 4).T
            return pose

    
    def has_errors(self):
        return self.frankx.has_errors()
    
    def start_cartesian_controller(self):
        motion = WaypointMotion([Waypoint(self.frankx.current_pose())], return_when_finished=False)
        thread = self.frankx.move_async(motion)
        return motion

    

