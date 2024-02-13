from scipy.spatial.transform.rotation import Rotation as R
import numpy as np

def quat_to_euler(quat):
    """xyz euler angles from xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")

def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()