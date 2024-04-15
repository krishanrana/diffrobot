import roboticstoolbox as rtb
import spatialmath as sm
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pdb

def compute_oriented_affordance_frame(transform_matrix):
    """
    Compute the angle needed to rotate the x-axis of a given transformation
    matrix so that it points towards the origin. Apply the rotation to the
    transformation matrix and return the resulting matrix.

    Parameters:
    - transform_matrix: A 4x4 numpy array representing the homogeneous transformation matrix.

    Returns:
    - The transformation matrix with the x-axis pointing towards the origin.
    """
    # Extract the translation components (P_x, P_y) from the matrix
    P_x, P_y = transform_matrix[0, 3], transform_matrix[1, 3]
    
    # Calculate the angle between the vector pointing from the frame's current position
    # to the origin and the global x-axis. This uses atan2 and is adjusted by 180 degrees
    # to account for the direction towards the origin.
    angle_to_origin = np.degrees(np.arctan2(-P_y, -P_x))
    
    # Calculate the initial orientation of the frame's x-axis relative to the global x-axis.
    # This is the angle of rotation about the z-axis that has already been applied to the frame.
    # We use the elements of the rotation matrix to find this angle.
    R11, R21 = transform_matrix[0, 0], transform_matrix[1, 0]
    initial_orientation = np.degrees(np.arctan2(R21, R11))
    
    # Compute the additional rotation needed from the frame's current orientation.
    # This is the difference between the angle to the origin and the frame's initial orientation.
    additional_rotation = angle_to_origin - initial_orientation
    
    # Normalize the result to the range [-180, 180]
    additional_rotation = (additional_rotation + 180) % 360 - 180

    # Create a new transformation matrix that applies the additional rotation to the original matrix.

    og_pose = sm.SE3(transform_matrix, check=False).norm()
    T2 = og_pose * sm.SE3.Rz(np.deg2rad(additional_rotation))

    return T2



# T1 = sm.SE3(4,0,0) * sm.SE3.Rz(30, 'deg') * sm.SE3.Ry(10, 'deg')

# sm.SE3().plot(frame='0', dims=[-5, 5], color='black')
# T1.plot(frame='1', color='red')
# plt.grid(True)

# alpha = compute_angle_to_origin(np.array(T1))

# print(f"Angle to origin: {alpha:.2f} degrees")

# # create a new SE3 object that is a rotation of T1 about the Z-axis by theta
# T2 = T1 * sm.SE3.Rz(np.deg2rad(alpha))

# plt.figure()
# sm.SE3().plot(frame='0', dims=[-5, 5], color='black')
# T1.plot(frame='1', color='red')
# T2.plot(frame='2', color='blue')
# plt.grid(True)
# plt.show()



dataset_path = "/home/krishan/work/2024/datasets/cup_rotate_X"

# sort numerically the episodes based on folder names
episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
for episode in episodes:
    # read the state.json file which consists of a list of dictionaries
    start_pose = json.load(open(os.path.join(dataset_path, "episodes", episode, "object_frame.json")))['X_BO']
    end_pose = json.load(open(os.path.join(dataset_path, "episodes", episode, "object_frame_last.json")))['X_BO']

    # as SE3 objects
    start_pose = sm.SE3(start_pose, check=False).norm()
    end_pose = sm.SE3(end_pose, check=False).norm()

    X_BOO = compute_oriented_affordance_frame(np.array(start_pose))

    sm.SE3().plot(frame='0', dims=[-5, 5], color='black')
    start_pose.plot(frame='1', color='red')
    end_pose.plot(frame='2', color='blue')
    X_BOO.plot(frame='3', color='green')
    plt.grid(True)

    plt.show()

    pdb.set_trace()