import os
import pdb
import cv2
import json
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
import pickle

from diffrobot.diffusion_policy.utils.rotation_transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
import roboticstoolbox as rtb
import spatialmath as sm


def create_sample_indices(sequence_length:int,
                          dataset_path:str):
    
    # iterate through all the episode folders
    indices = list()

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        state = json.load(open(os.path.join(dataset_path, "episodes", episode, "state.json")))

        # get the length of the episode
        episode_length = len(state)
        # iterate through the episode
        range_idx = episode_length - (sequence_length + 2)
        for idx in range(range_idx):
            # get the start and end index of the sequence
            buffer_start_idx = idx
            buffer_end_idx = idx + sequence_length
            indices.append([int(episode), buffer_start_idx, buffer_end_idx])
            assert buffer_end_idx - buffer_start_idx == sequence_length
        
    indices = np.array(indices)
    return indices


def flatten_2d_lists(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def get_data_stats(data: list):
    data = np.array(flatten_2d_lists(data))
    stats = {
       'min': np.min(data, axis=0),
       'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


# create an image based dataset from video
def decode_video(dataset_path:str):
    state = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
            # Paths to the video files
        vp_front = os.path.join(dataset_path, "episodes", str(episode), "video", "0.mp4")
        # vp_left = os.path.join(dataset_path, "episodes", str(episode), "video", "2.mp4")
        vp_hand = os.path.join(dataset_path, "episodes", str(episode), "video", "0_depth.mp4")

        # make images dir
        img_dir = os.path.join(dataset_path, "episodes", str(episode), "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # make dir for each stream
        # os.makedirs(os.path.join(img_dir, "top"), exist_ok=True)
        os.makedirs(os.path.join(img_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(img_dir, "depth"), exist_ok=True)

        # decode video and resize images 
        # os.system(f"ffmpeg -i {vp_top} -vf fps=10 -s 320x180 {img_dir}/top/%d.png")
        os.system(f"ffmpeg -i {vp_front} -vf fps=10 -s 1280x720 {img_dir}/rgb/%d.png")
        os.system(f"ffmpeg -i {vp_hand} -vf fps=10 -s 1280x720 {img_dir}/depth/%d.png")
    
    print("Decoding done")



def detect_aruco_markers(dataset_path:str):
    intrinsics_fpath = os.path.join(dataset_path, "calibration/hand_eye.json")
    with open(intrinsics_fpath, 'r') as f:
        meta_data = json.load(f)
        intrinsics = np.array(meta_data['intrinsics'])
        X_FC = np.array(meta_data["X_FC"])
        X_FE = np.array(meta_data["X_FE"])

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    for episode in tqdm.tqdm(episodes):
        # Paths to the image files
        video_dir = os.path.join(dataset_path, "episodes", str(episode), "video", "0.mp4")

        episode_path = os.path.join(dataset_path, "episodes", episode, "state.json")
        with open(episode_path, "r") as f:
            state_data = json.load(f)

        # read frames
        cap = cv2.VideoCapture(video_dir)
        tvecs = None
        ret = True
        frame_id = -1

        detection_list = []
        saved_first_frame = False

        while ret:
            ret, frame = cap.read()
            frame_id += 1
            if not ret:
                continue
            # detect markers
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            X_BE = np.array(state_data[frame_id]["X_BE"])
            X_BF = np.dot(X_BE, np.linalg.inv(X_FE))
            X_BC = np.dot(X_BF, X_FC)

            if ids is None:
                # print("No markers found for episode ", episode)
                detection_list.append({
                    'X_BO': None,
                    'frame_id': frame_id
                })
                continue

            if 3 in ids:
                idx = np.where(ids == 3)[0][0]
                corners = np.array([corners[idx]])
                ids = np.array([ids[idx]])
            
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # D455
                # distcoeffs = np.array([-5.78085221e-02,  6.55928180e-02,  8.11965656e-05,  4.67534643e-04, -2.07200460e-02])
                # L515
                distcoeffs = np.array([0.0,0.0,0.0,0.0,0.0])

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.025, intrinsics, distcoeffs)
                for i in range(len(rvecs)):
                    frame = cv2.drawFrameAxes(frame, intrinsics, distcoeffs, rvecs[i], tvecs[i], 0.1)

                # print(f"Marker found for episode {episode} at frame {frame_id}")

            else:
                # print("No marker 3 found for episode ", episode)
                detection_list.append({
                    'X_BO': None,
                    'frame_id': frame_id
                })
                continue
                

            #show frame
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            r = R.from_rotvec(np.array(rvecs[0]).flatten())
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, 3] = np.array(tvecs).flatten()
            T_cam_marker[:3, :3] = r.as_matrix()
            X_CO = T_cam_marker

            X_BO = np.dot(X_BC, X_CO)

            marker_info = {
                'X_BO': X_BO.tolist(),
                'frame_id': frame_id}
            

            # need to filter out bad object poses

            dist = np.dot(np.array([0,0,1]), X_BO[:3,2])
            if dist > 0.99:
                # print(dist)
                if not saved_first_frame:
                    with open(os.path.join(dataset_path, "episodes", str(episode), "object_frame.json"), 'w') as f:
                        json.dump(marker_info, f)
                    saved_first_frame = True
                    print("Good pose detected for episode {} at frame {}".format(episode, frame_id))
                    break
            else:
                # print("Bad pose")
                continue
                    
            
            # detection_list.append(marker_info)

        with open(os.path.join(dataset_path, "episodes", str(episode), "object_frame_last.json"), 'w') as f:
            json.dump(marker_info, f)

        cap.release()
        # pdb.set_trace()

    print("Done detecting markers")

    return


def extract_robot_pos_orien(poses: list):
    xyz = []
    oriens = []
    for episode in poses:
        temp_p = []
        temp_o = []
        for pose in episode:
            temp_p.append(pose[:3, 3])
            rot = pose[:3, :3]
            temp_o.append(matrix_to_rotation_6d(rot))
        xyz.append(temp_p)
        oriens.append(temp_o)
    return xyz, oriens


def extract_goal_positions(poses: list):
    xyz = []
    for episode in poses:
        xyz.append(episode[:3, 3])
    return xyz


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
    T = og_pose * sm.SE3.Rz(np.deg2rad(additional_rotation))
    
    return T



def adjust_orientation_to_z_up(matrix):
    # matrix must be in the base frame of robot
    # Extract the current Z direction
    current_z = matrix[:3, 2]
    target_z = np.array([0, 0, 1])

    # Calculate the axis of rotation and the angle needed for correction
    axis = np.cross(current_z, target_z)
    angle = np.arccos(np.dot(current_z, target_z) / (np.linalg.norm(current_z) * np.linalg.norm(target_z)))

    # Calculate the rotation matrix for aligning current_z to target_z
    if np.linalg.norm(axis) != 0:  # Avoid division by zero if the axes are already aligned
        axis = axis / np.linalg.norm(axis)
        rotation_matrix = R.from_rotvec(axis * angle).as_matrix()
    else:
        rotation_matrix = np.eye(3)  # No rotation needed if axes are already aligned

    # Construct the new transformation matrix
    new_matrix = np.eye(4)
    new_matrix[:3, :3] = np.dot(rotation_matrix, matrix[:3, :3])
    new_matrix[:3, 3] = matrix[:3, 3]  # Preserve the original xyz position

    return new_matrix

def parse_dataset(dataset_path:str):

    robot = rtb.models.Panda()
    # Transformation from flange to tool centre point
    X_FE = np.array([[0.70710678, 0.70710678, 0.0, 0.0], 
                     [-0.70710678, 0.70710678, 0, 0], 
                     [0.0, 0.0, 1.0, 0.2], #TODO: change this back to 0.2
                     [0.0, 0.0, 0.0, 1.0]])
    X_FE = sm.SE3(X_FE, check=False).norm()

    gello_poses = []
    ee_poses = []
    tactile_data = []
    joint_torques = []
    ee_forces = []
    object_poses = []
    oriented_object_poses = []

   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        raw_data = json.load(open(os.path.join(dataset_path, "episodes", episode ,"state.json")))
        raw_object_data = json.load(open(os.path.join(dataset_path, "episodes", episode ,"object_frame.json")))
        temp_poses = []
        temp_tactile = []
        temp_torques = []
        temp_forces = []
        temp_gello = []
        temp_object_poses = []
        temp_orien_object_poses = []

        for idx in range(len(raw_data)):
            pose = np.array(raw_data[idx]["X_BE"])
            temp_poses.append(pose)

            # get the tactile data
            # tactile = np.array(raw_data[idx]["tactile_sensors"])
            # temp_tactile.append(tactile)

            # get the joint torques
            torques = np.array(raw_data[idx]["joint_torques"])
            temp_torques.append(torques)

            # get the end effector forces
            forces = np.array(raw_data[idx]["ee_forces"])
            temp_forces.append(forces)

            # get gello q
            gello_q = np.array(raw_data[idx]["gello_q"])
            # convert joint positions to pose of tool centre point
            gello_pose = robot.fkine(gello_q, "panda_link8") * X_FE
            temp_gello.append(gello_pose.A)

        # get object poses and correct its z-axis
        object_pose = adjust_orientation_to_z_up(np.array(raw_object_data["X_BO"]))
        temp_object_poses.append(object_pose)

        # get oriented object pose
        X_BOO = compute_oriented_affordance_frame(object_pose)
        temp_orien_object_poses.append(X_BOO.A)

        ee_poses.append(temp_poses)
        # tactile_data.append(temp_tactile)
        joint_torques.append(temp_torques)
        ee_forces.append(temp_forces)
        gello_poses.append(temp_gello)
        object_poses.append(temp_object_poses)
        oriented_object_poses.append(temp_orien_object_poses)

    return {
        'ee_poses': ee_poses,
        # 'tactile_data': tactile_data,
        'joint_torques': joint_torques,
        'ee_forces': ee_forces,
        'gello_poses': gello_poses,
        'object_poses': object_poses,
        'oriented_object_poses': oriented_object_poses
    }



def compute_transforms(dataset_path:str):
    # read JSON file
    # TODO We assume that the object was always in the same position for all demos
    with open(os.path.join(dataset_path, "calibration/hand_eye.json"), 'r') as f:
        X_EC = np.array(json.load(f)['X_EC'])
    
    with open(os.path.join(dataset_path, "episodes/0/marker_pose.json"), 'r') as f:
        data = json.load(f)
        X_CO = np.array(data['X_CO'])
        frame_id = data['frame_id']

    with open(os.path.join(dataset_path, "episodes/0/state.json"), 'r') as f:
        data = json.load(f)[frame_id]
        X_BE = np.array(data['X_BE'])

    # compute X_BC
    X_BC = np.dot(X_BE, X_EC)
    # compute X_BO
    X_BO = np.dot(X_BC, X_CO)
    # save X_BC and X_BO
    transforms = {
        'X_BO': X_BO.tolist(),
        'X_EC': X_EC.tolist(),
    }

    with open(os.path.join(dataset_path, "calibration/transforms.json"), 'w') as f:
        json.dump(transforms, f)

    return


def extract_goal_poses(dataset_path:str):
    goal_poses = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        goal = np.array(json.load(open(os.path.join(dataset_path, "episodes", episode ,"marker_pose.json"))))
        goal_poses.append(goal)
    return goal_poses



# fpath = "/home/krishan/work/2024/datasets/cup_rotate_X"
# decode_video(fpath)
# detect_aruco_markers(fpath)
# out = extract_robot_poses(fpath)
# out = extract_goal_poses(fpath)
# compute_transforms(fpath)
# out = parse_dataset(fpath)

  
