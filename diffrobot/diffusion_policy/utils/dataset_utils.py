import os
import pdb
import cv2
import json
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
import pickle

from diffrobot.diffusion_policy.utils.rotation_transforms import rotation_6d_to_matrix, matrix_to_rotation_6d



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
        intrinsics = np.array(json.load(f)['intrinsics'])

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    for episode in tqdm.tqdm(episodes):
        # Paths to the image files
        video_dir = os.path.join(dataset_path, "episodes", str(episode), "video", "0.mp4")

        # read frames
        cap = cv2.VideoCapture(video_dir)
        tvecs = None
        ret = True
        frame_id = -1

        while ret:
            ret, frame = cap.read()
            frame_id += 1
            if not ret:
                continue
            # detect markers
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            if ids is None:
                # print("No markers found for episode ", episode)
                continue

            if 6 in ids:
                idx = np.where(ids == 6)[0][0]
                corners = np.array([corners[idx]])
                ids = np.array([ids[idx]])
            
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # D455
                # distcoeffs = np.array([-5.78085221e-02,  6.55928180e-02,  8.11965656e-05,  4.67534643e-04, -2.07200460e-02])
                # L515
                distcoeffs = np.array([0.0,0.0,0.0,0.0,0.0])

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, intrinsics, distcoeffs)
                for i in range(len(rvecs)):
                    frame = cv2.aruco.drawAxis(frame, intrinsics, distcoeffs, rvecs[i], tvecs[i], 0.1)

                print("Marker found for episode ", episode)

            else:
                print("No marker 8 found for episode ", episode)
                continue
                

            #show frame
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

            r = R.from_rotvec(np.array(rvecs[0]).flatten())
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, 3] = np.array(tvecs).flatten()
            T_cam_marker[:3, :3] = r.as_matrix()


            marker_info = {
                'X_CO': T_cam_marker.tolist(),
                'frame_id': frame_id}

            #save marker position as json
            with open(os.path.join(dataset_path, "episodes", str(episode), "marker_pose.json"), 'w') as f:
                json.dump(marker_info, f)
            break

        cap.release()

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
            # check if the rotation matrix is valid
            assert np.allclose(np.dot(rot, rot.T), np.eye(3))
            assert np.isclose(np.linalg.det(rot), 1)
            # convert to 6d representation
            temp_o.append(matrix_to_rotation_6d(rot))
        xyz.append(temp_p)
        oriens.append(temp_o)
    return xyz, oriens


def extract_goal_positions(poses: list):
    xyz = []
    for episode in poses:
        xyz.append(episode[:3, 3])
    return xyz


def parse_dataset(dataset_path:str):
    ee_poses = []
    tactile_data = []
    joint_torques = []
    ee_forces = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        raw_data = json.load(open(os.path.join(dataset_path, "episodes", episode ,"state.json")))
        temp_poses = []
        temp_tactile = []
        temp_torques = []
        temp_forces = []
        for idx in range(len(raw_data)):
            pose = np.array(raw_data[idx]["X_BE"])
            temp_poses.append(pose)

            # get the tactile data
            tactile = np.array(raw_data[idx]["tactile_sensors"])
            temp_tactile.append(tactile)

            # get the joint torques
            torques = np.array(raw_data[idx]["joint_torques"])
            temp_torques.append(torques)

            # get the end effector forces
            forces = np.array(raw_data[idx]["ee_forces"])
            temp_forces.append(forces)

        ee_poses.append(temp_poses)
        tactile_data.append(temp_tactile)
        joint_torques.append(temp_torques)
        ee_forces.append(temp_forces)

    return {
        'ee_poses': ee_poses,
        'tactile_data': tactile_data,
        'joint_torques': joint_torques,
        'ee_forces': ee_forces
    }



def compute_transforms(dataset_path:str):
    # read JSON file
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



fpath = "/home/krishan/work/2024/datasets/door_open"
# decode_video(fpath)
# detect_aruco_markers(fpath)
# out = extract_robot_poses(fpath)
# out = extract_goal_poses(fpath)
compute_transforms(fpath)

  
