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
import pandas as pd

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

def compute_oriented_affordance_frame(transform_matrix, base_frame=np.eye(4)):
        """
        Compute the angle needed to rotate the x-axis of a given transformation
        matrix so that it points towards the origin [0,0,0]. Apply the rotation to the
        transformation matrix and return the resulting matrix.

        Parameters:
        - transform_matrix: A 4x4 numpy array representing the homogeneous transformation matrix.

        Returns:
        - The transformation matrix with the x-axis pointing towards the origin.
        """

        # if the base frame is not the origin, we need to transform the matrix to the base frame
        if not np.array_equal(base_frame, np.eye(4)):
            transform_matrix = np.dot(np.linalg.inv(base_frame), transform_matrix) #X_OO

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

        # if the base frame is not the origin, we need to transform the matrix back to the base frame
        if not np.array_equal(base_frame, np.eye(4)):
            T = base_frame * T
            T = sm.SE3(T, check=False).norm()
        
        return T
class DatasetUtils:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.robot = rtb.models.Panda()
        self.X_FE = np.array([[0.70710678, 0.70710678, 0.0, 0.0], 
                            [-0.70710678, 0.70710678, 0, 0], 
                            [0.0, 0.0, 1.0, 0.2], 
                            [0.0, 0.0, 0.0, 1.0]])
        self.X_FE = sm.SE3(self.X_FE, check=False).norm()
        # self.affordance_transforms = json.load(open(os.path.join(self.dataset_path, "transforms", "to_afford.json"), "r"))  

    def create_rlds(self, num_noisy_variations=5):
        def add_noise(data, noise_level):
            return data + np.random.normal(scale=noise_level, size=data.shape)

        # Define noise levels for position and orientation
        pos_noise_level = 0.01  # Example noise level for position
        orien_noise_level = 0.01  # Example noise level for orientation

        rlds = {}
        episodes = sorted(os.listdir(os.path.join(self.dataset_path, "episodes")), key=lambda x: int(x))
        original_num_episodes = len(episodes)

        for episode_index, episode in enumerate(episodes):
            episode_path = os.path.join(self.dataset_path, "episodes", episode, "state.json")
            X_B_O1_path = os.path.join(self.dataset_path, "episodes", episode, "affordance_frames.json")

            with open(episode_path, "r") as f:
                data = json.load(f)
            with open(X_B_O1_path, "r") as f:
                object_data = json.load(f)

            df = pd.DataFrame(data)
            df['idx'] = range(len(df))

            if not isinstance(object_data, list):
                object_data = [object_data] * len(df)

            df_object = pd.DataFrame(object_data)
            phases = df['phase'].unique()

            rlds[episode_index] = {}
            for phase in phases:
                phase_data = df[df['phase'] == phase]

                # print("Processing episode {} phase {}".format(episode, phase))

                X_BE_follower = phase_data['X_BE'].tolist()
                X_BE_leader = [(self.robot.fkine(np.array(q), "panda_link8") * self.X_FE).A for q in phase_data['gello_q']]

                X_B_O1 = df_object[df_object['frame_id'].isin(phase_data['idx'])]

                X_B_O1 = [self.adjust_orientation_to_z_up(np.array(pose)) for pose in X_B_O1['X_BO']]
                base_frame = np.array(X_BE_follower[0]) if np.allclose(X_B_O1[0], X_B_O1[-1]) else np.eye(4)
                X_B_OO1 = [self.compute_oriented_affordance_frame(pose, base_frame=base_frame).A for pose in X_B_O1]

                progress = self.linear_progress(len(phase_data))
                X_OO1_O1 = [np.linalg.inv(x_b_oo1) @ x_bo1 for x_b_oo1, x_bo1 in zip(X_B_OO1, X_B_O1)]

                if phase == 0:     
                    X_OO_E_follower = [np.linalg.inv(x_b_oo1) @ x_be for x_b_oo1, x_be in zip(X_B_OO1, X_BE_follower)]
                    X_OO_E_leader = [np.linalg.inv(x_b_oo1) @ x_be for x_b_oo1, x_be in zip(X_B_OO1, X_BE_leader)]
                    orien_object = [matrix_to_rotation_6d(pose[:3, :3]) for pose in X_OO1_O1]

                pos_follower, orien_follower = self.extract_robot_pos_orien(X_OO_E_follower)
                pos_leader, orien_leader = self.extract_robot_pos_orien(X_OO_E_leader)

                rlds[episode_index][str(int(phase))] = {
                    'X_BE_follower': X_BE_follower,
                    'X_BE_leader': X_BE_leader,
                    'robot_q': phase_data['robot_q'].tolist(),
                    'gello_q': phase_data['gello_q'].tolist(),
                    'gripper_state': phase_data['gripper_state'].tolist(),
                    'gripper_action': phase_data['gripper_action'].tolist(),
                    'joint_torques': phase_data['joint_torques'].tolist(),
                    'ee_forces': phase_data['ee_forces'].tolist(),
                    'X_B_O1': X_B_O1,
                    'X_B_OO1': X_B_OO1,
                    'progress': progress,
                    'X_OO_E_follower': X_OO_E_follower,
                    'X_OO_E_leader': X_OO_E_leader,
                    'pos_follower': pos_follower,
                    'orien_follower': orien_follower,
                    'pos_leader': pos_leader,
                    'orien_leader': orien_leader,
                    'orien_object': orien_object,
                    'phase': phase_data['phase'].to_list()
                }


                # Add noisy variations
                for i in range(num_noisy_variations):
                    pos_follower_noisy = [add_noise(np.array(pos), pos_noise_level) for pos in pos_follower]
                    orien_follower_noisy = [add_noise(np.array(orien), orien_noise_level) for orien in orien_follower]

                    noisy_episode_index = original_num_episodes + episode_index * num_noisy_variations + i
                    rlds[noisy_episode_index] = {}
                    rlds[noisy_episode_index][str(int(phase))] = {
                        'X_BE_follower': X_BE_follower,
                        'X_BE_leader': X_BE_leader,
                        'robot_q': phase_data['robot_q'].tolist(),
                        'gello_q': phase_data['gello_q'].tolist(),
                        'gripper_state': phase_data['gripper_state'].tolist(),
                        'gripper_action': phase_data['gripper_action'].tolist(),
                        'joint_torques': phase_data['joint_torques'].tolist(),
                        'ee_forces': phase_data['ee_forces'].tolist(),
                        'X_B_O1': X_B_O1,
                        'X_B_OO1': X_B_OO1,
                        'progress': progress,
                        'X_OO_E_follower': X_OO_E_follower,
                        'X_OO_E_leader': X_OO_E_leader,
                        'pos_follower': pos_follower_noisy,
                        'orien_follower': orien_follower_noisy,
                        'pos_leader': pos_leader,
                        'orien_leader': orien_leader,
                        'orien_object': orien_object,
                        'phase': phase_data['phase'].to_list()
                    }

        stats = self.get_stats_rlds(rlds)
        rlds = self.normalize_rlds(rlds, stats)

        with open(os.path.join(self.dataset_path, "rlds.pkl"), 'wb') as f:
            pickle.dump(rlds, f)

        with open(os.path.join(self.dataset_path, "stats.pkl"), 'wb') as f:
            pickle.dump(stats, f)

        return rlds, stats
        

    def get_stats_rlds(self, rlds):

        all_pos_follower = []
        all_pos_leader = []
        all_gripper_state = []
        all_gripper_action = []
        all_progress = []
        all_phase = []

        for episode in rlds:
            ep_data = rlds[episode]
            for phase in ep_data:
                phase_data = ep_data[phase]
                all_pos_follower.append(phase_data['pos_follower'])
                all_pos_leader.append(phase_data['pos_leader'])
                all_gripper_state.append(phase_data['gripper_state'])
                all_gripper_action.append(phase_data['gripper_action'])
                all_progress.append(phase_data['progress'])
                # all_phase.append(phase_data['phase'])
        
        stats = dict()
        stats['pos_follower'] = self.get_data_stats(all_pos_follower)
        stats['pos_leader'] = self.get_data_stats(all_pos_leader)
        stats['gripper_state'] = self.get_data_stats(all_gripper_state)
        stats['gripper_action'] = self.get_data_stats(all_gripper_action)
        stats['progress'] = self.get_data_stats(all_progress)
        # stats['phase'] = self.get_data_stats(all_phase)

        return stats
    
    def normalize_rlds(self, rlds, stats):
        for episode in rlds:
            ep_data = rlds[episode]
            for phase in ep_data:
                phase_data = ep_data[phase]
                phase_data['pos_follower'] = self.normalize_data(phase_data['pos_follower'], stats['pos_follower'])
                phase_data['pos_leader'] = self.normalize_data(phase_data['pos_leader'], stats['pos_leader'])
                phase_data['gripper_state'] = self.normalize_data(phase_data['gripper_state'], stats['gripper_state'])
                phase_data['gripper_action'] = self.normalize_data(phase_data['gripper_action'], stats['gripper_action'])
                phase_data['progress'] = self.normalize_data(phase_data['progress'], stats['progress'])
                # phase_data['phase'] = self.normalize_data(phase_data['phase'], stats['phase'])
        return rlds
        

    def sigmoid_progress(self, length):
        x = np.linspace(-6, 6, length)
        return 1 / (1 + np.exp(-x))
    
    def linear_progress(self, length):
        return np.linspace(0, 1, length)
        
    def create_sample_indices(self, rlds_dataset, sequence_length=16):
        indices = list()

        # # WIP ------------------------------
        # # ensure the same 10 episodes are sampled each time
        # np.random.seed(0)
        # # sample only 10 episodes from the full set
        # episodes = list(rlds_dataset.keys())
        # episodes = np.random.choice(episodes, 11, replace=False)
        # rlds_dataset = {episode: rlds_dataset[episode] for episode in episodes}
        # # WIP ------------------------------

        for episode in rlds_dataset.keys():
            for phase in rlds_dataset[episode].keys():
                episode_length = len(rlds_dataset[episode][phase]['pos_follower'])
                range_idx = episode_length - (sequence_length + 2)
                for idx in range(range_idx):
                    buffer_start_idx = idx
                    buffer_end_idx = idx + sequence_length
                    indices.append([int(episode), int(phase), buffer_start_idx, buffer_end_idx])
                    assert buffer_end_idx - buffer_start_idx == sequence_length
        indices = np.array(indices)    
        return indices


    def flatten_2d_lists(self, list_of_lists):
        flattened_list = []
        for sublist in list_of_lists:
            for item in sublist:
                flattened_list.append(item)
        return flattened_list


    def get_data_stats(self, data: list):
        data = np.array(self.flatten_2d_lists(data))
        stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
        }
        return stats


    def normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data
    
    def extract_robot_pos_orien(self, poses):
        xyz = []
        oriens = []
        for pose in poses:
            pose = np.array(pose)
            xyz.append(pose[:3, 3])
            rot = pose[:3, :3]
            oriens.append(matrix_to_rotation_6d(rot))
        return xyz, oriens
    
    
    def compute_oriented_affordance_frame(self, transform_matrix, base_frame=np.eye(4)):
        """
        Compute the angle needed to rotate the x-axis of a given transformation
        matrix so that it points towards the origin [0,0,0]. Apply the rotation to the
        transformation matrix and return the resulting matrix.

        Parameters:
        - transform_matrix: A 4x4 numpy array representing the homogeneous transformation matrix.

        Returns:
        - The transformation matrix with the x-axis pointing towards the origin.
        """

        # if the base frame is not the origin, we need to transform the matrix to the base frame
        if not np.array_equal(base_frame, np.eye(4)):
            transform_matrix = np.dot(np.linalg.inv(base_frame), transform_matrix) #X_OO

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

        # if the base frame is not the origin, we need to transform the matrix back to the base frame
        if not np.array_equal(base_frame, np.eye(4)):
            T = base_frame * T
            T = sm.SE3(T, check=False).norm()
        
        return T



    def adjust_orientation_to_z_up(self, matrix):
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
    
    def transform_to_affordance_centric(self, pose, transform_matrix):
        return pose @ transform_matrix





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



def detect_aruco_markers(dataset_path:str, marker_id:int=3, file_name:str="cup_frames.json", dynamic_object:bool=True):
    intrinsics_fpath = os.path.join(dataset_path, "transforms/hand_eye.json")
    with open(intrinsics_fpath, 'r') as f:
        meta_data = json.load(f)
        intrinsics_b = np.array(meta_data['back']['intrinsics'])
        intrinsics_f = np.array(meta_data['front']['intrinsics'])

        distortion_b = np.array(meta_data['back']['distortion'])
        distortion_f = np.array(meta_data['front']['distortion'])

        X_EC_b = np.array(meta_data["X_EC_b"])
        X_EC_f = np.array(meta_data["X_EC_f"])
        X_FE = np.array(meta_data["X_FE"])

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    
    for episode in tqdm.tqdm(episodes):
        # Paths to the image files
        video_dir_b = os.path.join(dataset_path, "episodes", str(episode), "video", "0.mp4") #back
        video_dir_f = os.path.join(dataset_path, "episodes", str(episode), "video", "1.mp4") #front

        episode_path = os.path.join(dataset_path, "episodes", episode, "state.json")
        with open(episode_path, "r") as f:
            state_data = json.load(f)

        # read frames
        cap_b = cv2.VideoCapture(video_dir_b)
        cap_f = cv2.VideoCapture(video_dir_f)

        tvecs = None
        ret_b = True
        frame_id = -1

        detection_list = []
        saved_first_frame = False

        X_BO = None

        while ret_b:
            ret_b, frame_b = cap_b.read()
            ret_f, frame_f = cap_f.read()
            frame_id += 1
            if not ret_b:
                continue
            # detect markers
            corners_b, ids_b, rejectedImgPoints_b = cv2.aruco.detectMarkers(frame_b, aruco_dict, parameters=parameters)
            corners_f, ids_f, rejectedImgPoints_f = cv2.aruco.detectMarkers(frame_f, aruco_dict, parameters=parameters)

            X_BE = np.array(state_data[frame_id]["X_BE"])
            X_BF = np.dot(X_BE, np.linalg.inv(X_FE))

            X_BC_b = np.dot(X_BE, X_EC_b)
            X_BC_f = np.dot(X_BE, X_EC_f)

            if (ids_b is not None) and (marker_id in ids_b):
                    ids = ids_b
                    corners = corners_b
                    frame = frame_b
                    X_BC = X_BC_b
                    intrinsics = intrinsics_b
                    distcoeffs = distortion_b
                    print("Pose detected for episode {} at frame {} in the back camera".format(episode, frame_id))
            elif (ids_f is not None) and (marker_id in ids_f):
                    ids = ids_f
                    corners = corners_f
                    frame = frame_f
                    X_BC = X_BC_f
                    intrinsics = intrinsics_f
                    distcoeffs = distortion_f
                    print("Pose detected for episode {} at frame {} in the front camera".format(episode, frame_id))
            else:
                print("No markers found for episode ", episode)
                detection_list.append({
                    'X_BO': X_BO.tolist() if X_BO is not None else None,
                    'frame_id': frame_id
                })
                continue

        
            idx = np.where(ids == marker_id)[0][0]
            corners = np.array([corners[idx]])
            ids = np.array([ids[idx]])
    
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.025, intrinsics, distcoeffs)
            # tvecs[:, 0, 1] -= 0.094
            for i in range(len(rvecs)):
                frame = cv2.drawFrameAxes(frame, intrinsics, distcoeffs, rvecs[i], tvecs[i], 0.05)

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
                    
            detection_list.append(marker_info)

            if not dynamic_object:
                break

        cap_b.release()
        cap_f.release()

        with open(os.path.join(dataset_path, "episodes", str(episode), file_name), 'w') as f:
            json.dump(detection_list, f)

        if not dynamic_object:
            with open(os.path.join(dataset_path, "episodes", str(episode), file_name), 'w') as f:
                json.dump(marker_info, f)

        


    print("Done detecting markers")

    return



if __name__ == "__main__":

    fpath = "/home/krishan/work/2024/datasets/cup_rotate_eval_20"
    dataset_utils = DatasetUtils(fpath)
    detect_aruco_markers(fpath, marker_id=3, file_name="affordance_frames.json", dynamic_object=True)
    detect_aruco_markers(fpath, marker_id=8, file_name="saucer_frames.json", dynamic_object=False)
    rlds = dataset_utils.create_rlds()



  
