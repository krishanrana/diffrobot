import os
import pdb
import cv2
import json
import numpy as np
import tqdm
from scipy.spatial.transform import Rotation as R
import pickle
import os
import json


fpath = "/home/krishan/work/2024/datasets/franka_3D_reacher"

# create an image based dataset from video
def decode_video(dataset_path:str):
    state = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
            # Paths to the video files
        vp_front = os.path.join(dataset_path, "episodes", str(episode), "video", "1.mp4")
        # vp_left = os.path.join(dataset_path, "episodes", str(episode), "video", "2.mp4")
        vp_hand = os.path.join(dataset_path, "episodes", str(episode), "video", "0.mp4")

        # make images dir
        img_dir = os.path.join(dataset_path, "episodes", str(episode), "images")
        os.makedirs(img_dir, exist_ok=True)
        
        # make dir for each stream
        # os.makedirs(os.path.join(img_dir, "top"), exist_ok=True)
        os.makedirs(os.path.join(img_dir, "front"), exist_ok=True)
        os.makedirs(os.path.join(img_dir, "hand"), exist_ok=True)

        # decode video and resize images 
        # os.system(f"ffmpeg -i {vp_top} -vf fps=10 -s 320x180 {img_dir}/top/%d.png")
        os.system(f"ffmpeg -i {vp_front} -vf fps=10 -s 320x180 {img_dir}/front/%d.png")
        os.system(f"ffmpeg -i {vp_hand} -vf fps=10 -s 320x180 {img_dir}/hand/%d.png")
    
    print("Decoding done")



def detect_aruco_markers(dataset_path:str):
    intrinsics_fpath = os.path.join(dataset_path, "cam_front_intrinsics.json")
    extrinsics_fpath = os.path.join(dataset_path, "cameras.yaml")
    with open(intrinsics_fpath, 'r') as f:
        intrinsics = np.array(json.load(f))

    # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()

    for episode in tqdm.tqdm(episodes):
        # Paths to the image files
        video_dir = os.path.join(dataset_path, "episodes", str(episode), "video", "1.mp4")

        # read frames
        cap = cv2.VideoCapture(video_dir)
        tvecs = None
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret:
                break

            # detect markers
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

            if ids is None:
                print("No markers found for episode ", episode)
                continue

            if 8 in ids:
                idx = np.where(ids == 8)[0][0]
                corners = np.array([corners[idx]])
                ids = np.array([ids[idx]])
            
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # detect and draw the pose
                distcoeffs = np.zeros((4,1))
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, intrinsics, distcoeffs)
                for i in range(len(rvecs)):
                    frame = cv2.aruco.drawAxis(frame, intrinsics, distcoeffs, rvecs[i], tvecs[i], 0.1)

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

            #save marker position as json
            with open(os.path.join(dataset_path, "episodes", str(episode), "marker_pose.json"), 'w') as f:
                json.dump(T_cam_marker.tolist(), f)
            break

            
        cap.release()


def extract_robot_poses(dataset_path:str):
    state = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        raw_data = json.load(open(os.path.join(dataset_path, "episodes", episode ,"state.json")))
        temp = []
        for idx in range(len(raw_data)):
            pose = np.array(raw_data[idx]["X_BE"])[:3,3]
            temp.append(pose)
        
        state.append(temp)
    # save file as pickle
    with open(f'{dataset_path}/all_states.pkl', 'wb') as f:
        pickle.dump(state, f)
    print("Done saving state data")
    return


def extract_goal_poses(dataset_path:str):
    goals = []
   # sort numerically the episodes based on folder names
    episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
    for episode in episodes:
        # read the state.json file which consists of a list of dictionaries
        goal = np.array(json.load(open(os.path.join(dataset_path, "episodes", episode ,"marker_pose.json"))))
        goals.append(goal[:3,3])

    # save file as pickle
    with open(f'{dataset_path}/all_goals.pkl', 'wb') as f:
        pickle.dump(goals, f)
    print("Done saving goal data")
    return

# detect_aruco_markers(fpath)
# decode_video(fpath)
# extract_robot_poses(fpath)
extract_goal_poses(fpath)

        


            