from diffrobot.robot.visualizer import RobotViz
import os
import json
import pdb
import time
import spatialmath as sm
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import pyrealsense2 as rs


dataset_path = "/home/krishan/work/2024/datasets/door_open_v2.0"
episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
env = RobotViz()

# read in camera intrinsics
intrinsics_path = os.path.join(dataset_path, "calibration", "hand_eye.json")
with open(intrinsics_path, "r") as f:
    intrinsics_data = json.load(f)

intrinsics_matrix = np.array(intrinsics_data["intrinsics"])

fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

X_FE = np.array([[0.70710678, 0.70710678, 0.0, 0.0], 
                [-0.70710678, 0.70710678, 0, 0], 
                [0.0, 0.0, 1.0, 0.1], 
                [0.0, 0.0, 0.0, 1.0]])

X_FE = sm.SE3(X_FE, check=False).norm()

for episode in episodes:
    episode_path = os.path.join(dataset_path, "episodes", episode, "state.json")
    with open(episode_path, "r") as f:
        data = json.load(f)

    for idx, state in enumerate(data):

        # rgb_fpath = os.path.join(dataset_path, "episodes", episode, "images", "rgb", f"{idx+1}.png")
        rgb_fpath = os.path.join(dataset_path, "episodes", episode, "images", "rgb", "1.png")
        # depth_fpath = os.path.join(dataset_path, "episodes", episode, "images", "depth", f"{idx+1}.png")
        depth_fpath = os.path.join(dataset_path, "episodes", episode, "images", "depth", "1.png")
        rgb = cv2.imread(rgb_fpath)
        depth = cv2.imread(depth_fpath, -1) 
        # create a mask that cuts out a border of the image
        mask = np.zeros_like(depth)
        mask[150:-150, 150:-150] = 1
        depth = depth * mask
        rgb = rgb * mask[..., np.newaxis].astype(np.uint8)

        # mask 2
        mask = (depth*0.00025) > 0.5
        depth = depth * mask
        rgb = rgb * mask[..., np.newaxis].astype(np.uint8)
        


        height, width, _ = rgb.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


        # pdb.set_trace()

        # # show image with pyplot
        plt.figure()
        # use a different colormap for depth
        # color bar is in meters
        # add a colorbar to the image
        plt.colorbar(plt.imshow(depth*0.00025, cmap='jet'))
        plt.imshow(depth*0.00025, cmap='jet')
        

        # show rgb image as  well
        plt.figure()
        plt.imshow(rgb)
        




        # 

        # Create an Open3D RGBD image from the RGB and depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_scale=1.0/0.00025,  # Adjust based on your depth image format
            depth_trunc=2.0,  # Adjust to the maximum depth value you're interested in
            convert_rgb_to_intensity=False
        )
 
        # Create a point cloud from the RGBD image and the intrinsic parameters
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            intrinsics
        )

        
        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

        plt.show()

        env.ee_pose.T = sm.SE3(state["X_BE"], check=False).norm()
        target_pose = env.robot.fkine(state["gello_q"], "panda_link8") * X_FE
        env.policy_pose.T = target_pose 
        env.step(state["robot_q"])
        time.sleep(0.1)

    
    pdb.set_trace()
    




