import os
import pdb

fpath = "/home/krishan/work/2024/datasets/franka_reacher"

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


decode_video(fpath)

        


            