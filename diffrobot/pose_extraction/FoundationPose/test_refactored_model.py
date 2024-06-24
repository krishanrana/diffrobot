from diffrobot.pose_extraction.FoundationPose.estimater import *
from diffrobot.pose_extraction.FoundationPose.datareader import *
import pyrealsense2 as rs
from diffrobot.pose_extraction.FoundationPose.interactive_segmenter import InteractiveSegmenter
import numpy as np
import argparse
import cv2
import os
import pdb
from scipy.spatial.transform import Rotation as R
import json
from diffrobot.diffusion_policy.utils.dataset_utils import adjust_orientation_to_z_up

class MultiObjectTracker:
    def __init__(self, objects):
        self.objects = objects
        self.segmenter = InteractiveSegmenter()
        self.setup_camera()
        self.estimators = {}
        self.pose_dict = {}
        self.setup_estimators()

    def setup_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                           [0, intrinsics.fy, intrinsics.ppy],
                           [0, 0, 1]])

        # get distortion coefficients from depth stream
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.distcoeffs = np.array(depth_stream.as_video_stream_profile().get_intrinsics().coeffs)
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def setup_estimators(self):
        parser = argparse.ArgumentParser()
        code_dir = os.path.dirname(os.path.realpath(__file__))
        parser.add_argument('--est_refine_iter', type=int, default=5)
        parser.add_argument('--track_refine_iter', type=int, default=5)
        parser.add_argument('--debug', type=int, default=1)
        parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
        args = parser.parse_args()

        set_logging_format()
        set_seed(0)

        for obj in self.objects:
            mesh_file = f'{code_dir}/demo_data/{obj}/mesh/mesh_scaled.obj'
            mesh = trimesh.load(mesh_file)
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            with open(f'{code_dir}/demo_data/{obj}/mesh/affordance_transform.json', 'r') as f:
                saved_transform = np.array(json.load(f))

            scorer = ScorePredictor()
            refiner = PoseRefinePredictor()
            glctx = dr.RasterizeCudaContext()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                 scorer=scorer, refiner=refiner, debug_dir=args.debug_dir, debug=args.debug, glctx=glctx)
            self.estimators[obj] = {
                "estimator": est,
                "mesh": mesh,
                "bbox": bbox,
                "to_origin": to_origin,
                "debug_dir": args.debug_dir,
                "debug": args.debug,
                "est_refine_iter": args.est_refine_iter,
                "track_refine_iter": args.track_refine_iter,
                "saved_transform": saved_transform,
            }
        logging.info("All estimators initialized")

    def get_model_inputs(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None, None

        depth = np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale
        color = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        return color, depth, self.K


    def register_objects(self):
        color, depth, K = self.get_model_inputs()
        if color is None or depth is None:
            return

        for obj, data in self.estimators.items():
            est = data["estimator"]
            to_origin = data["to_origin"]

            print(f"Please select the mask points for {obj} in the image")
            mask = self.segmenter.segment_image(color)[0]
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=data["est_refine_iter"])

            center_pose = pose @ np.linalg.inv(to_origin)
            center_pose = center_pose @ np.linalg.inv(data["saved_transform"])
            self.pose_dict[obj] = center_pose

    def track_objects_once(self, track_objects_list):
        color, depth, K = self.get_model_inputs()
        if color is None or depth is None:
            return

        for obj in track_objects_list:
            if obj in self.estimators:
                data = self.estimators[obj]
                est = data["estimator"]
                to_origin = data["to_origin"]
                pose = est.track_one(rgb=color, depth=depth, K=K, iteration=data["track_refine_iter"])

                center_pose = pose @ np.linalg.inv(to_origin)
                center_pose = center_pose @ np.linalg.inv(data["saved_transform"])
                self.pose_dict[obj] = center_pose
        return color
    
    def visualize_poses(self, color):
        vis = color.copy()
        for obj, pose in self.pose_dict.items():
            vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.05, K=self.K, thickness=3, transparency=0.2)
        
        cv2.imshow('1', vis)
        cv2.waitKey(1)


    def track_objects(self, track_objects_list):
        while True:
            if cv2.waitKey(1) == 13:
                break
            color = self.track_objects_once(track_objects_list)
            self.visualize_poses(color)

        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    objects = ["teacup", "saucer", "teapot", "teaspoon"]
    tracker = MultiObjectTracker(objects)
    tracker.register_objects()
    tracker.track_objects(["teacup"])#, "teapot"])

