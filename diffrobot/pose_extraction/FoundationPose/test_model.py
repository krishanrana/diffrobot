from estimater import *
from datareader import *
import pyrealsense2 as rs
from interactive_segmenter import InteractiveSegmenter
import numpy as np
import argparse
import cv2
import os
import pdb
from scipy.spatial.transform import Rotation as R
import json
from diffrobot.diffusion_policy.utils.dataset_utils import adjust_orientation_to_z_up

selected_points = []

def pick_points(vis):
    print("Press [shift + left click] to select a point and [shift + right click] to finish.")
    vis.register_mouse_callback(mouse_callback)

def mouse_callback(vis, action, mods):
    if action == o3d.visualization.MouseAction.PICK_POINT:
        print(f"Selected point: {vis.get_picked_points()}") 
        selected_points.extend(vis.get_picked_points())

if __name__ == '__main__':

    segmenter = InteractiveSegmenter()
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # The K matrix (intrinsic matrix)
    K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
         [0, intrinsics.fy, intrinsics.ppy],
         [0, 0, 1]])

    print("K: ", K)

     # get distortion coefficients from depth stream
    depth_stream = profile.get_stream(rs.stream.depth)
    distcoeffs = np.array(depth_stream.as_video_stream_profile().get_intrinsics().coeffs)


    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    cam_K = np.array(K)

    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/teacup/mesh/mesh_scaled.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=5)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # aruco marker detector
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    marker_id = 3

    saved_transform = None


    i = 0
    while True:

        if cv2.waitKey(1) == 13:
            break

        print('waiting for frames')

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        

        print('frames received')

        if not aligned_depth_frame or not color_frame:
            print('frames not received')
            continue

        depth = np.asanyarray(aligned_depth_frame.get_data())* depth_scale
        color = np.asanyarray(color_frame.get_data())
        # conver to bgr
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth,depth,depth)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
                

        logging.info(f'i:{i}')
        if i == 0:
            # convert from bgr to rgb
            mask = segmenter.segment_image(color)
            mask = mask[0]
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, cam_K)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                #view  point cloud
                vis = o3d.visualization.VisualizerWithEditing()
                vis.create_window()
                vis.add_geometry(pcd)
                vis.run()
                vis.destroy_window()

                # Get the selected points
                picked_points = vis.get_picked_points()
                point = picked_points[0]
                loc_3d = np.asarray(pcd.points)[point]  # 3D location of the point
            
            if debug==1:
                with open(f'{os.path.dirname(args.mesh_file)}/affordance_transform.json', 'r') as f:
                       saved_transform = np.array(json.load(f))
             


                # o3d.visualization.draw_geometries([pcd])
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4, 4))

        if debug >= 1:

            center_pose = pose @ np.linalg.inv(to_origin)
            X_CO_fpose = center_pose

            

            # aruco marker detection
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(color, aruco_dict, parameters=parameters)
            if ids is not None and marker_id in ids:
                idx = np.where(ids==marker_id)
                corners = np.array(corners)[idx]
                ids = np.array(ids)[idx]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.025, K, distcoeffs)
                # for i in range(len(rvecs)):
                #     color = cv2.drawFrameAxes(color, K, distcoeffs, rvecs[i], tvecs[i], 0.05)

                r = R.from_rotvec(np.array(rvecs[0]).flatten())
                T_cam_marker = np.eye(4)
                T_cam_marker[:3, 3] = np.array(tvecs).flatten()
                T_cam_marker[:3, :3] = r.as_matrix()
                X_CO_aruco = T_cam_marker

                
                if debug >=3:

                    X_CO_aruco[:,3][:3] = loc_3d
                    # find transform from foundation pose to aruco marker
                    X_aruco_fpose = np.linalg.inv(X_CO_aruco) @ X_CO_fpose


                    if i>20:
                        if saved_transform is None:
                            print('Transforming to aruco marker frame')
                            pdb.set_trace()
                            # saved_transform = adjust_orientation_to_z_up(X_aruco_fpose)
                            saved_transform = X_aruco_fpose
                            # save pose to json in the location of the mesh file
                            with open(f'{os.path.dirname(args.mesh_file)}/affordance_transform.json', 'w') as f:
                                json.dump(saved_transform.tolist(), f)
                            


                
            if saved_transform is not None:
                center_pose = center_pose @ np.linalg.inv(saved_transform)


            # vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0.2)
            # vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            cv2.imshow('1', vis)
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)

        i += 1

    pipeline.stop()
    cv2.destroyAllWindows()