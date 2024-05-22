import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb

FONT = cv2.FONT_HERSHEY_SIMPLEX

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

class ArucoDetector:
    def __init__(self, cam, marker_size, marker_dict, marker_id, visualize=True):
        self.cam = cam
        self.marker_size = marker_size
        self.visualize = visualize
        self.camera_matrix = self.cam.get_intrinsics()
        self.marker_dict = eval(marker_dict) if isinstance(marker_dict, str) else marker_dict
        self.marker_id = marker_id
        self.dist_coeffs = self.cam.get_dist_coeffs()

    def detect_markers(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # set dictionary size depending on the aruco marker selected
        # aruco_dict = aruco.Dictionary_get(self.marker_dict)
        aruco_dict = aruco.getPredefinedDictionary(self.marker_dict)
        # detector parameters can be set here (List of detection parameters[3])
        # parameters = aruco.DetectorParameters_create()
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 10
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        # lists of ids and the corners belonging to each marker_id
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        # corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if ids is None:
            return {}
        return {_id[0]: [_corners] for _id, _corners in zip(ids, corners)}

    def detect_all_markers(self):
        rgb = self.cam.get()["color"]
        detected_markers = self.detect_markers(rgb)
        if len(detected_markers) == 0:
            # print("no marker detected")
            # code to show 'No Ids' when no markers are found

            cv2.putText(rgb, "No Ids", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("win2", rgb[:, :, ::-1])
            cv2.waitKey(1)

            return False
        for _id, corners in detected_markers.items():
            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            # (rvec-tvec).any() # get rid of that nasty numpy value array error

            # draw axis for the aruco markers
            cv2.drawFrameAxes(rgb, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.1)


            # draw a square around the markers
            aruco.drawDetectedMarkers(rgb, corners)

        cv2.putText(rgb, f"Ids: {list(detected_markers.keys())}", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"marker detected with marker_ids {detected_markers.keys()}")
        #
        cv2.imshow("win2", rgb[:, :, ::-1])
        cv2.waitKey(1)

    def detect_objects(self):
        rgb = self.cam.get()['color']
        detected_markers = self.detect_markers(rgb)
        return detected_markers

    def estimate_pose(self, detected_markers, marker_id):

        if len(detected_markers) == 0 or marker_id not in detected_markers:
            # print("no marker detected")
            # code to show 'No Ids' when no markers are found
            # if self.visualize:
            #     cv2.putText(rgb, "No Ids", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #     cv2.imshow("win2", rgb[:, :, ::-1])
            #     cv2.waitKey(1)

            return None

        corners = detected_markers[marker_id]
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
        rvec, tvec, _ = estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
        # (rvec-tvec).any() # get rid of that nasty numpy value array error

        # if self.visualize:
        #     cv2.drawFrameAxes(rgb, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.1)
        #     # draw a square around the markers
        #     aruco.drawDetectedMarkers(rgb, corners)

        #     cv2.putText(rgb, f"Id: {marker_id}", (0, 64), FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #     print(f"marker detected with marker_id {marker_id}")
        #     #
        #     cv2.imshow("win2", rgb[:, :, ::-1])
        #     cv2.waitKey(1)

        r = R.from_rotvec(np.array(rvec[0]).flatten())
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, 3] = np.array(tvec).flatten()
        T_cam_marker[:3, :3] = r.as_matrix()
        return T_cam_marker


