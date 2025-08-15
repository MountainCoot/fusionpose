
import json
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import cv2
import scipy.optimize
from cv2 import aruco
import os
import matplotlib.pyplot as plt
from typing import List, Optional




if __name__ == '__main__':
    # add two up to the root folder
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(base_dir) if base_dir not in sys.path else None

from deps.utils.minimial_square_funcs import points_to_description, description_to_points

def aspect_ratio_check(corners, threshold=0.7, return_vals=False):
    # Calculate distances between opposite sides
    AB = np.linalg.norm(corners[0] - corners[1])
    BC = np.linalg.norm(corners[1] - corners[2])
    CD = np.linalg.norm(corners[2] - corners[3])
    DA = np.linalg.norm(corners[3] - corners[0])
    
    # Aspect ratio (should be close to 1 if viewed frontally)
    aspect_ratio = (AB + CD) / (BC + DA)
    if return_vals:
        return aspect_ratio
    return threshold <= aspect_ratio <= (1 / threshold)


def angle_between(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def angle_check(corners, max_deviation=15, return_vals=False):
    # Calculate angles at each corner using dot product
    
    angles = [
        angle_between(corners[3], corners[0], corners[1]),
        angle_between(corners[0], corners[1], corners[2]),
        angle_between(corners[1], corners[2], corners[3]),
        angle_between(corners[2], corners[3], corners[0])
    ]
    if return_vals:
        return angles
    # Check if all angles are within the acceptable range
    return all(90 - max_deviation <= angle <= 90 + max_deviation for angle in angles)


def filter_markers(ids, corner_sets, aspect_threshold=0.3, angle_deviation=45):
    valid_ids = []
    n_angle = 0
    n_aspect = 0

    angles_list = []
    aspect_list = []

    for i, corners in zip(ids, corner_sets):
        angle_check_res = angle_check(corners, max_deviation=angle_deviation)
        # area_check_res = area_check(corners, min_area=min_area)
        aspect_ratio_check_res = aspect_ratio_check(corners, threshold=aspect_threshold)
        n_angle += int(not angle_check_res)
        # n_area += int(not area_check_res)
        n_aspect += int(not aspect_ratio_check_res)
        # populate lists for debugging
        angles_list.append(angle_check(corners, return_vals=True))
        aspect_list.append(aspect_ratio_check(corners, return_vals=True))

        if angle_check_res and aspect_ratio_check_res:        
            valid_ids.append(i)

    # # print(f'# of markers filtered out due to angle: {n_angle}, area: {n_area}, aspect ratio: {n_aspect}')
    for id, (angle, aspect) in zip(ids, zip(angles_list, aspect_list)):
        if id not in valid_ids:
            print(f'\tMarker {id}: angle: {np.round(angle, 1)}, aspect ratio: {aspect:.2f}')

    return valid_ids


def convert_markers_to_descriptions(marker_positions):
    # input is (n, 4, 3) array of corners 
    descriptions = []
    for corners in marker_positions:
        p1, p2, _, p4 = corners
        p1, r = points_to_description(p1, p2, p4)
        descriptions.append((p1, r))

    return np.asarray(descriptions)

def convert_descriptions_to_markers(descriptions):
    # input is (n, 2, 3) array of p1, r
    marker_positions = []
    for p1, r in descriptions:
        p1, p2, p3, p4 = description_to_points(p1, r)
        marker_positions.append([p1, p2, p3, p4])

    return np.asarray(marker_positions)


def check_marker_positions(marker_positions: dict[int, np.ndarray]):
    # check if the markers are square and have the same size
    markerlength = None
    for id, corners in marker_positions.items():
        # reshape to 4x3
        corners = np.array(corners).reshape(4, 3)
        # corners have length 4, find two sides and check if they are equal
        for c1, c2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            side = np.linalg.norm(corners[c1] - corners[c2])
            if markerlength is None:
                markerlength = side
            elif not np.isclose(markerlength, side):
               print(f'Marker {id} has sides of different lengths ({side:.3f}!={markerlength:.3f})')

    # check if the markers are square
    for id, corners in marker_positions.items():
        # reshape to 4x3
        corners = np.array(corners).reshape(4, 3)
        # must check if the enclosed angles are 90 deg
        angles = [
            angle_between(corners[3], corners[0], corners[1]),
            angle_between(corners[0], corners[1], corners[2]),
            angle_between(corners[1], corners[2], corners[3]),
            angle_between(corners[2], corners[3], corners[0])
        ]
        for i, angle in enumerate(angles):
            if not np.isclose(angle, 90):
                print(f'Marker {id} has angle {i+1} of {angle:.2f} deg')


def setup_aruco() -> aruco.ArucoDetector:
        aruco_params = aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        aruco_params.cornerRefinementWinSize = 2
        aruco_params.adaptiveThreshWinSizeMin = 15
        aruco_params.adaptiveThreshWinSizeMax = 15
        aruco_params.minMarkerPerimeterRate = 0.02
        aruco_params.maxMarkerPerimeterRate = 2
        aruco_params.minSideLengthCanonicalImg = 16
        aruco_params.adaptiveThreshConstant = 7
        # aruco_params.polygonalApproxAccuracyRate = 0.05
        # aruco_params.useAruco3Detection = True

        aruco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        return aruco.ArucoDetector(aruco_dic, aruco_params)   

def plot_face(ax, face, **kwargs):
    plot_face = np.vstack((face, face[0]))
    if kwargs.pop('plot_first_corner', True):
        ax.plot([face[0,0]], [face[0,1]], [face[0,2]], 'o', color='green') # plot first corner
    ax.plot(plot_face[:,0], plot_face[:,1], plot_face[:,2], **kwargs)
    # return color
    return ax.lines[-1].get_color()

# use tex
usetex = False
if usetex:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # fontsize to 14
    plt.rc('font', size=14)

def plot_marker_positions(marker_positions: dict, prev_marker_positions: Optional[np.array] = None, name: Optional[str] = None, output_path: Optional[str] = None, show=True, collage_mode=False):
    """Plots the marker positions in 3D."""
    if collage_mode:
        fig = plt.figure(figsize=(6, 6))
    else:
        fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # plot previous corners in red
    if prev_marker_positions is not None:
        for id in sorted(prev_marker_positions.keys()):
            corners = prev_marker_positions[id]
            plot_face(ax, np.array(corners), color='red', linestyle='--', plot_first_corner=not collage_mode, linewidth=1)
        # add to legend
        ax.plot([], [], 'red', label='Initial marker positions', linestyle='--')
        
    # sort by id
    for id in sorted(marker_positions.keys()):
        corners = marker_positions[id]
        plot_face(ax, np.array(corners), plot_first_corner=not collage_mode)
        if not collage_mode:
            ax.text(np.array(corners)[:,0].mean(), np.array(corners)[:,1].mean(), np.array(corners)[:,2].mean(), str(id))
    # add to legend if previous marker positions are given
    if prev_marker_positions is not None:
        ax.plot([], [], 'black', label='Calibrated marker positions')
        ax.legend()

    if not collage_mode:
        if usetex:
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            z_min, z_max = ax.get_zlim()
            # space in distances of 1
            ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max), 1))
            ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max), 1))
            ax.set_zticks(np.arange(np.floor(z_min), np.ceil(z_max), 1))
            # remove ax labels
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(name)
    else:
        # hide everything
        ax.set_axis_off()
        # hide legend
        ax.legend().set_visible(False)

    ax.axis('equal')
    # # look from front
    # ax.view_init(45, 90)
    plt.tight_layout()
    if name is not None and output_path is not None:
        if collage_mode:
            # create subfolder for collage
            collage_path = os.path.join(output_path, 'collages')
            os.makedirs(collage_path, exist_ok=True)
            # keep elevation and loop over azimuth in steps of 1
            for az in range(0, 360, 1):
                ax.view_init(-150, az)
                output_path = os.path.join(collage_path, name.replace(' ', '_') + f'_az{az}.png')
                plt.savefig(output_path, dpi=300, transparent=False)
                print(f'Saved plot "{name}" to {output_path}')
        else:
            if usetex:
                # save as svg
                output_path = os.path.join(output_path, name.replace(' ', '_') + '.svg')
                plt.savefig(output_path)
            else:
                # title, tight layout and save
                output_path = os.path.join(output_path, name.replace(' ', '_') + '.png')
                plt.savefig(output_path, dpi=300)
            print(f'Saved plot "{name}" to {output_path}')
    if show:
        plt.show()
    # close figure
    plt.close(fig)


Observation = dict[int, np.ndarray]  # Map from marker id to 4x3 array of corners

def normalize(marker_poses: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Normalizes the marker positions such that the first corner of the first marker is at (0, 0, 0) and all markers have side length 1."""
    # assert that ideal marker positions are square and all of the same size
    markerlength = None
    for id, corners in marker_poses.items():
        # corners have length 4, find two sides and check if they are equal
        for c1, c2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            side = np.linalg.norm(corners[c1] - corners[c2])
            if markerlength is None:
                markerlength = side
            else:
                assert np.isclose(markerlength, side), f'Marker {id} has sides of different lengths ({side:.3f}!={markerlength:.3f}), cannot normalize.'
    print(f'Normalizing marker positions to have side length 0.01 (1 cm), found marker length {markerlength:.2f}')
    # normalize marker length to 0.01 (1 cm)
    for id, corners in marker_poses.items():
        marker_poses[id] /= markerlength
    return marker_poses

def fix_corner(ids: list[int], marker_poses: dict[int, np.ndarray], fix_id: int) -> dict[int, np.ndarray]:
    """Moves the marker with id fix_id to the front of the list and shifts all markers so that the first corner of the first marker is at (0, 0, 0)."""
    # set such that fix_id is the first marker
    # move the fixed marker to the front
    if fix_id in ids:
        ids.remove(fix_id)
        ids.insert(0, fix_id)
    else:
        print(f'Fix id {fix_id} not found in marker ids. Fixing first marker.')

    # shift all markers so that the first corner of the first marker is at (0, 0, 0)
    first_corner = marker_poses[ids[0]][0, :].copy()
    for id in ids:
        marker_poses[id] -= first_corner


    return ids, marker_poses

class MarkerCalibrator:
    def __init__(self, initial_pos_path: str, marker_imgs_path: str, cam_calib_file_path: str, output_path: str, **params):
        self.debug = params.pop('debug', False)

        self.overwrite_detections = params.pop('overwrite_detections', False)
        self.enable_aruco_filtering = params.pop('enable_aruco_filtering', False)

        initial_json = json.load(open(initial_pos_path, "r"))

        self.ideal_marker_pos = {int(k): np.array(v) for k, v in initial_json.items()}

        # assert that ideal marker positions are square and all of the same size
        self.normalize = params.pop('normalize', False)
        if self.normalize:
            self.ideal_marker_pos = normalize(self.ideal_marker_pos)

        self.ids = sorted(self.ideal_marker_pos.keys())  
        # set such that fix_id is the first marker
        fix_id = params.get('fix_id', self.ids[0])
        self.ids, self.ideal_marker_pos = fix_corner(self.ids, self.ideal_marker_pos, fix_id)

        # check if the markers are square and have the same size
        check_marker_positions(self.ideal_marker_pos)

        self.use_descriptions = params.pop('use_descriptions', True)

        self.reject_outliers = params.pop('reject_outliers', False)
        self.outlier_threshold = params.pop('outlier_threshold', 1.5)


        self.marker_imgs_path = marker_imgs_path
        self.output_path = output_path
        # create output path if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        if self.overwrite_detections:
            # remove filtered folder if it exists
            filtered_path = os.path.join(output_path, 'filtered')
            if os.path.exists(filtered_path):
                for f in os.listdir(filtered_path):
                    os.remove(os.path.join(filtered_path, f))
                os.rmdir(filtered_path)

        self.calibrated = False        
        if os.path.isfile(cam_calib_file_path):
            # if npz file:
            if cam_calib_file_path.endswith('.npz'):
                with np.load(cam_calib_file_path) as X:
                    self.cam_mat, self.cam_dist = [X[i] for i in ('mtx', 'dist')]
                    self.calibrated = True
            # if json file:
            elif cam_calib_file_path.endswith('.json'):
                with open(cam_calib_file_path, "r") as f:
                    calib = json.load(f)
                    self.cam_mat = np.array(calib['mtx'])
                    self.cam_dist = np.array(calib['dist'])
                    self.calibrated = True
            else:
                print(f'Calibration file "{cam_calib_file_path}" not recognized. Must be .npz or .json')
                return
        else:
            print(f'Calibration file "{cam_calib_file_path}" not found.')
            return
        
        header_string = f'\n{"-" * 40}  {"MARKER CALIBRATION"} {"-" * 40}\n'
        
        summary_str = f'\tLoaded camera calibration from "{cam_calib_file_path}"\n'
        summary_str += f'\tLoaded initial marker positions from "{initial_pos_path}"\n'
        summary_str += f'\tCalibrating object with markers: {", ".join(map(str, self.ids))}, ({len(self.ids)} total), fixing marker {fix_id}\n'
        summary_str += f'\tLoading calibration images from "{marker_imgs_path}"\n'
        summary_str += f'\tOutputting results to "{output_path}"\n'
        summary_str += f'\tSettings:\n'
        summary_str += f'\t\tUse descriptions: {self.use_descriptions}\n'
        summary_str += f'\t\tDebug: {self.debug}\n'
        summary_str += f'\t\tNormalize marker positions: {self.normalize}\n'
        summary_str += f'\t\tOverwrite detections: {self.overwrite_detections}\n'
        summary_str += f'\t\tEnable Aruco filtering: {self.enable_aruco_filtering}\n'
        summary_str += f'\t\tReject outliers: {self.reject_outliers}\n'
        if self.reject_outliers:
            summary_str += f'\t\t\tOutlier threshold: {self.outlier_threshold}\n'
        # print enough dashes to cover the entire header string
        summary_str = header_string + summary_str + f'{"-" * len(header_string)}\n'
    
        print(summary_str)      

        self.marker_pose: Optional[np.ndarray] = None      

    def detectMarkers(self, img):
        corners2d, ids, rejected = self.aruco_detector.detectMarkers(img)
        # return corners2d, ids, rejected
        ids = [] if ids is None else [id[0] for id in ids]	
        corners2d = [] if corners2d is None else [np.array(corners[0]) for corners in corners2d]
        return corners2d, ids, rejected  
    
    def detect_calibration_imgs(self) -> Optional[List[Observation]]:
        """Reads in the set of images, and detects any markers in them."""
        observed_points: list[Observation] = []
        observed_ids: list[int] = []
        for img_name in os.listdir(self.marker_imgs_path):
            if 'calibrated' in img_name or 'ideal' in img_name:
                continue
            if not any([img_name.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']]):
                continue
            img = cv2.imread(os.path.join(self.marker_imgs_path, img_name))
            corners_all, ids_all, _ = self.detectMarkers(img)
            corners_all = tuple(np.array(corners).reshape(-1, 4, 2) for corners in corners_all) # to ensure compatibility with rest of program convert back to opencv format
            
            # retain only the markers we are interested in
            ids = np.array([id for id in ids_all if id in self.ids])
            corners = np.array([corner for corner, id in zip(corners_all, ids_all) if id in self.ids])

            # retain only the markers we are interested in
            ids = np.array([id for id in ids_all if id in self.ids])
            corners = np.array([corner for corner, id in zip(corners_all, ids_all) if id in self.ids])


            # if len(ids) < len(ids_all):
            #     # abs_path = os.path.abspath(os.path.join(self.marker_imgs_path, img_name))
            #     abs_path = img_name
            #     print(f'Found {len(ids_all)} markers in image "{abs_path}", but only {len(ids)} are in the set of markers to calibrate. Markers {np.setdiff1d(ids_all, ids)} are not used.')



            if self.enable_aruco_filtering:
                # filter markers based on aspect ratio and angle
                ids_valid = filter_markers(ids, corners.reshape(-1, 4, 2))
                filt_aspect_angle = np.setdiff1d(ids, ids_valid)
                if filt_aspect_angle.size > 0:
                    print(f'Filtered out markers due to aspect ratio or angle: {filt_aspect_angle}')
                    os.makedirs(os.path.join(self.output_path, 'filtered'), exist_ok=True)
                    # mark filtered markers
                    corners_filtered = np.array([corner for corner, id in zip(corners, ids) if id in filt_aspect_angle])
                    corners_filtered = corners_filtered.reshape(-1, 4, 2)
                    if False:
                        # into tuple
                        corners_filtered = (corners_filtered,)
                        img = aruco.drawDetectedMarkers(img, corners_filtered, filt_aspect_angle.squeeze())
                    else:
                        # implement own drawing function
                        for c in corners_filtered:
                            # connect corners with green line
                            for i in range(4):
                                cv2.line(img, tuple(int(x) for x in c[i]), tuple(int(x) for x in c[(i+1) % 4]), (0, 255, 0), 2)
                            # draw corners blue, red, green, yellow (1, 2, 3, 4)
                            for i, corner in enumerate(c):
                                cv2.circle(img, tuple(int(x) for x in corner), 5, (255, 0, 0), -1)
                                cv2.putText(img, str(i), tuple(int(x) for x in corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.imwrite(os.path.join(self.output_path, 'filtered', f'filter_aspect_angle_{img_name}'), img)
                
                corners = np.array([corner for corner, id in zip(corners, ids) if id in ids_valid])
                ids = np.array(ids_valid)


            observed_ids.extend(ids.flatten().tolist())  
            if self.debug:
                # show image with markers
                img = aruco.drawDetectedMarkers(img, corners, ids.squeeze())
                # rescale at 1500
                scale = 1500 / img.shape[1]
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                cv2.imshow('img', img)
                cv2.waitKey(0)
            if len(ids) == 0:
                print(f'No markers of interest found in image "{img_name}". Not using.')
                continue
            if len(ids) < 2:
                print(f'Only {len(ids)} markers of interest found in image "{img_name}". Not using.')
                continue
            corner_dict: Observation = {}
            for i, id in enumerate(ids):
                corner_dict[id] = corners[i][0]
            observed_points.append(corner_dict)

        cv2.destroyAllWindows()
        # check if all markers were observed
        if np.all(np.isin(self.ids, observed_ids)):
            print(f'All markers ({len(self.ids)}) were observed in calibration images.')
        else:
            print(f'Not all markers were observed. Missing markers: {np.setdiff1d(self.ids, np.unique(observed_ids))}')
            raise ValueError('Not all markers were observed in calibration images.')
        if not observed_points:
            print(f'No valid calibration images found in path {self.marker_imgs_path}')
            return None
        else:
            return observed_points
        

    def get_initial_estimate(self):
        """Computes an initial state vector based on:
        - The ideal marker positions
        - Camera poses for each image, estimated using PnP
        """
        marker_poses = np.zeros((len(self.ids), 4, 3), dtype=np.float32)
        for i, id in enumerate(self.ids):
            marker_poses[i, :, :] = self.ideal_marker_pos[id]

        camera_poses = np.zeros((len(self.observed_points), 6), dtype=np.float32)
        for img_id, img in enumerate(self.observed_points):
            validMarkers = []
            for marker_id, marker_corners_observed in img.items():
                if marker_id in self.ideal_marker_pos:
                    validMarkers.append(
                        (self.ideal_marker_pos[marker_id], marker_corners_observed)
                    )

            corners3d = np.concatenate([cornersIS for _, cornersIS in validMarkers])
            corners2d = np.concatenate([cornersPS for cornersPS, _ in validMarkers])
            success, rvec, tvec = cv2.solvePnP(
                corners2d,
                corners3d,
                self.cam_mat,
                self.cam_dist,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            camera_poses[img_id, :] = np.hstack((rvec.flatten(), tvec.flatten()))

        print(f'Tvecs averaged: {np.round(np.mean(camera_poses[:, 3:6], axis=0), 2)}')

        result = np.concatenate((marker_poses.flatten(), camera_poses.flatten()))
        return result[0:12], result[12:] # fixing first marker
    
    

    def residual(self, x: np.ndarray):
        marker_poses = np.concatenate((self.marker_pose, x[0 : (len(self.ids) - 1) * 4 * 3])).reshape((len(self.ids), 4, 3))
        camera_poses = x[(len(self.ids) - 1) * 4 * 3 :].reshape((-1, 6))
        res_all = []
        for img_id, img in enumerate(self.observed_points):
            rvec = camera_poses[img_id, 0:3]
            tvec = camera_poses[img_id, 3:6]

            for i, id in enumerate(self.ids):
                if id in img:
                    projected, jac = cv2.projectPoints(
                        objectPoints=marker_poses[i, :, :], # 4x3
                        rvec=rvec,
                        tvec=tvec,
                        cameraMatrix=self.cam_mat,
                        distCoeffs=self.cam_dist,
                    )

                    res = projected.flatten() - img[id].flatten()
                    res_all.append(res)

        # normalize by number of observations
        return np.concatenate(res_all) / len(self.observed_points)     
    

    def residual_descriptions(self, x: np.ndarray, binary_map=None, debug=False, return_res_non_concat=False):
        # each description is a tuple of (p1, r) (6 values)
        marker_descriptions_strp = x[0 : (len(self.ids) - 1) * 6].reshape((len(self.ids)-1, 2, 3))
        marker_poses_strp = convert_descriptions_to_markers(marker_descriptions_strp)
        marker_poses = np.concatenate((self.marker_pose.reshape(1, 4, 3), marker_poses_strp)).reshape((len(self.ids), 4, 3))
        camera_poses = x[(len(self.ids) - 1) * 6 :].reshape((-1, 6))

        if debug and return_res_non_concat:
            res_all_non_concat = []

        res_all = []
        for img_nr, img_ids in enumerate(self.observed_points):
            # if img_id == 2 and debug:
            #     print('Debugging: Skipping image 2')
            #     continue
            rvec = camera_poses[img_nr, 0:3]
            tvec = camera_poses[img_nr, 3:6]
            if debug:
                res_obs = []
            idx_id = 0
            for i, id in enumerate(self.ids):
                if id in img_ids:
                    idx_id += 1 # must already increment here due to 'continue' statement
                    if binary_map is not None:
                        if not binary_map[img_nr][idx_id-1]:
                            if debug:
                                print(f'Obs {img_nr}: Skipping marker {id} at index {idx_id} in image {img_nr} because it has too large error.')
                            continue

                    projected, jac = cv2.projectPoints(
                        objectPoints=marker_poses[i, :, :], # 4x3
                        rvec=rvec,
                        tvec=tvec,
                        cameraMatrix=self.cam_mat,
                        distCoeffs=self.cam_dist,
                    )

                    res = projected.flatten() - img_ids[id].flatten()
                    res_all.append(res)
                    if debug:
                        # print(f'Marker {id} in image {img_nr}: {np.linalg.norm(res):.3f}')
                        res_obs.append(np.linalg.norm(res))
                
            if debug and not return_res_non_concat:
                # print(f'Observation {img_id}: {np.linalg.norm(res_obs):.3f}')
                print(f'\tPer marker repr err. observation {img_nr}: {np.round(res_obs, 4)}')
            
            if debug and return_res_non_concat:
                res_all_non_concat.append(res_obs)

        if return_res_non_concat:
            return res_all_non_concat
        
        # normalize by number of observations
        return np.concatenate(res_all) / len(self.observed_points)
    

    def calibrate_markers(self):
        if self.calibrated:
            self.aruco_detector = setup_aruco()

            observed_points_path = os.path.join(self.output_path, 'observed_points.npy')
            if os.path.isfile(observed_points_path) and not self.overwrite_detections:
                print(f'Loaded observed points from {observed_points_path}')
                self.observed_points = np.load(observed_points_path, allow_pickle=True)
            else:
                self.observed_points = self.detect_calibration_imgs()
                np.save(observed_points_path, self.observed_points)


            # first plot the ideal marker positions
            plot_marker_positions(self.ideal_marker_pos, name='ideal marker poses', output_path=self.output_path, show=self.debug)
            # We fix the pose of the first marker, so that the problem is properly constrained.
            self.marker_pose, x0 = self.get_initial_estimate()

            if not self.use_descriptions:
                init_error = np.linalg.norm(self.residual(x0))
                print(f'Starting calibration with initial error {init_error:.2f}')
                opt_result = scipy.optimize.least_squares(
                    fun=self.residual,
                    x0=x0, 
                    max_nfev=400, 
                    verbose=2, 
                    ftol=1e-8
                )

                final_error = np.linalg.norm(self.residual(opt_result.x))
                # save results
                opt_marker_poses = np.concatenate((self.marker_pose, opt_result.x[0 : (len(self.ids) - 1) * 4 * 3])).reshape((len(self.ids), 4, 3))
            else:
                marker_poses_strp = x0[0 : (len(self.ids) - 1) * 4 * 3].reshape((len(self.ids)-1, 4, 3))
                camera_poses = x0[(len(self.ids) - 1) * 4 * 3 :].reshape((-1, 6))
                import time
                t_start = time.time()
                marker_descriptions_strp = convert_markers_to_descriptions(marker_poses_strp)
                # print(f'Initial marker descriptions:')
                # print(marker_descriptions_strp)
                # print(f'Initial marker poses:')
                # print(marker_poses_strp)
                # print(f'Conversion took {time.time() - t_start:.2f} s')

                x0_descriptions = np.concatenate((np.array(marker_descriptions_strp).flatten(), camera_poses.flatten()))

                init_error = np.linalg.norm(self.residual_descriptions(x0_descriptions, debug=True))

                print(f'Starting calibration with initial error {init_error:.2f}')

                opt_result = scipy.optimize.least_squares(
                    fun=self.residual_descriptions,
                    x0=x0_descriptions, 
                    max_nfev=400, 
                    verbose=2, 
                    ftol=1e-8
                )

                final_error = np.linalg.norm(self.residual_descriptions(opt_result.x, debug=True))


                res_non_concat = self.residual_descriptions(opt_result.x, debug=True, return_res_non_concat=True)
                binary_map = []
                for obs in res_non_concat:
                    binary_map.append([1 if r < self.outlier_threshold else 0 for r in obs])

                # check if any markers are filtered out
                if self.reject_outliers and not all([all(row) for row in binary_map]):
                    print(binary_map)
                    n_filtered = np.sum([1 for row in binary_map if not all(row)])
                    print(f'Filtered out {n_filtered} markers ({n_filtered / sum([len(row) for row in binary_map]) * 100:.2f}%) due to large error (outlier threshold {self.outlier_threshold}).')
                    print(f'First optimization run has error {final_error:.2f}, optimizing again with outliers removed.')

                    self.residual_descriptions(x0_descriptions, binary_map=binary_map)

                    # optimize again with binary map
                    opt_result = scipy.optimize.least_squares(
                        fun=self.residual_descriptions,
                        x0=opt_result.x, 
                        args=(binary_map,),
                        max_nfev=400, 
                        verbose=2, 
                        ftol=1e-8
                    )

                    final_error = np.linalg.norm(self.residual_descriptions(opt_result.x, debug=True, binary_map=binary_map))
                                    

                # save results
                marker_descriptions_strp = opt_result.x[0 : (len(self.ids) - 1) * 6].reshape((len(self.ids)-1, 2, 3))
                opt_marker_poses = convert_descriptions_to_markers(marker_descriptions_strp)

                opt_marker_poses = np.concatenate((self.marker_pose.reshape(1, 4, 3), opt_marker_poses)).reshape((len(self.ids), 4, 3))                


        

            print(f'Finished calibration with final error {final_error:.2f}')
            print(f'\nReduction by {100 * (init_error - final_error) / init_error:.2f}%\n')

            opt_marker_poses_dict = {id: opt_marker_poses[i, :, :].tolist() for i, id in enumerate(self.ids)}

            # check if the markers are square and have the same size
            check_marker_positions(opt_marker_poses_dict)

            output_filename = 'marker_poses_calibrated.json' if not self.normalize else 'marker_poses_calibrated_normalized.json'
            output_json = os.path.join(self.output_path, output_filename)
            with open(output_json, "w") as f:
                json.dump(opt_marker_poses_dict, f, indent=4)
            print(f'Saved calibrated marker poses to {output_json}')

            with open(output_json, "r") as f:
                opt_marker_poses_dict = json.load(f)

        
            # write initial and final error to file
            with open(os.path.join(self.output_path, 'calibration_errors.txt'), "w") as f:

                f.write(f'Initial error: {init_error:.2f}\n')
                f.write(f'Final error: {final_error:.2f}\n')
           

            # plot results
            plot_marker_positions(opt_marker_poses_dict, prev_marker_positions=self.ideal_marker_pos, name='calibrated marker poses', output_path=self.output_path, show=self.debug)


if __name__ == "__main__":

    os.chdir(base_dir)
    print(f'Changed directory to {os.getcwd()}')

    params = {}

    # sample object    
    cam_calib_file_path = r'files\rest\marker_calibrations\sample_oid\camera_calibration.json'
    base_dir = r'files\rest\marker_calibrations\sample_oid'
    marker_imgs_path = os.path.join(base_dir, 'calibration_frames')
    out_path = os.path.join(base_dir, 'calibrated')
    initialPath = os.path.join(base_dir, 'initial_poses.json')
    params['fix_id'] = 54 
    params['overwrite_detections'] = False # redetet aruco
    params['use_descriptions'] = True # right angle and equidistant points per marker constraints

    debug = False
    calibrator = MarkerCalibrator(
        initial_pos_path=initialPath,
        marker_imgs_path=marker_imgs_path,
        cam_calib_file_path=cam_calib_file_path,
        output_path=out_path,
        normalize=True,
        debug=debug,
        **params
    )
    calibrator.calibrate_markers()
