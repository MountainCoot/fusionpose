import os
import cv2
from cv2 import aruco

import numpy as np
from typing import Optional, Tuple, List
from deps.utils.logger_util import Logger

from deps.utils import visualization
from deps.utils.util import vector_rms, clamp
import json
import time

from functools import wraps


def consistent_debug(func: callable) -> callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Enable the lock
        self._debug_lock = True
        try:
            # Execute the wrapped function with consistent debug
            result = func(self, *args, **kwargs)
        finally:
            # Disable the lock and apply any pending debug update
            self._debug_lock = False
            if self._pending_debug is not None:
                self.debug = self._pending_debug
                self._pending_debug = None  # Clear the pending update
        return result
    return wrapper

def get_normed_repr_err(repr_err: np.ndarray, img_pts: np.ndarray) -> float:
    return np.linalg.norm(repr_err) / len(img_pts)

def solve_pnp(
        initialized: bool,
        prev_rvec: Optional[np.ndarray],
        prev_tvec: Optional[np.ndarray],
        obj_pts: np.ndarray,
        img_pts: np.ndarray,
        cam_mat: np.ndarray,
        dist_coeffs: np.ndarray,
        repr_err_thresh: float = 10,
                ) -> Tuple[bool, np.ndarray, np.ndarray, bool, float]:
    
    """Attempt to refine the previous pose. If this fails, fall back to SQPnP."""
    flags = cv2.SOLVEPNP_EPNP  
    if initialized:
        try:
            rvec_refine, tvec_refine = cv2.solvePnPRefineVVS(
                obj_pts,
                img_pts,
                cameraMatrix=cam_mat,
                distCoeffs=dist_coeffs,
                rvec=prev_rvec.copy(),
                tvec=prev_tvec.copy(),
            )

            rvec = rvec_refine
            tvec = tvec_refine

        except Exception as e:
            Logger.warning(f"Failed to refine pose with solvePnPRefineVVS: {e}. Falling back to SQPnP.")
            return solve_pnp(False, None, None, obj_pts, img_pts, cam_mat, dist_coeffs, repr_err_thresh)
            
        projected_image_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mat, dist_coeffs, None)
        projected_image_points = projected_image_points[:, 0, :]
        reprojection_error = vector_rms(projected_image_points - img_pts, axis=1)

        if reprojection_error < repr_err_thresh:
            return (True, rvec, tvec, False, get_normed_repr_err(reprojection_error, img_pts))
        else:
            # try normal pnp
            return solve_pnp(False, None, None, obj_pts, img_pts, cam_mat, dist_coeffs, repr_err_thresh)
    try:
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cameraMatrix=cam_mat, distCoeffs=dist_coeffs, flags=flags)
        projected_image_points, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mat, dist_coeffs, None)
        projected_image_points = projected_image_points[:, 0, :]
        reprojection_error = vector_rms(projected_image_points - img_pts, axis=1)
        if success and reprojection_error < repr_err_thresh:
            return (True, rvec, tvec, False, get_normed_repr_err(reprojection_error, img_pts))
        else:
            return (False, None, None, True, get_normed_repr_err(reprojection_error, img_pts))
        
    except Exception as e:
        Logger.error(f"Failed to solve PnP: {e}")
        return (False, None, None, False, -1)

def load_marker_positions(marker_positions_path: str, marker_length: Optional[float] = None, shift: Optional[List[float]] = None) -> dict[int, np.ndarray]:
        # load json with {id: corners3d}
        with open(marker_positions_path, "r") as f:
            marker_positions = json.load(f)
        # convert to np.ndarray
        marker_positions = {int(k): np.array(v) for k, v in marker_positions.items()}
        # scale to marker_length
        if marker_length is not None:
            for k, v in marker_positions.items():
                marker_positions[k] = v * marker_length

        if shift is not None:
            for k, v in marker_positions.items():
                marker_positions[k] += shift
        return marker_positions

def get_calibration_stats(calibration_path: str) -> dict[int, int]:
    calibration_stats = {}
    for root, _, files in os.walk(calibration_path):
        for file in files:
            if file.endswith('.png'):
                # of the form frame_1_2_3_ver0.png
                ids = [int(id) for id in file.split('_')[1:-1]]
                for id in ids:
                    calibration_stats[id] = calibration_stats.get(id, 0) + 1
    return calibration_stats

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
    aruco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    return aruco.ArucoDetector(aruco_dic, aruco_params)         


class MarkerTracker:
    def __init__(self, camera_name: str, calib_file_path: str, **params) -> None:

        self.setup_complete = False
        self.camera_name = camera_name
        self.debug = params.pop('debug', False)
        self._debug_lock = False  # Lock to control modifications
        self._pending_debug = None  # Store pending debug updates
        
        if not self.load_calibration(calib_file_path):
            return
                
        self.idx = 0
        # define interval to detect markers on full frame

        self.detector_mode = params.pop('detector_mode', 'aruco')

        self.aruco_detector = setup_aruco()

        self.FPS = params.pop('fps', 30)            
        self.FRAME_HEIGHT: Optional[int] = None
        self.FRAME_WIDTH: Optional[int] = None

        self.acquire_calibration = params.pop('acquire_calibration', False)
        calibration_params = params.pop('calibration_params', {})

        if not self.acquire_calibration:

            detect_FF_interval_sec = params.pop('detect_full_frame_interval', 0.5)
            self.detect_FF_interval = max(int(detect_FF_interval_sec * self.FPS), 1)
            self.repr_err_norm_thresh = 0.05 # TODO: define externally

            oid_config_dict = params.pop('oid_config_dict', '')
            self.oids = list(oid_config_dict.keys())
            oid_marker_lengths = [config['marker_length'] for config in oid_config_dict.values()]
            oid_pts_paths = [config['pts_file_path'] for config in oid_config_dict.values()]
            oid_shifts = [config.get('shift', [0, 0, 0]) for config in oid_config_dict.values()]
            self.marker_config3d_dict = {oid: load_marker_positions(pts_file, marker_length, shift) for oid, pts_file, marker_length, shift in zip(self.oids, oid_pts_paths, oid_marker_lengths, oid_shifts)}
            self.prev_markers_valid_dict = {oid: {} for oid in self.marker_config3d_dict.keys()}

            self.marker_config3d_flat_dict = {oid: np.concatenate(list(marker_positions.values())) for oid, marker_positions in self.marker_config3d_dict.items()}
            self.oids_ids = {oid: list(marker_positions.keys()) for oid, marker_positions in self.marker_config3d_dict.items()}

            self.rvec_dict: dict[str, Optional[np.ndarray]] = {oid: None for oid in self.oids}
            self.tvec_dict: dict[str, Optional[np.ndarray]] = {oid: None for oid in self.oids}
            self.initialized_dict: dict[str, bool] = {oid: False for oid in self.oids}
            self.oid_lost_in_prev: dict[str, bool] = {oid: False for oid in self.oids}
            self.prev_vel_mean_dict: dict[str, np.ndarray] = {oid: np.zeros(2) for oid in self.oids}
            self.search_area_dict: dict[str, Tuple[int, int, int, int]] = {oid: None for oid in self.oids}
            self.prev_ids_detec_all = []
            self.vel_expand = {oid: [0, 0, 0, 0] for oid in self.oids}

            self.timer = time.time()
            self.detec_fail_counter = 0
            self.oid_high_repr_err_count = {oid: 0 for oid in self.oids}

            self.opened_windows = []

            if self.debug:
                Logger.debug(f'\t"{self.camera_name}": Marker tracker debug mode is on.')
            
            if len(params) > 0:
                Logger.warning(f'"{self.camera_name}": Marker tracker: provided {len(params)} unused params:')
                for param, value in params.items():
                    Logger.warning(f'\t{param}: {value}')

        else:
            Logger.info(f'"{self.camera_name}": MarkerTracker: Acquiring calibration frames for marker calibration.')

            self.tracked_ids = {} # Used for saving frames for marker calibration
            self.calibration_path = calibration_params.pop('path', None)
            if self.calibration_path is None:
                Logger.error(f'"{self.camera_name}": MarkerTracker: No calibration path provided.')
                return
            else:
                Logger.info(f'"{self.camera_name}": MarkerTracker: Calibration path: {self.calibration_path}')
            # in calibration path, get all files, and count each id to store into self.calibration_stats
            self.calibration_stats = get_calibration_stats(self.calibration_path)

            self.ids_to_calibrate = calibration_params.pop('ids_to_calibrate', None)
            if self.ids_to_calibrate is not None:
                Logger.info(f'"{self.camera_name}": MarkerTracker: Calibration ids: {self.ids_to_calibrate}')
                
        self.setup_complete = True

    def load_calibration(self, calib_file_path: str, callback: bool = False) -> bool:
        if os.path.isfile(calib_file_path):
            visualization.CameraIntrinsics(calib_file_path)
            # see if it is a .npz file
            if calib_file_path.endswith('.npz'):
                with np.load(calib_file_path) as X:
                    self.cam_mat, self.cam_dist = [X[i] for i in ('mtx', 'dist')]
                if callback:
                    Logger.info(f'"{self.camera_name}": Loaded calibration from {calib_file_path}.')
                return True
            elif calib_file_path.endswith('.json'):
                with open(calib_file_path, "r") as f:
                    calib = json.load(f)
                    self.cam_mat = np.array(calib['mtx'])
                    self.cam_dist = np.array(calib['dist'])
                if callback:
                    Logger.info(f'"{self.camera_name}": Loaded calibration from {calib_file_path}.')
                return True
            else:
                msg = f'"{self.camera_name}": Calibration file "{calib_file_path}" has an unknown file extension.'
                if callback:
                    Logger.warning(f'{msg}. Keeping previous calibration.')
                else:
                    Logger.error(msg)
                return False
        else:
            msg = f'"{self.camera_name}": Calibration file "{calib_file_path}" not found.'
            if callback:
                Logger.warning(f'{msg}. Keeping previous calibration.')
            else:
                Logger.error(msg)
            return False

    def detectMarkers(self, img: np.ndarray) -> Tuple[list[np.ndarray], list[int], list[np.ndarray]]:
        corners2d, ids, rejected = self.aruco_detector.detectMarkers(img)
        ids = [] if ids is None else [id[0] for id in ids]	
        corners2d = [] if corners2d is None else [np.array(corners[0]) for corners in corners2d]
        return corners2d, ids, rejected
    
    def detect_markers(self, frame: np.ndarray, oid: str = None) -> Tuple[bool, list[np.ndarray], np.ndarray, list[np.ndarray]]:
        # t_start = time.time()
        ff_detection = False
        if oid is not None:
            if self.initialized_dict[oid]:
                (x0, x1, y0, y1), self.vel_expand[oid] = self.get_search_area(self.marker_config3d_flat_dict[oid], self.rvec_dict[oid], self.tvec_dict[oid], self.prev_vel_mean_dict[oid])
                self.search_area_dict[oid] = (x0, x1, y0, y1)
                # x0, y0 = max(x0, 0), max(y0, 0)
                frame_view = frame[y0:y1, x0:x1].copy()
                ids = None
                corners2d = []
                rejected = []
                try:
                    corners2d, ids, rejected = self.detectMarkers(frame_view)
                    corners2d = [[c + np.array([x0, y0]) for c in corners] for corners in corners2d]
                except cv2.error:
                    try:
                        # Logger.debug(f'Aruco Detection Error: Padding image (orginal: {x0}, {x1}, {y0}, {y1})')
                        padding = [9, 11, 7, 12] # random padding prevents this error in most cases
                        x0, x1 = max(x0 - padding[0], 0), min(x1 + padding[1], self.FRAME_WIDTH)
                        y0, y1 = max(y0 - padding[2], 0), min(y1 + padding[3], self.FRAME_HEIGHT)
                        frame_view = frame[y0:y1, x0:x1]
                        corners2d, ids, rejected = self.detectMarkers(frame_view)
                        corners2d = [[c + np.array([x0, y0]) for c in corners] for corners in corners2d]
                    except cv2.error as e:
                        self.detec_fail_counter += 1
                        ff_detection = True
                        corners2d, ids, rejected = self.detectMarkers(frame)
            else:
                corners2d, ids, rejected = [], [], []
                self.search_area_dict[oid] = None
        else:
            ff_detection = True    
            corners2d, ids, rejected = self.detectMarkers(frame)
        
        return ff_detection, corners2d, ids, rejected

    def get_search_area(self, marker_config3d_flat: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, velocity: np.array) -> Tuple[int, int, int, int]:
        """Returns a bounding box to search in the next frame, based on the current marker positions and velocity."""
        # Re-project all object points, to avoid cases where some markers were missed in the previous frame.
        projected_image_points, _ = cv2.projectPoints(marker_config3d_flat, rvec, tvec, self.cam_mat, self.cam_dist, None)
        projected_image_points = projected_image_points[:, 0, :]

        # Calculate the bounding box of the re-projected points.
        x0, y0 = np.min(projected_image_points, axis=0).astype(int)
        x1, y1 = np.max(projected_image_points, axis=0).astype(int)
        w = x1 - x0
        h = y1 - y0

        expand = (w // 2, h // 2)

        if velocity is not None:
            # expand box in direction of velocity
            vel_expand = [0, 0, 0, 0]
            vel_expand[0] = int(velocity[0]/self.FPS) 
            vel_expand[1] = int(velocity[0]/self.FPS)
            vel_expand[2] = int(velocity[1]/self.FPS)
            vel_expand[3] = int(velocity[1]/self.FPS)
            facs = [1.0+1.0*(clamp(np.abs(velocity[i]), 300, 2000)-300)/1700 for i in range(2)] # iterative tuning

            # make sure that fac squared is not bigger than 2 (i.e. expand by at most 2x)
            fac_prod_max = 2.0
            if np.prod(facs) > fac_prod_max:
                facs = [np.sqrt(fac_prod_max / np.prod(facs)) * max(1, f) for f in facs]

            expand = tuple([e*fac for e, fac in zip(expand, facs)])
        else:
            # expand initial box as no info about velocity available
            vel_expand = [-0.5*expand[0], 0.5*expand[1], -0.5*expand[0], 0.5*expand[1]]

        return (int(clamp(x0 - expand[0] + vel_expand[0], 0, self.FRAME_WIDTH)),
                int(clamp(x1 + expand[0] + vel_expand[1], 0, self.FRAME_WIDTH)),
                int(clamp(y0 - expand[1] + vel_expand[2], 0, self.FRAME_HEIGHT)),
                int(clamp(y1 + expand[1] + vel_expand[3], 0, self.FRAME_HEIGHT))), vel_expand
    
    def lost_oid(self, oid, was_initialized=None) -> None:
        if was_initialized is None:
            was_initialized = self.initialized_dict[oid]
        if was_initialized:
            self.oid_lost_in_prev[oid] = True

        self.initialized_dict[oid] = False
        self.prev_markers_valid_dict[oid] = {}
        self.prev_vel_mean_dict[oid] = np.zeros(2)
        self.search_area_dict[oid] = None

    def pnp_sequence(self, markers_valid_oid: dict[int, tuple[np.ndarray, np.ndarray]], oid: int, ids_invalid_all: dict = None) -> tuple[tuple, dict] | tuple:
        corner3d_valid, corners2d_valid = [corners3d for corners3d, _ in markers_valid_oid.values()], [corners2d for _, corners2d in markers_valid_oid.values()]
        ids_valid = np.array(list(markers_valid_oid.keys()))

        return_vals = solve_pnp(
            self.initialized_dict[oid],
            self.rvec_dict[oid],
            self.tvec_dict[oid],
            obj_pts=np.concatenate(corner3d_valid),
            img_pts=np.concatenate(corners2d_valid),
            cam_mat=self.cam_mat,
            dist_coeffs=self.cam_dist,
        )

        repr_err_norm = return_vals[-1]

        if not return_vals[0]:
            Logger.debug(f'"{self.camera_name}": Reprojection error too high: {repr_err_norm:.2f}')
        

        if repr_err_norm - len(ids_valid)*0.005 > self.repr_err_norm_thresh:
            if len(ids_valid) == 1: # when using single marker and error is high, discard at any rate
                return_vals = False, None, None, None, repr_err_norm
                # print(f'single marker high repr_err_norm: {repr_err_norm:.2f}')
                Logger.debug(f'"{self.camera_name}": Single marker high repr_err_norm: {repr_err_norm:.2f}')
            else:
                t_start = time.time()
                ignore_id = []
                repr_err_single_rejected = []
                # solve pnp for each marker individually
                for id, (corners3d, corners2d) in markers_valid_oid.items():
                        repr_err_norm_single = solve_pnp(
                            self.initialized_dict[oid],
                            self.rvec_dict[oid],
                            self.tvec_dict[oid],
                            obj_pts=corners3d,
                            img_pts=corners2d,
                            cam_mat=self.cam_mat,
                            dist_coeffs=self.cam_dist,
                        )[-1]

                        if repr_err_norm_single > 2*self.repr_err_norm_thresh:
                            ignore_id.append(id)
                            repr_err_single_rejected.append(repr_err_norm_single)

                repr_err_single_rejected = np.sort(repr_err_single_rejected)
                ignore_id = np.array(ignore_id)[np.argsort(repr_err_single_rejected)]

                # filtering step should not cause complete loss, so add back markers if too many removed
                min_markers = 2
                n_add_back = max(0, min_markers - (len(ids_valid) - len(ignore_id)))
                if n_add_back > 0:
                    ignore_id = ignore_id[n_add_back:]

                if len(ignore_id) > 0:
                    # print(f'"{self.camera_name}": Ignoring ids {ignore_id} with {repr_err_normalized_single:.2f} > {repr_err_norm:.2f}')          
                    idx_valid = [i for i, id in enumerate(ids_valid) if id not in ignore_id]
                    ids_valid = np.array([ids_valid[i] for i in idx_valid])
                    corners2d_valid = np.array([corners2d_valid[i] for i in idx_valid])
                    markers_valid_oid = {id: markers_valid_oid[id] for id in markers_valid_oid if id not in ignore_id}
                    corner3d_valid, corners2d_valid = [corners3d for corners3d, _ in markers_valid_oid.values()], [corners2d for _, corners2d in markers_valid_oid.values()]
                    return_vals_new = solve_pnp(
                        self.initialized_dict[oid],
                        self.rvec_dict[oid],
                        self.tvec_dict[oid],
                        obj_pts=np.concatenate(corner3d_valid),
                        img_pts=np.concatenate(corners2d_valid),
                        cam_mat=self.cam_mat,
                        dist_coeffs=self.cam_dist,
                    )
                    repr_err_norm_new = return_vals_new[-1]

                    info = f'"{self.camera_name}": oid {oid} reduced error by {(1-repr_err_norm_new/repr_err_norm)*100:.0f}% from {repr_err_norm:.2f} to {repr_err_norm_new:.2f}'
                    info += f' ({len(ignore_id)+len(ids_valid)} -> {len(ids_valid)} markers)'
                    info += f' in {(time.time() - t_start)*1000:.2f} ms'
                    info += f', repr single rejected: {np.round(repr_err_single_rejected, 3)}'
                    t_prev_parsed = np.round(return_vals[2].squeeze(), 4) if return_vals[2] is not None else 'None'
                    t_new_parsed = np.round(return_vals_new[2].squeeze(), 4) if return_vals_new[2] is not None else 'None'
                    info += f' [t_prev: {t_prev_parsed}, t_new: {t_new_parsed}]'

                    if (repr_err_norm_new < repr_err_norm) and (1 - repr_err_norm_new/repr_err_norm) > 0.2: # must have at least 20% reduction to be used
                        return_vals = return_vals_new
                        if self.debug:
                            if 'reprojection' not in ids_invalid_all:
                                ids_invalid_all['reprojection'] = []
                            ids_invalid_all['reprojection'].extend(ignore_id)
                    else:
                        info = f'NOT IMPROVED ENOUGH: {info}'
                    # Logger.debug(info)

                    if not return_vals_new[0]:
                        Logger.critical(f'"{self.camera_name}": AFTER REFINE Reprojection error too high: {repr_err_norm_new:.2f}')
                        Logger.debug(f'"{self.camera_name}": AFTER REFINE Reprojection error too high: {repr_err_norm_new:.2f}')


        # return false if z value is negative, wrong solution
        if return_vals[0] and return_vals[2][2] < 0:
            info = f'"{self.camera_name}": Negative z value, initialized {self.initialized_dict[oid]}, repr_err_norm: {repr_err_norm:.2f}, n_markers: {len(ids_valid)}'
            # do pnp again without initialization
            try:
                temp = solve_pnp(False, None, None, np.concatenate(corner3d_valid), np.concatenate(corners2d_valid), self.cam_mat, self.cam_dist)
                if temp[0]:
                    z_value = temp[2][2]
                    repr_err_norm = temp[-1]
                    info += f' ->resolved without init to z: {z_value}, repr_err_norm: {repr_err_norm}'
                else:
                    info += ' ->failed to resolve without init'
            except Exception as e:
                info += f' ->failed to resolve without init: {e}'

            print(info)
            Logger.debug(info)
            Logger.critical(info)
            return_vals = False, None, None, None, repr_err_norm

        if self.debug:
            return return_vals, ids_invalid_all        
        else:
            return return_vals

    @consistent_debug
    def process_frame(self, frame: np.ndarray) -> Optional[dict[str, Optional[np.ndarray]]]:

        if self.FRAME_HEIGHT is None or self.FRAME_WIDTH is None:
            self.FRAME_HEIGHT, self.FRAME_WIDTH = frame.shape[:2]

        if not self.acquire_calibration:
            ids_detec_all = []
            if self.debug:
                corners2d_all = []
                ids_all = []
                ids_invalid_all = {}

            T_dict = {oid: None for oid in self.oids}

            detect_FF = self.idx % self.detect_FF_interval == 0
            self.idx += 1

            corners2d_detec, ids_detec = None, None
            if detect_FF:
                _, corners2d_detec, ids_detec, _ = self.detect_markers(frame)
                ids_detec_all = ids_detec
            
            for oid in self.oids:

                self.oid_lost_in_prev[oid] = False
                if not detect_FF:
                    detect_FF, corners2d_detec, ids_detec, _ = self.detect_markers(frame, oid)
                    ids_detec_all.extend([id for id in ids_detec if id not in ids_detec_all])

                if self.debug:
                    # append all corners of ids that are not already in ids_all
                    corners2d_all.extend([corners2d_detec[i] for i, id in enumerate(ids_detec) if id not in ids_all])
                    ids_all.extend([id for id in ids_detec if id not in ids_all])

                idx_valid = [i for i, id in enumerate(ids_detec) if id in self.oids_ids[oid]]

                if len(idx_valid) == 0:
                    self.lost_oid(oid)
                    continue

                ids_valid = np.array([ids_detec[i] for i in idx_valid])
                corners2d_valid = np.array([corners2d_detec[i] for i in idx_valid])                

                if len(ids_valid) < 2: # less than 2 markers detected
                    self.lost_oid(oid)
                    continue

                # check for duplicate id, if so take the one which is closest to the other corners
                duplicate_idx = [i for i, id in enumerate(ids_valid) if np.sum(ids_valid == id) > 1]
                if len(duplicate_idx) > 0:
                    idx_to_del = []
                    duplicate_ids = {} # id -> idx
                    for idx in duplicate_idx:
                        id = ids_valid[idx]
                        if id not in duplicate_ids:
                            duplicate_ids[id] = [idx]
                        else:
                            duplicate_ids[id].append(idx)
                    
                    for id, idxs in duplicate_ids.items():
                        corners2d_duplicate = corners2d_valid[idxs]
                        corners2d_other = np.delete(corners2d_valid, idxs, axis=0)
                        # get distance between averaged corners of individual duplicates and average of all other corners
                        dists = np.linalg.norm(np.mean(corners2d_duplicate, axis=1) - np.mean(corners2d_other, axis=(0, 1)), axis=1)
                        
                        # get idx of closest corner
                        closest_idx = idxs[np.argmin(dists)]
                        # Logger.debug(f'"{self.camera_name}": Closest idx: {closest_idx}')
                        idx_to_del.extend([i for i in idxs if i != closest_idx])
                        Logger.debug(f'"{self.camera_name}": Found duplicate id {id} for oid {oid} with indexs: {idxs} with dists: {dists} with closest idx: {closest_idx}')
                    ids_valid = np.delete(ids_valid, idx_to_del)
                    corners2d_valid = np.delete(corners2d_valid, idx_to_del, axis=0)

                markers_valid_oid = {}
                for i, id, corners2d in zip(range(len(ids_valid)), ids_valid, corners2d_valid):
                    corners3d = self.marker_config3d_dict[oid][id]
                    markers_valid_oid[id] = (corners3d, corners2d)

                if self.debug:
                    return_vals, ids_invalid_all = self.pnp_sequence(markers_valid_oid, oid, ids_invalid_all)
                else:
                    return_vals = self.pnp_sequence(markers_valid_oid, oid)

                was_initialized = self.initialized_dict[oid]
                self.initialized_dict[oid], self.rvec_dict[oid], self.tvec_dict[oid], high_repr_err_count, _ = return_vals

                if self.initialized_dict[oid]:
                    # calculate mean velocity by averaging corner velocities of all markers between the current and previous frame
                    vels = []
                    for id, (corners3d, corners2d) in markers_valid_oid.items():
                        if id in self.prev_markers_valid_dict[oid]:
                            corners2d_prev = self.prev_markers_valid_dict[oid][id][1]
                            vels.append(np.mean(corners2d - corners2d_prev, axis=0) * self.FPS)
                    if len(vels) > 0:
                        vel_mean = np.mean(vels, axis=0)
                    else:
                        vel_mean = None
                        if not was_initialized:
                            self.oid_lost_in_prev[oid] = False

                    # calculate T
                    T = np.eye(4)
                    T[:3, :3] = cv2.Rodrigues(self.rvec_dict[oid])[0]
                    T[:3, 3] = self.tvec_dict[oid].squeeze()
                    T_dict[oid] = T

                    # store current markers as previous markers
                    self.prev_markers_valid_dict[oid] = markers_valid_oid
                    self.prev_vel_mean_dict[oid] = vel_mean       

                else:
                    T_dict[oid] = None
                    self.lost_oid(oid, was_initialized)
                    if high_repr_err_count:
                        self.oid_high_repr_err_count[oid] += 1

            self.prev_ids_detec_all = ids_detec_all

            report_interval = 100
            if time.time() - self.timer > report_interval:
                if self.detec_fail_counter > 0:
                    Logger.warning(f'"{self.camera_name}": Crop detection fell back to full frame {self.detec_fail_counter} times in the last {report_interval} seconds')
                    self.detec_fail_counter = 0
                for oid, high_repr_err_count in self.oid_high_repr_err_count.items():
                    if high_repr_err_count > 0:
                        Logger.warning(f'"{self.camera_name}": Reprojection error threshold for "oid {oid}" exceeded {high_repr_err_count} times in the last {report_interval} seconds')
                        self.oid_high_repr_err_count[oid] = 0

                self.timer = time.time()

            # DEBUG STUFF HEREINAFTER
            if self.debug:
                try:
                    disp_frame = frame.copy()
                    # covert to BGR if gray
                    if len(disp_frame.shape) == 2 or disp_frame.shape[2] == 1:
                        disp_frame = cv2.cvtColor(disp_frame, cv2.COLOR_GRAY2BGR)
                    # draw only valid markers
                    if ids_invalid_all:
                        ids_invalid_all_list = np.concatenate([ids_invalid for ids_invalid in ids_invalid_all.values()])
                    else:
                        ids_invalid_all_list = []

                    ids_valid_all = [id for id in ids_all if id not in ids_invalid_all_list]
                    corners2d_valid_all = [corners2d for corners2d, id in zip(corners2d_all, ids_all) if id not in ids_invalid_all_list]
                    for oid in self.oids:
                        if self.initialized_dict[oid]:
                            projected_image_points, _ = cv2.projectPoints(self.marker_config3d_flat_dict[oid], self.rvec_dict[oid], self.tvec_dict[oid], self.cam_mat, self.cam_dist, None)
                            projected_image_points = projected_image_points[:, 0, :]
                            for point in projected_image_points:
                                cv2.circle(disp_frame, tuple(point.astype(np.int32)), 1, (255, 0, 0), -1)

                    disp_frame = visualization.draw_markers(disp_frame, np.array(corners2d_valid_all), np.array(ids_valid_all))[0]
                    disp_frame = visualization.draw_pose(disp_frame, T_dict)

                    # different colored cross for different invalid reasons
                    for id, corners2d in zip(ids_all, corners2d_all):
                        if id in ids_invalid_all.get('reprojection', []):
                            cv2.line(disp_frame, tuple(corners2d[0].astype(np.int32)), tuple(corners2d[2].astype(np.int32)), (255, 255, 0), 2)
                            cv2.line(disp_frame, tuple(corners2d[1].astype(np.int32)), tuple(corners2d[3].astype(np.int32)), (255, 255, 0), 2)

                    shown = 0
                    for oid in self.oids:
                        if self.initialized_dict[oid]:
                            if detect_FF:
                                self.search_area_dict[oid], self.vel_expand[oid] = self.get_search_area(self.marker_config3d_flat_dict[oid], self.rvec_dict[oid], self.tvec_dict[oid], self.prev_vel_mean_dict[oid])
                            shown += 1
                            if shown > 1:
                                try:
                                    cv2.destroyWindow(f'"{self.camera_name}": oid {oid}')
                                except:
                                    pass
                                continue
                            else:
                                x0, x1, y0, y1 = self.search_area_dict[oid]
                                pad = 0
                                crop = disp_frame.copy()[max(y0-pad, 0):min(y1+pad, self.FRAME_HEIGHT), max(x0-pad, 0):min(x1+pad, self.FRAME_WIDTH)]
                                # draw search area which is now (pad, width-pad, pad, height-pad) in the cropped frame
                                cv2.rectangle(crop, (pad, pad), (crop.shape[1]-pad, crop.shape[0]-pad), (0, 255, 0), 2)
                                scale = 500 / max(crop.shape[:2])
                                crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale)
                                cv2.imshow(f'"{self.camera_name}": oid {oid}', crop)
                                self.opened_windows.append(f'"{self.camera_name}": oid {oid}')
                                cv2.waitKey(1)
                        else:
                            try:
                                cv2.destroyWindow(f'"{self.camera_name}": oid {oid}')
                                self.opened_windows.remove(f'"{self.camera_name}": oid {oid}')
                            except:
                                pass


                    # show full frame if any oid is not initialized
                    # if any([not self.initialized_dict[oid] for oid in self.oids]):
                    for oid in self.oids:
                        if self.search_area_dict[oid] is not None:
                            x0, x1, y0, y1 = self.search_area_dict[oid]
                            cv2.rectangle(disp_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    scale = visualization.get_scale(disp_frame.shape)
                    # undistort frame
                    disp_frame = cv2.resize(disp_frame, (0, 0), fx=scale, fy=scale)
                    # write number of detected markers
                    # draw search areas
                    cv2.putText(disp_frame, f'Detected aruco: {len(ids_detec_all)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(disp_frame, f'Detected oid: {len([oid for oid in self.oids if self.initialized_dict[oid]])}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # put legend for crosses
                    fontsize = 0.5
                    cv2.putText(disp_frame, f'#filtered due to high reprojection error: {len(ids_invalid_all.get("reprojection", []))}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 0), 1)
                    cv2.imshow(f'"{self.camera_name}": markers and poses', disp_frame)
                    self.opened_windows.append(f'"{self.camera_name}": markers and poses')
                    cv2.waitKey(1)
                except Exception as e:
                    Logger.debug(f'"{self.camera_name}": Could not show debug window: {e}')
            
            else:
                exception = False
                for window in self.opened_windows:
                    try:
                        cv2.destroyWindow(window)
                    except Exception as e:
                        exception = True
                        
                self.opened_windows = []
                if exception:
                    try:
                        cv2.destroyAllWindows()
                    except Exception as e:
                        pass

            return T_dict, len(ids_detec_all), self.oid_lost_in_prev
        
        else: # when doing calibration of object markers
            corners2d, ids, _ = self.detectMarkers(frame)
            disp_frame = frame.copy()
            if len(disp_frame.shape) == 2 or disp_frame.shape[2] == 1:
                disp_frame = cv2.cvtColor(disp_frame, cv2.COLOR_GRAY2BGR)

            # draw number of times each id was detected
            if self.ids_to_calibrate is not None:
                idx_keep = [i for i, id in enumerate(ids) if id in self.ids_to_calibrate]
                corners2d = [corners2d[i] for i in idx_keep]
                ids = [ids[i] for i in idx_keep]
            self.acquire_marker_calib_imgs(frame, corners2d, ids)

            # draw border around detected markers with thickness depending on number of times detected
            # create colormap that has five colors from red to green for thickness 1 to 5 and above
            # goes from red to orange to yellow to green to dark greens
            colormap = [(0, 0, 128), (0, 0, 255), (0, 128, 255), (0, 255, 255), (128, 255, 128), (0, 255, 0), (0, 200, 0)]
            for id, corners in zip(ids, corners2d):
                thickness = self.calibration_stats.get(id, 0)
                if thickness > 0:
                    color = colormap[min(thickness-1, len(colormap)-1)]
                    cv2.polylines(disp_frame, [corners.astype(np.int32)], isClosed=True, color=color, thickness=thickness)
            for id, corners in zip(ids, corners2d):
                tag = f'{id}: {self.calibration_stats.get(id, 0)}'
                center = tuple(np.mean(corners.reshape(-1, 2), axis=0).astype(np.int32) - [len(tag)*5, -10])
                cv2.putText(disp_frame, tag, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(disp_frame, tag, center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        visualization.show_guide_image(disp_frame, window_name=f'"{self.camera_name}": Calibration', wait=1)

        return {}, 0, {}
    
    def acquire_marker_calib_imgs(self, frame: np.ndarray, corners2d: list[np.ndarray], ids: np.ndarray) -> None:
        if self.calibration_path is None:
            return  
        if ids is None:
            return 
        if len(ids) > 1: # must at least detect 2
            # set self.tracked_ids to the ids of the markers that are currently being tracked and increment the counter
            ids_str = '_'.join([str(id) for id in sorted(ids)])
            self.tracked_ids[ids_str] = self.tracked_ids.get(ids_str, 0) + 1
            # save if tracking for at least FPS frames (due to dropped frames, it will take longer than 2 second to get FPS frames)
            if self.tracked_ids[ids_str] > 2*self.FPS:
                # save max of 10 versions:
                prev_img = None
                max_ver = 2
                for i in range(0, max_ver):
                    corners2d_prev = None
                    diff = None
                    filename = f'frame_{ids_str}_ver{i}.png'
                    filename = os.path.join(self.calibration_path, filename)
                    filename_next = filename.replace(f'ver{i}', f'ver{i+1}')
                    # create folder if it does not exist
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    if prev_img is not None:
                        corners2d_prev, ids_prev, _ = self.detectMarkers(prev_img)
                        if not ids_prev:
                            Logger.warning(f'"{self.camera_name}": Could not detect markers in saved calibration frame, skipping')
                            break
                        if self.ids_to_calibrate is not None:
                            idx_keep = [i for i, id in enumerate(ids_prev) if id in self.ids_to_calibrate]
                            corners2d_prev = [corners2d_prev[i] for i in idx_keep]
                            ids_prev = [ids_prev[i] for i in idx_keep]
                        # sort by id and subtract the first point from all other points
                        corners2d_prev = [corners2d_prev[i] for i in np.argsort(ids_prev)]
                        corner0_prev = corners2d_prev[0][0, 0]
                        corners2d_prev = [np.array([c-corner0_prev for c in cs]) for cs in corners2d_prev]
                        corners2d_copy = [corners2d[i] for i in np.argsort(ids)]
                        corner0 = corners2d_copy[0][0, 0]
                        corners2d_copy = [np.array([c-corner0 for c in cs]) for cs in corners2d_copy]
                        # calculate the difference between the current and previous frame
                        diff = np.mean([np.mean(np.abs(corners2d_copy[i] - corners2d_prev[i])) for i in range(len(corners2d_copy))])
                    if not os.path.isfile(filename):
                        if corners2d_prev is None or diff > 10:
                            cv2.imwrite(filename, frame)
                            self.calibration_stats = get_calibration_stats(self.calibration_path)
                            Logger.info(f'"{self.camera_name}": Saved frame to {filename} with difference {diff:.2f}') if diff is not None else Logger.info(f'"{self.camera_name}": Saved frame to {filename}')
                            break
                        self.tracked_ids[ids_str] = 0 # reset counter
                    elif not os.path.isfile(filename_next):
                        prev_img = cv2.imread(filename)

    def close(self) -> None:
        return

    def set_debug(self, debug_new: bool) -> None:
        # If locked, store the intended debug value in _pending_debug
        if self._debug_lock:
            self._pending_debug = debug_new
        else:
            self.debug = debug_new

