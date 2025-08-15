import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import numpy as np
from deps.utils.logger_util import Logger
import json
import ctypes


import os

class CameraIntrinsics:
    _loaded_intrinsics = None

    def __init__(self, calib_file_path: str) -> None:
        if os.path.isfile(calib_file_path):
            # chck if it is a .npz file
            if calib_file_path.endswith('.npz'):
                with np.load(calib_file_path) as X:
                    CameraIntrinsics._loaded_intrinsics = tuple(X[i] for i in ('mtx', 'dist'))
            # check if it is a .json file
            elif calib_file_path.endswith('.json'):
                with open(calib_file_path, 'r') as f:
                    data = json.load(f)
                    mtx = np.array(data['mtx'])
                    dist = np.array(data['dist'])
                    CameraIntrinsics._loaded_intrinsics = (mtx, dist)
            else:   
                Logger.error(f'Calibration file "{calib_file_path}" is not a .npz or .json file.')

        else:
            Logger.error(f'Calibration file "{calib_file_path}" not found.')

    @classmethod
    def get_intrinsics(cls) -> Tuple[np.ndarray, np.ndarray]:
        if cls._loaded_intrinsics is None:
            raise ValueError("Camera intrinsics not loaded. Call the constructor first.")
        return cls._loaded_intrinsics


def get_screen_resolution():
    user32 = ctypes.windll.user32
    return max(user32.GetSystemMetrics(0), 1920), max(user32.GetSystemMetrics(1), 1080)

global_screen_res = get_screen_resolution()

def get_scale(img_shape, screen_res=None, pads=None):
    if screen_res is None:
        screen_res = global_screen_res
    if pads is None:
        # 20% of corresponding screen dimension
        pad_x, pad_y = screen_res[0]//5, screen_res[1]//5
    else:
        pad_x, pad_y = pads
    return min(min(1, (screen_res[0]-pad_x) / img_shape[1]), min(1, (screen_res[1]-pad_y) / img_shape[0]))

def draw_markers(disp_frame: cv2.UMat, corners: list, ids: list, rejectedImgPoints: Optional[list] = None, draw_rejected: bool = False,
                 plot: bool = False) -> Tuple[cv2.UMat, cv2.UMat]:
    """Draws the markers in the frame."""
    if ids is None or len(ids) == 0:
        return disp_frame, None
    else:
        # assume they come as nxn_cornersx2 and draw using cv2.polylines
        draw_frame = disp_frame.copy()
        for i, corner in enumerate(corners):
            # shape is (4,2) or (5,2) for 4 or 5 corners
            # draw the corners
            cv2.polylines(draw_frame, [corner.astype(int)], True, (0, 255, 0), 1)
            # draw the id of the marker
            cv2.putText(draw_frame, str(ids[i]), tuple(corner[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if draw_rejected:
            assert rejectedImgPoints is not None, 'rejectedImgPoints must be provided if draw_rejected is True'
            rejected_frame = cv2.aruco.drawDetectedMarkers(disp_frame, rejectedImgPoints, borderColor=(0, 0, 255))
        else:
            rejected_frame = None
        if plot:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.imshow(draw_frame)
            ax.set_title(f'Found total of {len(corners)} markers')
            if rejected_frame is not None:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.imshow(rejected_frame)
                ax.set_title(f'Found total of {len(rejectedImgPoints)} rejected points')
            plt.show()
        return draw_frame, rejected_frame


def draw_pose(draw_frame: cv2.UMat, Ts: dict[str, Optional[np.ndarray]]) -> cv2.UMat:
    """Draws the poses of the markers in the frame."""
    names = np.array(list(Ts.keys())).astype(str)
    r_vecs, t_vecs = [], []
    for T in Ts.values():
        if T is None:
            r_vecs.append(None)
            t_vecs.append(None)
        else:
            r_vecs.append(cv2.Rodrigues(T[0:3, 0:3])[0])
            t_vecs.append(T[0:3, 3].reshape(3, 1))
    # draw the axes of the markers
    cam_mtx, cam_dist = CameraIntrinsics.get_intrinsics()
    # draw axes
    for r_vec, t_vec, name in zip(r_vecs, t_vecs, names):
        if r_vec is None or t_vec is None:
            continue
        # axes colors: x: red, y: green, z: blue
        draw_frame = cv2.drawFrameAxes(draw_frame, cam_mtx, cam_dist, r_vec, t_vec, 0.05)
        # project t_vec to image plane
        t_vec_proj, _ = cv2.projectPoints(t_vec, np.zeros((3, 1)), np.zeros((3, 1)), cam_mtx, cam_dist)
        t_vec_proj = t_vec_proj.squeeze()
        # draw text
        x, y = int(t_vec_proj[0]), int(t_vec_proj[1])
        if (x is not None and y is not None) and (x > 0 and y > 0):
            try:
                cv2.putText(draw_frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # draw red cross
                # write coordinates in 3D and angles
                # write ({np.linalg.norm(t_vec)*1000:.1f} mm)
                cv2.putText(draw_frame, f'Distance {np.linalg.norm(t_vec)*1000:.1f} mm',
                            (x-100, y+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 255), 2)
                # pos: [x | y | z] mm
                cv2.putText(draw_frame, 
                            f'[{t_vec[0][0]*1000:.1f} | {t_vec[1][0]*1000:.1f} | {t_vec[2][0]*1000:.1f}] mm',
                            # center
                            (x-100, y+55),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 255), 2)
                # angles: [roll | pitch | yaw] deg
                angles = np.rad2deg(cv2.Rodrigues(r_vec)[0].T[0]).flatten()
                cv2.putText(draw_frame, 
                            f'[{angles[0]:.1f} | {angles[1]:.1f} | {angles[2]:.1f}] deg',
                            # center
                            (x-100, y+75),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 255), 2)
            except:
                Logger.warning(f'Could not draw pose for {name}: name: {name}, x: {x}, y: {y}, t_vec: {t_vec}')
                
    return draw_frame


def get_word_length(text: str, fontsize: int = 1, constant: int = 18) -> int:
    return len(text)*constant*fontsize

def show_guide_image(img: np.ndarray, text: str = None, color: Tuple[int, int, int] = (255, 0, 255), window_name: str = 'frame', wait: int = 0, opague_box: bool = False, show: bool = True, fontsize: int = 1, thickness: int = 2) -> None:
    show_img = img.copy()
    # convert to BGR if gray
    if len(show_img.shape) == 2 or show_img.shape[2] == 1:
        show_img = cv2.cvtColor(show_img, cv2.COLOR_GRAY2BGR)
    scale = get_scale(show_img.shape, global_screen_res)
    show_img = cv2.resize(show_img, (0, 0), fx=scale, fy=scale)
    if text is not None:
        # replace \t with spaces
        text = text.replace('\t', '    ')
        lines = text.split('\n')
        # get max line length
        max_len = max([len(line) for line in lines])
        # if max_len exceets width of image, break lines into smaller parts
        fontsize = fontsize if len(lines) == 1 else 0.6
        thickness = thickness if len(lines) == 1 else 1
        if get_word_length(text, fontsize=fontsize) > show_img.shape[1]:
            new_lines = []
            for line in lines:
                # split at whitespace that occurs before the max_len    
                split = line.split(' ')
                new_line = ''
                for i, word in enumerate(split):
                    if get_word_length(new_line + word, fontsize=fontsize) < show_img.shape[1]:
                        new_line += word + ' '
                    else:
                        new_lines.append(new_line)
                        if get_word_length(word, fontsize=fontsize) < show_img.shape[1]:
                            new_line = word + ' '
                        else:
                            # break word in worst case
                            new_line = ''
                            for j, char in enumerate(word):
                                if get_word_length(new_line + char, fontsize=fontsize) < show_img.shape[1]:
                                    new_line += char
                                else:
                                    new_lines.append(new_line)
                                    new_line = char
                new_lines.append(new_line)
            lines = new_lines
            max_len = max([len(line) for line in lines])

        # draw box
        if opague_box:
            # x, y, w, h = 0, 0, max_len*18, len(lines)*29+15
            # x, y, w, h = 0, 0, max_len*18*size, len(lines)*29*size+15*size
            x, y, w, h = 0, 0, int(get_word_length(text, fontsize=fontsize)), int(len(lines)*29*fontsize+15*fontsize)

            sub_img = show_img[y:y+h, x:x+w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            show_img[y:y+h, x:x+w] = res
            
        for i, line in enumerate(lines):
            # cv2.putText(show_img, line, (10, 30 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
            # cv2.putText(show_img, line, (10, 30*size + 30*size*i), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
            cv2.putText(show_img, line, (10, int(30*fontsize + 30*fontsize*i)), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness, cv2.LINE_AA)
    if show:
        cv2.imshow(window_name, show_img)
        if wait > 0:
            cv2.waitKey(wait)
        elif wait < 0:
            # wait until key is pressed
            cv2.waitKey(0)

    return show_img


def try_to_destroy(window_name: str, condition: bool = False) -> None:
    try:
        if condition:
            if cv2.waitKey(0) & 0xFF == 27:
                cv2.destroyWindow(window_name)
        else:
            cv2.destroyWindow(window_name)
    except:
        pass

