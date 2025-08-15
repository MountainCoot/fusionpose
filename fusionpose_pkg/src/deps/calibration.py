import os
from typing import Optional
import numpy as np
from deps.utils.logger_util import Logger
import json

if __name__ == '__main__':
    # add one up to the root folder
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(base_dir) if base_dir not in sys.path else None
    from deps.utils.logger_util import reset_logging
    Logger = reset_logging(name='calibrator')


def print_unused_params(params: dict) -> None:
    if len(params) > 0:
        Logger.warning(f'Provided {len(params)} unused params:')
        for param, value in params.items():
            Logger.warning(f'\t{param}: {value}')

class CameraCalibration():
    def __init__(self, debug) -> None:
        self.calib_file_path: Optional[str] = None
        self.is_calibrated = False
        self.cam_dist: Optional[np.ndarray] = None
        self.cam_mtx: Optional[np.ndarray] = None
        self.debug = debug

    def load_calibration(self, calib_file_path: Optional[str] = None) -> bool:
        self.calib_file_path = calib_file_path
        if self.calib_file_path is not None:
            if not os.path.isfile(self.calib_file_path):
                Logger.warning(f'Camera calibration file "{self.calib_file_path}" is not a file.')
                self.calib_file_path = ''
                return False
            else:
                Logger.info(f'Loading camera intrinsics from "{self.calib_file_path}"')
                if self.calib_file_path.endswith('.npz'):
                    with np.load(self.calib_file_path) as X:
                        self.cam_mtx = X['mtx']
                        self.cam_dist = X['dist']
                        self.is_calibrated = True
                elif self.calib_file_path.endswith('.json'):
                    with open(self.calib_file_path, 'r') as f:
                        calib_data = json.load(f)
                        self.cam_mtx = np.array(calib_data['mtx'])
                        self.cam_dist = np.array(calib_data['dist'])
                        self.is_calibrated = True
                else:
                    Logger.error(f'Camera calibration file "{self.calib_file_path}" is not a .npz or .json file.')
                    self.calib_file_path = ''
                    return False   
                             
                if self.debug:
                    Logger.debug(f'\nCamera Matrix:\n{self.cam_mtx}')
                    Logger.debug(f'\nDistortion Coefficients:\n{self.cam_dist}')
                    Logger.info('Camera is calibrated.')
                return True
        else:
            Logger.warning('No calibration file found, not calibrated!')
            return False

    def get_path_to_calib_file(self) -> Optional[str]:
        if self.calib_file_path is None:
            Logger.error('Camera is uncalibrated, cannot get calibration location.')
            return ''
        # for windows paths, replace \ with /
        return self.calib_file_path.replace('\\', '/')
    