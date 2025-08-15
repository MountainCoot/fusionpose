import os
import cv2
from typing import Any, Optional, Generator

try:
    from deps.utils.logger_util import Logger
except ModuleNotFoundError:
    import sys
    # append one up and then src/
    to_append = [f'{os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))}']
    for p in to_append:
        sys.path.append(p) if p not in sys.path else None

    from deps.utils.logger_util import Logger

import rospy

import json
from threading import Thread

from deps.calibration import CameraCalibration
from deps.acq_core import OpenCVConnectionCore, opencv_variable_settings
from deps.acq_core import NeoapiConnectionCore, neoapi_variable_settings
from deps.acq_core import Frame
from deps.utils.visualization import show_guide_image, try_to_destroy

import time

class AcquisitionHandler:
    def __init__(self, debug: bool = False) -> None:
        self.n_frames = -1
        self.debug = debug
        self.calibrator = CameraCalibration(debug=debug)

    def get_n_frames(self) -> int:
        return self.n_frames
    
    def get_frame(self) -> Any:
        raise NotImplementedError
    
    def stop(self) -> None:
        raise NotImplementedError
    
    def get_path_to_calib_file(self) -> str:
        return self.calibrator.get_path_to_calib_file()
    
    def interrupt(self) -> None:
        pass


class CameraHandler(AcquisitionHandler):
    def __init__(self, backend: Optional[str] = 'opencv', camera_name: Optional[str] = 'camera', calib_file_path: Optional[str] = None, settings_path: Optional[str] = None, **params) -> None:
        self.init_settings: Optional[dict] = None
        self.current_settings: Optional[dict] = None
        self.camera_name: Optional[str] = None
        super().__init__(params.pop('debug', False))

        self.backend = backend
        if self.backend == 'neoapi':
            # pop params that are not needed for neoapi
            params.pop('api', None)
            self.core = NeoapiConnectionCore(
                cam_idx=params.pop('cam_idx', 0),
                cam_res=params.pop('cam_res', (1920, 1080)), 
                fps=params.pop('fps', 30), 
                camera_name=camera_name, 
                debug=self.debug, identifier=params.pop('identifier', None))
            self.variable_settings_dict = neoapi_variable_settings
        elif self.backend == 'opencv':
            params.pop('identifier', None)
            self.core = OpenCVConnectionCore(
                cam_idx=params.pop('cam_idx', 0),
                cam_res=params.pop('cam_res', (1920, 1080)), 
                fps=params.pop('fps', 30), 
                api=params.pop('api', cv2.CAP_DSHOW),
                camera_name=camera_name, 
                debug=self.debug)
            self.variable_settings_dict = opencv_variable_settings    
        else:
            raise ValueError(f'Backend "{self.backend}" not supported. Use either "opencv" or "neoapi"')
            
        self.camera_name = self.core.camera_name
        self._interrupt_daq = False
        self._monitor_active = False
        self._daq_active = False
        if self.core.start(): # meaning that camera is opened
            self.cam_idx = self.core.cam_idx
            # save initial settings
            self.init_settings = self.core.read_settings(self.variable_settings_dict)
            self.current_settings = self.init_settings
            if settings_path is not None:
                self.load_settings(settings_path)    
            # calibrate
            if self.calibrator.load_calibration(calib_file_path=calib_file_path):
                Logger.info(f'Camera setup for camera "{self.camera_name}" with index {self.cam_idx} is complete.')

    def _daq(self) -> None:
        start_time = time.time()
        self._daq_active = True
        for _ in self.core._grab_next_frame():
            # report frequency every 100 frames
            if self.core.curr_frame_cls.idx % 100 == 0:
                Logger.info(f'Camera "{self.camera_name}" with index {self.cam_idx} frequency: {100/(time.time() - start_time):.2f} Hz')
                start_time = time.time()
            if self._interrupt_daq:
                break
        self._daq_active = False

    def run_daq(self) -> None:
        time.sleep(0.5)
        self._interrupt_daq = False
        daq_thread = Thread(target=self._daq, daemon=True)
        daq_thread.start()

    def set_settings(self, settings: dict) -> bool:
        if not self.is_connected():
            Logger.warning('Camera not opened. Cannot set settings.')
            return False
        # then set the new settings
        if self.debug:
            Logger.info('Setting new settings:')
        for setting_name, value in settings.items():
            self.current_settings[setting_name] = value
            self.core.set(setting_name, value, self.variable_settings_dict)
            if self.debug:
                Logger.info(f'\t{setting_name} to {value}')
        # save settings to temp
        self.save_settings(f'files/{self.camera_name}_settings_last.json', print_msg=False)
        return True
    
    def save_settings(self, path: str = 'camera_settings.json', print_msg: bool = True) -> None:
        settings_to_save = self.current_settings
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
        if print_msg:
            Logger.info(f'Saved camera settings to {path}')

    def load_settings(self, path: str) -> None:
        if not os.path.exists(path):
            Logger.warning(f'Camera settings file "{path}" does not exist. Cannot load settings.')
            return
        with open(path, 'r') as f:
            settings = json.load(f)
        # replace keys with int keys from settings_mapper
        if self.set_settings(settings):
            Logger.info(f'Loaded camera settings from "{path}"')

    def get_frame(self) -> Optional[Frame]:
        return self.core.curr_frame_cls

    def track_bar_init(self, window_name: str) -> None:
        for setting_name, (_, settings) in self.variable_settings_dict.items():
            # if settings dict is empty, not meant to be displayed
            if not settings:
                continue
            is_manual_setting = settings['manual']
            if self.backend == 'neoapi':
                is_autoexposure = not self.current_settings.get('autoexposure', True)
            elif self.backend == 'opencv':
                is_autoexposure = self.current_settings.get('autoexposure', False)
            else:
                raise ValueError(f'Backend "{self.backend}" not supported. Use either "opencv" or "neoapi"')
            if is_autoexposure and is_manual_setting: # dont create trackbar for settings that are not auto and camera is in auto mode
                continue
            min, max = settings['min'], settings['max']
            scale = settings.get('scale', 1)
            cv2.createTrackbar(setting_name, window_name, int(self.current_settings[setting_name]*scale), int((max-min)*scale), lambda x, setting_name=setting_name, scale=scale: self.set_settings({setting_name: x/scale}))
            cv2.setTrackbarMin(setting_name, window_name, int(min*scale))

    def set_track_bars(self, window_name: str) -> None:
        for setting_name, (_, settings) in self.variable_settings_dict.items():
            # if settings dict is empty, not meant to be displayed
            if not settings:
                continue
            scale = settings.get('scale', 1)

            cv2.setTrackbarPos(setting_name, window_name, int(self.current_settings[setting_name]*scale))

    def _monitor(self, start_daq: bool = False) -> None:
        # sleep for 1 second to allow triggers to reset
        if start_daq:
            self.run_daq()
        self._monitor_active = True
        window_name = f'Monitoring camera "{self.camera_name}" with index {self.cam_idx}'
        # define key bindings
        key_bindings = {
            'autofocus': 'a',
            'autoexposure': 'e',
            'autowb': 'w',
            'save_settings': 's',
            'load_settings': 'l',
            'quit': 'q',
            'left': 2424832,
            'up': 2490368,
            'right': 2555904,
            'down': 2621440,
            'reset_crop': 'r',
        }
        last_frame = None
        # create named window
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 0, 0) 


        y_shift_crop, x_shift_crop = 0, 0
        crop_step = 150
        closed_in_last = True
        while True:
            current_frame_cls = self.get_frame()
            # show and continue
            if current_frame_cls.frame is not None and current_frame_cls.frame is not last_frame:
                current_frame_disp = current_frame_cls.frame.copy()
                # make bgr if it is grayscale
                if len(current_frame_disp.shape) == 2 or current_frame_disp.shape[2] == 1:
                    current_frame_disp = cv2.cvtColor(current_frame_disp, cv2.COLOR_GRAY2BGR)
                last_frame = current_frame_cls.frame
                frame_height, frame_width = current_frame_disp.shape[:2]

                if closed_in_last:
                    self.track_bar_init(window_name)

                # crop inner 0.4:0.6 of the frame and display next to the frame
                length_crop = int(frame_width * 0.2)
                # get a square in the middle of the frame of side length 0.2 of the frame width
                xm, ym = frame_width//2, frame_height//2
                x0_crop, y0_crop = max(0, xm - length_crop//2 + x_shift_crop), max(0, ym - length_crop//2 + y_shift_crop)
                x1_crop, y1_crop = min(frame_width, xm + length_crop//2 + x_shift_crop), min(frame_height, ym + length_crop//2 + y_shift_crop)

                cutout = current_frame_disp[y0_crop:y1_crop, x0_crop:x1_crop]
                scale = frame_height / cutout.shape[0]
                cutout = cv2.resize(cutout, (0, 0), fx=scale, fy=scale)                
                # red border to the left of the cutout
                cutout = cv2.copyMakeBorder(cutout, 0, 0, 5, 0, cv2.BORDER_CONSTANT, value=(0, 0, 255))
                # create green border where cutout is in the frame
                current_frame_disp = cv2.rectangle(current_frame_disp, (x0_crop, y0_crop), (x1_crop, y1_crop), (0, 255, 0), 2)
                current_frame_disp = cv2.hconcat([current_frame_disp, cutout])                


                key = cv2.waitKeyEx(1)
                if key == ord(key_bindings['quit']) or self._interrupt_daq or not self._monitor_active:
                    try_to_destroy(window_name)
                    break
                
                if key == ord(key_bindings['autoexposure']):
                    if self.backend == 'neoapi':
                        self.set_settings({'autoexposure': 1 if self.current_settings['autoexposure'] == 0 else 0})
                    elif self.backend == 'opencv':
                        self.set_settings({'autoexposure': 1- self.current_settings['autoexposure']})
                    else:
                        raise ValueError(f'Backend "{self.backend}" not supported. Use either "opencv" or "neoapi"')
                    try_to_destroy(window_name)
                elif key == ord(key_bindings['save_settings']):
                    self.save_settings(f'files/{self.camera_name}_settings_monitor.json')
                    show_guide_image(current_frame_disp, f'Saved current settings to "files/{self.camera_name}_settings_monitor.json"', color=(255, 0, 255), window_name=window_name, wait=1000)
                elif key == ord(key_bindings['load_settings']):
                    self.load_settings(f'files/{self.camera_name}_settings_monitor.json')
                    show_guide_image(current_frame_disp, f'Loaded settings from "files/{self.camera_name}_settings_monitor.json"', color=(255, 0, 255), window_name=window_name, wait=1000)

                if key == key_bindings['left']:
                    x_shift_crop -= crop_step
                elif key == key_bindings['right']:
                    x_shift_crop += crop_step
                elif key == key_bindings['up']:
                    y_shift_crop -= crop_step
                elif key == key_bindings['down']:
                    y_shift_crop += crop_step
                elif key == ord(key_bindings['reset_crop']):
                    x_shift_crop, y_shift_crop = 0, 0

                # limit x_shift_crop and y_shift_crop to be within the frame
                x_shift_crop = max(-frame_width//2 + length_crop//2, min(frame_width//2 - length_crop//2, x_shift_crop))
                y_shift_crop = max(-frame_height//2 + length_crop//2, min(frame_height//2 - length_crop//2, y_shift_crop))

                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        closed_in_last = True
                    else:
                        closed_in_last = False
                        self.set_track_bars(window_name)
                                
                except:
                    closed_in_last = True

                disp_str = ''   
                disp_str += f'Change settings of camera "{self.camera_name}" with index {self.cam_idx}:\n'
                disp_str += '--------------------------\n'
                if self.backend == 'neoapi':
                    disp_str += f'Autoexposure: {"on" if self.current_settings["autoexposure"] == 0 else "off"} (press \'e\' to change)\n'
                elif self.backend == 'opencv':
                    disp_str += f'Autoexposure: {"on" if self.current_settings["autoexposure"] == 1 else "off"} (press \'e\' to change)\n'
                else:
                    raise ValueError(f'Backend "{self.backend}" not supported. Use either "opencv" or "neoapi"')

                disp_str += '--------------------------\n'
                disp_str += f'Use arrow keys to move crop window, press \'r\' to reset crop window\n'
                disp_str += f'Press \'s\' to save current settings to "files/{self.camera_name}_settings_monitor.json"\n'
                disp_str += f'Press \'l\' to load settings from "files/{self.camera_name}_settings_monitor.json"\n'
                disp_str += f'Press \'q\' to stop monitoring camera "{self.camera_name}" at index {self.cam_idx}\n'
                
                current_frame_disp = cv2.putText(current_frame_disp, f'Frame {current_frame_cls.idx}', (frame_width - len(f'Frame {current_frame_cls.idx}')*20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                current_frame_disp = show_guide_image(current_frame_disp, disp_str, window_name=window_name, color=(0, 0, 0), wait=1, opague_box=True, show=False)
                
                                
                cv2.imshow(window_name, current_frame_disp)
                cv2.waitKey(1)

            if self._interrupt_daq:
                break

        self._monitor_active = False
        if start_daq:
            self.interrupt()

    def grab_next_frame(self) -> Generator[Frame, None, None]:
        if self.calibrator.is_calibrated:
            for frame in self.core._grab_next_frame():
                if self._interrupt_daq:
                    break
                yield frame

    def monitor(self, start_daq: bool = False, start: bool = True) -> None:
        if start:
            time.sleep(0.5)
            if self._monitor_active:
                Logger.warning(f'Monitoring for camera "{self.camera_name}" with index {self.cam_idx} is already active.')
                return
            monitor_thread = Thread(target=self._monitor, args=(start_daq,), daemon=True)
            monitor_thread.start()
        else:
            self._monitor_active = False

    def interrupt(self) -> None:
        self._interrupt_daq = True

    def is_connected(self) -> bool:
        if self.core.cam_class is not None:
            return self.core.is_connected()
        else:
            return False
    
    def is_calibrated(self) -> bool:
        return self.calibrator.is_calibrated

    def stop(self) -> None:
        self.core.stop()
        


def main():
    camera_handler = CameraHandler(
        camera_name='test',
        settings_path=None,
        cam_res=(2464, 2048),
        fps=40,
        cam_idx=0,
        calib_file_path=r'calibration.json',
        debug=True,
    )   

    for frame in camera_handler.grab_next_frame():
        if camera_handler._interrupt_daq:
            break
        cv2.imshow('frame', frame.frame)
        cv2.waitKey(1)
        if rospy.is_shutdown():
            break


if __name__ == '__main__':
    rospy.init_node('acquisition_node', anonymous=True)
    main()