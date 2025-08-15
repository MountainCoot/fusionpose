import cv2
try:
    import neoapi
except ImportError:
    print('Neoapi not available')

from typing import Any, Tuple, Optional, Generator
import numpy as np
import time
try:
    from deps.utils.logger_util import Logger
except ImportError:
    class Logger:
        @staticmethod
        def info(msg: str) -> None:
            print(msg)
        @staticmethod
        def warning(msg: str) -> None:
            print(msg)
        @staticmethod
        def error(msg: str) -> None:
            print(msg)
        @staticmethod
        def debug(msg: str) -> None:
            print(msg)


opencv_variable_settings = {
    'brightness': (cv2.CAP_PROP_BRIGHTNESS, {'manual': True, 'min': 0, 'max': 100}),
    'contrast': (cv2.CAP_PROP_CONTRAST, {'manual': True, 'min': 0, 'max': 100}),
    'saturation': (cv2.CAP_PROP_SATURATION, {'manual': True, 'min': 0, 'max': 100}),
    'hue': (cv2.CAP_PROP_HUE, {'manual': True, 'min': 0, 'max': 100}),
    'exposure': (cv2.CAP_PROP_EXPOSURE, {'manual': True, 'min': -16, 'max': 1}),
    'gain': (cv2.CAP_PROP_GAIN, {'manual': True, 'min': 0, 'max': 3000}),
    'autoexposure': (cv2.CAP_PROP_AUTO_EXPOSURE, {}),
}

opencv_fixed_settings = {
    'frame_width': cv2.CAP_PROP_FRAME_WIDTH,
    'frame_height': cv2.CAP_PROP_FRAME_HEIGHT,
    'fps': cv2.CAP_PROP_FPS,
}

neoapi_fixed_settings = {
    'fps_enable': 'AcquisitionFrameRateEnable',
    'fps': 'AcquisitionFrameRate',
    'binning_horizontal': 'BinningHorizontal',
    'binning_vertical': 'BinningVertical',
    'pixel_format': 'PixelFormat',
}

neoapi_variable_settings = {
    'autoexposure': ('ExposureAuto', {}),
    'exposure': ('ExposureTime', {'manual': True, 'min': 1, 'max': 1000*50, 'scale': 1/100}), # in us
    'gain': ('Gain', {'manual': True, 'min': 1, 'max': 50}), 
    'gamma_enable': ('LUTEnable', {}),
    'gamma': ('Gamma', {'manual': True, 'min': 0.1, 'max': 2, 'scale': 10}),
}

def parse_setting(setting: Any, settings_mapper: dict) -> Any:
    try:
        setting_parsed = settings_mapper[setting]
        if isinstance(setting_parsed, tuple): # for variable settings, only get the first element
            setting_parsed = setting_parsed[0]
    except KeyError:
        Logger.warning(f'{setting} is not a valid setting. Possible settings are: {", ".join(settings_mapper.keys())}')
        setting_parsed = None
    return setting_parsed

class Frame():
    def __init__(self, frame: np.ndarray, frame_idx: int, timestamp_ms: float) -> None:
        self.frame = frame
        self.idx = frame_idx
        self.timestamp_s = None
        if timestamp_ms is not None:
            self.timestamp_s = timestamp_ms / 1000

    @property
    def timestamp_ms(self) -> float:
        if self.timestamp_s is not None:
            return self.timestamp_s * 1000
        else:
            return None

    def as_tuple(self) -> Tuple[np.ndarray, int, float]:
        return self.frame, self.idx, self.timestamp_s


class GenericConnectionCore():
    def __init__(self, backend: str, cam_idx: int, cam_res: Tuple[int, int], fps: int, camera_name: str, debug: bool = False, identifier: Optional[str] = None) -> None:
        assert backend in ['opencv', 'neoapi'], f'Backend {backend} is not valid. Use "opencv", "neoapi" instead.'
        self.backend = backend
        self.cam_idx = cam_idx
        self.cam_class = None
        self.cam_res = cam_res
        self.cam_fps = fps
        self.camera_name = camera_name
        self.curr_frame_cls = Frame(None, -1, None)
        self.equal_count = 0
        self.debug = debug
        self.settings_mapper_dict = None
        self.identifier = identifier
        if identifier is not None:
            self.cam_idx = identifier

    def start(self) -> bool:
        try:
            # first check if camera is already opened
            if self.cam_class is not None:
                if not self._is_valid_connection():
                    Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx} was disconnected. Retrying...')
                    self.stop()
                    return self.start()
                else:
                    if self.debug:
                        Logger.info(f'Camera "{self.camera_name}" with index {self.cam_idx} is already opened, not adjusting frame rate or resolution.')
                    return True
            else:
                self._connect()

                if not self._is_valid_connection():
                    Logger.error(f'Could not open camera "{self.camera_name}" with index {self.cam_idx} at resolution {self.cam_res} and fps {self.cam_fps}')
                    self.stop()
                    return False
                else:
                    if self.debug:
                        Logger.info(f'Opened camera "{self.camera_name}" with index {self.cam_idx} at resolution {self.cam_res} and fps {self.cam_fps}')
                    try:
                        if next(self._grab_next_frame()).frame.shape[:2][::-1] != self.cam_res:
                            Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx}: Actual resolution {next(self._grab_next_frame()).frame.shape[:2][::-1]} is not as desired ({self.cam_res}).')
                        return True
                    except StopIteration:
                        self.stop()
                        return False
        except Exception as e:
            Logger.error(f'Error opening camera "{self.camera_name}" with index {self.cam_idx}: {e}')
            self.stop()
            return False
                
    def stop(self) -> None:
        if self.cam_class is not None:
            self._disconnect()
            self.cam_class = None
            Logger.info(f'Closed camera "{self.camera_name}" with index {self.cam_idx}')

    def read_settings(self, dict: Optional[dict] = None) -> dict:
        if dict is None:
            dict = self.settings_mapper_dict
        settings = {}
        for setting_name in dict.keys():
            settings[setting_name] = self.get(setting_name)
        return settings

    def _is_valid_connection(self) -> bool:
        assert self.cam_class is not None, 'ConnectionCore.cap is None'
        if self.is_connected():
            # get first 10 frames
            frames = []
            for i, frame_cls in enumerate(self._grab_next_frame()):
                frames.append(frame_cls.frame)
                if i == 5:
                    break
            else:
                if self.debug:
                    Logger.debug(f'Could not get 10 frames from camera "{self.camera_name}" with index {self.cam_idx}.')
                return False
            # return false if all frames are the same or black
            if all([np.array_equal(frame, frames[0]) for frame in frames]):
                if self.debug:
                    Logger.debug(f'Camera "{self.camera_name}" with index {self.cam_idx} is not opened. All frames are the same.')
                return False
            elif all([np.mean(frame) == 0 for frame in frames]):
                if self.debug:
                    Logger.debug(f'Camera "{self.camera_name}" with index {self.cam_idx} is not opened. All frames are black.')
                return False
            return True
        else:
            if self.debug:
                Logger.debug(f'Camera "{self.camera_name}" with index {self.cam_idx} is not opened.')
            return False

class OpenCVConnectionCore(GenericConnectionCore):
    def __init__(self, api: str = 'MSMF', **kwargs) -> None:
        super().__init__(backend='opencv', **kwargs)
        apis = {'DSHOW': cv2.CAP_DSHOW, 'MSMF': cv2.CAP_MSMF}
        if api not in apis:
            Logger.warning(f'API {api} is not valid. Using DSHOW instead.')
            self.apiPreference = apis['DSHOW']
        else:
            self.apiPreference = apis[api]
        self.settings_mapper_dict = opencv_fixed_settings | opencv_variable_settings

    def _connect(self, **kwargs) -> None:
        info_str = f'Opening OpenCV camera "{self.camera_name}" with index {self.cam_idx} with resolution {self.cam_res} and fps {self.cam_fps} with API {self.apiPreference}'
        Logger.info(info_str)
        self.cam_class = cv2.VideoCapture(kwargs.get('cam_idx', 0), apiPreference=kwargs.get('api', cv2.CAP_MSMF))
        self.set('frame_width', self.cam_res[0])
        self.set('frame_height', self.cam_res[1])
        self.set('fps', self.cam_fps)

    def _disconnect(self) -> None:
        self.cam_class.release()

    def is_connected(self) -> bool:
        return self.cam_class.isOpened()

    def set(self, setting: str, value: Any, dict: Optional[dict] = None) -> None:
        if dict is None:
            dict = self.settings_mapper_dict
        setting = parse_setting(setting, dict)
        if setting is not None:
            # print(f'Setting {setting} to {value}')
            self.cam_class.set(setting, value)

    def get(self, setting: str) -> Any:
        return self.cam_class.get(parse_setting(setting, self.settings_mapper_dict))

    def _grab_next_frame(self) -> Generator[Frame, None, None]:
        t_cam_ms_init = None
        t_cam_ms_prev = None
        t_timer = time.time()
        t_interval = 100 # seconds
        n_dropped_tot = 0

        while(self.cam_class is not None and self.cam_class.isOpened()):
            try:
                ret = self.cam_class.grab()
                if ret:
                    t_frame_ms = time.time()*1000
                    # check if dt matches desired fps, if not continue
                    if self.apiPreference == cv2.CAP_MSMF:
                        t_cam_ms = self.cam_class.get(cv2.CAP_PROP_POS_MSEC)
                        if t_cam_ms_init is None:
                            t_cam_ms_init = t_cam_ms
                        if t_cam_ms_prev is not None:
                            dt = t_cam_ms - t_cam_ms_prev
                            dt_desired = 1000 / self.cam_fps
                            if dt*1.01 < dt_desired:
                                continue # discard this frame
                        t_cam_ms_prev = t_cam_ms
                    else:
                        if self.curr_frame_cls.timestamp_ms is not None:
                            dt = t_frame_ms - self.curr_frame_cls.timestamp_ms
                            dt_desired = 1000 / self.cam_fps
                            if dt*1.2 < dt_desired:
                                continue # discard this frame

                    _, frame = self.cam_class.retrieve()
                    if np.array_equal(self.curr_frame_cls.frame, frame):
                        if self.equal_count == 0:
                            if not self.apiPreference == cv2.CAP_MSMF:
                                Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx} returned the same frame')
                        self.equal_count += 1
                        if self.equal_count == 5:
                            Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx} returned same frame 5 times. Stopping camera.')
                            self.stop()
                            break
                        continue
                    else:
                        self.equal_count = 0
                    self.curr_frame_cls.frame = frame
                    if self.apiPreference == cv2.CAP_MSMF:
                        next_idx = int((t_cam_ms - t_cam_ms_init) / (1000 / self.cam_fps))
                        if next_idx > self.curr_frame_cls.idx + 1:
                            n_dropped = int(next_idx - self.curr_frame_cls.idx - 1)
                            if 1000 / self.cam_fps < (n_dropped + 1) * 1000 / self.cam_fps:
                                dt = t_frame_ms - self.curr_frame_cls.timestamp_ms
                                Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx} dropped {n_dropped} frames. Dt: {dt:.0f} ms')
                        self.curr_frame_cls.idx = next_idx
                    else:
                        self.curr_frame_cls.idx += 1
                        if self.curr_frame_cls.timestamp_ms is not None:
                            dt = (t_frame_ms - self.curr_frame_cls.timestamp_ms)
                            dt_desired = 1000 / self.cam_fps
                            n_dropped = np.round(dt/dt_desired)-1
                            n_dropped_tot += n_dropped
                            if time.time() - t_timer > t_interval:
                                t_timer = time.time()
                                Logger.warning(f'Camera "{self.camera_name}": Dropped {int(n_dropped_tot)} frames in last {t_interval} seconds') if n_dropped_tot > 0 else None
                                n_dropped_tot = 0

                    if not ret:
                        Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx} disconnected, cannot get next frame.')
                        self.stop()
                        break
                    self.curr_frame_cls = Frame(frame, self.curr_frame_cls.idx, t_frame_ms)
                    yield self.curr_frame_cls

            except Exception as e:
                Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx}: Error getting next frame: {e}')
                self.stop()
                break
    
class NeoapiConnectionCore(GenericConnectionCore):
    def __init__(self, **kwargs) -> None:
        super().__init__(backend='neoapi', **kwargs)
        self.settings_mapper_dict = neoapi_fixed_settings | neoapi_variable_settings

        # variables for drift correction
        self.initial_correction = False

        self.t_py_start_abs = None
        self.t_cam_start_abs = None
        self.t_freq = time.time()
        self.t_freq2 = time.time()

        self.sync_interval = 20 # seconds

        self.correction_time_ms = 0
        self.adjust_step_ms_max = 3 # seconds delay after which to adjust
        # self.adjust_step_ms = 1 # milliseconds to adjust by when delay is above adjust threshold
        self.delay_thresh_reset = 0.05 # seconds delay after which to reset or print critical error
        self.delay_corrected_hist = []

    def _connect(self) -> None:
        self.cam_class = neoapi.Cam()
        if self.identifier is not None:
            self.cam_class.Connect(self.identifier)
        else:
            self.cam_class.Connect()
        self.cam_class.SetImageBufferCount(50)      # set the size of the buffer queue to 10
        self.cam_class.SetImageBufferCycleCount(1)  # sets the cycle count to 1 
        # neoapi.PixelFormat_Mono8
        self.set('pixel_format', neoapi.PixelFormat_Mono8)
        self.cam_class.f.TriggerMode.value = neoapi.TriggerMode_Off
        self.set('fps_enable', True)
        self.set('fps', self.cam_fps)
        self.set('binning_horizontal', 1)
        self.set('binning_vertical', 1)
        frame = self.cam_class.GetImage().GetNPArray()
        # find out binning settings
        horizontal_ratio = frame.shape[1] / self.cam_res[0]
        vertical_ratio = frame.shape[0] / self.cam_res[1]
        # round to closest integer
        horizontal_ratio = max(1, round(horizontal_ratio))
        vertical_ratio = max(1, round(vertical_ratio))
        # pick the lowest one such that aspect ratio is preserved
        binning = min(horizontal_ratio, vertical_ratio)
        self.set('binning_horizontal', binning)
        self.set('binning_vertical', binning)
        info_str = (f'Opening Neoapi camera "{self.camera_name}" with index {self.cam_idx} with resolution {self.cam_res} and fps {self.cam_fps} and binning {binning}.')
        if self.identifier is not None:
            info_str += f' Identifier: "{self.identifier}"'
        if self.get("binning_horizontal") != binning:
            Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx} could not set binning to {binning}. Actual binning: {self.get("binning_horizontal")}')
        
    def _disconnect(self) -> None:   
        self.cam_class.Disconnect()

    def is_connected(self) -> bool:
        return self.cam_class.IsConnected()
    
    def activate_trigger(self) -> None:
        self.cam_class.f.TriggerMode.value = neoapi.TriggerMode_On
        Logger.info(f'Activated trigger for camera "{self.camera_name}" at time {time.time()}')

    def deactivate_trigger(self) -> None:
        self.cam_class.f.TriggerMode.value = neoapi.TriggerMode_Off
        Logger.info(f'Deactivated trigger for camera "{self.camera_name}" at time {time.time()}')

    def empty_buffer(self) -> None:
        while True:
            try:
                img = self.cam_class.GetImage()
                shape = img.GetNPArray().shape
                if shape[0] == 0 or shape[1] == 0:
                    break
                print(f'Emptying buffer (img shape: {shape})')
            except neoapi.NoImageBufferException:
                break
        print('Buffer emptied') 
    
    def trigger_frame(self) -> bool:
        try:
            self.cam_class.f.TriggerSoftware.Execute()
            return True
        except (neoapi.neoapi.FeatureAccessException, AttributeError):
            print(f'Camera "{self.camera_name}": Trigger not available')
            return False

    def retrieve_frame(self) -> Frame:
        img = self.cam_class.GetImage()
        t_py_abs = time.time()

        frame = img.GetNPArray()
        t_cam_abs = img.GetTimestamp()/10**9 # convert from ns to s

        frame_cls = Frame(frame, -1, t_cam_abs*1000)

        if frame.shape[0] == self.cam_res[1] and frame.shape[1] == self.cam_res[0]:
            frame_cls.idx = self.curr_frame_cls.idx + 1
            self.curr_frame_cls = frame_cls

        return frame_cls
    
    def set(self, setting: str, value: Any, dict: Optional[dict] = None) -> None:
        if dict is None:
            dict = self.settings_mapper_dict
        func_str = parse_setting(setting, dict)
        if func_str is not None:
            try:
                getattr(self.cam_class.f, func_str).Set(value)
            except Exception as e:
                Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx}: {e}')
                return None
        else:
            Logger.warning(f'Setting {setting} is not valid for neoapi.')

    def get(self, setting: str) -> Any:
        func_str = parse_setting(setting, self.settings_mapper_dict)
        if func_str is not None:
            try:
                value = getattr(self.cam_class.f, func_str).Get()
                return value
            except Exception as e:
                Logger.warning(f'Camera "{self.camera_name}" with index {self.cam_idx}: {e}')
                return None
        else:
            Logger.warning(f'Setting {setting} is not valid for neoapi.')

    def _grab_next_frame(self) -> Generator[Frame, None, None]:
        while(self.cam_class is not None and self.cam_class.IsConnected()):
            try:
                img = self.cam_class.GetImage()
                if not img.IsEmpty():
                    t_py_abs = time.time()
                    frame = img.GetNPArray()
                    self.curr_frame_cls.frame = frame
                    self.curr_frame_cls.idx += 1
                    self.curr_frame_cls.timestamp_s = t_py_abs
                    del img
                    yield self.curr_frame_cls
                else:
                    if True:
                        continue
                    else:
                        Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx} disconnected, cannot get next frame.')
                        self.stop()
                        break
            except neoapi.NoImageBufferException:
                Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx} returned no image buffer available')
                print(f'Error: Camera "{self.camera_name}" with index {self.cam_idx} returned no image buffer available')
                # disconnect and connect again
                self.stop()
                break

            except Exception as e:
                Logger.error(f'Camera "{self.camera_name}" with index {self.cam_idx}: Error getting next frame: {e}')
                print(f'Error: Camera "{self.camera_name}" with index {self.cam_idx}: Error getting next frame: {e}')
                self.stop()
                break
