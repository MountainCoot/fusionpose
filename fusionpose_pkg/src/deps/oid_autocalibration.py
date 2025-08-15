from deps.utils.logger_util import Logger
from deps.utils.util import keep_max_n_files
try:
    from vicon2gt.srv import CalibrateBag, CalibrateBagRequest
except ImportError:
    print('Cannot import CalibrateBag service, probably not running in ROS environment')

import time
import os
import numpy as np
import threading

import rosbag
import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage


def load_calibration(filepath: str) -> dict: # TODO: maybe change cpp implementation to a YAML or JSON file because this is a bit hacky
    with open(filepath, 'r') as file:
        text = file.read()

    lines = text.strip().splitlines()  # Split the text into lines
    result = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for specific keywords and extract values
        if "R_BtoI:" in line:
            result["R_BtoI"] = []
            for _ in range(3):  # 3 lines for the 3x3 matrix
                i += 1
                result["R_BtoI"].extend(map(float, lines[i].strip().split()))
            # create a 3x3 matrix
            result["R_BtoI"] = np.array(result["R_BtoI"]).reshape(3, 3)
        elif "q_BtoI:" in line:
            result["q_BtoI"] = []
            for _ in range(4):  # 4 values for quaternion
                i += 1
                result["q_BtoI"].append(float(lines[i].strip()))
            # create a quaternion
            result["q_BtoI"] = np.array(result["q_BtoI"])
        elif "p_BinI:" in line:
            result["p_BinI"] = []
            for _ in range(3):  # 3 values for the position vector
                i += 1
                result["p_BinI"].append(float(lines[i].strip()))
            # create a position vector
            result["p_BinI"] = np.array(result["p_BinI"])
        elif "R_GtoV:" in line:
            result["R_GtoV"] = []
            for _ in range(3):  # 3 lines for the 3x3 matrix
                i += 1
                result["R_GtoV"].extend(map(float, lines[i].strip().split()))
            # create a 3x3 matrix
            result["R_GtoV"] = np.array(result["R_GtoV"]).reshape(3, 3)
        elif "t_off_vicon_to_imu:" in line:
            i += 1
            result["t_off_vicon_to_imu"] = float(lines[i].strip())
        
        i += 1

    for key, value in result.items():
        Logger.debug(f"{key}: {value}")
        
    return result

def check_calibration(result: dict) -> bool:
    if np.all(np.abs(result["p_BinI"]) < 0.05): # indicates a 'good' calibration
        return True
    else:
        return False

def call_calibration_service(frame_id: str, oid_name: str, bag_dir: str) -> tuple[bool, dict] | None:
    rospy.wait_for_service('/run_calibration')  # Wait until the service is available
    try:
        # Create a service proxy
        calibrate_bag = rospy.ServiceProxy('/run_calibration', CalibrateBag)
        
        # Create a request object and set the values
        request = CalibrateBagRequest()
        request.camera_name = frame_id
        request.oid_name = oid_name
        
        # Call the service
        response = calibrate_bag(request)
        
        # Print the result
        if response.success:
            Logger.info(f'Calibration service for "{frame_id}" and "{oid_name}" was successful.')
            # in subfolder vicon2gt find {oid_name}_autocalibration_{frame_id}_vicon2gt_info.yaml
            info_file_path = os.path.join(bag_dir, 'vicon2gt', f'{oid_name}_autocalibration_{frame_id}_vicon2gt_info.txt')
            result = load_calibration(info_file_path)
            if check_calibration(result):
                return True, result
            else:
                return False, None
        else:
            return False, None
    except rospy.ServiceException as e:
        Logger.error(f'Service call failed: {e}')
        return False, None


class AutoCalibration():
    """
    AutoCalibration class automatically creates rosbag files while a specific oid is significantly 
    moved to allow for gravity and time offset calibration using vicon2gt. As vicon2gt requires
    a bag with constant IMU and pose frequency (i.e., no dropped frames), bag files are filtered to 
    ensure useful convergence in vicon2gt. 

    If autocalibration is enabled, new bags are continously created given significant movement.
    """
    def __init__(self, oid_name: str, master_camera: str, **params: dict) -> None:
        self.oid_name = oid_name
        self.master_camera = master_camera

        self.enabled = params.get('enabled')
        if not self.enabled:
            Logger.info(f'AutoCalibration for "{self.oid_name}" is disabled.')
            return
        self.bag_dir = params.get('bag_dir')
        # create path
        os.makedirs(self.bag_dir, exist_ok=True)

        Logger.info(f'AutoCalibration for "{self.oid_name}" enabled. Bag directory: "{self.bag_dir}"')

        self.bag_duration_s = params.get('bag_duration_s', 20) # continous bag duration in seconds
        self.rec_interval_s = params.get('rec_interval_s', 240) # record interval in seconds
        self.pos_diff_thresh = params.get('pos_diff_thresh', 1)*self.bag_duration_s/10 # minimum pose difference threshold for calibration

        pose_freq_desired = params['pose_freq_des']
        imu_freq_desired = params['imu_freq_des']
        self.pose_dt_thresh = 1.0 / pose_freq_desired * 1.7
        self.imu_dt_thresh = 1.0 / imu_freq_desired * 7.0 # can allow for quite some slack in imu dt
        
        self.bag_name = f'{self.oid_name}_autocalibration'

        self.tf_msg = []
        rospy.Subscriber('/tf_static', PoseStamped, self.tf_callback)

        # create subscriber that is able to reset calibration status, such that recalibration is required
        rospy.Subscriber(f'/{self.oid_name}/reset_calibration', Bool, self.reset_callback)

        self.bag_paths = dict()

        self.last_modified = dict()
        self.calibration_active = dict()

        self.reset_calibration_data()

        # create publisher for /oid_name/latency_delta
        self.t_off_imu_master_new = None
        self.imu_latency_pub = rospy.Publisher(f'/{self.oid_name}/latency_delta', Float32, queue_size=10)

        self.slave_latencies_unpublished = dict() # can only publish after master camera calibration, store here until then
        self.slave_cams_latency_subs = dict() # subscribers time offset of slave cameras to master camera are corrected, master camera is the reference
        self.slave_cams_latency_pubs = dict() # publishers time offset of slave cameras to master camera are corrected, master camera is the reference

    def reset_callback(self, msg: Bool) -> None:
        # if true, reset calibration data for all frames
        # if false, set last_modified to now for all frames
        if msg.data:
            Logger.info(f'Resetting calibration data for "{self.oid_name}", recalibration required.')
            print(f'{self.pretty_time()}: Resetting calibration data for "{self.oid_name}", recalibration required.')
            self.reset_calibration_data()
        else:
            Logger.info(f'Setting frames {", ".join(self.calibration_active.keys())} to calibrated for "{self.oid_name}".')
            print(f'{self.pretty_time()}: Setting frames {", ".join(self.calibration_active.keys())} to calibrated.')
            for frame_id, _ in self.calibration_active.items():
                self.last_modified[frame_id] = time.time()
                self.calibration_active[frame_id] = False

    def tf_callback(self, tf_msgs: list[TFMessage]) -> None:
        # append to tf_msgs and replace if already exists
        for tf_msg in tf_msgs.transforms:
            p_frame_id_new, c_frame_id_new = tf_msg.header.frame_id, tf_msg.child_frame_id
            for i, msg in enumerate(self.tf_msg):
                p_frame_id, c_frame_id = msg.header.frame_id, msg.child_frame_id
                if set([p_frame_id, c_frame_id]) == set([p_frame_id_new, c_frame_id_new]):
                    self.tf_msg[i] = tf_msg
                    break
            else:
                self.tf_msg.append(tf_msg)

    def reset_calibration_data(self, frame_id=None, success=True) -> None:
        if frame_id is None:
            self.pose_stats = dict()
            self.last_modified = dict()
            self.calibration_active = dict()
            self.t_off_imu_master_new = None
        else:
            if success:
                self.pose_stats.pop(frame_id, None)
                self.calibration_active[frame_id] = False
            else: # must continue calibration directly
                self.calibration_active[frame_id] = True
                # just empty the pose data
                self.pose_stats[frame_id] = {'ts': [], 'pos': [], 'msgs': [], 'imu_data': {'ts': [], 'msgs': []}}

    def recalibration_required(self, frame_id: str, t_msg: float) -> bool:
        # check if bag file exists and if its last modified time is older than the record interval
        if not self.last_modified.get(frame_id):
            self.bag_paths[frame_id] = os.path.join(self.bag_dir, f'{self.bag_name}_{frame_id}.bag')
            # if not os.path.exists(self.bag_paths[frame_id]):
            #     Logger.debug(f'Bag file does not exist: "{os.path.basename(self.bag_paths[frame_id])}", so calibration is required.')
            # else:
            #     Logger.debug(f'No recent calibration performed for "{self.oid_name}" and "{frame_id}", so starting calibration.')
            return True


        if required:=(t_msg - self.last_modified[frame_id]) > self.rec_interval_s:
            # Logger.debug(f'Calibration required for {frame_id}: {t_msg - self.last_modified[frame_id]:.2f} > {self.rec_interval_s} is {required}')
            self.reset_calibration_data(frame_id)

        return required

    @staticmethod
    def get_pos(msg: PoseStamped) -> np.ndarray:
        try:
            pos = msg.pose.pose.position
        except AttributeError:
            pos = msg.pose.position
        return np.array([pos.x, pos.y, pos.z])

    def log_stats(self, frame_id=None) -> None:
        info_str = ''
        if frame_id is None:
            for key, val in self.pose_stats.items():
                if len(val['ts']) > 0:
                    info_str += f'Pose from {key}: {len(val["ts"])}, time passed: {val["ts"][-1] - val["ts"][0]:.2f} s '
                    info_str += f'Pos diff {np.sum(np.abs(np.diff(val["pos"], axis=0))):.3f} '
                    info_str += f'IMU msgs: {len(val["imu_data"]["ts"])}\n'
            info_str = info_str[:-1] # remove last newline
            Logger.debug(info_str)     
        else:
            val = self.pose_stats[frame_id]
            if len(val['ts']) > 0:
                info_str += f'Pose from {frame_id}: {len(val["ts"])}, time passed: {val["ts"][-1] - val["ts"][0]:.2f} s '
                info_str += f'Pos diff {np.sum(np.abs(np.diff(val["pos"], axis=0))):.3f} '
                info_str += f'IMU msgs: {len(val["imu_data"]["ts"])}\n'
            return info_str

    @staticmethod
    def parse_f(frame_id: str) -> str:
        return frame_id.replace('_lost', '')
    
    @staticmethod
    def pretty_time() -> str:
        # return time in HH:MM:SS format
        return time.strftime('%H:%M:%S', time.localtime())

    def msg_callback(self, msg: PoseStamped | Imu, is_pose=False) -> None:
        if not self.enabled:
            return
        # self.log_stats()
        t_msg = msg.header.stamp.to_sec()
        if is_pose:
            frame_id = msg.header.frame_id
            frame_id_parsed = self.parse_f(frame_id)

            # if it is slave camera, initialize latency publisher and subscriber for that camera
            if frame_id_parsed != self.master_camera:
                if frame_id_parsed not in self.slave_cams_latency_subs:
                    # if getting new delta from any other camera, must end the calibration because otherwise offset might be published twice
                    self.slave_cams_latency_subs[frame_id_parsed] = rospy.Subscriber(f'/{frame_id_parsed}/latency_delta', Float32, lambda msg: self.reset_calibration_data(frame_id_parsed, success=True))
                if frame_id_parsed not in self.slave_cams_latency_pubs:
                    self.slave_cams_latency_pubs[frame_id_parsed] = rospy.Publisher(f'/{frame_id_parsed}/latency_delta', Float32, queue_size=10)
            
            if self.calibration_active.get(frame_id_parsed, False):
                if frame_id_parsed not in self.pose_stats:
                    master_str = '(master camera)' if frame_id_parsed == self.master_camera else '(slave camera)'
                    Logger.info(f'Starting calibration for "{self.oid_name}" and "{frame_id_parsed}" {master_str}')
                    print(f'{self.pretty_time()}: Starting calibration for "{self.oid_name}" and "{frame_id_parsed}" {master_str}')
                    self.pose_stats[frame_id_parsed] = {'ts': [], 'pos': [self.get_pos(msg)], 'msgs': [], 'imu_data': {'ts': [], 'msgs': []}}
                if frame_id_parsed != frame_id:
                    # Logger.critical(f'Pose message from "{self.oid_name}" and "{frame_id_parsed}" is lost')
                    self.check_for_bag_release(frame_id_parsed)
                    return

                if len(self.pose_stats[frame_id_parsed]['ts']) > 0:
                    if t_msg - self.pose_stats[frame_id_parsed]['ts'][0] > self.bag_duration_s:
                        # Logger.debug(f'Bag duration exceeded for "{self.oid_name}" and "{frame_id_parsed}"')
                        self.check_for_bag_release(frame_id_parsed)
                        return
                    dt = t_msg - self.pose_stats[frame_id_parsed]['ts'][-1]
                    if dt > self.pose_dt_thresh:
                        Logger.debug(f'Threshold for pose dt exceeded for "{self.oid_name}" and "{frame_id_parsed}": {dt:.2f} > {self.pose_dt_thresh:.2f}')
                        self.check_for_bag_release(frame_id_parsed)
                        return
                        
                self.pose_stats[frame_id_parsed]['ts'].append(t_msg)
                self.pose_stats[frame_id_parsed]['pos'].append(self.get_pos(msg))
                self.pose_stats[frame_id_parsed]['msgs'].append(msg)
            else:
                if self.recalibration_required(frame_id_parsed, t_msg=t_msg):
                    self.calibration_active[frame_id_parsed] = True

        elif any(self.calibration_active.values()):
            if self.pose_stats: # if there is pose data
                for frame_id in self.pose_stats:
                    if self.calibration_active.get(frame_id, False) and len(self.pose_stats[frame_id]['imu_data']['ts']) > 0:
                        dt = t_msg - self.pose_stats[frame_id]['imu_data']['ts'][-1]
                        if dt > self.imu_dt_thresh:
                            Logger.debug(f'Threshold for imu dt exceeded for "{self.oid_name}" and "{frame_id}": {dt:.2f} > {self.imu_dt_thresh:.2f}')
                            self.check_for_bag_release(frame_id)
                            return
                    self.pose_stats[frame_id]['imu_data']['ts'].append(t_msg)
                    self.pose_stats[frame_id]['imu_data']['msgs'].append(msg)

    def get_relevant_transforms(self, frame_id: str) -> list:
        tf_relevant = []
        for tf_msg in self.tf_msg:
            c_frame_id, p_frame_id = tf_msg.child_frame_id, tf_msg.header.frame_id
            relevant_child = c_frame_id in [frame_id, 'gravity'] or self.oid_name in c_frame_id
            relevant_parent = p_frame_id in [frame_id, 'gravity'] or self.oid_name in p_frame_id
            if relevant_child and relevant_parent:
                tf_relevant.append(tf_msg)
        return tf_relevant            

    def check_for_bag_release(self, frame_id: str) -> None:
        # check if data in bag is meets the pos diff threshold
        # get relevant frames, all if frame_id is None
        if self.pose_stats and self.calibration_active.get(frame_id, False):
            if len(self.pose_stats[frame_id]['ts']) == 0 or len(self.pose_stats[frame_id]['imu_data']['ts']) == 0:
                self.reset_calibration_data(frame_id, success=False)
            elif self.pose_stats[frame_id]['ts'][-1] - self.pose_stats[frame_id]['ts'][0] < 0.6*self.bag_duration_s:
                bag_duration = self.pose_stats[frame_id]['ts'][-1] - self.pose_stats[frame_id]['ts'][0]
                Logger.debug(f'Bag duration {bag_duration:.2f} < {0.6*self.bag_duration_s}, not enough data for calibration for "{self.oid_name}" and "{frame_id}"\nstats: {self.log_stats(frame_id)}')
                self.reset_calibration_data(frame_id, success=False)
            elif np.sum(np.abs(np.diff(self.pose_stats[frame_id]['pos'], axis=0))) < self.pos_diff_thresh:
                pos_diff = np.sum(np.abs(np.diff(self.pose_stats[frame_id]['pos'], axis=0)))
                Logger.debug(f'Pos diff {pos_diff:.3f} < {self.pos_diff_thresh}, not enough movement for calibration for "{self.oid_name}" and "{frame_id}"\nstats: {self.log_stats(frame_id)}')
                self.reset_calibration_data(frame_id, success=False)
            else:
                Logger.info(f'Enough data for calibration for "{self.oid_name}" and "{frame_id}", saving, stats: {self.log_stats(frame_id)}')
                # must do these first to avoid race conditions
                self.last_modified[frame_id] = self.pose_stats[frame_id]['msgs'][-1].header.stamp.to_sec()
                self.calibration_active[frame_id] = False
                # get all relevant tf messages (either with self.oid_name, frame or 'gravity' in names)
                self.pose_stats[frame_id]['tf'] = self.get_relevant_transforms(frame_id)
                # print(f'{self.pretty_time()}: Data collection for "{self.oid_name}" and "{frame}" calibration complete, calibrating...')
                self.calibrate(frame_id)

    def calibrate(self, frame_id: str) -> None:  
        # create a new thread for saving the bag
        thread = threading.Thread(target=self.calibrate_thread, args=(frame_id,))
        thread.start()

    def calibrate_thread(self, frame_id: str) -> None:
        bag_path = self.bag_paths[frame_id]
        keep_max_n_files(bag_path, n=4)
        
        # receive all messages in tf tree
        msgs_combined = []
        for pose_msg in self.pose_stats[frame_id]['msgs']:
            msgs_combined.append((f'/{self.oid_name}/pose', pose_msg.header.stamp, pose_msg))
        for imu_msg in self.pose_stats[frame_id]['imu_data']['msgs']:
            msgs_combined.append((f'/{self.oid_name}/imu', imu_msg.header.stamp, imu_msg))
        # sort by timestamp
        msgs_combined.sort(key=lambda x: x[1])
        with rosbag.Bag(bag_path, 'w') as outbag:
            outbag.write('/tf_static', TFMessage(self.pose_stats[frame_id]['tf']))
            for topic, ts, msg in msgs_combined:
                outbag.write(topic, msg, ts)

        # call calibration service
        success, result = call_calibration_service(frame_id, self.oid_name, self.bag_dir)
        if success:
            t_off_camera_imu = result["t_off_vicon_to_imu"]
            t_off_camera_imu = round(t_off_camera_imu*1000)/1000
            if frame_id == self.master_camera:
                t_off_master_imu = t_off_camera_imu
                # round to ms
                Logger.info(f'Calibration for "{frame_id}" and "{self.oid_name}" is successful with a time offset between IMU and Master Camera of {t_off_master_imu:.3f} s.')
                print(f'{self.pretty_time()}: Calibration for "{frame_id}" and "{self.oid_name}" is successful with a time offset between IMU and Master Camera of {t_off_master_imu:.3f} s.')
                # # must release any other bags at this time because we are about to change the latency which might affect the other bags
                for f in self.pose_stats:
                    if f != frame_id:
                        self.check_for_bag_release(f)

                if self.t_off_imu_master_new is None:
                    self.t_off_imu_master_new = {'t_off': 0, 'last_modified': self.last_modified[frame_id]}
                if np.abs(t_off_master_imu) > 0.001: # only publish if significant (1 ms)
                    self.imu_latency_pub.publish(Float32(t_off_master_imu))
                    self.t_off_imu_master_new['t_off'] = t_off_master_imu
                else:
                    # publish 0 if not significant
                    self.imu_latency_pub.publish(Float32(0))
                # check if there are any slave cameras that have not been published yet
                for f, t_off_slave_imu in self.slave_latencies_unpublished.items():
                    t_off_slave_master = t_off_slave_imu - self.t_off_imu_master_new['t_off']
                    if np.abs(t_off_slave_master) > 0.001:
                        self.slave_cams_latency_pubs[f].publish(Float32(t_off_slave_master))
                    else:
                        # publish 0 if not significant
                        self.slave_cams_latency_pubs[f].publish(Float32(0))
                    Logger.info(f'Master calibration available. Can now publish time offset between Slave Camera "{f}" and Master Camera of {t_off_slave_master:.3f} s.')
                    print(f'{self.pretty_time()}: Master calibration available. Can now publish time offset between Slave Camera "{f}" and Master Camera of {t_off_slave_master:.3f} s.')
                self.slave_latencies_unpublished = dict()

            else: # for slave cameras, do not calibrate IMU time but camera time
                t_off_slave_imu = t_off_camera_imu
                if self.t_off_imu_master_new is None or self.t_off_imu_master_new['last_modified'] + self.rec_interval_s < self.last_modified[frame_id]:
                    Logger.warning(f'Calibration for "{frame_id}" and "{self.oid_name}" is successful but no recent time offset between IMU and Master Camera is available, waiting for master camera calibration...')
                    print(f'{self.pretty_time()}: Calibration for "{frame_id}" and "{self.oid_name}" is successful but no recent time offset between IMU and Master Camera is available, waiting for master camera calibration...')
                    self.slave_latencies_unpublished[frame_id] = t_off_slave_imu
                else:
                    t_off_slave_master = t_off_slave_imu - self.t_off_imu_master_new['t_off']
                    if np.abs(t_off_slave_master) > 0.001:
                        self.slave_cams_latency_pubs[frame_id].publish(Float32(t_off_slave_master))
                    else:
                        # publish 0 if not significant
                        self.slave_cams_latency_pubs[frame_id].publish(Float32(0))
                    print(f'{self.pretty_time()}: Calibration for "{frame_id}" and "{self.oid_name}" is successful with a time offset between Slave Camera and Master Camera of {t_off_slave_master:.3f} s.')
                    Logger.info(f'Calibration for "{frame_id}" and "{self.oid_name}" is successful with a time offset between Slave Camera and Master Camera of {t_off_slave_master:.3f} s.')
        else:
            Logger.error(f'Calibration for "{frame_id}" and "{self.oid_name}" failed.')
            print(f'{self.pretty_time()}: Calibration for "{frame_id}" and "{self.oid_name}" failed.')
            self.last_modified[frame_id] = None
            
        self.reset_calibration_data(frame_id, success=success)

