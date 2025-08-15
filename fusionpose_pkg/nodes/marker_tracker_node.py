#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from fusionpose_pkg.msg import NodeStatus
from cv_bridge import CvBridge
import os
from node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TransformStamped
from std_msgs.msg import Bool
import tf2_ros 
import tf

from deps.utils.logger_util import Logger
from deps.acquisition import CameraHandler
from deps.marker_tracker import MarkerTracker

import roslaunch

import threading
import random

import numpy as np
import json
from copy import deepcopy

from functools import wraps

from typing import Any, Optional

# create wrapper that shuts down the node on exception
def shutdown_on_exception(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            Logger.error(f"Exception in {func.__name__}: {e}")
            rospy.signal_shutdown(f"Exception in {func.__name__}: {e}")
    return wrapper

def prepare_static_transform(frame_id: str, child_id: str, R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None, T: Optional[np.ndarray] = None) -> TransformStamped:
    if T is None:
        assert R is not None and t is not None, "Either T or R and t must be provided, not None"
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        q = tf.transformations.quaternion_from_matrix(T)
        t = tf.transformations.translation_from_matrix(T)
    else:
        assert R is None and t is None, "Either T or R and t must be provided, not both"
        q = tf.transformations.quaternion_from_matrix(T)
        t = tf.transformations.translation_from_matrix(T)
    t = t.flatten()
    q = q.flatten()
    # normalize quaternion
    q /= np.linalg.norm(q)
    transf = TransformStamped()
    transf.header.stamp = rospy.Time.now()
    transf.header.frame_id = frame_id
    transf.child_frame_id = child_id
    transf.transform.translation.x = t[0]
    transf.transform.translation.y = t[1]
    transf.transform.translation.z = t[2]
    transf.transform.rotation.x = q[0]
    transf.transform.rotation.y = q[1]
    transf.transform.rotation.z = q[2]
    transf.transform.rotation.w = q[3]
    return transf

def setup_object_markers(oid_files_path: str) -> dict[str, dict[str, dict[str, any]]]:
    # find all .json files in oid_files_path
    if not os.path.isdir(oid_files_path):
        Logger.warning(f'oid files path "{oid_files_path}" not found.')
        return {}
    oid_files = [f for f in os.listdir(oid_files_path) if f.endswith('.json')]
    if len(oid_files) == 0:
        Logger.warning(f'No .json files found in oid files path "{oid_files_path}"')
        return {}
    oid_config_dict = {}
    oid_ids = {}
    for oid_file in oid_files:
        with open(os.path.join(oid_files_path, oid_file), "r") as f:
            # try to access keys 'name', 'marker_length', 'pts_file_path', 'T_MtoI',
            raw_config = json.load(f)
            new_entry = {}
            json_entries = ['name', 'marker_length', 'pts_file_path', 'T_MtoI']
            # json_entries_optional = ['t_shift']
            for entry in json_entries:
                if entry not in raw_config.keys():
                    Logger.warning(f'oid file "{oid_file}" is missing key "{entry}"')
                    continue
            try:
                int(raw_config['name'])
            except ValueError:
                Logger.warning(f'oid file "{oid_file}" has invalid name "{raw_config["name"]}", must be an integer')
                continue

            # check if pts_file_path exists
            if not os.path.isfile(os.path.join(oid_files_path, raw_config['pts_file_path'])):
                Logger.warning(f'oid file "{oid_file}" has invalid pts_file_path "{raw_config["pts_file_path"]}"')
                continue

            new_entry['pts_file_path'] = os.path.join(oid_files_path, raw_config['pts_file_path'])
            with open(new_entry['pts_file_path'], "r") as f:
                ids = [int(k) for k in json.load(f).keys()]

            duplicates = False
            for oid, ids_existing in oid_ids.items():
                if set(ids).intersection(ids_existing):
                    Logger.warning(f'oid {oid} and oid {raw_config["name"]} have overlapping ids, not loading oid {raw_config["name"]}')
                    duplicates = True
                    break

            if duplicates:
                continue
            else:
                oid_ids[int(raw_config['name'])] = ids


            try:
                new_entry['marker_length'] = float(raw_config['marker_length'])
            except ValueError:
                Logger.warning(f'oid file "{oid_file}" has invalid marker_length "{raw_config["marker_length"]}"')
                continue

            try:
                T_MtoI = np.eye(4)
                T_MtoI[:3, :3] = np.transpose(np.array(raw_config['T_MtoI']['r']).reshape((3, 3)))
                T_MtoI[:3, 3] = np.array(raw_config['T_MtoI']['t']).reshape((3,))
                new_entry['T_MtoI'] = T_MtoI
            except Exception as e:
                Logger.warning(f'oid file "{oid_file}" has invalid T_MtoI "{raw_config["T_MtoI"]}"')
                continue
            
            # optional shift
            if 't_shift' in raw_config.keys():
                try:
                    new_entry['shift'] = np.array(raw_config['t_shift']).reshape((3,))
                except Exception as e:
                    Logger.warning(f'oid file "{oid_file}" has invalid shift "{raw_config["shift"]}", ignoring')

            oid_config_dict[int(raw_config['name'])] = new_entry

    Logger.info(f'Loaded config for oids: {", ".join([str(oid) for oid in oid_config_dict.keys()])}') if len(oid_config_dict.keys()) > 0 else Logger.warning(f'No valid oid files found in oid files path "{oid_files_path}"')
    return oid_config_dict  

class MarkerTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__(name=f'marker_tracker_node_{random.randint(0, 1000)}')

        # create Node status publisher
        self.status_pub = rospy.Publisher('/node_status', NodeStatus, queue_size=1, latch=True)
        self.is_cam_ready = False
        self.bridge = CvBridge()

        self.camera_handler: CameraHandler = None
        self.camera_name: str = None
        self.camera_latency_s: float = None

        self.marker_tracker: MarkerTracker = None

        self.image_pub = None
        self.image_sub = None

        self.t_cam_start_abs = None
        self.t_py_start_abs = None

        self.tfs_to_publish = []

        self.is_tracker_enabled = False

        self.tracker_queue = []

        self.idx = 0

        self.camera_name = rospy.get_param('~camera_name')
        self.master = self.camera_name == self.config['acquisition']['master_camera']

        self.inference_rate = self.config['acquisition'][self.camera_name].get('fps', 30)

        self.calibrate_new_oid = self.config['marker_tracker'].get('marker_calibration', {}).get('enabled', False)

        if not self.calibrate_new_oid:
            if self.master:
                Logger.info(f'Registering camera "{self.camera_name}" as master camera!')
            else:
                Logger.info(f'Registering camera "{self.camera_name}" as slave camera!')

            self.oid_config_dict = setup_object_markers(self.config['marker_tracker'].get('oid_files_path', ''))
        else:
            self.oid_config_dict = {}   

        self.subscribe = self.config['acquisition'][self.camera_name].get('subscribe', False)
        camera_stream_topic = f'/{self.camera_name}/stream'
        if not self.subscribe:
            if self.config['acquisition'][self.camera_name].get('publish', False):
                Logger.info(f"Publishing frames to topic {camera_stream_topic}")
                self.image_pub = rospy.Publisher(camera_stream_topic, Image, queue_size=10)

            if self.init_camera_handler():
                rospy.Subscriber(f'/{self.camera_name}/latency', Float32, self.latency_callback)
                rospy.Subscriber(f'/{self.camera_name}/latency_delta', Float32, self.latency_delta_callback)
                rospy.Subscriber(f'/{self.camera_name}/acquisition_debug', Bool, self.acquisition_debug_callback)
                self.is_cam_ready = True

            # else:
            #     self.shutdown()
        else:
            Logger.info(f"Subscribing to topic {camera_stream_topic}, make sure framerate is low enough for processing")
            self.image_sub = rospy.Subscriber(camera_stream_topic, CompressedImage, self.image_callback)
            self.is_cam_ready = True

        self.is_tracker_enabled = self.config['marker_tracker'].get('enabled', True)        
        if self.is_cam_ready:
            if self.is_tracker_enabled:
                self.marker_poses: dict[int, np.ndarray] = None
                # create a ros subscriber for marker tracker debug
                self.init_marker_tracker()
                # example: rostopic pub /arducam1/marker_debug std_msgs/Bool true 
                rospy.Subscriber(f'/{self.camera_name}/marker_debug', Bool, self.marker_debug_callback)
            else:
                Logger.info("Marker tracker is disabled")

            if not self.calibrate_new_oid and self.master: # only master camera should start other nodes
                self.start_other_nodes()
                tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
                tf_broadcaster.sendTransform(self.tfs_to_publish)

        self.publish_node_status(is_healthy=True, info=f'Camera is ready', is_just_started=True)

    def publish_node_status(self, is_healthy: bool, info: str, is_unrecoverable_error: bool = False, is_just_started: bool = False, is_busy: bool = False) -> None:
        status = NodeStatus()
        status.node_name = f'{self.node_name}/{self.camera_name}'
        status.is_healthy = is_healthy
        status.info = info
        status.is_unrecoverable_error = is_unrecoverable_error
        status.is_just_started = is_just_started
        status.is_busy = is_busy
        self.status_pub.publish(status)

    def latency_delta_callback(self, data: Float32) -> None:
        if self.master:
            Logger.warning(f'Master camera "{self.camera_name}" has master clock, latency delta cannot be set, ignoring')
        else:
            info_str = f'Camera "{self.camera_name}" decrementing current latency of {self.camera_latency_s*1000:.2f} ms by {data.data*1000:.2f} ms'
            self.camera_latency_s -= data.data
            info_str += f' to {self.camera_latency_s*1000:.2f} ms'
            Logger.info(info_str)

    def latency_callback(self, data: Float32) -> None:
        if self.master:
            Logger.warning(f'Master camera "{self.camera_name}" has master clock, latency cannot be set, ignoring')
        else:
            Logger.info(f'Camera "{self.camera_name}" changing latency from {self.camera_latency_s*1000:.2f} ms to {data.data*1000:.2f} ms')
            self.camera_latency_s = data.data

    def marker_debug_callback(self, req: Bool) -> None:
        self.marker_tracker.set_debug(req.data)

    def acquisition_debug_callback(self, req: Bool) -> None:
        self.camera_handler.monitor(start=req.data)

    def image_callback(self, data: CompressedImage) -> None: # compressed image callback
        frame = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        time_ros = data.header.stamp
        if self.is_tracker_enabled:
            self.tracker_queue.append((frame, time_ros))
    
    def init_camera_handler(self) -> bool:
        if self.subscribe:
            return True
        acq_debug = self.config['acquisition']['debug']
        camera_params = self.config['acquisition'][self.camera_name]
        idx = camera_params.get('idx', 0)
        resolution = (camera_params.get('width', 640), camera_params.get('height', 480))
        fps = camera_params.get('fps', 30)

        settings_path = camera_params.get('settings_file', None)

        if self.calibrate_new_oid and settings_path is not None:
            settings_dir = os.path.dirname(settings_path)
            for file in os.listdir(settings_dir):
                camera_parsed = ''.join([i for i in self.camera_name if not i.isdigit()])
                if 'quality' in file and camera_parsed in file:
                    Logger.info(f'Due to calibration acquisition, using settings file "{file}"')
                    settings_path = os.path.join(settings_dir, file)
                    break
        
        self.camera_handler = CameraHandler(
            backend=camera_params.get('backend', 'opencv'),
            camera_name=self.camera_name,
            settings_path=settings_path,
            cam_res=resolution,
            fps=fps,
            cam_idx=idx,
            calib_file_path=camera_params.get('calibration_file', None),
            api=camera_params.get('api', 'DSHOW'),
            debug=acq_debug,
            identifier=camera_params.get('identifier', None),
        )     

        if not self.master:
            self.camera_latency_s = camera_params.get('latency', 0.0)
            Logger.info(f'Camera "{self.camera_name}" latency: {self.camera_latency_s} s')
        else:
            self.camera_latency_s = 0.0

        if self.camera_handler.is_connected() and self.camera_handler.is_calibrated():
            print(f'Camera "{self.camera_name}" connected and calibrated')
            # camera daq thread
            cam_daq_thread = threading.Thread(target=self.image_daq, args=(camera_params.get('backend', 'opencv'),))
            cam_daq_thread.daemon = True
            cam_daq_thread.start()
 
            if self.master: # must only publish gravity transform for master camera
                R_transf = np.eye(3)
                # load from file
                gravity_transform_file = camera_params.get('gravity_transform_file', None)
                if gravity_transform_file is not None and os.path.isfile(gravity_transform_file):
                    try:
                        with open(gravity_transform_file, 'r') as f:
                            R_transf = np.array(eval(f.read()))
                        assert R_transf.shape == (3, 3), "Gravity transform must be a 3x3 matrix"
                        # linalg inverse
                        R_transf = np.linalg.inv(R_transf)
                        Logger.info(f'Camera "{self.camera_name}": Loaded gravity transform from file "{gravity_transform_file}"')
                    except Exception as e:
                        Logger.error(e)
                        Logger.warning(f'Camera "{self.camera_name}": Error loading gravity transform from file "{gravity_transform_file}", using identity')

                elif gravity_transform_file is not None:
                    Logger.warning(f'Camera "{self.camera_name}": Gravity transform file "{gravity_transform_file}" not found, using identity')
                else:
                    Logger.warning(f'Camera "{self.camera_name}": No gravity transform file provided, using identity')
                print(f'Gravity transform for {self.camera_name}: {R_transf}, loaded from file: {gravity_transform_file}')
                self.tfs_to_publish.append(prepare_static_transform(self.config['fusion']['world_frame'], self.camera_name, R=R_transf, t=np.zeros(3)))
                
                extrinsics_file = camera_params.get('extrinsics_file', None)
                if extrinsics_file is not None and os.path.isfile(extrinsics_file):
                    try:
                        with open(extrinsics_file, 'r') as f:
                            extrinsics = json.load(f)
                        for _, extrinsic in extrinsics.items():
                            from_frame = extrinsic['from_frame']
                            to_frame = extrinsic['to_frame']
                            if not (from_frame == self.camera_name or to_frame == self.camera_name):
                                Logger.warning(f'Camera "{self.camera_name}": Extrinsics file "{extrinsics_file}" contains extrinsics not related to this camera, ignoring')
                                continue
                            R = np.array(extrinsic['R']).reshape((3, 3))
                            t = np.array(extrinsic['t']).reshape((3,))
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = t
                            self.tfs_to_publish.append(prepare_static_transform(from_frame, to_frame, T=T))
                            Logger.info(f'Camera "{self.camera_name}": Found extrinsics from "{from_frame}" to "{to_frame}"')
                    except Exception as e:
                        Logger.error(e)
                        Logger.warning(f'Camera "{self.camera_name}": Error loading extrinsics from file "{extrinsics_file}"')
                elif extrinsics_file is not None:
                    Logger.warning(f'Camera "{self.camera_name}": Extrinsics file "{extrinsics_file}" not found, ignoring')

            return True
        else:
            Logger.error(f'Camera "{self.camera_name}" failed to connect or calibrate')
            print(f'Camera "{self.camera_name}" failed to connect or calibrate')
            return False

    def init_marker_tracker(self) -> None:
        calib_file_path = self.config['acquisition'][self.camera_name]['calibration_file']
        marker_tracker_config = deepcopy(self.config['marker_tracker'])

        # pop some parameters that are used later
        marker_tracker_config.pop('enabled', False)
        marker_tracker_config.pop('launch_imu_acquisition', False)
        marker_tracker_config.pop('launch_fusion', False)
        marker_tracker_config.pop('fusion_exclude_oids', [])
        marker_tracker_config.pop('launch_udp_server', False)
        marker_tracker_config.pop('oid_files_path', '')

        self.marker_tracker = MarkerTracker(
            camera_name=self.camera_name,
            calib_file_path=calib_file_path,
            acquire_calibration=self.calibrate_new_oid, 
            calibration_params=marker_tracker_config.pop('marker_calibration', {}),      
            fps=self.inference_rate,
            oid_config_dict=self.oid_config_dict,  
            debug=marker_tracker_config.pop('debug'),
            **marker_tracker_config
            )
                
        if self.marker_tracker.setup_complete:
            if not self.calibrate_new_oid:
                self.use_cov = False
                # create a publisher for each marker
                self.marker_pose_pub = {}
                for oid in self.marker_tracker.oids:
                    if self.use_cov:
                        self.marker_pose_pub[oid] = rospy.Publisher(f'/oid_{oid}/pose', PoseWithCovarianceStamped, queue_size=1000)
                    else:
                        self.marker_pose_pub[oid] = rospy.Publisher(f'/oid_{oid}/pose', PoseStamped, queue_size=1000)

    def start_other_nodes(self) -> None:
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        package = self.config['pkg_name']

        # rosparm imu/devices/names
        ble_names = [f'oid_{oid}' for oid in self.oid_config_dict.keys()]
        # publish these rosparams
        rospy.set_param('/imu/devices/names', ble_names)
        
        if self.config['marker_tracker'].pop('launch_imu_acquisition', False) and not self.subscribe:
            node = roslaunch.core.Node(package=package, node_type='imu_acquisition_node.py', name='imu_acquisition_node', output='screen')
            launch.launch(node)
        elif self.subscribe:
            Logger.info(f'Master camera "{self.camera_name}": Not launching imu acquisition node as image subscription is enabled')
        else:
            Logger.info(f'Master camera "{self.camera_name}": Not launching imu acquisition node')

        for oid in self.oid_config_dict.keys():
            # publish marker to imu transform for this oid
            self.tfs_to_publish.append(prepare_static_transform(f'oid_{oid}/marker', f'oid_{oid}/imu', T=self.oid_config_dict[oid]['T_MtoI']))
        
        if self.config['marker_tracker'].pop('launch_fusion', False):
            # set rosparam /fusion/autocalibration/enabled to false if it is true
            if self.subscribe and rospy.get_param('/fusion/auto_calibration/enabled', False):
                Logger.info(f'Master camera "{self.camera_name}": Disabling fusion auto-calibration as image subscription is enabled')
                rospy.set_param('/fusion/auto_calibration/enabled', False)
            fusion_exlcude_oids = self.config['marker_tracker'].get('fusion_exclude_oids', [])
            for oid in self.oid_config_dict.keys():
                if oid in fusion_exlcude_oids:
                    Logger.info(f'Skipping fusion node for oid {oid}')
                else:
                    name = f'fusion_node_{oid}'
                    Logger.info(f'Launching fusion node for oid {oid}')
                    node = roslaunch.core.Node(package=package, node_type='fusion_node.py', name=name, output='screen', args=f'_fusion/oid_name:=oid_{oid}')
                    launch.launch(node)
        else:
            Logger.info(f'Master camera "{self.camera_name}": Not launching fusion node')

    @shutdown_on_exception
    def trigger_callback(self, data: Any) -> None:
        success = self.camera_handler.core.trigger_frame()
        if success:
            curr_tdaq_ros = rospy.Time.now()
            # Logger.debug(f'Triggered camera "{self.camera_name}" at time (s): {rospy.Time.now().to_sec()}')
            # threading.Thread(target=self.trigger_thread, args=(curr_tdaq_ros)).start()
            self.trigger_thread(curr_tdaq_ros)
        else:
            Logger.warning(f'Camera "{self.camera_name}" trigger failed')
            # check if camera is still connected
            if not self.camera_handler.is_connected():
                raise Exception(f'Camera "{self.camera_name}" is not connected')
                # self.shutdown()

    @shutdown_on_exception
    def trigger_thread(self, curr_tdaq_ros: rospy.Time) -> None:
        frame_cls = self.camera_handler.core.retrieve_frame()
        t_py_daq_abs = rospy.Time.now()
        curr_frame, curr_idx, t_cam_daq = frame_cls.as_tuple()
        # check if frame is empty
        if curr_frame.shape[0] == 0 or curr_frame.shape[1] == 0:
            return
        self.idx = curr_idx

        if self.t_py_start_abs is None:
            self.t_py_start_abs = t_py_daq_abs
        
        t_cam_daq_ros = curr_tdaq_ros - rospy.Duration.from_sec(self.camera_latency_s)

        if self.is_tracker_enabled:
            self.tracker_queue.append((curr_frame, t_cam_daq_ros))
            
        # checks
        if self.report_iter is None:
            self.report_iter = -1
        if self.report_timer is None:
            self.report_timer = rospy.Time.now().to_sec()

        self.report_iter += 1
        report_interval = 20 # seconds
        if rospy.Time.now().to_sec() - self.report_timer > report_interval:
            acq_rate = self.report_iter / (rospy.Time.now().to_sec() - self.report_timer)
            if acq_rate < self.inference_rate * 0.9:
                Logger.warning(f'''Camera {f'"{self.camera_name}"':10}: Average rate (triggered) in last {report_interval} seconds was {acq_rate:.1f} Hz (< {self.inference_rate} Hz)''')
            self.report_timer = rospy.Time.now().to_sec()
            self.report_iter = 0

    @shutdown_on_exception
    def image_daq(self, backend: str = 'opencv') -> None:
        if self.config['acquisition'][self.camera_name].get('monitor', False):
            self.camera_handler.monitor()
        report_iter = 0
        report_timer = rospy.Time.now().to_sec()


        self.triggered_daq = self.config['acquisition'][self.camera_name].get('triggered_daq', False)
        if backend == 'opencv' and self.triggered_daq:
            Logger.warning(f'Camera "{self.camera_name}": Triggered DAQ is not supported for OpenCV backend, ignoring')
            self.triggered_daq = False


        if self.triggered_daq:
            threading.Thread(target=self.camera_handler.core.activate_trigger).start()
            threading.Thread(target=self.camera_handler.core.empty_buffer).start()
            # debug variables
            self.report_iter = None
            self.report_timer = None
            # setup trigger subscriber
            rospy.Subscriber(f'/acquisition_trigger', Bool, self.trigger_callback)
            Logger.info(f'Camera "{self.camera_name}": Triggered DAQ activated')
            # if master camera, start up publisher that will publish a trigger
            if self.master:
                trigger_pub = rospy.Publisher(f'/acquisition_trigger', Bool, queue_size=10)   
                # continously trigger at 1/fps rate
                while not rospy.is_shutdown():
                    trigger_pub.publish(True)
                    rospy.sleep(1/self.inference_rate)    
            

        else:
            for frame_cls in self.camera_handler.grab_next_frame():
                curr_frame, curr_idx, curr_tdaq = frame_cls.as_tuple()
                # if curr_idx != self.idx + 1 and self.idx != 0:
                #     Logger.warning(f'"{self.camera_name}" DAQ: Dropped {curr_idx - self.idx - 1} frames')
                self.idx = curr_idx
                if self.t_py_start_abs is None:
                    self.t_py_start_abs = rospy.Time.now()
                if self.t_cam_start_abs is None:
                    self.t_cam_start_abs = curr_tdaq
                    # self.t_start_ros = rospy.Time.now()
                curr_tdaq_ros = self.t_py_start_abs + rospy.Duration.from_sec(curr_tdaq - self.t_cam_start_abs - self.camera_latency_s)
            

                if self.is_tracker_enabled:
                    self.tracker_queue.append((curr_frame, curr_tdaq_ros))
                # Debugging
                report_iter += 1
                report_interval = 20 # seconds
                if rospy.Time.now().to_sec() - report_timer > report_interval:
                    acq_rate = report_iter / (rospy.Time.now().to_sec() - report_timer)
                    if acq_rate < self.inference_rate * 0.9:
                        Logger.warning(f'''Camera {f'"{self.camera_name}"':10}: Average rate in last {report_interval} seconds was {acq_rate:.1f} Hz (< {self.inference_rate} Hz)''')
                    report_timer = rospy.Time.now().to_sec()
                    report_iter = 0

        self.is_cam_ready = False

    def publish_pose(self, publisher: rospy.Publisher, time: rospy.Time, quaternion: list, position: list, lost=False, use_cov=None) -> None:
        time = time
        frame_id = self.camera_name if not lost else f'{self.camera_name}_lost'
        if use_cov is None:
            use_cov = self.use_cov
        if use_cov:
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = time
            # frame is the camera frame
            msg.header.frame_id = frame_id
            msg.pose.pose.position.x = position[0]
            msg.pose.pose.position.y = position[1]
            msg.pose.pose.position.z = position[2]
            msg.pose.pose.orientation.x = quaternion[0]
            msg.pose.pose.orientation.y = quaternion[1]
            msg.pose.pose.orientation.z = quaternion[2]
            msg.pose.pose.orientation.w = quaternion[3]
        else:
            msg = PoseStamped()
            msg.header.stamp = time
            # frame is the camera frame
            msg.header.frame_id = frame_id
            msg.pose.position.x = position[0]
            msg.pose.position.y = position[1]
            msg.pose.position.z = position[2]
            msg.pose.orientation.x = quaternion[0]
            msg.pose.orientation.y = quaternion[1]
            msg.pose.orientation.z = quaternion[2]
            msg.pose.orientation.w = quaternion[3]

        publisher.publish(msg)

    def process_frame(self, frame: np.ndarray, time_ros: rospy.Time) -> None:
        if self.image_pub is not None:
            if frame.ndim == 3 and frame.shape[2] == 3:  # color image
                img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            else:
                img_msg = self.bridge.cv2_to_imgmsg(frame, "mono8")
            img_msg.header.frame_id = self.camera_name
            img_msg.header.stamp = time_ros
            self.image_pub.publish(img_msg)

        # return
        self.marker_poses, _, lost_in_prev_frame = self.marker_tracker.process_frame(frame)

        if self.marker_poses is None:
            Logger.warning(f"Marker tracker failed for camera {self.camera_name}")
            return
        for oid, pose in self.marker_poses.items():
            if pose is not None:
                # pose is 4x4 transformation matrix, convert to position and quaternion
                position = pose[:3, 3] # convert to m
                quaternion = tf.transformations.quaternion_from_matrix(pose)
                self.publish_pose(self.marker_pose_pub[oid], time_ros, quaternion, position)
            if lost_in_prev_frame[oid]:
                self.publish_pose(self.marker_pose_pub[oid], time_ros, [np.nan]*4, [np.nan]*3, lost=True)

    def shutdown(self) -> None:
        # publish a lost message for all markers in self.marker_poses
        self.publish_node_status(is_healthy=False, info=f'Camera is shutting down', is_unrecoverable_error=True)
        try:
            if self.marker_tracker is not None:
                if self.marker_poses is not None:
                    for oid in self.marker_poses.keys():
                        if self.marker_poses[oid] is not None:
                            self.publish_pose(self.marker_pose_pub[oid], rospy.Time.now(), oid, [np.nan]*4, [np.nan]*3, lost=True)
                self.marker_tracker.close()
            if self.camera_handler is not None:
                if self.triggered_daq:
                    self.camera_handler.core.deactivate_trigger()
                self.camera_handler.stop()
        except Exception as e:
            pass

        Logger.info(f'Shutting down marker tracker node for camera "{self.camera_name}"')
        rospy.loginfo(f'Shutting down marker tracker node for camera "{self.camera_name}"')

if __name__ == '__main__':
    node = None      

    def attempt_shutdown():
        try:
            node.shutdown()
        except:
            pass

    rospy.on_shutdown(attempt_shutdown)
    try:
        node = MarkerTrackerNode()
        max_queue_size = 5
        n_dropped = 0
        t_curr = rospy.Time.now().to_sec()
        t_prev = 0
        if node.is_cam_ready:
            rate = rospy.Rate(node.inference_rate)
            while not rospy.is_shutdown() and node.is_cam_ready:
                if node.is_tracker_enabled:
                    while len(node.tracker_queue) > 0:
                        if not node.calibrate_new_oid:
                            if len(node.tracker_queue) > max_queue_size:
                                n_dropped += len(node.tracker_queue) - max_queue_size
                                # report number of dropped frames every 1 second
                                report_interval = 100
                                if rospy.Time.now().to_sec() - t_curr > report_interval:
                                    if n_dropped > 0:
                                        Logger.warning(f'Marker Tracker camera "{node.camera_name}": Dropped {n_dropped} frames in last {report_interval} seconds')
                                    n_dropped = 0
                                    t_curr = rospy.Time.now().to_sec()                             
                                node.tracker_queue = node.tracker_queue[-max_queue_size:]
                        else:
                            # only keep the last frame if calibrating
                            node.tracker_queue = node.tracker_queue[-1:]
                        frame, time_ros = node.tracker_queue.pop(0)
                        t_prev = time_ros.to_sec()
                        node.process_frame(frame, time_ros)
                rate.sleep()
    except Exception as e:
        Logger.exception("message")
