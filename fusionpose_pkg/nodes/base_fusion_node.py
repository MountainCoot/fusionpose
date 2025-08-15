import rospy
import rosbag
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from tf import transformations
import tf2_ros
import tf2_msgs.msg
import os
from node import Node
import heapq
import threading

from typing import Optional

import time
import sys

import numpy as np
np.set_printoptions(precision=4, suppress=True)

from deps.oid_autocalibration import AutoCalibration
from deps.utils.logger_util import Logger
from deps.utils.fusion_saver import save_to_file
from deps.utils.util import load_rosbag

from math import floor

class BaseFusionNode(Node):
    def __init__(self, node_name: str, usecore: bool) -> None:
        super().__init__(node_name, use_ros=usecore)

        # Logger.warning('NOT ENOUGH IMU SAMPLES WARNING IS SUPPRESSED')

        if any(['fusion/oid_name' in arg for arg in sys.argv]):
            idx = [i for i, arg in enumerate(sys.argv) if 'fusion/oid_name' in arg][0]
            self.config['fusion']['oid_name'] = sys.argv[idx].split('=')[1]
            #  turn off replay if on 
            self.config['fusion']['replay']['enabled'] = False

        fusion_params = self.config['fusion']

        self.save_dir = fusion_params['save_dir']
        self.replay = fusion_params['replay']['enabled']
        self.plot_results = self.replay
        if self.plot_results:
            self.occlusions = {}
            self._fusion_name = "OTS + IMU"
            self._results = {'IMU': [], 'OTS': [], self._fusion_name: [], 'OTS_frame': []}

        self.world_frame = fusion_params['world_frame']
        self.master_pose_frame = self.config['acquisition']['master_camera']
        self.output_pose_frame = fusion_params.get('output_frame', self.master_pose_frame)

        self.pose_frames_occluded = {} # dict of occluded pose frames
        self.curr_occlusion_ts = {} 

        self.lock = threading.Lock()
        self.imu_queue = []
        self.pose_queue = []
        self.imu_queue_pose_lost = [] # queue for IMU messages when pose is lost
        
        self.transformations_map = {self.world_frame: {}}

        self.timer = time.time()
        
        self.t_prev = None

        # variables
        self._last_imu = None
        self._last_pose = None
        
        self._fusion_core = None
        self.fusion_core_initialized = False
        self.fusion_core_reset = False
        self.init_pos = None

        self.t_start_bag = None

        self.last_t_processed = 0

        self.lost_before = False


        if self.replay:
            Logger.info('Fusion node in replay mode')
            self.replay_params = fusion_params['replay']
            self.oid_filter = self.replay_params.get('oid_filter', None)
            self.bag_file_path = os.path.join(self.replay_params['bag_dir'], self.replay_params['bag_name'])
            if not os.path.exists(self.bag_file_path):
                Logger.error(f'Bag file does not exist: "{self.bag_file_path}"')
                return
            else:
                Logger.info(f'Replaying from bag file: "{self.bag_file_path}"')


            self.pose_topic, self.imu_topic = None, None
            if self.replay_params.get('pose_topic', '') != '':
                self.pose_topic = self.replay_params['pose_topics']
            if self.replay_params.get('imu_topic', '') != '':
                self.imu_topic = self.replay_params['imu_topic']   
            
            self.get_tf_from_bag = self.replay_params.get('get_tf_from_bag', True)

            if not self.get_tf_from_bag:
                self.extrinsics = self.replay_params.get('extrinsics', [])

                # Transformation between IMU and atracsys
                R_AtoI = np.array(self.replay_params['b2s_rotmat']).reshape(3, 3)
                R_ItoA = np.linalg.inv(R_AtoI)
                t = np.array(self.replay_params['b2s_pos'])
                t_ItoA = -R_ItoA.dot(t)
                self.T_ItoA = np.eye(4)
                self.T_ItoA[0:3, 0:3] = R_ItoA
                self.T_ItoA[0:3, 3] = t_ItoA
                self.oid_name = self.replay_params['oid_name']
            else:
                self.tf_msgs = []
                self.T_ItoA = None
        
            if not self.prepare_replay(bag=rosbag.Bag(self.bag_file_path)):
                return	

            self.record_out_bag = self.replay_params.get('export_bag', False)
            if self.record_out_bag:
                export_bag_path_dir = os.path.join(self.replay_params['bag_dir'], 'export')
                os.makedirs(export_bag_path_dir, exist_ok=True)
                append = self.replay_params.get('append_to_bag', True)
                if append:
                    export_bag_path = os.path.join(export_bag_path_dir, self.replay_params['bag_name'])
                else:
                    export_bag_path = os.path.join(export_bag_path_dir, self.replay_params.get('export_bag_name', 'output.bag'))
                if append and os.path.exists(export_bag_path):
                    info_str = f'Appending to existing bag file: "{export_bag_path}"'
                    try:
                        existing_bag = load_rosbag(export_bag_path)
                        # print all the topics in the bag
                        topics = existing_bag.get_type_and_topic_info().topics
                        for topic in topics:
                            info_str += f'\n\tExisting topic: "{topic}"'
                    
                        # find all topics that contain the oid_name
                        overwrite_topics = [f'/{self.oid_name}/fused', f'/{self.oid_name}/fused_smooth',\
                                             f'/{self.oid_name}/pose', f'/{self.oid_name}/imu', \
                                             f'/tf', f'/tf_static']
                        oid_topics = []
                        curr_msgs = []
                        for topic in topics:
                            if topic in overwrite_topics:
                                oid_topics.append(topic)
                                # skip
                                continue
                            for _, msg, t in existing_bag.read_messages(topics=[topic]):
                                curr_msgs.append((topic, msg, t))
                        info_str += f'\nOverwriting topics: {oid_topics}'
                        Logger.info(info_str)
                        existing_bag.close()
                        # now can overwrite the bag file
                        self.out_bag = rosbag.Bag(export_bag_path, 'w')
                        for topic, msg, t in curr_msgs:
                            self.out_bag.write(topic, msg, t)
                    except rosbag.bag.ROSBagException:
                        Logger.warning(f'Bag file "{export_bag_path}" is not indexed, cannot append to it. Creating new bag file..')
                        self.out_bag = rosbag.Bag(export_bag_path, 'w')                   
                else:
                    Logger.info(f'Exporting to new bag file: "{export_bag_path}"')
                    self.out_bag = rosbag.Bag(export_bag_path, 'w')

            self.show_position = self.replay_params.get('show_position', False)
            self.show_orientation = self.replay_params.get('show_orientation', False)
            self.show_3d = self.replay_params.get('show_3d', False)

        else:
            self.oid_name = fusion_params['oid_name']
            if not self.get_imu_transform(): # can get IMU transform at this point as it should be available in the tf tree
                return
            
            self.record_out_bag = False

            # get imu topic and pose topics from ROS parameters
            self.imu_topic = f'/{self.oid_name}/imu'
            self.pose_topic = f'/{self.oid_name}/pose'

            # setup subscribers
            self.pose_subscriber = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_callback)
            if self.imu_topic:
                self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
            # subscribe to tf messages
            self.tf_static_subscriber = rospy.Subscriber('/tf_static', tf2_msgs.msg.TFMessage, self.tf_static_callback)

        if self.replay: 
            fusion_params['auto_calibration']['enabled'] = False # no auto calibration in replay mode

        self.auto_calibration = AutoCalibration(self.oid_name, self.master_pose_frame, **fusion_params['auto_calibration'])

    def initialize_fusion_core(self, t_init: float, init_pos: np.ndarray, init_ori: np.ndarray) -> None:
        raise NotImplementedError

    def get_imu_transform(self) -> bool:
        oid_marker_frame = f'{self.oid_name}/marker'
        oid_imu_frame = f'{self.oid_name}/imu'

        T_AtoI = self.get_transform(oid_marker_frame, oid_imu_frame)

        if T_AtoI is None:
            Logger.error(f'No transformation found from {oid_marker_frame} to {oid_imu_frame}')
            return False
        
        R_AtoI = T_AtoI[0:3, 0:3]
        t = T_AtoI[0:3, 3]

        R_ItoA = np.linalg.inv(R_AtoI)
        t_ItoA = -R_ItoA.dot(t)
        self.T_ItoA = np.eye(4)

        euler_angles = transformations.euler_from_matrix(R_ItoA)
        quat = transformations.quaternion_from_euler(*euler_angles)
        R_ItoA = transformations.quaternion_matrix(quat)[:3, :3]

        self.T_ItoA[0:3, 0:3] = R_ItoA
        self.T_ItoA[0:3, 3] = t_ItoA
        
        return True

    def prepare_replay(self, bag: rosbag.Bag) -> bool:
        """Check if the bag file contains the necessary topics and if all transformations are found"""
        # get topics from the bag file
        topics = bag.get_type_and_topic_info().topics
        pose_topic_temp = None
        imu_topic_temp = None

        if self.oid_filter is not None:
            self.oid_name = f'oid_{self.oid_filter}'
            # filter topics that contain the oid_name
            # print other topics
            # for topic in topics:
            #     if self.oid_name not in topic:
            #         print(f'Ignoring topic "{topic}" as it does not contain "{self.oid_name}"')
            topics = {topic: topics[topic] for topic in topics if self.oid_name in topic}
            assert len(topics) > 0, f'No topics found for "{self.oid_name}"'
            Logger.info(f'Filtering bag file for "{self.oid_name}"')

        n_msgs_fused_smooth = None
        n_msgs_imu = None
        for topic in topics:
            n_msgs = topics[topic].message_count
            info_str = f'Found topic "{topic}" of type "{topics[topic].msg_type}" with {n_msgs} messages'
            if topics[topic].frequency is not None:
               info_str += f'and frequency {topics[topic].frequency:.1f} Hz'
            else:
                info_str += 'and unknown frequency'
            print(info_str)
            if 'fused_smooth' in topic:
                n_msgs_fused_smooth = n_msgs
            if topics[topic].msg_type in ['geometry_msgs/PoseWithCovarianceStamped', 'geometry_msgs/PoseStamped'] and 'fuse' not in topic:
                if pose_topic_temp is not None:
                    Logger.warning(f'Additional pose topic "{topic}" found in the bag file, ignoring..')
                else:
                    pose_topic_temp = topic
            if topics[topic].msg_type == 'sensor_msgs/Imu':
                n_msgs_imu = n_msgs
                if imu_topic_temp is not None:
                    Logger.warning(f'Additional IMU topic "{topic}" found in the bag file, ignoring..')
                else:
                    imu_topic_temp = topic

        if n_msgs_fused_smooth is not None and n_msgs_imu is not None:
            if 2*n_msgs_imu < n_msgs_fused_smooth:
                raise 

        # find oid_#number in pose topics
        self.oid_name = None
        if 'oid' in pose_topic_temp:
            self.oid_name = pose_topic_temp.split('/')[1]

        Logger.info(f'Found pose topic: {pose_topic_temp} for {self.oid_name}')        
                
        if self.pose_topic is None:
            self.pose_topic = pose_topic_temp
        if self.imu_topic is None:
            self.imu_topic = imu_topic_temp

        if self.pose_topic not in topics:
            Logger.error(f'Pose topic "{self.pose_topic}" not found in the bag file, exiting..')
            return False

        if self.imu_topic not in topics:
            Logger.error(f'IMU topic "{self.imu_topic}" not found in the bag file, exiting..')
            return False
        
        if self.pose_topic is None:
            Logger.error(f'No valid pose topic found in the bag file, exiting..')
            return False
        
        if self.imu_topic is None:
            Logger.error(f'No valid IMU topic found in the bag file, exiting..')
            return False
        
        
        if self.get_tf_from_bag:
            for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
                # print type of msg and raise
                self.tf_msgs.append(msg)

            Logger.debug(f'Found {len(self.tf_msgs)} tf messages')
            self.get_imu_transform()

        else:
            pose_frames = []
            for topic, msg, t in bag.read_messages(topics=[self.pose_topic]):
                pose_frames.append(msg.header.frame_id) if msg.header.frame_id not in pose_frames else None


            self.transformations_map = {frame: {} for frame in [*pose_frames, self.world_frame]}
            for camera_extrinsic in self.extrinsics:
                from_frame, to_frame = camera_extrinsic['from'], camera_extrinsic['to']
                if from_frame in pose_frames and to_frame in pose_frames:
                    T = np.eye(4)
                    T[0:3, 0:3] = np.array(camera_extrinsic['rotmat']).reshape(3, 3)
                    T[0:3, 3] = np.array(camera_extrinsic['pos'])
                    self.transformations_map[from_frame][to_frame] = T
                    self.transformations_map[to_frame][from_frame] = transformations.inverse_matrix(T)
                elif from_frame in pose_frames and to_frame == self.world_frame or to_frame in pose_frames and from_frame == self.world_frame:
                    T = np.eye(4)
                    T[0:3, 0:3] = np.array(camera_extrinsic['rotmat']).reshape(3, 3)
                    T[0:3, 3] = np.array(camera_extrinsic['pos'])
                    if from_frame == self.world_frame:
                        self.transformations_map[self.world_frame][to_frame] = T
                        self.transformations_map[to_frame][self.world_frame] = transformations.inverse_matrix(T)
                    else:
                        self.transformations_map[self.world_frame][from_frame] = transformations.inverse_matrix(T)
                        self.transformations_map[from_frame][self.world_frame] = T

                else:
                    Logger.warning(f'Transformation from "{camera_extrinsic["from"]}" to "{camera_extrinsic["to"]}" not used as one frame not found in frames {[*pose_frames, self.world_frame]}')

            # check if all combinations of extrinsics are found
            for frame in pose_frames:
                for frame2 in pose_frames:
                    if frame == frame2:
                        continue
                    if frame2 not in self.transformations_map[frame]:
                        Logger.error(f'Transformations from "{frame}" to "{frame2}" not found, exiting..')
                        return False
                                    
            if self.transformations_map.get(self.world_frame, {}) == {}:
                Logger.error(f'No transformation found for gravity frame "{self.world_frame}", exiting..')
                return False
                        
            # Logger.debug(self.transformations_map)
            Logger.info(f'Found transformations for frames: {pose_frames}') if len(pose_frames) > 1 else None
        Logger.info(f'\nReplay using:\n\t\tpose topic: {self.pose_topic}\n\t\tIMU topic: "{self.imu_topic}"\n\t\tgravity frame: "{self.world_frame}"\n\t\toutput frame: "{self.output_pose_frame}"')
        return True
    
    @staticmethod
    def get_parsed_time(t: float, exponent: int = 9, n_round=3) -> str:
        t_rounded = floor(t/10**3)/10**(exponent-3)
        t_remainder = t - t_rounded*10**exponent
        return f'{t_rounded}.e{exponent}+{t_remainder:.{n_round}f} s'

    def run(self) -> None:
        """Either process bag file via rosbag API or subscribe to topics"""
        if self.replay:
            bag = rosbag.Bag(self.bag_file_path)

            start = self.replay_params.get('start', 0)
            stop = self.replay_params.get('stop', bag.get_end_time())

            n_imu = 0
            n_pose = 0
            
            msgs = bag.read_messages(topics=[self.pose_topic, self.imu_topic])

            self.t_start_bag = bag.get_start_time()
            
            t_start = time.time()

            # start time is the first time a pose message is received
            for topic, msg, t in msgs:
                if topic == self.pose_topic:
                    start_time = msg.header.stamp.to_sec()
                    print(f'Time of first message from {self.pose_topic} is {start_time - bag.get_start_time():.3f} s after bag start')
                    break

            start += start_time
            started = False
            t = rospy.Time.from_sec(start_time)
            for i, (topic, msg, _) in enumerate(msgs):
                # skip transforms
                if topic in ['/tf', '/tf_static']:
                    continue
                t = msg.header.stamp
                if t.to_sec() < start:
                    continue
                elif not started:
                    started = True
                    print(f'START AT {t.to_sec() - start_time:.3f} SECS')
                if i % (bag.get_message_count() // 10) == 0:
                    Logger.debug(f'Progress: {int(i / bag.get_message_count() * 100)}%')
                elapsed_time_secs = t.to_sec() - start_time
                if topic == self.imu_topic:
                    self.imu_callback(msg)
                    n_imu += 1
                elif topic == self.pose_topic:
                    frame_id = msg.header.frame_id
                    n_pose += 1
                    self.pose_callback(msg)

                if elapsed_time_secs > stop:
                    print(f'\nSTOP AT {elapsed_time_secs} SECS\n')
                    Logger.info(f'Stopping at {elapsed_time_secs:.1f} s')
                    break
            
            if self.plot_results:
                for frame, t0 in self.curr_occlusion_ts.items():
                    if frame not in self.occlusions:
                        self.occlusions[frame] = []
                    self.occlusions[frame].append([t0, t.to_sec()])
                

            print(f'number of pose messages: {n_pose}, imu messages: {n_imu}, relevant bag duration {msg.header.stamp.to_sec() - start_time:.1f} s processed in {time.time() - t_start:.1f} s')
            bag.close()

            self.stop()
        else:

            while not rospy.is_shutdown():
                rospy.spin()
            self.stop()

    def imu_callback(self, msg: Imu) -> None:
        self.auto_calibration.msg_callback(msg)
        with self.lock:
            tol = 0.0001
            try:
                dt = 0
                # check if any within tolerance
                if any([abs(ts - msg.header.stamp.to_sec()) < tol for ts, _ in self.imu_queue]):
                    dt = tol
                heapq.heappush(self.imu_queue, (msg.header.stamp.to_sec()+dt, msg))
            except TypeError:
                # make queue sortable by adding a small number to the timestamp
                heapq.heappush(self.imu_queue, (msg.header.stamp.to_sec()+0.0001, msg))
            self.process_queues()

    def pose_callback(self, msg: PoseStamped) -> None:
        self.auto_calibration.msg_callback(msg, is_pose=True)
        with self.lock:
            tol = 0.0001
            try:
                dt = 0
                # check if any within tolerance
                if any([abs(ts - msg.header.stamp.to_sec()) < tol for ts, _ in self.pose_queue]):
                    dt = tol
                heapq.heappush(self.pose_queue, (msg.header.stamp.to_sec()+dt, msg))
            except TypeError:
                # make queue sortable by adding a small number to the timestamp
                heapq.heappush(self.pose_queue, (msg.header.stamp.to_sec()+tol, msg))

            self.process_queues()

    @staticmethod
    def parse_f(frame: str) -> str:
        return frame.replace('_lost', '')

    def get_active_frames(self) -> list[str]:
        # get frames that are neither occluded nor not calibrated
        return [frame for frame in self.pose_frames_occluded if not self.pose_frames_occluded[frame]]

    def get_unocc_frames(self) -> list[str]:
        # get frames that are not occluded
        return [frame for frame in self.pose_frames_occluded if not self.pose_frames_occluded[frame]]

    def render_info_str(self, info_str: str, active_frames_prev: list[str]) -> str:
        return # deactivated for now
        active_frames = self.get_active_frames()
        if info_str:
            if len(active_frames) == 0 and len(active_frames_prev) > 0:
                sub_str = ' resulting in loss of all active frames'
            elif len(active_frames) < len(active_frames_prev):
                sub_str = ' resulting in loss of some active frames'
            elif len(active_frames) > len(active_frames_prev):
                sub_str = ' resulting in gain of active frames'
            else:
                sub_str = ' yielding same number of active frames'
                
            info_str += ',' if info_str else ''
            info_str += f'{sub_str} ({len(active_frames_prev)} -> {len(active_frames)})'
            Logger.debug(info_str)
        return ''

    def process_queues(self) -> None:
        info_str = ''

        while self.pose_queue and self.imu_queue:
            # Logger.debug(f'{len(self.pose_queue)} pose messages and {len(self.imu_queue)} IMU messages in queue')

            # first check if at least one message per unoccluded camera frame is available
            if not set(self.get_unocc_frames()).issubset(set([self.parse_f(msg.header.frame_id) for _, msg in self.pose_queue])):
                break

            active_frames_prev = self.get_active_frames()
            # Logger.debug(f'Pose queue: {[(self.get_parsed_time(ts), msg.header.frame_id) for ts, msg in self.pose_queue]}')
            # get oldest pose and imu message
            pose_time, pose_msg = heapq.heappop(self.pose_queue)
            imu_time, imu_msg = heapq.heappop(self.imu_queue)

            # Check which message has the earlier timestamp
            t_process = min(pose_time, imu_time)
            # if len(self.get_active_frames()) == 0:
            #     if t_process != pose_time:
            #         Logger.debug(f'overwriting t_process with {pose_time:.3f} s')
            #         Logger.debug(f't_prev is {self.t_prev:.3f} s') if self.t_prev is not None else None
            #     t_process = pose_time
            if self.t_prev is None:
                self.t_prev = t_process
            else:
                t_delta = t_process - self.t_prev
                if len(self.pose_frames_occluded) == 1: # indicating that only one camera is used
                    if t_delta < 0:
                        pass
                        # Logger.error(f'Negative time delta: {t_delta:.3f} s, t_prev: {self.t_prev:.3f} s, t_process: {t_process:.3f} s')
                self.t_prev = t_process
                    # assert t_delta >= 0, f'Negative time delta: {t_delta:.3f} s, pose_time: {pose_time:.3f} s, imu_time: {imu_time:.3f} s'

            
            if pose_time <= imu_time or len(self.get_active_frames()) == 0:
                # Logger.debug(f'Processing pose message at {pose_time:.3f} s (imu_time: {imu_time:.3f} s)')
                pose_msg_frame = pose_msg.header.frame_id
                pose_msg_frame_parsed = self.parse_f(pose_msg_frame)
                # push back the IMU message
                heapq.heappush(self.imu_queue, (imu_time, imu_msg))

                if pose_msg_frame != pose_msg_frame_parsed: # if pose message is lost
                    if not self.pose_frames_occluded.get(pose_msg_frame_parsed, True):
                        info_str = f'"{self.oid_name}": Pose message from "{pose_msg_frame_parsed}" is lost'
                        info_str += f' ({self.get_parsed_time(pose_time)})' if self.replay else ''
                        self.curr_occlusion_ts[pose_msg_frame_parsed] = pose_time
                        self.pose_frames_occluded[pose_msg_frame_parsed] = True

                    if len(self.pose_queue) == 0: # keep at least one pose message in queue during occlusion
                        heapq.heappush(self.pose_queue, (pose_time, pose_msg))

                # check if any frames came back
                else:
                    if self.pose_frames_occluded.get(pose_msg_frame, False) and pose_msg_frame in self.curr_occlusion_ts:
                        info_str += f'"{self.oid_name}": Pose message from "{pose_msg_frame}" back'
                        info_str += f' after {pose_time - self.curr_occlusion_ts[pose_msg_frame]:.3f} s ({self.get_parsed_time(pose_time)})' if self.replay else ''
                        
                        if self.plot_results:
                            # ensure that occlusion length is positive
                            assert pose_time - self.curr_occlusion_ts[pose_msg_frame] > 0, f'Negative occlusion length: {pose_time - self.curr_occlusion_ts[pose_msg_frame]:.2f} s'
                            if pose_msg_frame not in self.occlusions:
                                self.occlusions[pose_msg_frame] = []
                            self.occlusions[pose_msg_frame].append((self.curr_occlusion_ts[pose_msg_frame], pose_time))
                        
                        self.curr_occlusion_ts.pop(pose_msg_frame)

                    self.pose_frames_occluded[pose_msg_frame] = False

                    use_first_pose = False

                    # first check if all pose messages are lost
                    if len(active_frames := self.get_active_frames()) > 0:
                        if self.lost_before:
                            # push back the pose message
                            heapq.heappush(self.pose_queue, (pose_time, pose_msg))
                            while self.pose_queue:
                                next_pose_time, next_pose_msg = heapq.heappop(self.pose_queue)
                                next_frame_id = next_pose_msg.header.frame_id
                                if next_frame_id in active_frames: # if 'lost' not in frame_id, it is the first message after occlusion
                                    break
                            else:
                                raise ValueError('No next pose message found after occlusion')


                            # dont put back the next pose message or initialize core with it because it is typically noisy 

                            if self._last_pose:
                                info_str += f' ({next_pose_time - self._last_pose[0]:.3f} s after last processed) (resetting fusion core)' if self.replay else ''
                                self.fusion_core_reset = True
                                self.t_prev = None
                                # put into imu queue from imu_queue_pose_lost from 0.1 s before pose message until pose message
                                for imu_time_lost, imu_msg_lost in self.imu_queue_pose_lost:
                                    if imu_time_lost < next_pose_time - 0.1:
                                        # Logger.info(f'Skip IMU message at {imu_time_lost:.3f} s, which is {next_pose_time - imu_time_lost:.3f} s before next pose message at {next_pose_time:.3f} s')
                                        continue
                                    heapq.heappush(self.imu_queue, (imu_time_lost, imu_msg_lost))

                                if use_first_pose:
                                    try:
                                        self.initialize_fusion_core(*self.process_pose_message(next_pose_msg, return_input=True))
                                    except TypeError:
                                        Logger.error(f'Error in pose message: {next_pose_msg}')

                            if use_first_pose:
                                heapq.heappush(self.pose_queue, (next_pose_time, next_pose_msg))

                            self.imu_queue_pose_lost = []
                            self.lost_before = False

                            info_str = self.render_info_str(info_str, active_frames_prev)
                            continue

                        elif len(active_frames) > len(active_frames_prev):
                            if not use_first_pose:
                                # Logger.debug(f'Pose message from "{pose_msg_frame}" not used as it is not the first message after partial occlusion or initialization')
                                info_str = self.render_info_str(info_str, active_frames_prev)
                                continue
                        
                if len(self.get_active_frames()) == 0 and self.pose_frames_occluded:

                    # empty all imu samples
                    imu_time = None
                    while self.imu_queue:
                        imu_time, imu_msg = heapq.heappop(self.imu_queue)
                        self.process_imu_message(imu_msg, pose_lost=True)
                        # put into queue for IMU messages when pose is lost
                        heapq.heappush(self.imu_queue_pose_lost, (imu_time, imu_msg))

                    lost_msgs = []
                    while self.pose_queue:
                        pose_time, pose_msg = heapq.heappop(self.pose_queue)
                        if self.pose_frames_occluded[pose_msg_frame_parsed]:
                            # append to lost messages
                            lost_msgs.append((pose_time, pose_msg))
                            self.lost_before = True
                            
                    # put back the lost messages
                    for pose_time, pose_msg in lost_msgs:
                        heapq.heappush(self.pose_queue, (pose_time, pose_msg))
                    info_str = self.render_info_str(info_str, active_frames_prev)
                    continue

                occluded = self.pose_frames_occluded.get(pose_msg_frame_parsed, None)
                if occluded is None:
                    Logger.critical(f'\n\nFrame "{pose_msg_frame_parsed}" not found in occluded frames: {self.pose_frames_occluded}\n\n')
                    occluded = True
                    self.pose_frames_occluded[pose_msg_frame_parsed] = True

                if occluded:
                    # pop the lost pose message
                    # heapq.heappop(self.pose_queue) # TODO: can maybe already pop here, investigate
                    info_str = self.render_info_str(info_str, active_frames_prev)
                    continue

                else: # have active frames and no change occured
                    self.process_pose_message(pose_msg)

                info_str = self.render_info_str(info_str, active_frames_prev)

            else:
                # Logger.debug(f'Processing IMU message at {imu_time:.3f} s (pose_time: {pose_time:.3f} s)')
                # push back the pose message
                heapq.heappush(self.pose_queue, (pose_time, pose_msg))
                # Process IMU message
                self.process_imu_message(imu_msg)


        # keep max 30 pose messages and 300 imu messages in the queue # TODO: define in constructor
        if len(self.pose_queue) > 20:
            self.pose_queue = self.pose_queue[-20:]
        if len(self.imu_queue) > 100:
            self.imu_queue = self.imu_queue[-100:]
        if len(self.imu_queue_pose_lost) > 100:
            self.imu_queue_pose_lost = self.imu_queue_pose_lost[-100:]

    @staticmethod
    def quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        q_diff = transformations.quaternion_multiply(transformations.quaternion_inverse(q1), q2)
        euler_diff = np.array(transformations.euler_from_quaternion(q_diff))
        return euler_diff

    def transform_marker_into_imu(self, t_MtoC: np.ndarray, q_MtoC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return t_MtoC, q_MtoC

    def transform_imu_into_marker(self, t_ItoC: np.ndarray, q_ItoC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return t_ItoC, q_ItoC

    def process_imu_message(self, msg: Imu, pose_lost=False) -> None:
        raise NotImplementedError

    def process_pose_message(self, msg: PoseStamped, return_input=False) -> None:
        raise NotImplementedError

    def process_smooth_result(self, result: list) -> None:
        raise NotImplementedError

    def transform_pose(self, pos: np.ndarray, ori: np.ndarray, from_frame: str, to_frame: str, allow_missing=False):
        """Transform pose from one frame to another"""
        if from_frame == to_frame:
            return pos, ori
        if from_frame in self.transformations_map and to_frame in self.transformations_map[from_frame]:
            T = self.transformations_map[from_frame][to_frame]
            ori_mat = transformations.quaternion_matrix(ori)
            pos_transf = T.dot(np.concatenate((pos, [1])))
            ori_mat_transf = T.dot(ori_mat)
            ori_transf = transformations.quaternion_from_matrix(ori_mat_transf)
        else:
            Logger.warning(f'Didn\'t lookup transformation from "{from_frame}" to "{to_frame}", trying to do so now')
            transf = self.get_transform(from_frame, to_frame)
            if transf is None:
                if allow_missing:
                    return None, None
                else:
                    Logger.error(f'Unable to transform pose from "{from_frame}" to "{to_frame}"')
                    raise ValueError(f'Unable to transform pose from "{from_frame}" to "{to_frame}"')
            if from_frame not in self.transformations_map:
                self.transformations_map[from_frame] = {}
            self.transformations_map[from_frame][to_frame] = transf
            return self.transform_pose(pos, ori, from_frame, to_frame)
        return pos_transf[0:3], ori_transf

    def tf_static_callback(self, msg: tf2_msgs.msg.TFMessage) -> None:
        # log tf messages
        for transform in msg.transforms:
            T = np.eye(4)
            T[0:3, 0:3] = transformations.quaternion_matrix([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])[0:3, 0:3]
            T[0:3, 3] = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            child_id, parent_id = transform.child_frame_id, transform.header.frame_id
            # if it already exists
            if child_id in self.transformations_map and parent_id in self.transformations_map[child_id]:
                Logger.info(f'Fusion "{self.oid_name}": Updated static transform "{child_id}" -> "{parent_id}" received')                           

            if child_id not in self.transformations_map:
                self.transformations_map[child_id] = {}
            if parent_id not in self.transformations_map:
                self.transformations_map[parent_id] = {}
            self.transformations_map[child_id][parent_id] = T
            self.transformations_map[parent_id][child_id] = transformations.inverse_matrix(T)

    def get_transform(self, source_frame: str, target_frame: str) -> Optional[np.ndarray]:
        """Use TF2 to get the transform between source frame and target frame"""
        trans = None
        inverse = False
        if self.replay and self.get_tf_from_bag:
            # Logger.debug(f'Looking up transform from "{source_frame}" to "{target_frame}"')
            for msg in self.tf_msgs:
                for tf in msg.transforms:
                    if tf.header.frame_id == source_frame and tf.child_frame_id == target_frame:
                        inverse = True
                        trans = tf
                        break   
                    elif tf.header.frame_id == target_frame and tf.child_frame_id == source_frame:
                        trans = tf
                        break
            if trans is None:
                Logger.error(f'Unable to lookup transform from "{source_frame}" to "{target_frame}"')
                # raise
                return None
        else:
            tfBuffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tfBuffer)
            try:
                trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time().now(), rospy.Duration(2.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                Logger.error(f'Unable to lookup transform from "{source_frame}" to "{target_frame}"')
                # rospy.signal_shutdown("Required TF missing")
                return None
        # Convert to rotation matrix
        target_T_source = transformations.quaternion_matrix(np.array([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]))
        target_T_source[0, 3] = trans.transform.translation.x
        target_T_source[1, 3] = trans.transform.translation.y
        target_T_source[2, 3] = trans.transform.translation.z
        if inverse:
            target_T_source = transformations.inverse_matrix(target_T_source)
        return target_T_source

    def publish(self, stamp: rospy.Time, frame_id: str, position: np.ndarray, orientation: np.ndarray, publisher=None, bag=None, topic='/pose') -> None:
        """Publish PoseWithCovarianceStamped"""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        if bag is not None:
            bag.write(topic, msg, stamp)
        else:
            assert publisher is not None, 'Publisher is None'
            publisher.publish(msg)

    def stop(self) -> None:
        if self.replay:
            if self.record_out_bag:
                self.out_bag.close()
            # Logger.debug(get_timing_stats())
        if self.plot_results:
            Logger.info(f'Saving results to "{self.save_dir}"')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            save_to_file(self._results, self._fusion_name, self.save_dir, self.occlusions, t_bag_start=self.t_start_bag, show_position=self.show_position, show_orientation=self.show_orientation, show_3d=self.show_3d)
        Logger.info('Stopping fusion node')