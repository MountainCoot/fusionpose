from typing import Optional
import rospy
from geometry_msgs.msg import PoseStamped
from tf import transformations

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped

import random

import time
import sys

from copy import deepcopy

import numpy as np
np.set_printoptions(precision=4, suppress=True)

from copy import deepcopy

usecore = True
try:
    import deps.fusion_core as gtsam_fusion_core
except:
    usecore = False
    import sys
    # append one up and then src/
    to_append = [f'{sys.path[0]}/../src']
    for p in to_append:
        sys.path.append(p) if p not in sys.path else None
    import deps.fusion_core as gtsam_fusion_core
    # overwrite rospy.get_param
    def get_param(_, default):
        return default
    rospy.get_param = get_param
    rospy.spin = lambda: None


from deps.utils.logger_util import Logger
from base_fusion_node import BaseFusionNode

class FusionNode(BaseFusionNode):
    def __init__(self) -> None:
        super().__init__(node_name=f'fusion_node_{random.randint(0, 1000)}', usecore=usecore)       

        self.publisher = rospy.Publisher(f'{self.oid_name}/fused', PoseStamped, queue_size=100)
        self.publisher_smooth = rospy.Publisher(f'{self.oid_name}/fused_smooth', PoseStamped, queue_size=100)

    def initialize_fusion_core(self, t_init: rospy.Time, init_pos: np.ndarray, init_ori: np.ndarray) -> None:
        if not self.fusion_core_initialized:
            print(f'"{self.oid_name}": Setting up fusion core...')
        fusion_params = deepcopy(self.config['fusion'])
        ori_quat = transformations.quaternion_from_matrix(self.T_ItoA)
        fusion_params['b2s_ori'] = list(ori_quat)
        fusion_params['b2s_pos'] = list(self.T_ItoA[0:3, 3])

        if 'initial_state' not in fusion_params:
            fusion_params['initial_state'] = {}

        fusion_params['initial_state']['t'] = t_init
        fusion_params['initial_state']['pos'] = list(np.round(init_pos, 4))
        fusion_params['initial_state']['ori'] = list(init_ori)

        if self._fusion_core is not None: # indicating it is a reset
            if 'bias' not in fusion_params['initial_state']:
                fusion_params['initial_state']['bias'] = {}
            fusion_params['initial_state']['bias']['gyro'] = self._fusion_core._current_bias.gyroscope()
            fusion_params['initial_state']['bias']['acc'] = self._fusion_core._current_bias.accelerometer()

        # Start fusion core
        self._fusion_core = gtsam_fusion_core.GtsamFusionCore(fusion_params, log=not self.fusion_core_initialized)

        self.fusion_core_initialized = True
        self.fusion_core_reset = False

    def process_imu_message(self, msg: Imu, pose_lost=False) -> None:
        """Handle IMU message"""
        if not self.fusion_core_initialized:
            return
        t_imu_msg = msg.header.stamp.to_sec()
        if pose_lost and self._last_pose:
            return
        
        if t_imu_msg < self.last_t_processed:
            Logger.error(f'IMU message at {t_imu_msg} s is older than last processed message at {self.last_t_processed} s')
            print(f'ERROR: IMU message at {t_imu_msg} s is older than last processed message at {self.last_t_processed} s')
            return
        
        self.last_t_processed = t_imu_msg

        # Convert to numpy
        lin_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        ang_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])


        if self._last_imu:
            dt = t_imu_msg - self._last_imu[0]
            # IMU update
            if dt == 0:
                Logger.critical(f'IMU message at {t_imu_msg} s: dt is 0, should not happen')
                return
            
            res = self._fusion_core.add_imu_measurement(t_imu_msg, lin_acc, ang_vel, dt)
            if not res:
                Logger.error(f'Fusion "{self.oid_name}": IMU prediction failed.')
                return None
            else:
                world_pos, world_ori, vel, acc_bias, gyr_bias = res

            # warn if abs(accel) above 39 or abs(gyro) above 39 because this is limit of the sensor
            if np.any(np.abs(lin_acc) > 39) or np.any(np.abs(ang_vel) > 39): # TODO: define externally
                Logger.warning(f'Fusion "{self.oid_name}": IMU message at {t_imu_msg} s: IMU measurements saturated: acc: {np.round(lin_acc, 3)}, gyro: {np.round(ang_vel, 3)}')

            world_pos, world_ori = self.transform_imu_into_marker(world_pos, world_ori)
            output_pos, output_ori = self.transform_pose(world_pos, world_ori, self.world_frame, self.output_pose_frame)

            # publish pose
            if usecore:
                self.publish(msg.header.stamp, self.output_pose_frame, output_pos, output_ori, self.publisher)
            if self.record_out_bag:
                self.publish(msg.header.stamp, self.output_pose_frame, output_pos, output_ori, bag=self.out_bag, topic=f'/{self.oid_name}/fused')
            # data for plots
            if self.plot_results:
                # store input
                self._results['IMU'].append(
                    np.concatenate((np.array([t_imu_msg, dt]), lin_acc, ang_vel), axis=0))
                self._results[self._fusion_name].append(
                    np.concatenate((np.array([t_imu_msg]), output_pos, output_ori, vel, acc_bias, gyr_bias), axis=0))
            
            # print biases
            report_interval = 30
            if time.time() - self.timer > report_interval:
                Logger.debug(f'Fusion "{self.oid_name}" biases: acc: {np.round(acc_bias, 2)}, gyr: {np.round(gyr_bias, 2)}')
                self.timer = time.time()

        self._last_imu = (t_imu_msg, lin_acc, ang_vel)

    def process_pose_message(self, msg: PoseStamped, return_input=False) -> None:
        """Handle 6DOF pose message"""
        cam_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        cam_ori = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])    

        t_pose_msg = msg.header.stamp.to_sec()        
        frame_id = msg.header.frame_id

        # transform from marker to imu frame
        cam_pos, cam_ori = self.transform_pose(cam_pos, cam_ori, frame_id, self.master_pose_frame, allow_missing=True)
        if cam_pos is None and cam_ori is None:
            return
        
        # change coordinate system from marker into imu frame
        cam_pos, cam_ori = self.transform_marker_into_imu(cam_pos, cam_ori)        
        
        world_pos, world_ori = self.transform_pose(cam_pos, cam_ori, self.master_pose_frame, self.world_frame, allow_missing=True)
        if world_pos is None and world_ori is None:
            return
        
        if return_input:
            return (t_pose_msg, world_pos, world_ori)
        
        if self.plot_results:
            cam_pos, cam_ori = self.transform_imu_into_marker(cam_pos, cam_ori)
            pos, ori = self.transform_pose(cam_pos, cam_ori, self.master_pose_frame, self.output_pose_frame)
            ori_euler = np.asarray(transformations.euler_from_quaternion(ori)) / np.pi * 180.
            self._results['OTS'].append(
                np.concatenate((np.array([t_pose_msg]), pos, ori_euler), axis=0))
            frame_id_plot = frame_id if frame_id in self.get_active_frames() else frame_id + '_inactive'
            self._results['OTS_frame'].append(frame_id_plot)
        
        self.last_t_processed = t_pose_msg

        if not self.fusion_core_initialized or self.fusion_core_reset:
            self.init_pose = (t_pose_msg, world_pos, world_ori)
            self.initialize_fusion_core(t_pose_msg, world_pos, world_ori)

        # change noise if multiple cameras are active
        result = self._fusion_core.add_absolute_pose_measurement(
            t_pose_msg, 
            world_pos, 
            world_ori, 
            high_imu_noise=False, 
        )

        if result is None:
            Logger.error(f'Fusion "{self.oid_name}": Absolute pose measurement failed.')
            return

        if result[0] is None and len(self.get_active_frames()) == 1: # add camera pose to result if only one camera is active and no result
            result = ([], [(t_pose_msg, (world_pos, world_ori, np.nan*np.ones(3), np.nan*np.ones(3), np.nan*np.ones(3)))])

        self._last_pose = (t_pose_msg, world_pos, world_ori, frame_id)

        self.process_smooth_result(result)

    def process_smooth_result(self, result: Optional[tuple]) -> None:

        if result is not None:
            (sm_pose, sm_imu) = result
            # publish each pose in the smoothed trajectory with dt of their messages
            for t, (p_world, o_world, v_world, ba, bg) in sm_imu:
                p_world, o_world = self.transform_imu_into_marker(p_world, o_world)
                pos, ori = self.transform_pose(p_world, o_world, self.world_frame, self.output_pose_frame)
                if usecore:
                    self.publish(rospy.Time(t), self.output_pose_frame, pos, ori, self.publisher_smooth)
                if self.record_out_bag:
                    self.publish(rospy.Time(t), self.output_pose_frame, pos, ori, bag=self.out_bag, topic=f'/{self.oid_name}/fused_smooth')

            if self.plot_results:
                if 'ISAM2' not in self._results:
                    self._results['ISAM2'] = []
                for t, (p_world, o_world, v_world, ba, bg) in sm_pose:
                    p_world, o_world = self.transform_imu_into_marker(p_world, o_world)
                    pos, ori = self.transform_pose(p_world, o_world, self.world_frame, self.output_pose_frame)
                    self._results['ISAM2'].append(
                        np.concatenate((np.array([t]), pos, ori, ba, bg, v_world), axis=0)
                    )

                if 'IMU_ISAM2' not in self._results:
                    self._results['IMU_ISAM2'] = []
                for t, (p_world, o_world, v_world, ba, bg) in sm_imu:
                    p_world, o_world = self.transform_imu_into_marker(p_world, o_world)
                    pos, ori = self.transform_pose(p_world, o_world, self.world_frame, self.output_pose_frame)
                    self._results['IMU_ISAM2'].append(
                        np.concatenate((np.array([t]), pos, ori, ba, bg, v_world), axis=0)
                    )


def main():
    """Main"""
    try:
        node = FusionNode()
        node.run()
    except Exception as e:
        Logger.exception("message")
        try:
            node.stop() if not node.replay else None
        except UnboundLocalError:
            pass
        raise e
    except KeyboardInterrupt:
        Logger.info("Keyboard interrupt")
        print("Keyboard interrupt")
        try:
            node.stop()
        except UnboundLocalError:
            pass
          

    rospy.loginfo("Exiting..")

if __name__ == '__main__':
    main()
