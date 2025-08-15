import asyncio
from node import Node
import sys

USE_ROS = True
try:
    from deps.utils.logger_util import Logger
except ImportError:
    USE_ROS = False
    # append one up and then src/
    to_append = f'{sys.path[0]}/../src'
    sys.path.append(to_append) if to_append not in sys.path else None

    from deps.utils.logger_util import Logger

from deps.imu_daq import BLEDevice

from sensor_msgs.msg import Imu
from fusionpose_pkg.msg import NodeStatus

import rospy 
from node import Node

import signal

class IMUAcquisitionNode(Node):
    def __init__(self, use_ros: bool) -> None:
        super().__init__('imu_acquisition_node', use_ros=use_ros)

        # create Node status
        self.status_pub = rospy.Publisher('/node_status', NodeStatus, queue_size=1, latch=True)

        self.use_ros = use_ros
        
        imu_config = self.config['imu']
        self.device_names = imu_config.get('devices', {}).get('names', [])
        self.latency = imu_config.get('latency', 0.035)
        self.frequency = self.config.get('fusion').get('auto_calibration', {}).get('imu_freq_des', 100)

        if len(self.device_names) == 0:
            Logger.warning("No IMU devices to monitor.")
            return        

        # Create publishers for each IMU device
        self.publishers = {}
        for name in self.device_names:
            if USE_ROS:
                self.publishers[name] = rospy.Publisher('/' + name + '/imu', Imu, queue_size=1000)
            else:
                self.publishers[name] = None
        

        self.publish_node_status(is_healthy=True, is_just_started=True, info="IMU acquisition node ready", is_busy=False)

    def publish_node_status(self, is_healthy: bool, info: str, is_unrecoverable_error: bool = False, is_just_started: bool = False, is_busy: bool = False) -> None:
        status = NodeStatus()
        status.node_name = f'{self.node_name}'
        status.is_healthy = is_healthy
        status.info = info
        status.is_unrecoverable_error = is_unrecoverable_error
        status.is_just_started = is_just_started
        status.is_busy = is_busy
        self.status_pub.publish(status)

    async def run_ble(self) -> None:
        print(f'Running BLE IMU DAQ for {len(self.device_names)} devices...')
        # Start monitoring BLE devices
        tasks = [BLEDevice(name, self.publishers[name], self.latency, self.frequency, use_ros=self.use_ros).monitor_ble_async() for name in self.device_names]
        await asyncio.gather(*tasks)

    def shutdown(self) -> None:
        self.publish_node_status(
            is_healthy=False,
            info="IMU acquisition node shutting down",
            is_unrecoverable_error=True,
            is_just_started=False,
            is_busy=False
        )


def signal_handler(sig, frame):
    print("Shutdown signal received: Exiting...")
    # publish a final “down” status if the node exists
    for task in asyncio.all_tasks():
        task.cancel()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        node = IMUAcquisitionNode(use_ros=USE_ROS)
        rospy.on_shutdown(node.shutdown)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(node.run_ble())
    except asyncio.CancelledError:
        pass

    
