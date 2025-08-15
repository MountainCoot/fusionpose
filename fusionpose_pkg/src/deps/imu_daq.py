from bleak import BleakClient, BleakScanner, BleakGATTCharacteristic
import asyncio
import struct
import time
import numpy as np
from deps.utils.logger_util import Logger

from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Imu
import rospy 

def calc_accel(a: np.ndarray) -> np.ndarray:
    """Remap raw accelerometer measurements to Gs."""
    accel_range = 4  # Should match settings.accelRange in microcontroller code
    return a * 0.061 * (accel_range / 2) / 1000 * 9.81

def calc_gyro(g: np.ndarray) -> np.ndarray:
    """Remap raw gyro measurements to degrees per second."""
    gyro_range = 2000  # Should match settings.gyroRange in microcontroller code
    return g * 4.375 * (gyro_range / 125) / 1000 * np.pi / 180.0

def imu_from_raw_data(data: bytearray, n_packets=1) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Unpacks multiple IMUDataPacket structs from the given BLE buffer."""
    packet_size = struct.calcsize("<I3h3h")
    packets = []
    i_msg = 0
    for i in range(0, len(data), packet_size):
        if i_msg >= n_packets:
            break
        i_msg += 1
        t, ax, ay, az, gx, gy, gz = struct.unpack_from("<I3h3h", data, i)
        accel = calc_accel(np.array([ax, ay, az], dtype=np.float64))
        gyro = calc_gyro(np.array([gx, gy, gz], dtype=np.float64))
        packets.append((t / 1000, accel, gyro))
    if len(packets) != n_packets:
        print(f"Warning: Expected {n_packets} packets, but got {len(packets)}.")
    return packets

def soc_from_raw_data(data: bytearray) -> tuple[float, int]:
    """Unpacks an BatteryDataPacket struct from the given data buffer."""
    # struct BatteryDataPacket {
    # uint32_t time;
    # float voltage;
    # };
    # unpack a struct of uint32_t, float
    t, soc = struct.unpack("<If", data)
    # round up if above .5 and down if below and cast to int
    soc = round(soc)
    return t, soc

def generate_ble_config(oid_number: int) -> dict:
    # Format OID string as two-digit (e.g., 7 -> "07", 12 -> "12")
    oid_str = f"{oid_number:02}"
    uuid_suffix = oid_str + "00"

    base_uuid = "19B100{:02X}-E8F2-537E-4F6C-D104768A" + uuid_suffix

    return {
        "device_name": f"oid_{oid_str}",
        "service_uuid": base_uuid.format(0),
        "imu_char_uuid": base_uuid.format(1),
        "packet_count_uuid": base_uuid.format(2),
        "freq_uuid": base_uuid.format(3),
        "battery_char_uuid": base_uuid.format(4)
    }

class BLEDevice:
    def __init__(
            self, name: str, 
            imu_publisher: rospy.Publisher, 
            latency: float = 0, 
            frequency: float = 100,
            use_ros: bool = True) -> None:
        self.oid_name = name
        self.imu_publisher = imu_publisher
        self.use_ros = use_ros
        self.t_imu_start_abs = None # start time in ms as measured by IMU clock
        self.t_py_start_abs = None # start time in ms as measured by PC clock
        self.t_timer_sync = time.time()
        self.i_message = 0
        self.latency = latency
        
        self.sync_interval_max = 10 # seconds
        self.sync_interval_min = 1 # seconds
        self.sync_delay_thresh = 0.01 # seconds
        self.sync_interval = self.sync_interval_max
        self.delay_hist = None
        self.low_freq = False

        self.t_timer_history_append = time.time()
        self.t_corr_hist = []

        self.correction_time_ms = 0
        self.reset_delay_thresh = 0.1 # seconds

        self.acquisition_frequency = frequency
        self.packet_count = 1

        # just use oid number to generate
        oid_nr = int(name.split('_')[1])
        print(f'oid_nr: {oid_nr}')
        ble_config = generate_ble_config(oid_nr)
        self.imu_characteristic = ble_config['imu_char_uuid']
        # self.packet_char = ble_config['packet_count_uuid']
        # self.freq_char = ble_config['freq_uuid']
        self.voltage_characteristic = ble_config['battery_char_uuid']

        self.freq = 0

        self.ble_exception = False
        self.last_publish_time = None

        # create battery level publisher
        self.battery_publisher = rospy.Publisher(f'/{self.oid_name}/battery', Int32, queue_size=10)

        # create latency subscriber
        rospy.Subscriber(f'/{self.oid_name}/latency_abs', Float32, self.latency_abs_callback)
        rospy.Subscriber(f'/{self.oid_name}/latency_delta', Float32, self.latency_callback)

        # create a frequency publisher
        self.freq_publisher = rospy.Publisher(f'/{self.oid_name}/imu/frequency', Float32, queue_size=10)

        self.t_imu_abs_last = 0
        # self.t_py_abs_last = 0

    def latency_abs_callback(self, data: Float32) -> None:
        Logger.info(f'IMU "{self.oid_name}" changing latency from {self.latency*1000:.2f} ms to {data.data*1000:.2f} ms')
        print(f'IMU "{self.oid_name}" changing latency from {self.latency*1000:.2f} ms to {data.data*1000:.2f} ms')
        self.latency = data.data

    def latency_callback(self, data: Float32) -> None:
        info_str = f'IMU "{self.oid_name}" incrementing current latency of {self.latency*1000:.2f} ms by {data.data*1000:.2f} ms'
        self.latency += data.data
        info_str += f' to {self.latency*1000:.2f} ms'
        Logger.info(info_str)
        print(info_str)
    
    async def monitor_ble_async(self) -> None:
        """Monitor the BLE device and publish IMU data."""
        while True:
            try:
                counter = 0
                t_start = time.perf_counter()
                print(f'Scanning for BLE device "{self.oid_name}"...')
                device = await BleakScanner.find_device_by_name(self.oid_name, timeout=5)
                self.ble_exception = False
                if device is None:
                    print(f'Could not find device with name "{self.oid_name}". Retrying in 1 second...')
                    await asyncio.sleep(1)
                    continue

                def soc_notification_handler(_: BleakGATTCharacteristic, data: bytearray):
                    t, soc = soc_from_raw_data(data)
                    # publish battery level
                    self.battery_publisher.publish(soc)

                def imu_notification_handler(_: BleakGATTCharacteristic, data: bytearray):
                    try:
                        # ignore first 200 messages
                        if self.i_message <= 200:
                            self.i_message += 1
                            if self.i_message < 200:
                                return

                        nonlocal counter
                        nonlocal t_start
                        
                        packets = imu_from_raw_data(data)


                        for i, packet in enumerate(packets):
                            counter = self.calculate_frequency(counter, time.perf_counter() - t_start)
                            if counter == 0:
                                t_start = time.perf_counter()

                            # t_imu_abs, accel, gyro = imu_from_raw_data(data)
                            t_imu_abs, accel, gyro = packet

                            self.t_imu_abs_last = t_imu_abs

                            t_py_daq_abs = rospy.Time.now().to_sec() if self.use_ros else time.time()

                            if self.t_imu_start_abs is None and self.t_py_start_abs is None:
                                self.t_imu_start_abs = t_imu_abs
                                self.t_py_start_abs = t_py_daq_abs
                                print(f'"{self.oid_name}" First IMU message received with intial latency {self.latency*1000:.2f} ms')

                            if self.use_ros:
                                t_imu_daq_ros = t_imu_abs - self.t_imu_start_abs + self.t_py_start_abs # use relative time from IMU but absolute time from python as start
                                t_imu_daq_ros += self.correction_time_ms/1000 # adjust for drift
                                t_imu_daq_ros = rospy.Time.from_sec(t_imu_daq_ros)
                                # t_py_daq_ros = rospy.Time.from_sec(self.t_py_daq_abs) # use absolute time from python instead of IMU
                                imu_data = Imu()
                                imu_data.header.stamp = t_imu_daq_ros - rospy.Duration.from_sec(self.latency) # adjust for latency
                                if self.last_publish_time is not None and np.abs(t_imu_daq_ros.to_sec()-self.last_publish_time.to_sec()) < 0.001:
                                    print(f'"{self.oid_name}" Timestamps are too close with difference of {(t_imu_daq_ros.to_sec()-self.last_publish_time.to_sec())*1000:.3f} ms')
                                    # just skip
                                    return
                                

                                self.last_publish_time = t_imu_daq_ros

                                imu_data.angular_velocity.x = gyro[0]
                                imu_data.angular_velocity.y = gyro[1]
                                imu_data.angular_velocity.z = gyro[2]

                                imu_data.linear_acceleration.x = accel[0]
                                imu_data.linear_acceleration.y = accel[1]
                                imu_data.linear_acceleration.z = accel[2]

                                # Publish the populated Imu message
                                self.imu_publisher.publish(imu_data)


                            # only do drift correction on last packet because it is the one with the actual delay
                            if i == len(packets)-1:
                                # drift correction 
                                curr_time = time.time()

                                if curr_time - self.t_timer_history_append > 0.25:
                                    self.t_timer_history_append = time.time()
                                    delay_corrected = t_py_daq_abs-t_imu_daq_ros.to_sec()
                                    self.t_corr_hist.append(delay_corrected)

                                if curr_time - self.t_timer_sync > self.sync_interval and t_py_daq_abs > self.t_py_start_abs:
                                    self.t_timer_sync = time.time()

                                    t_corr_avg = np.mean(self.t_corr_hist)

                                    if self.delay_hist is None:
                                        self.delay_hist = np.zeros(10)
                                        print(f'Initial avg delay: {t_corr_avg*1000:.0f} ms, correcting start time once')
                                        self.t_py_start_abs += t_corr_avg
                                        t_py_daq_abs += t_corr_avg
                                        t_corr_avg = 0

                                    delay_corrected = t_py_daq_abs-t_imu_daq_ros.to_sec()

                                    self.delay_hist = np.roll(self.delay_hist, 1)
                                    self.delay_hist[0] = delay_corrected

                                    if not self.low_freq:
                                        if np.abs(np.mean(self.delay_hist)) > self.sync_delay_thresh and self.sync_interval == self.sync_interval_max:
                                            print(f'"{self.oid_name}" averaged corrected delay is above {self.sync_delay_thresh*1000} ms: {np.mean(self.delay_hist):.3f} s, setting sync interval to {self.sync_interval_min} seconds')
                                            self.sync_interval = self.sync_interval_min
                                        elif np.abs(np.mean(self.delay_hist)) < self.sync_delay_thresh and self.sync_interval == self.sync_interval_min:
                                            print(f'"{self.oid_name}" averaged corrected delay is again below {self.sync_delay_thresh*1000} ms: {np.mean(self.delay_hist):.3f} s, setting sync interval to {self.sync_interval_max} seconds')
                                            self.sync_interval = self.sync_interval_max

                                    corrected_drift_rate = (self.correction_time_ms/1000)/(t_py_daq_abs-self.t_py_start_abs)*1000*1000

                                    increment = 0

                                    debug_str = ''

                                    if self.low_freq:
                                        self.low_freq = False
                                        debug_str += f'Low frequency detected -> not adjusting delay, '
                                    
                                    else:

                                        if np.abs(t_corr_avg) < self.reset_delay_thresh and np.abs(t_corr_avg) > 0.002:
                                            increment = np.sign(t_corr_avg)*1
                                            self.correction_time_ms += increment

                                        elif np.abs(delay_corrected) > self.reset_delay_thresh:
                                            Logger.critical(f'"{self.oid_name}" corrected delay is above {self.reset_delay_thresh*1000} ms: {delay_corrected:.3f} s, resetting start times')
                                            print(f'"{self.oid_name}" Corrected delay is above {self.reset_delay_thresh*1000} ms: {delay_corrected:.3f} s, resetting start times')
                                            self.t_imu_start_abs = None
                                            self.t_py_start_abs = None
                                            self.sync_interval = self.sync_interval_max
                                            increment = -self.correction_time_ms
                                            self.correction_time_ms = 0
                                            self.delay_hist = np.zeros(10)
                                            self.last_publish_time = None
                                            self.t_corr_hist = []

                                    debug_str += f'Delay corrected: {delay_corrected*1000:.0f} ms, avg: {t_corr_avg*1000:.0f} ms, correction: {self.correction_time_ms-increment:.0f} ms + {increment:.0f} ms, drift rate: {corrected_drift_rate:.0f} ppm'
                                    debug_str = f'"{self.oid_name}" ' + debug_str
                                    print(debug_str)
                                    
                                    self.t_corr_hist = []

                    except Exception as e:
                        print(f'IMU Exception: {e}')
                        Logger.error(f'IMU Exception: {e}')

                print(f'Connecting to BLE device "{self.oid_name}"...')
                Logger.info(f'Connecting to BLE device "{self.oid_name}"...')
                disconnected_event = asyncio.Event()
                try:
                    async with BleakClient(device, disconnected_callback=lambda _: disconnected_event.set()) as client:
                        print(f'Connected to BLE device "{self.oid_name}".')
                        Logger.info(f'Connected to BLE device "{self.oid_name}".')
                        await client.start_notify(self.imu_characteristic, imu_notification_handler)
                        await client.start_notify(self.voltage_characteristic, soc_notification_handler)
                        await disconnected_event.wait()
                        print(f'Disconnected from BLE device "{self.oid_name}".')
                        Logger.warning(f'Disconnected from BLE device "{self.oid_name}".')
                except Exception as e:
                    print(f'BLE Exception: {e}')
                    print('Retrying in 1 second...')
                    Logger.error(f'BLE Exception: {e}')
                    Logger.error('Retrying in 1 second...')
                    await asyncio.sleep(1)
                finally:
                    self.i_message = 0
                    self.t_imu_start_abs = None
                    self.t_py_start_abs = None
                    self.sync_interval = self.sync_interval_max
                    self.correction_time_ms = 0
                    self.delay_hist = None
                    self.last_publish_time = None
                    self.t_corr_hist = []
                    self.t_timer_sync = time.time()
                    self.t_timer_history_append = time.time()
                    self.low_freq = False

                    
            except asyncio.CancelledError:
                try:
                    await client.disconnect()
                    print(f'Disconnected from BLE device "{self.oid_name}".')
                    Logger.warning(f'Disconnected from BLE device "{self.oid_name}".')
                except:
                    pass
                return
        
            except Exception as e:
                if not self.ble_exception:
                    print(f'BLE Exception: {e}')
                    Logger.error(f'BLE Exception: {e}. Trying again in 5 seconds...')
                self.ble_exception = True
                await asyncio.sleep(5)

    def calculate_frequency(self, counter: int, time: float, t_thresh: float = 1) -> int:
        """Calculate the frequency of a signal given the number of cycles and the time it took to complete them."""
        # every t_thresh seconds, calculate frequency
        if time > t_thresh:
            self.freq = counter / time
            # publish frequency
            self.freq_publisher.publish(self.freq)

            # log warning if frequency low
            if counter > 0 and self.freq < self.acquisition_frequency*0.9:
                Logger.warning(f'"{self.oid_name}" frequency is significantly below {self.acquisition_frequency} Hz: {self.freq:.2f} Hz')
                self.low_freq = True

            return 0
        else:
            return counter + 1
