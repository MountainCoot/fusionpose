""" FUSION core.
Provides funtionality to fuse IMU and 6DOF (absolute) pose data using GTSAM's 
incremental smoothing and mapping based on the Bayes tree (ISAM2).
"""

from typing import Optional
from gtsam import Point3
import gtsam
import gtsam.noiseModel
import gtsam_unstable
import numpy as np
import heapq
from deps.utils.logger_util import Logger


def gtsam_pose_to_numpy(gtsam_pose: gtsam.gtsam.Pose3) -> tuple[np.ndarray, np.ndarray]:
    """Convert GTSAM pose to numpy arrays 
    (position, orientation)"""
    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()])
    # print out all members of gtsam_pose.rotation()
    quat = gtsam_pose.rotation().toQuaternion()
    orientation = np.array([quat.x(), quat.y(), quat.z(), quat.w()]) # xyzw
    return position, orientation

def numpy_pose_to_gtsam(position: np.ndarray, orientation: np.ndarray) -> gtsam.Pose3:
    """Convert numpy arrays (position, orientation)
    to GTSAM pose"""
    # print arguments that Pose3 takes
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2]), # wxyz for GTSAM
        Point3(
            position[0],
            position[1],
            position[2])) 

class GtsamFusionCore():
    """Core functions for ISAM2 fusion."""

    def __init__(self, params: dict, log: bool = True) -> None:
        """Initialize ISAM2, IMU preintegration, and set prior factors"""
        self._imu_measurements_predict = [] # IMU measurements for real-time pose prediction
        self._imu_measurements_optimize = [] # IMU measurements for pose prediction between measurement updates
        self._imu_samples = []
        self._opt_measurements = [] # measurements for optimization

        self.high_noise_poses = []

        # ISAM2 keys
        self._pose_key = gtsam.symbol('x', 0)
        self._vel_key = gtsam.symbol('v', 0)
        self._bias_key = gtsam.symbol('b', 0)

        opt_yaml = params.get('optimizer', {})

        self.setup_optimizer(opt_yaml, log=log)

        preint_yaml = params.get('preintegration', {})
        sigmas_yaml = params.get('sigmas', {})
        init_sigmas_yaml = params.get('init_sigmas', {})
        initial_state_yaml = params.get('initial_state', {})
        
        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimate = gtsam.Values()
        self._prev_pose_key = None

        self._min_imu_sample_count_for_integration = preint_yaml.pop('min_imu_integration', 1)
        assert self._min_imu_sample_count_for_integration > 0, "min_imu_integration must be greater than 0"

        # IMU preintegration
        self._pre_integration_params = gtsam.PreintegrationParams(np.asarray(preint_yaml.pop('gravity', [0, 0, -9.81])))
        ds = sigmas_yaml.pop('ds', 1.0)

        acc_cov = np.eye(3) * np.power(ds*np.array(sigmas_yaml.pop('acc', [3e-2, 3e-2, 3e-2])), 2)
        gyro_cov = np.eye(3) * np.power(ds*np.array(sigmas_yaml.pop('gyro', [1e-3, 1e-3, 1e-3])), 2)

        self._pre_integration_params.setAccelerometerCovariance(acc_cov)
        self._pre_integration_params.setGyroscopeCovariance(gyro_cov)
        self._pre_integration_params.setIntegrationCovariance(np.eye(3) * np.power(sigmas_yaml.pop('integration', 0.01), 2)) # [m/s^2]
        self._pre_integration_params.setUse2ndOrderCoriolis(preint_yaml.pop('use_2nd_order_coriolis', False))
        self._pre_integration_params.setOmegaCoriolis(np.array(preint_yaml.pop('omega_coriolis', [0, 0, 0])))
        self._pre_integration_params.setBodyPSensor(numpy_pose_to_gtsam(params['b2s_pos'], params['b2s_ori']))
        self._imu_accum = gtsam.PreintegratedImuMeasurements(self._pre_integration_params)

        # initial state
        self._current_time = initial_state_yaml.pop('t', 0)
        
        self._current_pose = numpy_pose_to_gtsam(initial_state_yaml.pop('pos'), initial_state_yaml.pop('ori'))
        self._predicted_pose = self._current_pose
        self._current_vel = np.asarray(initial_state_yaml.pop('vel', [0, 0, 0]))
        self._current_bias = gtsam.imuBias.ConstantBias(
            np.asarray(initial_state_yaml.get('bias', {}).pop('acc', [0, 0, 0])),
            np.asarray(initial_state_yaml.get('bias', {}).pop('gyro', [0, 0, 0]))
        )

        self._current_bias_avg = self._current_bias
        self._current_bias_n_avg = preint_yaml.pop('bias_n_avg', 1)

        # store for predict
        self._last_opt_time = self._current_time
        self._last_opt_pose = self._current_pose
        self._last_opt_vel = self._current_vel
        self._last_opt_bias = self._current_bias

        # uncertainty of the initial state
        self._sigma_init_pose = gtsam.noiseModel.Diagonal.Sigmas(np.array(init_sigmas_yaml.pop('pose', [10e5, 10e5, 10e5, 10e5, 10e5, 10e5])))
        self._sigma_init_vel = gtsam.noiseModel.Diagonal.Sigmas(np.array(init_sigmas_yaml.pop('vel', [10e5, 10e5, 10e5])))
        self._sigma_init_bias = gtsam.noiseModel.Diagonal.Sigmas(np.array(init_sigmas_yaml.pop('bias', [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])))

        # measurement noise
        self._pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((
            sigmas_yaml.pop('pose_rot', [1*np.pi/180, 1*np.pi/180, 1*np.pi/180]), # [rad]
            sigmas_yaml.pop('pose_pos', [0.001, 0.001, 5*0.001])
        )))

        self.acc_bias_evol = ds*np.array(sigmas_yaml.pop('acc_bias_evolution', [2e-2, 2e-2, 2e-2])) # [m/s^2]
        self.gyro_bias_evol = ds*np.array(sigmas_yaml.pop('gyro_bias_evolution', [1e-3, 1e-3, 1e-3])) # [rad/s]

        _bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((self.acc_bias_evol, self.gyro_bias_evol)))

        self._bias_noise = _bias_noise
        
        if log:
            # # print summary of all set sigmas
            # summary_str = f'\n{30*"-"} SIGMAS {30*"-"}\n'
            # init_len = len(summary_str.strip('\n'))
            # summary_str += f'Acceleration: {self._pre_integration_params.getAccelerometerCovariance().diagonal()}\n'
            # summary_str += f'Gyroscope: {self._pre_integration_params.getGyroscopeCovariance().diagonal()}\n'
            # summary_str += f'Accelerometer bias evolution: {_bias_noise.sigmas()[:3]}\n'
            # summary_str += f'Gyroscope bias evolution: {_bias_noise.sigmas()[3:]}\n'
            # summary_str += f'Pose rotation: {self._pose_noise.sigmas()[:3]}\n'
            # summary_str += f'Pose position: {self._pose_noise.sigmas()[3:]}\n'
            # summary_str += f'{init_len * "-"}\n'
            # Logger.info(summary_str)

            def nested_dict_values(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        yield from nested_dict_values(v)
                    else:
                        yield (k, v)

            # also print name of param_dict, alongside unused parameters
            for idx, param_dict in enumerate([sigmas_yaml, init_sigmas_yaml, preint_yaml, opt_yaml, initial_state_yaml]):
                # get all values (nested)   
                param_dict_str = ['sigmas', 'init_sigmas', 'preintegration', 'optimizer', 'initial_state'][idx]
                for k, v in nested_dict_values(param_dict):
                    Logger.warning(f'Unused parameter in "{param_dict_str}": "{k}" = {v}')


        # variabes to use for live smoothing
        self.result_prev = None
        self.keys_history = []
        self.imu_samples_per_pose = {}
        self.oldest_key_time = None


        self.set_initial_state(
            self._current_time, 
            self._pose_key, 
            self._current_pose, 
            self._vel_key, 
            self._current_vel, 
            self._bias_key, 
            self._current_bias
        )

    def setup_optimizer(self, opt_yaml: dict, log: bool = False) -> dict:
        opt_method = opt_yaml.pop('method', 'FIXED_LAG')
        self.fixed_lag = max(opt_yaml.pop('fixed_lag_time', 1/20*3), 0.0) # cannot be negative
        
        self.optimize_func = None
        self.use_timestamps = False

        if opt_method == 'ISAM2':
            isam2_yaml = opt_yaml.pop('isam2', {})
            # ISAM2 initialization
            isam2_params = gtsam.ISAM2Params()
            isam2_params.setRelinearizeThreshold(isam2_yaml.pop('relinearize_threshold', 0.01))
            factorization = isam2_yaml.pop('factorization', 'QR')
            assert factorization in ['QR', 'CHOLESKY'], "Factorization must be either 'QR' or 'CHOLESKY'"
            isam2_params.setFactorization(factorization)
            isam2_params.relinearizeSkip = isam2_yaml.pop('relinearize_skip', 1)
            self._isam2 = gtsam.ISAM2(isam2_params)
            self.optimize_func = self._isam2_update
            if log:
                Logger.info(f'Initializing ISAM2 for optimization with fixed lag of {self.fixed_lag:.2f} seconds')
        
        elif opt_method == 'FIXED_LAG':
            self._timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
            self._smoother_batch = gtsam_unstable.BatchFixedLagSmoother(self.fixed_lag)
            self.optimize_func = self._fixed_lag_update
            self.use_timestamps = True
            if log:
                Logger.info(f'Initializing Fixed-lag smoothing for optimization with fixed lag of {self.fixed_lag:.2f} seconds')
        else:
            raise ValueError(f'Unknown optimization method: "{opt_method}" (must be either "ISAM2" or "FIXED_LAG")')

        return opt_yaml

    def set_initial_state(
            self,
            time: float,
            pose_key: int,
            pose: gtsam.gtsam.Pose3,
            vel_key: int,
            vel: np.ndarray,
            bias_key: int,
            bias: gtsam.gtsam.imuBias.ConstantBias,
            sigma_init_pose: Optional[gtsam.noiseModel.Base] = None,
            sigma_init_vel: Optional[gtsam.noiseModel.Base] = None,
            sigma_init_bias: Optional[gtsam.noiseModel.Base] = None
    ) -> None:

        if sigma_init_pose is None:
            sigma_init_pose = self._sigma_init_pose
        if sigma_init_vel is None:
            sigma_init_vel = self._sigma_init_vel
        if sigma_init_bias is None:
            sigma_init_bias = self._sigma_init_bias
        prior_pose_factor = gtsam.PriorFactorPose3(
            pose_key, 
            pose, 
            sigma_init_pose)
        self._graph.add(prior_pose_factor)
        prior_vel_factor = gtsam.PriorFactorVector(
            vel_key,
            vel,
            sigma_init_vel)
        self._graph.add(prior_vel_factor)
        prior_bias_factor = gtsam.PriorFactorConstantBias(
            bias_key, 
            bias,
            sigma_init_bias)
        self._graph.add(prior_bias_factor)
        # initial estimates
        self._initial_estimate.insert(pose_key, pose)
        self._initial_estimate.insert(vel_key, vel)
        self._initial_estimate.insert(bias_key, bias)

        if self.use_timestamps:
            self._timestamps.insert((pose_key, time))
            self._timestamps.insert((vel_key, time))
            self._timestamps.insert((bias_key, time))

    def add_absolute_pose_measurement(self, time: float, position: np.ndarray, orientation: np.ndarray, high_imu_noise=False) -> Optional[tuple[list, list]]:
        heapq.heappush(self._opt_measurements, (time, 'absolute_pose_', position, orientation))
        try:
            return self._abs_pose_update(heapq.heappop(self._opt_measurements), high_imu_noise=high_imu_noise)
        except RuntimeError:
            return None

    def add_imu_measurement(self, time: float, linear_acceleration: np.ndarray, angular_velocity: np.ndarray, dt: float, high_acc_var=False) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        # Add measurement
        heapq.heappush(self._imu_measurements_predict, (time, linear_acceleration, angular_velocity, dt, high_acc_var))
        heapq.heappush(self._imu_measurements_optimize, (time, linear_acceleration, angular_velocity, dt, high_acc_var))
        if self._pose_key not in self.imu_samples_per_pose:
            self.imu_samples_per_pose[self._pose_key] = []
        self.imu_samples_per_pose[self._pose_key].append((time, linear_acceleration, angular_velocity, dt, high_acc_var))
        try:
            return self._imu_predict()
        except RuntimeError:
            return None

    def _abs_pose_update(self, measurement: tuple, high_imu_noise=False) -> tuple[list, list]:
        """Trigger update based on the measurement type"""
        meas_time = measurement[0]
        meas_type = measurement[1]

        imu_samples = []
        while True:
            if not self._imu_measurements_optimize:
                break
            imu_sample = heapq.heappop(self._imu_measurements_optimize)
            if imu_sample[0] < meas_time:
                imu_samples.append(imu_sample)
            else:
                break
        if len(imu_samples) < self._min_imu_sample_count_for_integration:
            # Must have at least _min_imu_sample_count_for_integration samples for integration
            for imu_sample in imu_samples:
                heapq.heappush(self._imu_measurements_optimize, imu_sample)
            # Logger.warning(f'Not enough IMU samples ({len(imu_samples)}) since last measurement update. Ignoring measurement.')
            return (None, None)
        # new pose & velocity estimate
        self._pose_key += 1
        self._vel_key += 1
        self._bias_key += 1
        # add keys to back of history
        heapq.heappush(self.keys_history, (meas_time, (self._pose_key, self._vel_key, self._bias_key)))
        if 'absolute_pose' in meas_type:
            pos, quat = measurement[2], measurement[3]
            pose = numpy_pose_to_gtsam(pos, quat)

            pose_noise = self._pose_noise
            pose_noise = gtsam.noiseModel.Robust.Create(
                                    gtsam.noiseModel.mEstimator.Huber(k=1),
                                    pose_noise)
        
            if high_imu_noise:
                heapq.heappush(self.high_noise_poses, self._pose_key)
                # make zero noise for pose
                pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((
                    np.array([0, 0, 0]),
                    np.array([0, 0, 0])
                )))

            # add pose factor
            pose_factor = gtsam.PriorFactorPose3(
                self._pose_key,
                pose,
                pose_noise
            )
            self._graph.add(pose_factor)
            
        else:
            raise ValueError(f'Unknown measurement type: {meas_type}')


        # optimize
        return self._optimize(meas_time, imu_samples)

    def _imu_predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict with IMU"""
        if self._current_time > self._last_opt_time: # when new optimized pose is available
            # store state
            self._last_opt_time = self._current_time
            self._last_opt_pose = self._current_pose
            self._last_opt_vel = self._current_vel
            self._last_opt_bias = self._current_bias
            # start new integration onwards from optimization time
            self._imu_accum.resetIntegration()
            new_imu_samples = []
            for sample in self._imu_samples:
                if sample[0] > self._last_opt_time:
                    self._imu_accum.integrateMeasurement(sample[1], sample[2], sample[3])
                    new_imu_samples.append(sample)
            self._imu_samples = new_imu_samples
        else: # when no new optimized pose is available
            self._last_opt_bias = self._current_bias_avg
        # get new sample from the queue
        (time, linear_acceleration, angular_velocity, dt, high_acc_var) = heapq.heappop(self._imu_measurements_predict)
        # store sample for re-integration after new measurement is available
        self._imu_samples.append((time, linear_acceleration, angular_velocity, dt, high_acc_var))
        # integrate
        self._imu_accum.integrateMeasurement(linear_acceleration, angular_velocity, dt)
        bias = self._last_opt_bias
        # predict pose
        predicted_nav_state = self._imu_accum.predict(
            gtsam.NavState(self._last_opt_pose, self._last_opt_vel), bias)
        self._predicted_pose = (predicted_nav_state.pose(), time)
        # return pose
        pos, ori = gtsam_pose_to_numpy(predicted_nav_state.pose())
        return (
            pos,
            ori,
            predicted_nav_state.velocity(),
            bias.accelerometer(),
            bias.gyroscope()
        )

    def _imu_update(self, imu_samples: list) -> gtsam.NavState:
        """Create new IMU factor and perform bias evolution"""
        # reset integration done for the prediction
        imu_accum = gtsam.PreintegratedImuMeasurements(self._pre_integration_params)

        # preintegrate IMU measurements up to meas_time
        for imu_sample in imu_samples:
            imu_accum.integrateMeasurement(imu_sample[1], imu_sample[2], imu_sample[3])

        # predict pose at meas_time for the optimization
        last_opt_pose = self._current_pose
        last_opt_vel = self._current_vel
        last_opt_bias = self._current_bias
        # predict the pose using the last optimized state and current bias estimate
        predicted_nav_state = imu_accum.predict(
            gtsam.NavState(last_opt_pose, last_opt_vel), last_opt_bias)
        # add IMU factor
        imu_factor = gtsam.ImuFactor(
            self._pose_key - 1, self._vel_key - 1, 
            self._pose_key, self._vel_key, 
            self._bias_key, imu_accum)
        self._graph.add(imu_factor)
        return predicted_nav_state

    def _optimize(self, meas_time: float, imu_samples: list) -> tuple[list, list]:
        """Perform optimization"""

        # perform IMU preintegration until meas_time
        predicted_nav_state = self._imu_update(imu_samples)
        # add current pose to initial estimates
        self._initial_estimate.insert(self._pose_key, predicted_nav_state.pose())
        self._initial_estimate.insert(self._vel_key, predicted_nav_state.velocity())
        self._initial_estimate.insert(self._bias_key, self._current_bias)

        if self.use_timestamps:
            self._timestamps.insert((self._pose_key, meas_time))
            self._timestamps.insert((self._vel_key, meas_time))
            self._timestamps.insert((self._bias_key, meas_time))

        ratio_high_acc_var = len([imu_sample for imu_sample in imu_samples if imu_sample[4]])/len(imu_samples)
        # allow for high noise in bias evolution
        if ratio_high_acc_var > 0:
            # high_acc_var = any([imu_sample[4] for imu_sample in imu_samples])
            # print(f'% of samples with high noise: {100*ratio_high_acc_var:.2f}%')
            fac = 40
            self._bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((self.acc_bias_evol*fac, self.gyro_bias_evol*fac)))
        else:
            self._bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((self.acc_bias_evol, self.gyro_bias_evol))) 

        # add factor for bias evolution
        bias_factor = gtsam.BetweenFactorConstantBias(
            self._bias_key - 1, self._bias_key, gtsam.imuBias.ConstantBias(), self._bias_noise)
            
        self._graph.add(bias_factor)
        
        smoothened_pose_result, smoothened_imu_result = self.check_for_release(meas_time)

        result = self.optimize_func()

        if result:
            # update current state
            self._current_time = meas_time
            self._current_pose = result.atPose3(self._pose_key)
            self._current_vel = result.atVector(self._vel_key)
            self._current_bias = result.atConstantBias(self._bias_key)
            # assign average bias 
            avg_accel_bias = self._current_bias_avg.accelerometer() * (self._current_bias_n_avg - 1) / self._current_bias_n_avg + self._current_bias.accelerometer() / self._current_bias_n_avg
            avg_gyro_bias = self._current_bias_avg.gyroscope() * (self._current_bias_n_avg - 1) / self._current_bias_n_avg + self._current_bias.gyroscope() / self._current_bias_n_avg
            self._current_bias_avg = gtsam.imuBias.ConstantBias(avg_accel_bias, avg_gyro_bias)

        self.result_prev = result
        # Logger.info(f'# smoothened poses: {len(smoothened_imu_result)}') if smoothened_imu_result else None
        return smoothened_pose_result, smoothened_imu_result

    def check_for_release(self, meas_time: float, pose_lost=False, debug=False) -> tuple[tuple[list, list], bool] | tuple[list, list]:
        smoothened_imu_result = []
        smoothened_pose_result = []
        lost_message = False
        if self.result_prev:
            while True and self.keys_history:
                oldest_keys = heapq.heappop(self.keys_history)
                if meas_time - oldest_keys[0] <= self.fixed_lag:
                    self.oldest_key_time = oldest_keys[0]
                    heapq.heappush(self.keys_history, oldest_keys) # put back
                    break
                else:
                    if debug:
                        try:
                            Logger.debug(f'Age is {(meas_time - oldest_keys[0])*1000:.0f} ms, has {len(self.imu_samples_per_pose[oldest_keys[1][0]])} IMU samples')
                        except KeyError:
                            Logger.debug(f'Age is {(meas_time - oldest_keys[0])*1000:.0f} ms, has no IMU samples')
                    pose_key, vel_key, bias_key = oldest_keys[1]
                    try:
                        pose, vel, bias = self.result_prev.atPose3(pose_key), self.result_prev.atVector(vel_key), self.result_prev.atConstantBias(bias_key)
                    except RuntimeError as e:
                        if pose_key in self.high_noise_poses:
                            self.high_noise_poses.remove(pose_key)
                        Logger.error(f'Error in accessing pose, vel, bias: {e}')
                        Logger.debug(f'Age of pose: {(meas_time - oldest_keys[0])*1000:.0f} ms')
                        Logger.debug(f'{len(self.keys_history)=}')
                        Logger.debug(f'{100*"-"}\n{self.result_prev}\n{100*"-"}')
                        Logger.debug(self.optimize_func())
                        continue
                    pos, ori = gtsam_pose_to_numpy(pose)
                    smoothened_pose_result_i = (oldest_keys[0], (pos, ori, vel, bias.accelerometer(), bias.gyroscope()))
                    if pose_key in self.high_noise_poses:
                        self.high_noise_poses.remove(pose_key)
                        # temporarilly allow for high noise in bias evolution
                        self._bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((self.acc_bias_evol*20, self.gyro_bias_evol*20)))
                        t_pose = oldest_keys[0] # only add optimized camera pose
                        smoothened_imu_result_i = [(t_pose, (pos, ori, vel, bias.accelerometer(), bias.gyroscope()))]
                        _ = self.imu_samples_per_pose.pop(pose_key, []) # remove from imu samples
                    else:
                        smoothened_imu_result_i = self.get_smooth_poses(pose, vel, bias, pose_key)
                        self._bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((self.acc_bias_evol, self.gyro_bias_evol)))

                    lim_len = 1
                    if len(self.keys_history) >= lim_len: # only extend if key history has certain length, indicating that results have been smoothed
                        smoothened_pose_result.append(smoothened_pose_result_i)
                        smoothened_imu_result.extend(smoothened_imu_result_i)
                        if len(self.keys_history) == lim_len:
                            lost_message = True

        if pose_lost:
            return (smoothened_pose_result, smoothened_imu_result), lost_message
        else:
            return smoothened_pose_result, smoothened_imu_result

    def get_smooth_poses(self, pose: gtsam.gtsam, vel: np.ndarray, bias: gtsam.gtsam.imuBias.ConstantBias, pose_key: int) -> list:
        smoothened_result = []
        # get imu samples
        imu_samples = self.imu_samples_per_pose.pop(pose_key, [])
        # create new imu preintegration
        imu_accum = gtsam.PreintegratedImuMeasurements(self._pre_integration_params)
        for imu_sample in imu_samples:
            imu_accum.integrateMeasurement(imu_sample[1], imu_sample[2], imu_sample[3])
            new_state = imu_accum.predict(gtsam.NavState(pose, vel), bias)
            pos, ori = gtsam_pose_to_numpy(new_state.pose())
            new_state_np = (pos, ori, new_state.velocity(), bias.accelerometer(), bias.gyroscope())
            smoothened_result.append((imu_sample[0], new_state_np))
        return smoothened_result

    def _isam2_update(self) -> gtsam.gtsam.Values:
        """ISAM2 update and pose estimation""" 
        result = None
        try:
            self._isam2.update(self._graph, self._initial_estimate)
            result = self._isam2.calculateEstimate()
        except RuntimeError as e:
            print(f'Runtime error in optimization: {e}')
            raise e
        except IndexError as e:
            print(f'Index error in optimization: {e}')
            raise e
        except TypeError as e:
            print(f'Type error in optimization: {e}')
            raise e
        # # reset
        self._graph.resize(0)
        self._initial_estimate.clear()
        return result

    def _fixed_lag_update(self) -> gtsam.gtsam.Values:
        """Fixed-lag smoothing update"""
        result = None
        try:
            self._smoother_batch.update(self._graph, self._initial_estimate, self._timestamps)
            result = self._smoother_batch.calculateEstimate()

            self._graph.resize(0)
            self._initial_estimate.clear()
            self._timestamps.clear()

        except RuntimeError as e:
            print(f'ERROR: Runtime error in optimization: {e}, resetting')
            Logger.error(f'Runtime error in optimization: {e}, resetting')
            Logger.debug(f'{100*"-"}\n{self._graph}\n{100*"-"}')
            Logger.debug(f'{100*"-"}\n{self._initial_estimate}\n{100*"-"}')
            raise e
        
        return result
    