# vicon2gt Calibration
Based on the work of [Geneva et al.](https://github.com/rpng/vicon2gt), we have incorporated a calibration framework that allows IMU and camera calibration directly within our tracking pipeline. In basic terms, the `vicon2gt` is run by starting the `calibration_node` (refer [to the main readme](../README.md#running-the-tracking-pipeline)) and takes a set of 6D camera poses and IMU measurements to calculate the following quantities:

- `R_BtoI`: Rotation from the ArUco marker frame to the IMU frame
- `p_BtoI`: Translation from the ArUco marker frame to the IMU frame
- `R_GtoV`: Rotation from the gravity frame to the camera frame
- `t_off_vicon_to_imu`: Time offset in seconds from the camera to the IMU

We use the calibration pipeline for three tasks during the tracking process, which are explained in the following sections.

## Gravity Vector Calibration (a) / Marker to IMU Frame Calibration (`T_MtoI`) (b)

The gravity vector and marker to IMU frame calibration are obtained in similar ways. Where specified, follow either **(a)** for the gravity vector or **(b)** for `T_MtoI`, respectively.

**(a)** For the gravity vector calibration, we are interested in the `R_GtoV` rotation matrix, which transforms the gravity vector from the gravity frame to the camera frame. 

**(b)** For the marker to IMU frame calibration, we are interested in the `R_BtoI` rotation matrix and `p_BtoI` translation vector, which transform the ArUco marker frame to the IMU frame.

To perform calibration, follow these steps:

1. Prepare at least one fiducial object and its corresponding `json` file (refer to [the main readme](../README.md#preparing-the-tracking-system-for-first-use)).  
**(a)** If you have not calibrated `T_MtoI` yet in the `json` file, you can simply set it to the identity transformation:  
**(b)** Set `T_MtoI` initially to the identity transformation:
    ```bash
    "T_MtoI": {"r": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [[0, 0, 0]]}
    ```
2. **(b)** If you have already calibrated your gravity vector, you can skip this step! Otherwise follow instructions for **(a)**:  
**(a)** Create a gravity transform file with the identity matrix in the [`gravity_transforms`](../fusionpose_pkg/src/files/gravity_transforms) directory that ideally corresponds to your camera name. E.g. if your camera is called `baumer1`, create a file called `baumer1.txt` with the following contents: 
    ```
    np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
    ```
3. Modify the [`cam_config.yaml`](../fusionpose_pkg/config/cam_config.yaml) file:
    - Specify the path of the gravity transform file (`gravity_transform_file`) in the parameter section of the camera you want to calibrate
    - Ensure `launch_fusion` of the `marker_tracker` section is set to `true` such that the fusion nodes are launched during calibration.
    - Ensure that `enabled` of the `auto_calibration` section (as part of `fusion` section) is set to `true`. This activates the auto-calibration process and ensures that IMU and pose data are continuously being recorded. You can also change the other `auto_calibration` settings, they are further detailed [below](#imu-camera-time-offset-calibration-auto-calibration)
4. Launch the `marker_tracker_node`, `imu_acquisition_node`, and the `calibration_node` (refer to [the main readme](../README.md#running-the-tracking-pipeline)). Make sure that BLE connection to the fiducial object is established.
5. You can make sure that everything is working properly by checking the ROS topics for the camera and IMU data. E.g. if your OID is `0`, check the frequency of topics
    ```bash
    rostopic hz /oid_0/pose
    rostopic hz /oid_0/imu
    ```
    while the fiducial object is visible to the camera. You should see the camera frame rate and IMU frequency.
6. Start moving the fiducial object around in the camera's field of view to collect diverse poses and IMU readings (e.g., moving it up and down, left and right, and rotating it). Make sure the camera never loses sight of the marker, or otherwise the calibration will restart. [Monitor the logs of the tracking software](../README.md#monitoring). The debug information will log if the auto-calibration has been aborted. After continuous recording for the time specified in the `cam_config.yaml` file (`bag_duration_s`), calibration will be automatically performed. Watch out for an `info` log statement that says that calibration has been completed successfully. If anything went wrong an `error` log message is shown, and you have to continue moving the fiducial object until another successful calibration occurs.
7. Upon successful calibration, the results are both displayed in the terminal of the `calibration_node` and saved to the [`oid_autocalib/vicon2gt`](../fusionpose_pkg/src/files/oid_autocalib/vicon2gt) directory with the naming convention `oid_<OID>_autocalibration_<camera_name>_vicon2gt_info.txt`.  
**(a)** Replace the identity transformation in your `gravity_transform_file` by the `R_GtoV` rotation matrix obtained from the calibration results.  
**(b)** Replace the identity transformation `R` and the zero translation vector `t` in your `T_MtoI` transformation with `R_BtoI` and `t_BtoI` obtained from the calibration results.


**Some additional notes:**
- `bag_duration_s` is set to 20 seconds by default. To ensure good and stable calibrations, it might be advisable to increase this duration.
- **(a)** Recalibration of the gravity vector is recommended if there are significant changes in the setup or if the calibration results seem inaccurate (high bias fluctuations during sensor fusion of all fiducial objects).
- **(b)** If no changes are made to the fiducial object, `T_MtoI` will also remain the same. Only if you see high bias fluctuations for the specific fiducial object you calibrated, we would recommend recalibrating `T_MtoI`.
- The time offset `t_off_vicon_to_imu` is never used during these calibrations, but is relevant for the auto-calibration (see [next section](#imu-camera-time-offset-calibration-auto-calibration))


## IMU-Camera Time Offset Calibration (Auto-Calibration)
In case you have set the `enabled` flag for the `auto_calibration` block in the `fusion` section of the [cam_config.yaml](../fusionpose_pkg/config/cam_config.yaml) file to true, the `fusion_node` will automatically launch a calibration process for each active fiducial object. The purpose of this auto-calibration is to determine the time offset between the IMU and the master camera (`t_off_imu_to_camera`). As for the previous calibration steps, IMU and pose data will autonomously be collected during periods of unoccluded visibility of the fiducial object. During this period, move the fiducial object around in the camera's field of view to collect diverse poses and IMU readings (e.g., moving it up and down, left and right, and rotating it). If enough data was gathered and the calibration was successful, this is indicated [in the log file](../README.md#monitoring), alongside additional debug information. The new time offset will be automatically published to [the appropriate latency topic](../README.md#topics-overview). Moreover, if you use multiple cameras, the time offset between cameras will also automatically be calculated (only in case the cameras are not triggered together, i.e., `triggered_daq = False` in the `cam_config.yaml` file).

The most important settings of the auto-calibration are:
- `enabled`: Set to true to enable auto-calibration.
- `bag_duration_s`: Duration for which data is recorded during calibration
- `rec_interval_s`: How often to re-calibrate

Additionally, there are some additional optimizer settings (refer to `cam_config.yaml` file for more details).