# 6D Object Pose Tracking for Orthopedic Surgical Training using Visual-Inertial Sensor Fusion
### Authors: [Maarten Hogenkamp (ETH Zürich)](https://pdz.ethz.ch/the-group/people/maarten-hogenkamp.html),  [Tobias Stauffer (ETH Zürich)](https://pdz.ethz.ch/the-group/people/stauffer.html), [Quentin Lohmeyer (ETH Zürich)](https://pdz.ethz.ch/the-group/people/lohmeyer.html), [Mirko Meboldt (ETH Zürich)](https://pdz.ethz.ch/the-group/people/meboldt.html), MICCAI 2025
---
We present a robust, low-cost visual-inertial 6D object tracking system for accurate, real-time motion capture.

<div style="display: flex; justify-content: center;">
  <img src="assets/drilling.gif" alt="drilling" width="49%"  style="margin-right:15px;" />
  <img src="assets/screwing.gif" alt="screwing" width="49%" />
</div>


<!-- :
- [fusionpose_pkg](fusionpose_pkg): The main package for 6D object pose tracking using visual-inertial sensor fusion.
- [vicon2gt](vicon2gt): A package adapted from the [vicon2gt repository](https://github.com/rpng/vicon2gt) that enables spatial and temporal calibration of the IMU. -->
## Getting Started
Welcome to the official implementation of FusionPose! Get the pipeline running by following the instructions below step by step. Note that some instructions are stored in additional read me files which are linked from the main read me. If you encounter any issues during installation or usage that are not addressed in the read mes, feel free to open an issue on this GitHub repository, and we will try to help you as soon as possible!

## Setting up the ROS Environment
This pipeline is implemented using two ROS Noetic packages. Due to our specific project requirements, the pipeline is developed and tested in Windows 11, and we provide the installation instructions for this specific operating system. However, we expect that it will work on Linux as well with only minor modifications. To facilitate deployment, we leverage [ROS inside of a Anaconda environment](https://robostack.github.io/GettingStarted.html).


### Cloning Repository
First off, clone the repository using Git bash:

```bash
# Create a catkin workspace folder with a name of your choice
mkdir catkin_ws
cd catkin_ws
# clone directly into src folder
git clone git@github.com:MountainCoot/fusionpose.git src
```

### Conda/Mamba Environment Installation

Secondly, download a distribution of Anaconda, ideally [`miniforge`](https://github.com/conda-forge/miniforge/releases/) for Windows, and open the `miniforge` shell. Install the required packages:

```bash
cd catkin_ws/src
mamba create -n fusionpose_env -f env/environment.yml
```

This installs all necessary dependencies directly from the `YAML` file. Note that the specific versions of the packages are pinned to ensure compatibility. However, we have included a more flexible environment configuration in `env/environment_flexible.yml` which will be useful when migrating the repo to a different OS.

### Visual Studio C++ Build Tools

To ensure that `catkin_make` works, you must set up Visual Studio C++ build tools for Windows. For compatibility with our implementation, use Visual Studio 2019. Download using PowerShell:
```bash
Invoke-WebRequest "https://aka.ms/vs/16/release/vs_buildtools.exe" -OutFile vs_buildtools.exe
```
and select "Visual C++ build tools" during the installation process.

### Baumer Camera SDK

By default, you can use any camera that is supported using the `OpenCV` camera API. Optionally, if you aim to use Baumer cameras just like in our implementation, you must install the [Baumer Camera Explorer](https://www.baumer.com/int/en/product-overview/industrial-cameras-image-processing/software/baumer-camera-explorer/c/42504) and the `.whl` file located in the `wheels` directory:

```bash
pip install env/wheels/baumer_neoapi-1.4.1-cp34.cp35.cp36.cp37.cp38.cp39.cp310.cp311.cp312-none-win_amd64.whl
```

### Building the Workspace
Before using the pipeline for the first time, you need to build the ROS workspace. For this, ensure first that the Visual Studio 2019 build tools are available in your current `cmd` session:
```bash
call "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
where cl && where nmake # should return two paths
```
Then build the workspace. Please be patient as especially the [vicon2gt](vicon2gt) package may take a while to build. 


```bash
mamba activate fusionpose_env
cd catkin_ws
catkin_make
```

## Preparing the Tracking System for First Use
In addition to setting up the environment and building your ROS workspace, some additional setup steps are required. If you wish to test the system with recorded data we provide, head directly to the [Simulate Without Hardware](#simulate-without-hardware) section and skip this section.

### Hardware Preparations
The hardware preparation steps are detailed [here](docs/HARDWARE_SETUP.md).

### ROS Parameter Configuration
The tracking pipeline uses the following main ROS nodes:

- [`marker_tracker_node`](fusionpose_pkg/nodes/marker_tracker_node.py): Handles camera acquisition, marker detection on image and Vision-Based Pose Estimation using PnP
- [`imu_acquisition_node`](fusionpose_pkg/nodes/imu_acquisition_node.py): Handles IMU data acquisition and preprocessing.
- [`fusion_node`](fusionpose_pkg/nodes/fusion_node.py): Performs visual-inertial sensor fusion using data from the camera and IMU.
- [`calibration_node`](vicon2gt/nodes/calibration_node.cpp): Takes a set of 6D camera poses and IMU measurements to calculate temporal and spatial calibration parameters between IMU and camera.

These nodes communicate via ROS topics and can be configured using the [`cam_config.yaml`](fusionpose_pkg/config/cam_config.yaml) file. The config file includes comments explaining each configuration parameter. The default parameters as provided can be used for a working pipeline and small adjustments should be made if you want to change camera settings (e.g., name, resolution, fps, etc.), the calibration file locations (see next sections) and fusion parameters, in addition to some other parameters. If you aim to use a camera different to the default `baumer1` camera as your main camera, make sure to rename the `master_camera` and `output_frame` parameters in the config file to the name of your camera.


### Camera Calibrations
- **Intrinsics**: To use your camera, you need to intrinsically calibrate it using a calibration board ([ideally with a CharUco pattern like this one](https://calib.io/products/charuco-targets?variant=9400454971439)). Intrinsic camera calibration is not implemented as part of this repo. However, we provide a [dummy calibration file](fusionpose_pkg/src/files/calibrations/baumer1/calibration.json) and refer to the great [multical](https://github.com/oliver-batchelor/multical) GitHub repository for simple intrinsic calibration. After intrinsic calibration, be sure to set the `calibration_file` parameter in the [`cam_config.yaml`](fusionpose_pkg/config/cam_config.yaml#15) file for your specific camera. Unless you change the focus or aperture, this file should remain the same.
- **Gravity Vector Calibration**: Since the IMU measures gravity, the transformation between the camera frame and the gravity vector must be known for accurate sensor fusion. You can obtain this transform as described [in the calibration readme (Gravity Vector Calibration (a))](docs/VICON2GT_CALIBRATION.md). This transformation must be recalculated whenever the camera is re-mounted (i.e., its attitude is changes w.r.t gravity).

- **(optional) Extrinsics**: This step is only required if you aim to use multiple cameras. For extrinsic calibration, we again refer to [a sample file](fusionpose_pkg/src/files/extrinsics/baumer1.json) and the [multical](https://github.com/oliver-batchelor/multical) repository. After extrinsic calibration, set the `extrinsics_file` parameter in the [`cam_config.yaml`](fusionpose_pkg/config/cam_config.yaml#18) file for your specific camera. Recalibration is required as soon as you move any of your cameras.

### Fiducial Object Calibrations
To use a new fiducial object, you must properly configure the [`oid_files` folder](fusionpose_pkg/src/files/oid_files). The process involves two calibration steps and is described in [the OID readme](docs/OID_README.md).
- **Calibration of the ArUco Marker Positions**: To use the hand-crafted fiducial objects, you need to calibrate the position of their ArUco markers with respect to each other.
- **Calibration of the IMU w.r.t to the Camera**: To successfully fuse camera and IMU data, you need to spatially and temporally calibrate both sensors.

## Running the Tracking Pipeline

### Preparing Terminals

To configure a ROS terminal session, use the following commands:

```bash
mamba activate fusionpose_env
cd catkin_ws
call devel/setup.bat # Sourcing the workspace
```

### Running Nodes
In one node, start the ROS core using `roscore`. Now open additional terminals with the same configuration as above. From the `catkin_ws` directory, load the `cam_config.yaml` file using the following command:

```bash
rosparam load src/fusionpose_pkg/config/cam_config.yaml
```

Now launch the `marker_tracker_node` for your master camera (in our case `baumer1`):

```bash
rosrun fusionpose_pkg src/fusionpose_pkg/nodes/marker_tracker_node.py _camera_name:=baumer1
```
If you did not set `launch_imu_acquisition` to `true` in the `marker_tracker` section of the [config file](fusionpose_pkg/config/cam_config.yaml), you need to launch the IMU acquisition node manually. Note that you should only launch this node once the `marker_tracker_node` of the master camera has fully initialized as it publishes the names of the fiducial objects that the IMU node will connect to. Launch it in a new terminal using:

```bash
rosrun fusionpose_pkg src/fusionpose_pkg/nodes/imu_acquisition_node.py
```
Similarly, if `launch_fusion` is set to `false` in the config, you must start the `fusion_node` for each fiducial object with an associated OID in a new terminal using:
```bash
rosrun fusionpose_pkg src/fusionpose_pkg/nodes/fusion_node.py _fusion/oid_name:=oid_<OID>
```

Lastly, launch the `calibration_node` in a new terminal using

```bash
rosrun vicon2gt calibration_node
```


**Note**: Instead of using `roslaunch` files, we launch our nodes individually to separate them over multiple terminals to have better control. We recommend setting `launch_fusion` to `true` and `launch_imu_acquisition` to `false` such that you have a terminal for the `marker_tracker_node`, `imu_acquisition_node`, and `calibration_node`, respectively. Keep an additional terminal to interact with nodes. We provide [a python script](fusionpose_pkg/src/files/rest/bat_files/generate_bat_files.py) that generates a `bat` file which launches this specific terminal configuration.

Optionally, you can launch additional cameras (which are slave to the master camera) by launching another `marker_tracker_node` with another camera name. Just make sure that the [extrinsic transformation between the slave and master camera](#camera-calibrations) is properly configured.

### Monitoring

In general, all nodes write to a common log file stored [here](fusionpose_pkg/logs/live/logging_all.log). There are different log levels and during normal operation you should only see `INFO` and `DEBUG` messages. Any other level indicates an issue that may require attention.

Moreover, during operation, nodes publish their health on the `/node_status` topic. So use `rostopic echo /node_status` to monitor their status. Moreover, the `marker_tracker_node` has some subscribers that allow additional debugging. To show the camera view and change exposure settings use (in case your camera is `baumer1`):

```bash
rostopic pub /baumer1/acquisition_debug std_msgs/Bool true 
```
Publish `false` to the same topic to stop the camera view. Moreover, you can look at the tracked fiducial objects using the following command:

```bash
rostopic pub /baumer1/marker_debug std_msgs/Bool true
```

Additional calibration steps  are required if auto-calibration is active. Please refer to [this readme section](docs/VICON2GT_CALIBRATION.md#imu-camera-time-offset-calibration-auto-calibration).

### Topics Overview
Use `rostopic list` to see all active topics. In general, there are the following topics per camera that is active
```bash
/baumer1/acquisition_debug # Accepts std_msgs/Bool to show camera view and change exposure settings
/baumer1/marker_debug # Accepts std_msgs/Bool to show tracked fiducial objects
/baumer1/latency # Accepts std_msgs/Float32 to change absolute latency of camera (Only for non-master cameras)
/baumer1/latency_delta # Accepts std_msgs/Float32 to change relative latency of camera (Only for non-master cameras)
/baumer1/stream # Publishes the acquired images (only if `publish` is set to true in the config file)
```
and the following topics for each fiducial object that the marker tracker is tracking
```bash
/oid_<OID>/pose # Publishes poses from vision-based tracking, `frame_id` indicates which camera the measurement comes from
```

as well as each fiducial object that is connected via BLE
```bash
/oid_<OID>/imu # Publishes IMU data
/oid_<OID>/imu/frequency # Publishes IMU frequency
/oid_<OID>/battery # Publishes IMU SoC every couple of seconds
/oid_<OID>/latency_abs # Accepts std_msgs/Float32 to change absolute latency of IMU
/oid_<OID>/latency_delta # Accepts std_msgs/Float32 to change relative latency of IMU
```
and each fiducial object for which the sensor fusion node is active
```bash
/oid_<OID>/fused # Publishes real time fusion output (with lag window = 0) (DO NOT USE)
/oid_<OID>/fused_smooth # Publishes smoothed sensor fusion output delayed by `fixed_lag_time`in `cam_config.yaml` -> USE
```

If auto-calibration is activated (refer to [this readme](docs/VICON2GT_CALIBRATION.md#imu-camera-time-offset-calibration-auto-calibration)), there is one additional subscriber for each fiducial object:
```bash
/oid_<OID>/reset_calibration # Accepts std_msgs/Bool that either resets time calibration of fiducial object (true) or sets it to calibrated (false)
```
Lastly the `/tf_static` topic contains all published transforms between a variety of coordinate frames, obtained from the calibration steps.

### Simulate Without Hardware
Some bag files for testing the tracking system without an actual hardware setup can be downloaded [here](https://drive.google.com/drive/folders/1d1aALN5MHx_zVtRj_FLIvHzOxJhieklC?usp=drive_link). To use them, launch only the `marker_tracker_node` for `baumer1` and set `subscribe: True` and `launch_fusion: True` in the `cam_config.yaml` file. There is no need to launch the `calibration_node`. Once the marker tracker is fully initialized, play the ROS bags from the terminal using

```bash
rosbag play <path_to_bag_file>
```
[Monitor the log file](#monitoring) and the [`fused_smooth` topics](#topics-overview) to see if sensor fusion is working correctly.

**Note**: Do not replay the same bag twice without restarting the fusion nodes as this will lead to errors due to negative time deltas.


## Related Work

We would like to thank the authors of the following repositories for their previous work

- [Vicon-IMU fusion for groundtruth trajectory generation](https://github.com/rpng/vicon2gt)
- [D-POINT: Digital Pen with Optical-Inertial Tracking](https://github.com/Jcparkyn/dpoint)
- [Sensor Fusion using GTSAM](https://github.com/PaulKemppi/gtsam_fusion)


## Citation


```
To be added  
```