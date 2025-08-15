# OID Configurations
To let the tracking system know which fiducial objects should be used, we use the [`oid_files` folder](../fusionpose_pkg/src/files/oid_files) to store OID configuration files of active fiducial objects. 

## Configuration File Overview

There is a `json` file for each active fiducial object in the `oid_files` folder that has the following structure:

```bash
{
    "name": 20,
    "marker_length": 0.01415,
    "pts_file_path": "pts/oid0_pts.json",
    "T_MtoI": {
        "r": [
            [0.00481387095099, -0.0035851101478, -0.999981986654],
            [0.589002918808, -0.808110573083, 0.00573265272715],
            [-0.808116568498, -0.589019905145, -0.00177849967972]
        ],
        "t": [
            [-0.02, 0.017, 0.003]
        ]
}
```
A couple of sample configuration files for fiducial objects are provided in the `oid_files` folder. They are explained in the following sections.

## Configuration File Creation

To configure the tracking system for a new fiducial object, you must create a new `json` file in the `oid_files` folder with the same keys. The following sections explain each key of the `json` file.

### `name`
The `name` key corresponds to the unique identifier of the fiducial object and must also correspond to the OID number flashed onto the IMU (refer to [the hardware documentation](HARDWARE_SETUP.md)). It can be freely chosen as long as it is an integer without leading zeros and unique within the `oid_files` folder.

### `marker_length`
The `marker_length` key corresponds to the physical length in meters of the ArUco marker side and can be extracted from the [ArUco printing script](../fusionpose_pkg/src/files/rest/aruco/generate_aruco_markers_for_printing.py) (refer to [the hardware documentation](HARDWARE_SETUP.md)).

### `pts_file_path`
The `pts_file_path` key points to a file containing the calibrated 3D coordinates of the ArUco markers in the fiducial object. This file is obtained by calibrating the fiducial object after gluing on the ArUco markers. For this purpose, you can leverage a built-in functionality of the `marker_tracker_node` that allows acquisition of calibration frames for a new fiducial object. The steps are:
1. Edit the [`cam_config.yaml`](../fusionpose_pkg/config/cam_config.yaml) `marker_calibration` section for the `marker_tracker` such that `enabled` is set to `true` and the `ids_to_calibrate` list corresponds to the ArUco markers of your new fiducial object. Also set the `path` to a location where you want to store the calibration images
2. Launch the `marker_tracker_node` (refer to [the main read me](../README.md#running-the-tracking-pipeline)) and follow the instructions to record images.
3. For the iterative calibration given these images, you additionally need an initial guess of our ArUco marker positions. Depending on your layout, you should extract these points (`json` with `dict` format that maps your `n` ArUco markers to a `3x4` array, corresponding to the corner points) directly from your CAD model or write them out manually.
4. Now that you have a folder with your calibration images and an initial calibration, use the [`marker_calibration`](../fusionpose_pkg/src/deps/utils/marker_calibration.py) script to calibrate the fiducial marker. You will additionally need the intrinsic calibration of your camera. Be sure to set `params['use_descriptions'] = True` such that the right angle and unit-length constraints (refer to page 4 of the paper) are enforced. The script will directly output the file you should specify for `pts_file_path` (again a `json` with `dict` format that maps your `n` ArUco markers to a `3x4` array, corresponding to the corner points).

**Note:** We have included the data structure for a sample fiducial object in the [`marker_calibrations` folder](../fusionpose_pkg/src/files/rest/marker_calibrations/sample_oid). Please refer to this structure for guidance.

### `T_MtoI`
The `T_MtoI` key contains the transformation matrix from the marker frame to the IMU frame, which is crucial for sensor fusion. This transformation is obtained using the `vicon2gt` package of this repository and details are provided in the [vicon2gt documentation (Marker to IMU Frame Calibration (T_MtoI) (b))](VICON2GT_CALIBRATION.md).

## Activating and Deactivating Configurations
To activate a configuration, simply move it out of the [`inactive`](../fusionpose_pkg/src/files/oid_files/inactive) subfolder and into the main `oid_files` folder. To deactivate a configuration, move it into the `inactive` subfolder.