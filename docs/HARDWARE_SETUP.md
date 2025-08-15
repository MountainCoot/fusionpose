# Hardware Setup
## PC Requirements
- Developed for Windows 11, likely portable to Linux with minor adjustments.
- Fairly strong CPU required with many cores, no GPU required
- BLE support for wireless communication

## Setting up the Camera
### Required Components
- At least one camera that has reasonably high resolution and at least 20 FPS
- Tripod to ensure that camera is stationary
- (recommended) external light source

## Building a Fiducial Object
### Required Tools
- 3D Printer with filament (e.g. white PLA)
- Inkjet or Laser Printer
- Soldering iron

### Required Components per Fiducial Object
- 1x [Seeed Studio XIAO Sense](https://wiki.seeedstudio.com/XIAO_BLE/)
- 1x 3.7 V Rechargeable Battery (We recommend LiPo batteries, [for example this one](https://www.ebay.com/itm/114292221242?var=414545856354))
- (recommended) Some adhesive paper for ArUco markers ([for example this one](https://www.amazon.de/dp/B0BWV51YGD?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1)) 

### Assembly Instructions
1. 3D-print a Fiducial Object of any shape that has some planar surfaces. The choice of shape is not limited to the Dodecahedron shape as suggested in our work.[^1]
2. Generate ArUco markers for printing using [this script](../fusionpose_pkg/src/files/rest/aruco/generate_aruco_markers_for_printing.py) and print them ideally on adhesive paper
3. Cut out the ArUco markers and stick them onto the planar surfaces of the Fiducial Object
4. Solder the battery to the charging circuit of the XIAO Sense and insert both into the Fiducial Object

### Flashing Software onto the IMU
The IMU software is provided in the [`imu_handler` directory](../fusionpose_pkg/src/files/rest/imu_handler). We recommend using the [PlatformIO extension for VS Code](https://platformio.org/install/ide?install=vscode) to flash the firmware onto the XIAO Sense. In the [main.cpp](../fusionpose_pkg/src/files/rest/imu_handler/src/main.cpp) file, you can configure the IMU DAQ settings. If using the default settings, you only need to change [the OID number](../fusionpose_pkg/src/files/rest/imu_handler/src/main.cpp#L71) in the script, which will define the OID number of your fiducial object.

### Pairing IMU to PC
Note that you have to pair the IMU to your PC when connecting for the first time. Otherwise, the data acquisition will be lower than anticipated. [Read more here](https://support.microsoft.com/en-us/windows/pair-a-bluetooth-device-in-windows-2be7b51f-6ae9-b757-a3b9-95ee40c3e242).

[^1]: Due to license constraints, we unfortunately cannot provide the exact 3D model files used in our work. However, it should be quite straightforward to create your own models using common 3D design software with an inset for the XIAO Sense. Moreover, [other resources for this specific microcontroller can be found on GitHub and used as a starting point](https://github.com/Jcparkyn/dpoint/tree/main/print/export).
