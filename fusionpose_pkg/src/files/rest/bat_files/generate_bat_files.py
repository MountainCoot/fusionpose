import os

def run_command(cmd):   
    if cmd is None:
        return ""
    
    if isinstance(cmd, list):
        full_cmd = ""
        for c in cmd:
            full_cmd += f"""
rem Print the command
echo {c}
rem Run the command
{c}
"""

    else:
    

        full_cmd = f"""
rem Print the command
echo {cmd}
rem Run the command
{cmd}
"""
    return full_cmd



def generate_ros_bat_files(mamba_env_name, catkin_ws_path, config_path, commands):

    # save into folder of curr file dir
    bat_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(bat_dir, exist_ok=True)
    bat_subdir = "subfiles"
    os.makedirs(os.path.join(bat_dir, bat_subdir), exist_ok=True)

    # delete all existing batch files
    for f in os.listdir(bat_dir):
        if f.endswith(".bat"):
            os.remove(os.path.join(bat_dir, f))

    # replace / or \ with \\ for Windows paths
    catkin_ws_path = catkin_ws_path.replace("/", os.sep)
    config_path = config_path.replace("/",os.sep)
    bat_dir = bat_dir.replace("/", os.sep)
    # make sure it always ends with a backslash
    if not catkin_ws_path.endswith(os.sep):
        catkin_ws_path += os.sep
    if not config_path.startswith(os.sep):
        config_path = os.sep + config_path
    if not bat_dir.endswith(os.sep):
        bat_dir += os.sep    


    roscore_bat_path = os.path.join(bat_dir, bat_subdir, "start_roscore.bat")

    roscore_lock_file = os.path.join(bat_dir, bat_subdir,  "roscore_ready.lock")

    ros_init_str = f"""@echo off
rem Activate {mamba_env_name} environment
call mamba activate {mamba_env_name}
rem Source ROS setup
call "{catkin_ws_path}\\devel\\setup.bat"
rem Change to workspace directory
cd "{catkin_ws_path}" """

    # Generate the "roscore" batch file
    with open(roscore_bat_path, "w") as f:
        f.write(f"""@echo off
rem Delete any existing lock file
if exist "{roscore_lock_file}" del "{roscore_lock_file}"

{ros_init_str}

rem stop any running roscore
rosnode kill /rosout >nul 2>&1

rem Start roscore
start /b roscore

rem Wait for roscore to be fully started
:wait_for_roscore
timeout /t 1 >nul
rosparam list >nul 2>&1
if %ERRORLEVEL% neq 0 goto wait_for_roscore

rem Create lock file
echo "roscore ready" > "{roscore_lock_file}"

rem Load ROS parameters
rosparam load "{catkin_ws_path}{config_path}"

rem Clear the screen for a clean look
cls

{run_command(commands[0])}

""")

    node_bat_paths = []
    for i, command in enumerate(commands[1:]):
        node_bat_path = os.path.join(bat_dir, bat_subdir, f"start_sub_node_{i+1}.bat")
        node_bat_paths.append(node_bat_path)

        with open(node_bat_path, "w") as f:
            f.write(f"""{ros_init_str}

rem Wait for roscore to be ready
:wait_for_roscore
if exist "{roscore_lock_file}" goto load_rosparam
timeout /t 1 >nul
goto wait_for_roscore

:load_rosparam
rem Load ROS parameters
rosparam load "{catkin_ws_path}{config_path}"

rem Clear the screen
cls

{run_command(command)}

""")


    windows_terminal_cmd = f"""wt cmd /k "{roscore_bat_path}" """
    orders = [('V', 'left'), ('H', 'right'), ('H', None), ('V', None), ('H', None), ('V', None)]
    for i, node_bat_path in enumerate(node_bat_paths):
        split_cmd = f"; split-pane -{orders[i][0]} cmd /k \"{node_bat_path}\""
        if orders[i][1]:
            split_cmd += f" ; move-focus {orders[i][1]}"
        windows_terminal_cmd += split_cmd


    terminal_script_path = os.path.join(bat_dir, "LAUNCH_ME.bat")
    with open(terminal_script_path, "w") as f:
        f.write(windows_terminal_cmd)

    print(f"Batch files and Windows Terminal script generated in: {bat_dir}")
    print(f"Run Windows Terminal startup: {terminal_script_path}")


mamba_env_name = "fusionpose_env"
fusionpose_package_name = "fusionpose_pkg"

catkin_ws_path = r"../../../../../.." 
catkin_ws_path = os.path.join(os.path.dirname(__file__), catkin_ws_path)
catkin_ws_path = os.path.abspath(catkin_ws_path)

config_path = "src/fusionpose_pkg/config/cam_config.yaml"

# raise error if any paths are not found
if not os.path.exists(catkin_ws_path):
    raise FileNotFoundError(f"Catkin workspace path not found: {catkin_ws_path}")
if not os.path.exists(os.path.join(catkin_ws_path, config_path)):
    raise FileNotFoundError(f"Config path not found: {os.path.join(catkin_ws_path, config_path)}")

info_str = f'{50*"="}\n'
info_str += 'RUN THIS SCRIPT FROM FOR CATKIN WORKSPACE'
info_str += 'VERIFY THAT THE FOLLOWING PATHS ARE CORRECT:\n'
info_str += f'{50*"="}\n'
info_str += 'CATKIN_WS_PATH (abs) = ' + catkin_ws_path + '\n'
info_str += 'CONFIG_PATH (abs) = ' + os.path.join(catkin_ws_path, config_path) + '\n'
info_str += 'NAME OF MAMBA ENVIRONMENT = ' + mamba_env_name + '\n'
info_str += 'FUSIONPOSE_PACKAGE_NAME = ' + fusionpose_package_name + '\n'
info_str += f'{50*"="}\n'
info_str += f'INFO: IF YOU GET A MESSAGE THAT PATH IS NOT FOUND, TRY RUNNING DIRECTLY FROM MAMBA PROMPT\n'
info_str += f'{50*"="}\n'
print(info_str)

commands = [
    f'rosrun {fusionpose_package_name} {os.path.join(catkin_ws_path, "src", fusionpose_package_name, "nodes", "marker_tracker_node.py")} _camera_name:=baumer1',
    ['timeout /t 10 /nobreak', f'rosrun {fusionpose_package_name} {os.path.join(catkin_ws_path, "src", fusionpose_package_name, "nodes", "imu_acquisition_node.py")}'], # sleep a bit until marker_tracker_node is ready
    f'rosrun vicon2gt calibration_node',
    'echo "Use this terminal for interactive commands"'
]

assert len(commands) > 0, "Please provide at least one command to run."
assert len(commands) < 6, "Please provide at most five commands to run."

generate_ros_bat_files(mamba_env_name, catkin_ws_path, config_path, commands)
