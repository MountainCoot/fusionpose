import rospy
import rospkg
import yaml
import os
import subprocess

cwd = os.path.dirname(os.path.abspath(__file__))
cam_config_path = os.path.join(os.path.dirname(cwd), 'config', 'cam_config.yaml')

class Node():
    def __init__(self, name: str, use_ros: bool = True, load_config: bool = False) -> None:
        self.config = {}
        self.node_name = name

        if use_ros:
            if load_config:
                # run rosparam load config/cam_config.yaml
                # go up from cwd
                subprocess.run(['rosparam', 'load', cam_config_path])
            rospy.init_node(self.node_name, anonymous=True)
            rospy.loginfo(f"Initialized node {self.node_name}")

            param_names = rospy.get_param_names()
            # Iterate through parameter names and get their values
            for param_name in param_names:
                # Split parameter name based on slashes
                keys = param_name.split('/')
                # Remove the first element, which is the node name
                keys = keys[1:]
                current_dict = self.config

                # Iterate through keys to create nested dictionaries
                for key in keys[:-1]:
                    if key not in current_dict:
                        current_dict[key] = {}
                    current_dict = current_dict[key]

                # Assign the value to the innermost dictionary
                current_dict[keys[-1]] = rospy.get_param(param_name)

            try:
                self.root_dir = rospkg.RosPack().get_path(self.config['pkg_name'])
            except KeyError as e:
                print(f'Could not find key: {e}')
                raise e
        else:
            # get yaml from config.cam_config.yaml
            print("Loading config from config/cam_config.yaml")
            import os
            with open(os.path.join('config', 'cam_config.yaml'), 'r') as f:
                self.config = yaml.safe_load(f)
            self.root_dir = os.path.dirname(os.path.abspath(__file__))
            # if subfolder is /nodes, then go up one level
            if os.path.basename(self.root_dir) == 'nodes':
                self.root_dir = os.path.dirname(self.root_dir)

        # loop over all parameters and check if it is a relative path, if so replace it with os.path.join(self.root_dir, ...)
        self._replace_relative_paths(self.config)

    def _replace_relative_paths(self, config: dict) -> None:
        """Helper function to replace relative paths in config with absolute paths."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._replace_relative_paths(value)
            elif isinstance(value, str):
                if 'path' in key or 'dir' in key or 'file' in key:
                    if not os.path.isabs(value):
                        config[key] = os.path.join(self.root_dir, value)
                    # if not os.path.isabs(value) and os.path.exists(os.path.join(self.root_dir, value)):
                    #     config[key] = os.path.join(self.root_dir, value)
                    # elif '/' or '\\' in value:
                    