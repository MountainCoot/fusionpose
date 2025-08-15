#ifndef BAG_CALIBRATOR_H
#define BAG_CALIBRATOR_H

#include <Eigen/Eigen>
#include <cmath>
#include <memory>
#include <vector>
#include <windows.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include "meas/Interpolator.h"
#include "meas/Propagator.h"
#include "solver/ViconGraphSolver.h"
#include "utils/general.h"


class BagCalibrator {
public:
    BagCalibrator(ros::NodeHandle &nh);  
    
    bool runCalibration(const std::string oid_name, const std::string &camera_name);

private:

    // base path
    std::string bag_dir_;

    int state_freq_;
    double sigma_w;
    double sigma_a;
    double sigma_wb;
    double sigma_ab;
    Eigen::Matrix3d R_q_, R_p_;

    // node handle
    ros::NodeHandle nh_;
};

#endif // BAG_CALIBRATOR_H
