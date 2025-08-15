#include <ros/ros.h>
#include "vicon2gt/CalibrateBag.h"
#include "BagCalibrator.hpp"

class CalibrationService {
public:
    CalibrationService(ros::NodeHandle &nh) {
        
        // Instantiate the BagCalibrator with the parameters
        bag_calibrator_ = std::make_shared<BagCalibrator>(nh);

        // Advertise the service
        service_ = nh.advertiseService("run_calibration", &CalibrationService::runCalibration, this);
        ROS_INFO("Calibration service ready to receive requests.");
    }

private:
    std::shared_ptr<BagCalibrator> bag_calibrator_;  // Only store BagCalibrator as a class member
    ros::ServiceServer service_;

    // Service callback method
    bool runCalibration(vicon2gt::CalibrateBag::Request &req, vicon2gt::CalibrateBag::Response &res) {
        std::string camera_name = req.camera_name;
        std::string oid_name = req.oid_name;

        ROS_INFO("Received calibration request: camera_name = %s, oid_name = %s", camera_name.c_str(), oid_name.c_str());

        // Use the BagCalibrator instance to perform calibration
        bool calibration_success = bag_calibrator_->runCalibration(oid_name, camera_name);
        
        std::string message = calibration_success ? "Calibration successful" : "Calibration failed";

        res.success = calibration_success;
        res.message = message;

        return true;
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "calibration_node");
    ros::NodeHandle nh;

    // Create an instance of the CalibrationService class
    CalibrationService calibration_service(nh);

    // Keep the node alive
    ros::spin();

    return 0;
}
