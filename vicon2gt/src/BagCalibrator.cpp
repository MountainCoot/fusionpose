#include "BagCalibrator.hpp"

bool copyFile(const std::string &source, const std::string &destination) {
    // CopyFile returns 0 on failure
    if (!CopyFile(source.c_str(), destination.c_str(), FALSE)) {
        std::cerr << "Error: " << GetLastError() << " - Could not copy file." << std::endl;
        return false;
    }
    return true;
}

bool removeFile(const std::string &filePath) {
    // DeleteFile returns 0 on failure
    if (!DeleteFile(filePath.c_str())) {
        std::cerr << "Error: " << GetLastError() << " - Could not delete file." << std::endl;
        return false;
    }
    return true;
}

BagCalibrator::BagCalibrator(ros::NodeHandle &nh) {
    nh_ = nh;

    nh_.param<int>("/fusion/auto_calibration/imu_freq_des", state_freq_, 100);

    // Load the IMU noise parameters
    std::string fusion_sigmas_ns = "/fusion/sigmas";

    double ds;
    nh_.param<double>(fusion_sigmas_ns + "/ds", ds, 1);
    nh_.param<double>(fusion_sigmas_ns + "/gyro", sigma_w, 1.e-2);
    nh_.param<double>(fusion_sigmas_ns + "/acc", sigma_a, 2.e-2);
    nh_.param<double>(fusion_sigmas_ns + "/gyro_bias_evolution", sigma_wb, 3.e-3);
    nh_.param<double>(fusion_sigmas_ns + "/acc_bias_evolution", sigma_ab, 1.e-2);
    // multiply by ds
    sigma_w *= ds;
    sigma_a *= ds;
    sigma_wb *= ds;
    sigma_ab *= ds;
    
    // Load the Vicon noise parameters
    std::vector<double> vicon_sigma_pos;
    std::vector<double> vicon_sigma_rot;
    nh_.param<std::vector<double>>(fusion_sigmas_ns + "/pose_pos", vicon_sigma_pos, {2.e-3, 2.e-3, 6.e-3});
    nh_.param<std::vector<double>>(fusion_sigmas_ns + "/pose_rot", vicon_sigma_rot, {0.0174533, 0.0174533, 0.0174533});
    
    // Initialize the noise covariance matrices
    R_q_ = Eigen::Matrix3d::Zero();
    R_p_ = Eigen::Matrix3d::Zero();
    R_q_(0, 0) = std::pow(vicon_sigma_rot.at(0), 2);
    R_q_(1, 1) = std::pow(vicon_sigma_rot.at(1), 2);
    R_q_(2, 2) = std::pow(vicon_sigma_rot.at(2), 2);
    R_p_(0, 0) = std::pow(vicon_sigma_pos.at(0), 2);
    R_p_(1, 1) = std::pow(vicon_sigma_pos.at(1), 2);
    R_p_(2, 2) = std::pow(vicon_sigma_pos.at(2), 2);

    // base path is found by getting the path of the package "/pkg_name" and appending "/fusion/auto_calibration/bag_dir"
    std::string package_name;
    nh_.param<std::string>("pkg_name", package_name, "fusionpose_pkg");
    std::string bag_subdir;
    nh_.param<std::string>("/fusion/auto_calibration/bag_dir", bag_subdir, "/src/files/oid_autocalib");
    // bag dir is the package path + bag subdir and make sure there is a / in between
    if (bag_subdir[0] != '/') {
        bag_subdir = "/" + bag_subdir;
    }
    bag_dir_ = ros::package::getPath(package_name) + bag_subdir;

    // Log some stuff
    ROS_INFO("Initialized BagCalibrator with parameters:");
    ROS_INFO("state_freq: %d", state_freq_);

    ROS_INFO("gyroscope_noise_density: %f", sigma_w);
    ROS_INFO("accelerometer_noise_density: %f", sigma_a);
    ROS_INFO("gyroscope_random_walk: %f", sigma_wb);
    ROS_INFO("accelerometer_random_walk: %f", sigma_ab);

    ROS_INFO("vision_position_noise: %f %f %f", vicon_sigma_pos.at(0), vicon_sigma_pos.at(1), vicon_sigma_pos.at(2));
    ROS_INFO("vision_orientation_noise: %f %f %f", vicon_sigma_rot.at(0), vicon_sigma_rot.at(1), vicon_sigma_rot.at(2));

    ROS_INFO("bag_dir: %s", bag_dir_.c_str());
}

bool BagCalibrator::runCalibration(const std::string oid_name, const std::string &camera_name) {
    // setup topics and bag
    std::string dataset_name = oid_name + "_autocalibration_" + camera_name;
    std::string bag_path = bag_dir_ + "/" + dataset_name + ".bag";
    std::string imu_topic = "/" + oid_name + "/imu";
    std::string vicon_topic = "/" + oid_name + "/pose";

    std::string frame_vicon;
    nh_.param<std::string>("/fusion/auto_calibration/vicon2gt/frame_filter", frame_vicon, "all");   
    
    // create a copy of the bag by replacing name.bag with name_temp.bag
    std::string bag_path_temp = bag_path;
    // first strip .bag away
    bag_path_temp = bag_path_temp.substr(0, bag_path_temp.size() - 4);
    // add _temp.bag
    bag_path_temp += "_temp.bag";

    // copy the bag using windows command
    if (!copyFile(bag_path, bag_path_temp)) {
        return false;
    }


    // Open the bag
    rosbag::Bag bag;
    bag.open(bag_path_temp, rosbag::bagmode::Read);

    ROS_INFO("Opened bag %s", bag_path.c_str());

    // We should load the bag as a view
    // Here we go from beginning of the bag to the end of the bag
    rosbag::View view;

    // Start a few seconds in from the full view time
    // If we have a negative duration then use the full bag length
    view.addQuery(bag, rosbag::TopicQuery({imu_topic, vicon_topic}));
    ros::Time time_init = view.getBeginTime();
    ros::Time time_finish = view.getEndTime();
    ROS_INFO("loading rosbag into memory...");
    ROS_INFO("    - time start = %.6f", time_init.toSec());
    ROS_INFO("    - time end   = %.6f", time_finish.toSec());
    ROS_INFO("    - duration   = %.2f (secs)", time_finish.toSec() - time_init.toSec());
    view.addQuery(bag, rosbag::TopicQuery({imu_topic, vicon_topic}), time_init, time_finish);

    // Our data storage objects
    std::shared_ptr<Propagator> propagator = std::make_shared<Propagator>(sigma_w, sigma_wb, sigma_a, sigma_ab);
    std::shared_ptr<Interpolator> interpolator = std::make_shared<Interpolator>();

    // Counts on how many measurements we have
    int ct_imu = 0;
    int ct_vic = 0;
    double start_time = -1;
    double end_time = -1;

    // Check if we have any interval with long gap of no vicon measurements
    // We can still try to process, but if it is too long, then the IMU can drift
    // Thus just warn the user that there might be an issue!
    double last_vicon_time = -1;
    double max_vicon_lost_time = 1.0; // seconds
    auto warn_amount_vicon_rate = [&](double timestamp, double timestamp_last) {
        double vicon_dt = timestamp - timestamp_last;
        if (last_vicon_time == -1 || vicon_dt < max_vicon_lost_time)
        return false;
        double dist_from_start = timestamp_last - time_init.toSec();
        ROS_WARN("over %.2f seconds of no vicon!! (starting %.2f sec into bag)", vicon_dt, dist_from_start);
    };

    // list all topics
    std::vector<std::string> topics;
    for (const rosbag::ConnectionInfo *info : view.getConnections()) {
        topics.push_back(info->topic);
    }
    ROS_INFO("topics in the rosbag:");
    for (size_t i = 0; i < topics.size(); i++) {
        ROS_INFO("    - %s", topics[i].c_str());
    }

    // Step through the rosbag
    ROS_INFO("load custom data into memory...");
    for (const rosbag::MessageInstance &m : view) {

        // If ros is wants us to stop, break out
        if (!ros::ok())
        break;

        // Handle IMU messages
        sensor_msgs::Imu::ConstPtr s0 = m.instantiate<sensor_msgs::Imu>();
        if (s0 != nullptr && m.getTopic() == imu_topic) {
        Eigen::Vector3d wm, am;
        wm << s0->angular_velocity.x, s0->angular_velocity.y, s0->angular_velocity.z;
        am << s0->linear_acceleration.x, s0->linear_acceleration.y, s0->linear_acceleration.z;
        propagator->feed_imu(s0->header.stamp.toSec(), wm, am);
        ct_imu++;
        }

        // Handle VICON messages as PoseStamped
        geometry_msgs::PoseStamped::ConstPtr s4 = m.instantiate<geometry_msgs::PoseStamped>();
        if (s4 != nullptr && m.getTopic() == vicon_topic) {
        // skip if frame is not any and the frame is not the one we want
        if (frame_vicon != "all" && s4->header.frame_id != frame_vicon) {
            continue;
        }
        // load orientation and position of the vicon
        Eigen::Vector4d q;
        Eigen::Vector3d p;
        q << s4->pose.orientation.x, s4->pose.orientation.y, s4->pose.orientation.z, s4->pose.orientation.w;
        p << s4->pose.position.x, s4->pose.position.y, s4->pose.position.z;
        // feed it!
        interpolator->feed_pose(s4->header.stamp.toSec(), q, p, R_q_, R_p_);
        ct_vic++;
        // update timestamps
        if (start_time == -1) {
            start_time = s4->header.stamp.toSec();
        }
        if (start_time != -1) {
            end_time = s4->header.stamp.toSec();
        }
        warn_amount_vicon_rate(s4->header.stamp.toSec(), last_vicon_time);
        last_vicon_time = s4->header.stamp.toSec();
        }
    }

    // Create our camera timestamps at the requested fix frequency
    int ct_cam = 0;
    std::vector<double> timestamp_cameras;
    if (start_time != -1 && end_time != -1 && start_time < end_time) {
        double temp_time = start_time;
        while (temp_time < end_time) {
        timestamp_cameras.push_back(temp_time);
        temp_time += 1.0 / (double)state_freq_;
        ct_cam++;
        }
    }

    // Print out how many we have loaded
    ROS_INFO("done loading the rosbag...");
    ROS_INFO("    - number imu   = %d", ct_imu);
    ROS_INFO("    - number cam   = %d", ct_cam);
    ROS_INFO("    - number vicon = %d", ct_vic);

    // Check to make sure we have data to optimize
    if (ct_imu == 0 || ct_cam == 0 || ct_vic == 0) {
        ROS_ERROR("Not enough data to optimize with!");
        return false;
    }

    // Create the graph problem, and solve it
    ViconGraphSolver solver(nh_, propagator, interpolator, timestamp_cameras);
    solver.build_and_solve();

    std::string states_csv_file = bag_dir_ + "/vicon2gt/" + dataset_name + "_vicon2gt_states.csv";
    std::string info_txt_file = bag_dir_ + "/vicon2gt/" + dataset_name + "_vicon2gt_info.txt";

    // Write the results to file
    keep_max_n_files(info_txt_file, 4);
    solver.write_to_file(states_csv_file, info_txt_file);

    // Get the calibration parameters
    double toff;
    Eigen::Matrix3d R_BtoI, R_GtoV;
    Eigen::Vector3d p_BinI;
    solver.get_calibration(toff, R_BtoI, p_BinI, R_GtoV);

    // Close the bag
    bag.close();

    std::cout << "finished with success" << std::endl;

    // remove temp bag using filesystem
    removeFile(bag_path_temp);

    return true;        
}