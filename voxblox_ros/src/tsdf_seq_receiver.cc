#include "voxblox_ros/tsdf_seq_receiver.h"

namespace voxblox {

    TsdfSeqReceiver::TsdfSeqReceiver(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private) : nh_(nh), nh_private_(nh_private){
        
        provides_weight_image = false;
        early_depth = false;
        projects_weight = false;
        aligned = false;
    }


    void TsdfSeqReceiver::initialize(boost::function<void (const sensor_msgs::ImageConstPtr&, 
                                                           const sensor_msgs::ImageConstPtr&, 
                                                           const sensor_msgs::CameraInfoConstPtr&, 
                                                           const sensor_msgs::CameraInfoConstPtr&)> func){
        server_callback = func;

    }

    void TsdfSeqReceiver::initialize_weight(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&,
                                                                  const cv::Mat, const std::unordered_map<int, std::string>, const std::unordered_map<int, std::vector<cv::Point>>)> func){
        server_weight_callback = func;
    }

    void TsdfSeqReceiver::initialize_weight2(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                   const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> func){
        server_weight_callback2 = func;
    }

    void TsdfSeqReceiver::initialize_weight3(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                   const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> func){
        server_weight_callback3 = func;
    }

    void TsdfSeqReceiver::init_early_depth_callback(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> func){
        early_depth_callback = func;
    }
}