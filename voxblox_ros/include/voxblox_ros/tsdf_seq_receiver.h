#ifndef VOXBLOX_TSDF_SEQ_RECEIVER_H_
#define VOXBLOX_TSDF_SEQ_RECEIVER_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <unordered_map>

#include <opencv2/highgui.hpp>

#include <voxblox/core/voxel.h>


namespace voxblox {

    class TsdfSeqReceiver {


        public:
        TsdfSeqReceiver(const ros::NodeHandle& nh, 
                        const ros::NodeHandle& nh_private);

        virtual void initialize(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> func);

        boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> server_callback;
        
        virtual void initialize_weight(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&,
                                                  const cv::Mat, const std::unordered_map<int, std::string>, const std::unordered_map<int, std::vector<cv::Point>>)> func);

        virtual void initialize_weight2(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                              const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> func);
                                        
        virtual void initialize_weight3(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                              const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> func);


        virtual void init_early_depth_callback(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> func);

        boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                              const cv::Mat, const std::unordered_map<int, std::string>, const std::unordered_map<int, std::vector<cv::Point>>)> server_weight_callback;


        boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                              const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> server_weight_callback2;
        
        boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                              const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> server_weight_callback3;
        


        
        boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&)> early_depth_callback;
        //flags
        bool provides_weight_image;
        bool early_depth;
        bool projects_weight;
        bool aligned;

        protected:

        ros::NodeHandle nh_;
        ros::NodeHandle nh_private_;

        
        

        std::unordered_map<int, std::string> label_lookup;
        std::unordered_map<int, std::vector<cv::Point>> label_map; //like example
                    
    };
    //abstract version

}

#endif