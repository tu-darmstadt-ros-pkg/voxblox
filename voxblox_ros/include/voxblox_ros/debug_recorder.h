#ifndef VOXBLOX_DEBUG_RECORDER_H_
#define VOXBLOX_DEBUG_RECORDER_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <sensor_msgs/CameraInfo.h>
#include <ros/publisher.h>
#include <ros/subscriber.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>


#include <mask_rcnn_ros/Result.h>
#include "voxblox_ros/tsdf_seq_receiver.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/rgbd.hpp>

#include "voxblox/core/voxel.h"

#include <image_geometry/pinhole_camera_model.h>

#include <voxblox_ros/image_operations.h>

namespace voxblox{


    class DebugRecorder {

        public:
            DebugRecorder(ros::NodeHandle& nh, ros::NodeHandle& nh_private,
                          const std::string color_image_topic, const std::string depth_image_topic,
                          const std::string color_info_topic, const std::string depth_info_topic);
    

            void callback(const sensor_msgs::ImageConstPtr& color_img, 
                          const sensor_msgs::ImageConstPtr& depth_img, 
                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                          const sensor_msgs::CameraInfoConstPtr& depth_info);
        
            void cnn_callback(const mask_rcnn_ros::Result res);


        protected:

            message_filters::Subscriber<sensor_msgs::Image> color_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub;
            message_filters::Subscriber<sensor_msgs::Image> depth_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;

            message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>> sync;

            ros::Publisher cnn_image_pub;
            ros::Subscriber cnn_result_sub;

            bool published_to_cnn;
            bool got_callback;
            bool first_cb;

            ros::NodeHandle nh_;
            ros::NodeHandle nh_private_;


            ros::Publisher debug_color_image_pub;
            ros::Publisher debug_depth_image_pub;
            ros::Publisher debug_color_info_pub;
            ros::Publisher debug_depth_info_pub;
            ros::Publisher debug_cnn_result_pub;


            //storage
            sensor_msgs::Image color_img_cam;
            sensor_msgs::Image depth_img_cam;
            sensor_msgs::CameraInfo color_info_cam;
            sensor_msgs::CameraInfo depth_info_cam;
    };
}


#endif