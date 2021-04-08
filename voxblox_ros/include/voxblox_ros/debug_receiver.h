#ifndef VOXBLOX_DEBUG_RECEIVER_H_
#define VOXBLOX_DEBUG_RECEIVER_H_

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

namespace voxblox {

    class DebugReceiver : public TsdfSeqReceiver {

        public:
            DebugReceiver(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                          const std::string color_image_topic, const std::string depth_image_topic,
                          const std::string color_info_topic, const std::string depth_info_topic, 
                          const std::string cnn_result_topic);
    

            void callback(const sensor_msgs::ImageConstPtr& color_img, 
                          const sensor_msgs::ImageConstPtr& depth_img, 
                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                          const sensor_msgs::CameraInfoConstPtr& depth_info,
                          const mask_rcnn_ros::ResultConstPtr& cnn_result);

            void callback2(const sensor_msgs::ImageConstPtr& color_img, 
                          const sensor_msgs::ImageConstPtr& depth_img, 
                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                          const sensor_msgs::CameraInfoConstPtr& depth_info,
                          const mask_rcnn_ros::ResultConstPtr& cnn_result);

            void initialize_weight2(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                          const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> func);

            void initialize_weight3(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                          const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> func);


            double scale;
        protected:
            message_filters::Subscriber<sensor_msgs::Image> color_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub;
            message_filters::Subscriber<sensor_msgs::Image> depth_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;
            message_filters::Subscriber<mask_rcnn_ros::Result> cnn_result_sub;

            message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo, mask_rcnn_ros::Result>> sync;


            std::vector<std::vector<LabelVoxel>> label_map;
            std::vector<cv::Vec3f> coord_map; 
            std::unordered_map<std::string, int> label_string_mapping;
            int id_counter;


            bool got_callback;
            bool aligned;


             

            //storage
            sensor_msgs::Image color_img_cam;
            sensor_msgs::Image depth_img_cam;
            sensor_msgs::CameraInfo color_info_cam;
            sensor_msgs::CameraInfo depth_info_cam;
            mask_rcnn_ros::Result cnn_res;
    };
}


#endif