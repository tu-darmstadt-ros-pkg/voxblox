#ifndef SEG_DEBUG_RECEIVER_H_
#define SEG_DEBUG_RECEIVER_H_

#include "voxblox_ros/SegReceiver.h"
#include "voxblox_ros/SegToolbox.h"


namespace voxblox{
    template<class T>
    class SegDebugReceiver : public SegReceiver{

        public:
        SegDebugReceiver(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                          const std::string color_image_topic, const std::string depth_image_topic,
                          const std::string color_info_topic, const std::string depth_info_topic, 
                          const std::string cnn_result_topic);


        message_filters::Subscriber<sensor_msgs::Image> color_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub;
            message_filters::Subscriber<sensor_msgs::Image> depth_img_sub;
            message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;
            message_filters::Subscriber<mask_rcnn_ros::Result> cnn_result_sub;

            message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo, mask_rcnn_ros::Result>> sync;

        void callback(const sensor_msgs::ImageConstPtr& color_img, 
                          const sensor_msgs::ImageConstPtr& depth_img, 
                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                          const sensor_msgs::CameraInfoConstPtr& depth_info,
                          const mask_rcnn_ros::ResultConstPtr& cnn_result);



        void setServerPointer(std::shared_ptr<SegServer<T>> server_ptr);
        protected:

        int id_counter;
        bool aligned_rgbd;
        bool use_weight;

        std::unordered_map<std::string, int> label_lookup;


        //ros::Publisher cnn_image_pub;
        //ros::Publisher seg_img_pub;
        //ros::Subscriber cnn_result_sub;
        //ros::Publisher debug_seg_cloud_pub;

         std::shared_ptr<rawDataPointer> data;

         double scale;

         std::string body_frame;

         std::shared_ptr<SegServer<T>> server;

         int downsample_receiver_factor;

         bool skip_tsdf;
         uint16_t highest_depth;

         int bilateral_filter_size;
         double bilateral_filter_diff;
    
    };
}

#endif