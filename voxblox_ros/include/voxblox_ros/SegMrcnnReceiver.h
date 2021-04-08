#ifndef SEG_MRCNN_RECEIVER_H_
#define SEG_MRCNN_RECEIVER_H_

#include "voxblox_ros/SegReceiver.h"
#include "voxblox_ros/SegToolbox.h"

namespace voxblox {
template <class T>
class SegMrcnnReceiver : public SegReceiver {
 public:
  SegMrcnnReceiver(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                   const std::string color_image_topic,
                   const std::string depth_image_topic,
                   const std::string color_info_topic,
                   const std::string depth_info_topic);

  message_filters::Subscriber<sensor_msgs::Image> color_img_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub;
  message_filters::Subscriber<sensor_msgs::Image> depth_img_sub;
  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub;

  message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
      sensor_msgs::CameraInfo>>
      sync;

  void callback(const sensor_msgs::ImageConstPtr& color_img,
                const sensor_msgs::ImageConstPtr& depth_img,
                const sensor_msgs::CameraInfoConstPtr& color_info,
                const sensor_msgs::CameraInfoConstPtr& depth_info);

  void cnn_callback(const mask_rcnn_ros::Result res);

  void setServerPointer(std::shared_ptr<SegServer<T>> server_ptr);

 protected:
  std::unordered_map<std::string, int> label_lookup;

  bool aligned_rgbd;
  bool published_to_cnn;
  bool use_weight;
  bool got_callback;
  int id_counter;

  double scale;

  std::shared_ptr<SegServer<T>> server;

  ros::Publisher cnn_image_pub;
  ros::Publisher seg_img_pub;
  ros::Subscriber cnn_result_sub;
  ros::Publisher debug_seg_cloud_pub;

  std::shared_ptr<rawDataPointer> data;

  std::string body_frame;

  int downsample_receiver_factor;

  int bilateral_filter_size;
  double bilateral_filter_diff;
};
}  // namespace voxblox

#endif