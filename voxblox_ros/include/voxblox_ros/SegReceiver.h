#ifndef SEG_RECEIVER_H_
#define SEG_RECEIVER_H_

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <voxblox_ros/SegServer.h>
#include <voxblox_ros/SegToolbox.h>

// TODO check includes

#include <ros/publisher.h>
#include <ros/subscriber.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <mask_rcnn_ros/Result.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/rgbd.hpp>
#include "voxblox_ros/tsdf_seq_receiver.h"

#include "voxblox/core/voxel.h"

#include <image_geometry/pinhole_camera_model.h>

namespace voxblox {

// TODO try to fix templates and set server here
class SegReceiver {
 public:
  SegReceiver(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);

  // virtual void setServerPointer(std::shared_ptr<SegServer<T>> server_ptr);

  std::vector<std::string> label_lookup;

  // std::shared_ptr<SegServer<T>> server;
  Transformer transformer_;

  void initLookup(std::shared_ptr<LabelLookup> lookup);
  std::shared_ptr<LabelLookup> lookup;

 protected:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  bool initedLookup;
};

}  // namespace voxblox
#endif