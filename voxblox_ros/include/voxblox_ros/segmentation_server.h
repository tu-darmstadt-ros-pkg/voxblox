#ifndef VOXBLOX_ROS_SEGMENTATION_SERVER_H_
#define VOXBLOX_ROS_SEGMENTATION_SERVER_H_

#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <voxblox/core/voxel.h>
#include <voxblox/integrator/intensity_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox/utils/segment_tools.h>

#include "voxblox_ros/tsdf_server.h"
#include "voxblox_ros/ros_params.h"
#include "voxblox_ros/segmenter.h"

#include <voxblox_msgs/PointCloudList.h>

#include <pcl/keypoints/uniform_sampling.h>

namespace voxblox {

class SegmentationServer : public TsdfServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> RgbdSyncPolicy;

  SegmentationServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  SegmentationServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private, const SegmentedTsdfIntegrator::Config& seg_integrator_config);
  virtual ~SegmentationServer() {}

  virtual void updateMesh();
  virtual void publishPointclouds();
  virtual void processPointCloudMessageAndInsert(const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
                                                 const Transformation& T_G_C,
                                                 const bool is_freespace_pointcloud);

  void rgbdCallback(const sensor_msgs::PointCloud2ConstPtr& pointcloud, const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img,
                    const sensor_msgs::CameraInfoConstPtr& color_cam_info, const sensor_msgs::CameraInfoConstPtr& depth_cam_info);

 protected:

  void recolorVoxbloxMeshMsgBySegmentation(voxblox_msgs::Mesh* mesh_msg);
  void integrateSegmentation(const sensor_msgs::PointCloud2ConstPtr& pointcloud, const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img,
                             const sensor_msgs::CameraInfoConstPtr& color_cam_info, const sensor_msgs::CameraInfoConstPtr& depth_cam_info);
  inline void fillPointcloudWithMesh(const MeshLayer::ConstPtr& mesh_layer, pcl::PointCloud<pcl::PointNormal>& pointcloud);
  void publishSegmentPointclouds();

  // Publish markers for visualization.
  ros::Publisher segment_pointclouds_pub_;
  ros::Publisher segmentation_mesh_pub_;

  std::shared_ptr<SegmentedTsdfMap> seg_tsdf_map_;
  std::unique_ptr<SegmentedTsdfIntegrator> seg_tsdf_integrator_;
  Segmenter segmenter_;
  SegmentTools::Ptr segment_tool_;

  message_filters::Subscriber<sensor_msgs::PointCloud2> point_cloud_sub_;
  message_filters::Subscriber<sensor_msgs::Image> color_image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> color_info_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> depth_info_sub_;
  message_filters::Synchronizer<RgbdSyncPolicy> msg_sync_;

  Transformation T_G_C_current_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_SEGMENTATION_SERVER_H_
