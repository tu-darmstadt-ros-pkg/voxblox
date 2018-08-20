#ifndef VOXBLOX_ROS_SEGMENTATION_SERVER_H_
#define VOXBLOX_ROS_SEGMENTATION_SERVER_H_

#include <cv_bridge/cv_bridge.h>
#include <memory>
#include <sensor_msgs/Image.h>

#include <voxblox/core/voxel.h>
#include <voxblox/integrator/intensity_integrator.h>
#include <voxblox/utils/color_maps.h>

#include "voxblox_ros/tsdf_server.h"
#include "voxblox_ros/ros_params.h"
#include "voxblox_ros/segmenter.h"

#include <pcl/keypoints/uniform_sampling.h>

namespace voxblox {

class SegmentationServer : public TsdfServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SegmentationServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  SegmentationServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private, const SegmentedTsdfIntegrator::Config& seg_integrator_config);
  virtual ~SegmentationServer() {}

  virtual void updateMesh();
  virtual void processPointCloudMessageAndInsert(const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
                                                 const Transformation& T_G_C,
                                                 const bool is_freespace_pointcloud);
  virtual void publishPointclouds();

 protected:

  void recolorVoxbloxMeshMsgBySegmentation(voxblox_msgs::Mesh* mesh_msg);
  void integrateSegmentation(const sensor_msgs::PointCloud2::Ptr pointcloud_msg, const Transformation& T_G_C);

  // Publish markers for visualization.
  ros::Publisher segmentation_pointcloud_pub_;
  ros::Publisher segmentation_mesh_pub_;

  std::shared_ptr<SegmentedTsdfMap> seg_tsdf_map_;
  std::unique_ptr<SegmentedTsdfIntegrator> seg_tsdf_integrator_;
  Segmenter segmenter_;

  // Subscriber for the camera info needed to do the reprojections
  ros::Subscriber depth_cam_info_sub_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_SEGMENTATION_SERVER_H_
