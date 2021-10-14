#ifndef VOXBLOX_ROS_DYNAMIC_MAPPING_SERVER_H_
#define VOXBLOX_ROS_DYNAMIC_MAPPING_SERVER_H_

#include <memory>
#include <queue>
#include <string>

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>

#include <voxblox/alignment/icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>
#include <voxblox_msgs/FilePath.h>
#include <voxblox_msgs/MultiMesh.h>
#include <voxblox_msgs/Mesh.h>

#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"

#include "voxblox_ros/dynamic_mapping/common.h"
#include "voxblox_ros/dynamic_mapping/dynamic_mapper.h"

namespace voxblox {

class DynamicMappingServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DynamicMappingServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  DynamicMappingServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
             const DynamicMapper::Config& dynamic_mapper_config);
  virtual ~DynamicMappingServer() {}

  void getServerConfigFromRosParam(const ros::NodeHandle& nh_private);

  void DynamicMappingCallback( const sensor_msgs::PointCloud2::Ptr& pointcloud_msg);

  virtual void updateMesh();

  bool saveMeshCallback(std_srvs::Empty::Request& request,
                        std_srvs::Empty::Response& response);

  bool saveObjectTrajectories(std_srvs::Empty::Request& request,
                              std_srvs::Empty::Response& response);

  void visualizeBBoxes(const sensor_msgs::PointCloud2::Ptr& pointcloud_msg);

  void updateMeshEvent(const ros::TimerEvent& event);

 protected:

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  /// Data subscribers.
  ros::Subscriber pointcloud_sub_;

  /// Publish markers for visualization.
  ros::Publisher multi_mesh_pub_;

  ros::Publisher bbox_vis_pub_;
  ros::Publisher label_vis_pub_;

  // Services.
  ros::ServiceServer generate_mesh_srv_;
  ros::ServiceServer generate_obj_trajectories_srv_;

  bool verbose_;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  std::string world_frame_;

  DynamicMapper dynamic_mapper;

  /// Delete blocks that are far from the system to help manage memory
  double max_block_distance_from_body_;

  /// How to color the mesh.
  ColorMode color_mode_;

  std::vector<std::string> semantic_classes_;
  std::vector<int> non_rigid_classes_;

  /// Whether to save the latest mesh message sent (for inheriting classes).
  bool cache_mesh_;

  /// Subscriber settings.
  int pointcloud_queue_size_;

  /// Optionally cached mesh message.
  voxblox_msgs::MultiMesh cached_mesh_msg_;

  /**
   * Transformer object to keep track of either TF transforms or messages from
   * a transform topic.
   */
  Transformer transformer_;
  Transformation T_G_C_;

};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_DYNAMIC_MAPPING_SERVER_H_
