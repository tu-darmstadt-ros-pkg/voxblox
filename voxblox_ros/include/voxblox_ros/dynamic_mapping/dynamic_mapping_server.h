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

constexpr float kDefaultMaxIntensity = 100.0;

class DynamicMappingServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DynamicMappingServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  DynamicMappingServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
             const TsdfMap::Config& config,
             const TsdfIntegratorBase::Config& integrator_config,
             const MeshIntegratorConfig& mesh_config,
             const DynamicMapper::Config& dynamic_mapper_config);
  virtual ~DynamicMappingServer() {}

  void getServerConfigFromRosParam(const ros::NodeHandle& nh_private);

  void DynamicMappingCallback( const sensor_msgs::PointCloud2::Ptr& pointcloud_msg);

  virtual void updateMesh();
  /// Batch update.
  virtual bool generateMesh();

  bool saveBackgroundMeshCallback(std_srvs::Empty::Request& request, // NOLINT
                        std_srvs::Empty::Response& response);        // NOLINT
  bool saveObjectsCallback(std_srvs::Empty::Request& request,        // NOLINT
                        std_srvs::Empty::Response& response);        // NOLINT
  bool clearMapCallback(std_srvs::Empty::Request& request,           // NOLINT
                        std_srvs::Empty::Response& response);        // NOLINT

  bool generateMeshCallback(std_srvs::Empty::Request& request,       // NOLINT
                            std_srvs::Empty::Response& response);    // NOLINT

  void visualizeBBoxes(std::string frame_id);

  void updateMeshEvent(const ros::TimerEvent& event);

  void setWorldFrame(const std::string& world_frame) {
    world_frame_ = world_frame;
  }
  std::string getWorldFrame() const { return world_frame_; }

  /// CLEARS THE ENTIRE MAP!
  virtual void clear();


 protected:

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  /// Data subscribers.
  ros::Subscriber pointcloud_sub_;
  ros::Subscriber pointcloud_debug_sub_;
  ros::Subscriber freespace_pointcloud_sub_;

  /// Publish markers for visualization.
  ros::Publisher mesh_pub_;
  ros::Publisher multi_mesh_pub_;
  ros::Publisher tsdf_pointcloud_pub_;
  ros::Publisher surface_pointcloud_pub_;
  ros::Publisher tsdf_slice_pub_;
  ros::Publisher occupancy_marker_pub_;
  ros::Publisher icp_transform_pub_;

  ros::Publisher bbox_vis_pub_;
  ros::Publisher label_vis_pub_;

  ros::Publisher debug_pointcloud_pub_;

  /// Publish the complete map for other nodes to consume.
  ros::Publisher tsdf_map_pub_;

  /// Subscriber to subscribe to another node generating the map.
  ros::Subscriber tsdf_map_sub_;

  // Services.
  ros::ServiceServer generate_mesh_srv_;
  ros::ServiceServer clear_map_srv_;
  ros::ServiceServer save_map_srv_;
  ros::ServiceServer load_map_srv_;
  ros::ServiceServer publish_pointclouds_srv_;
  ros::ServiceServer publish_tsdf_map_srv_;

  /// Tools for broadcasting TFs.
  tf::TransformBroadcaster tf_broadcaster_;

  // Timers.
  ros::Timer update_mesh_timer_;
  ros::Timer publish_map_timer_;

  bool verbose_;

  /**
   * Global/map coordinate frame. Will always look up TF transforms to this
   * frame.
   */
  std::string world_frame_;
  /**
   * Name of the ICP corrected frame. Publishes TF and transform topic to this
   * if ICP on.
   */
  std::string icp_corrected_frame_;
  /// Name of the pose in the ICP correct Frame.
  std::string pose_corrected_frame_;

  /// Delete blocks that are far from the system to help manage memory
  double max_block_distance_from_body_;

  /// Pointcloud visualization settings.
  double slice_level_;

  /// If the system should subscribe to a pointcloud giving points in freespace
  bool use_freespace_pointcloud_;

  /**
   * Mesh output settings. Mesh is only written to file if mesh_filename_ is
   * not empty.
   */
  std::string mesh_filename_;
  /// How to color the mesh.
  ColorMode color_mode_;

  std::vector<std::string> semantic_classes_;

  /// Colormap to use for intensity pointclouds.
  std::shared_ptr<ColorMap> color_map_;

  /// Will throttle to this message rate.
  ros::Duration min_time_between_msgs_;

  /// What output information to publish
  bool publish_pointclouds_on_update_;
  bool publish_slices_;
  bool publish_pointclouds_;
  bool publish_tsdf_map_;

  /// Whether to save the latest mesh message sent (for inheriting classes).
  bool cache_mesh_;

  /**
   *Whether to enable ICP corrections. Every pointcloud coming in will attempt
   * to be matched up to the existing structure using ICP. Requires the initial
   * guess from odometry to already be very good.
   */
  bool enable_icp_;
  /**
   * If using ICP corrections, whether to store accumulate the corrected
   * transform. If this is set to false, the transform will reset every
   * iteration.
   */
  bool accumulate_icp_corrections_;

  /// Subscriber settings.
  int pointcloud_queue_size_;
  int num_subscribers_tsdf_map_;

  DynamicMapper dynamic_mapper;

  // Maps and integrators.
  std::shared_ptr<TsdfMap> tsdf_map_;
  std::unique_ptr<TsdfIntegratorBase> tsdf_integrator_;

  /// ICP matcher
  std::shared_ptr<ICP> icp_;

  // Mesh accessories.
  std::shared_ptr<MeshLayer> mesh_layer_;
  std::unique_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;
  /// Optionally cached mesh message.
  voxblox_msgs::MultiMesh cached_mesh_msg_;

  /**
   * Transformer object to keep track of either TF transforms or messages from
   * a transform topic.
   */
  Transformer transformer_;

  Transformation T_G_C_;
  float offset_; //TODO remove
  /**
   * Queue of incoming pointclouds, in case the transforms can't be immediately
   * resolved.
   */
  std::queue<sensor_msgs::PointCloud2::Ptr> pointcloud_queue_;
  std::queue<sensor_msgs::PointCloud2::Ptr> freespace_pointcloud_queue_;

  // Last message times for throttling input.
  ros::Time last_msg_time_ptcloud_;
  ros::Time last_msg_time_freespace_ptcloud_;

  /// Current transform corrections from ICP.
  Transformation icp_corrected_transform_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_DYNAMIC_MAPPING_SERVER_H_
