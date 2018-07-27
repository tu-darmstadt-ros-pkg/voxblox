#ifndef VOXBLOX_ROS_SEGMENTER_H_
#define VOXBLOX_ROS_SEGMENTER_H_

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/MarkerArray.h>
#include <memory>
#include <string>

#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/io/png_io.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/filters/fast_bilateral.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox_msgs/Mesh.h>

#include <voxblox_msgs/FilePath.h>
#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"

#include <cv_bridge/cv_bridge.h>

namespace voxblox {

class Segmenter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Segmenter(const ros::NodeHandle& nh);

  void segmentPointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, Labels& segments, LabelIndexMap& segment_map);

  Color getSegmentColor(uint segment);

 protected:

  void publishImg(const cv::Mat& img, const std_msgs::Header& header, ros::Publisher& pub);

  void detectConcaveBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                           const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                           cv::Mat &edge_img);

  void detectGeometricalBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                           const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                           cv::Mat& edge_img);

  void getNeighbors(int row, int col, int height, int width, pcl::PointIndices& neighbors);

  void enumerateSegments(const LabelIndexMap &segment_map, cv::Mat &img);

  ros::NodeHandle nh_private_;

  bool verbose_;

  // Global/map coordinate frame. Will always look up TF transforms to this
  // frame.
  std::string world_frame_;

  // Delete blocks that are far from the system to help manage memory
  double max_block_distance_from_body_;

  // Pointcloud visualization settings.
  double slice_level_;

  // If the system should subscribe to a pointcloud giving points in freespace
  bool use_freespace_pointcloud_;

  std::map<uint, Color> segment_colors_;

  ros::Publisher edge_img_pub_;
  ros::Publisher segmentation_pub_;
  ros::Publisher concave_edges_pub_;
  ros::Publisher depth_disc_edges_pub_;

};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_SEGMENTER_H_
