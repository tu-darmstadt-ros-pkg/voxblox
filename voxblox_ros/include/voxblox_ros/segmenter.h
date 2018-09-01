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
#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/io/png_io.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/io/point_cloud_image_extractors.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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


namespace voxblox {

class Segmenter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Segmenter(const ros::NodeHandle& nh);

  void segmentPointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const pcl::PointCloud<int>& sub_cloud_indices, LabelIndexMap& segment_map);

  Color getSegmentColor(uint segment);

  const std::map<uint, Color>& getColorMap() { return segment_colors_; }

 protected:

  void initColorMap(int num_entries);

  void publishImg(const cv::Mat& img, const std_msgs::Header& header, ros::Publisher& pub);
  void publishNormalsImg(pcl::PointCloud<pcl::Normal>::ConstPtr normals, const std_msgs::Header& header, ros::Publisher& pub);

  void detectConcaveBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                           const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                           cv::Mat &edge_img);

  void detectGeometricalBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                           const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                           cv::Mat& edge_img);

  void detectRgbBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                           cv::Mat& edge_img);

  void getNeighbors(int row, int col, int height, int width, pcl::PointIndices& neighbors);

  void enumerateSegments(const LabelIndexMap& segment_map, const ImageIndexList& segment_centroids,
                         cv::Mat& img);

  cv::Mat colorizeSegmentationImg(const cv::Mat& seg_img, const LabelIndexMap& segment_map);

  ros::NodeHandle nh_private_;

  std::map<uint, Color> segment_colors_;

  float canny_sigma_;
  int canny_kernel_size_;
  float min_concavity_;
  float max_dist_step_;

  ros::Publisher edge_img_pub_;
  ros::Publisher segmentation_pub_;
  ros::Publisher concave_edges_pub_;
  ros::Publisher depth_disc_edges_pub_;
  ros::Publisher rgb_edges_pub_;
  ros::Publisher normals_pub_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_SEGMENTER_H_
