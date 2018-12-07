#ifndef VOXBLOX_ROS_SEGMENTER_H_
#define VOXBLOX_ROS_SEGMENTER_H_

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

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
#include <pcl/octree/octree_pointcloud_pointvector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/rgbd.hpp>
#include <opencv2/ximgproc.hpp>

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
#include "voxblox_ros/image_operations.h"

namespace voxblox {

class Segmenter {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Segmenter(const ros::NodeHandle& nh, float voxel_size);

  typedef boost::shared_ptr<cv_bridge::CvImage> CvImagePtr;
  typedef boost::shared_ptr<cv_bridge::CvImage const> CvImageConstPtr;

  typedef pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ> Octree;

  void segmentRgbdImage(const cv::Mat& color_img, const sensor_msgs::CameraInfoConstPtr& color_cam_info,
                        const cv::Mat& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
                        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, Pointcloud& cloud_out, LabelIndexMap& segment_map);

  Color getSegmentColor(uint segment);

  int getNormalsWindowSize() const { return normals_window_size_; }
  const std::map<uint, Color>& getColorMap() { return segment_colors_; }

 protected:

  void initColorMap(int num_entries);

  void publishImg(const cv::Mat& img, const std_msgs::Header& header, ros::Publisher& pub);
  void publishNormalsImg(pcl::PointCloud<pcl::Normal>::ConstPtr normals, const std_msgs::Header& header, ros::Publisher& pub);
  void publishNormalsImg(const cv::Mat& normals, const std_msgs::Header& header, ros::Publisher& pub);

  cv::Mat detectConcaveBoundaries(const cv::Mat& points, const cv::Mat& normals);
  cv::Mat detectDepthDiscBoundaries(const cv::Mat& points, const cv::Mat& normals);
  cv::Mat detectCannyEdgesMono(const cv::Mat& color_img);
  cv::Mat detectCannyEdgesH1H2H3(const cv::Mat& color_img);
  cv::Mat detectStructuredEdges(const cv::Mat& color_img);

  cv::Mat inpaintDepth(const cv::Mat& depth_img);
  cv::Mat filterImage(cv::Mat& depth_img);

  cv::Mat applyCanny(const cv::Mat& gray_img);

  void getNeighbors(int row, int col, int height, int width, int step_size, std::vector<cv::Point2i>& neighbors);

  void enumerateSegments(const LabelIndexMap& segment_map,
                         const ImageIndexList& segment_centroids, cv::Mat& img);
  ImageIndexList computeSegmentCentroids(const LabelIndexMap& segment_map, cv::Mat& segmentation_img);

  cv::Mat colorizeSegmentationImg(const cv::Mat& seg_img, const LabelIndexMap& segment_map);
  cv::Mat estimateNormals(const cv::Mat& points_3d, const cv::Matx33d& intrinsic_matrix);
  cv::Mat estimateNormalsCrossProduct(const cv::Mat& depth_img);

  void applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, const cv::Mat& segmentation_img, Pointcloud& cloud_out, LabelIndexMap& segment_map);

  void assignEdgePoints(int radius, double max_distance, const cv::Mat& points_3d, cv::Mat& img);
  ushort assignEdgePoint(int row, int col, int radius, double max_distance, const cv::Mat& points_3d, const cv::Mat& img);

  ros::NodeHandle nh_private_;

  std::map<uint, Color> segment_colors_;

  float voxel_size_;

  float canny_sigma_;
  int canny_kernel_size_;
  float min_concavity_;
  float max_dist_step_;

  double concave_weight_;
  double color_weight_;
  double edge_treshold_;

  int edges_window_size_;
  int normals_window_size_;

  cv::Ptr<cv::ximgproc::StructuredEdgeDetection> structured_edges_;

  ros::Publisher edge_img_pub_;
  ros::Publisher segmentation_pub_;
  ros::Publisher concave_edges_pub_;
  ros::Publisher depth_disc_edges_pub_;
  ros::Publisher rgb_edges_pub_;
  ros::Publisher normals_pub_;
  ros::Publisher depth_inpainted_pub_;
  ros::Publisher depth_filtered_pub_;
  ros::Publisher depth_input_pub_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_SEGMENTER_H_
