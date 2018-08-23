#include "voxblox_ros/segmenter.h"

namespace voxblox {

Segmenter::Segmenter(const ros::NodeHandle& nh_private) :
  nh_private_(nh_private) {

  edge_img_pub_ = nh_private_.advertise<sensor_msgs::Image>("all_edges", 1, true);
  segmentation_pub_ = nh_private_.advertise<sensor_msgs::Image>("segmentation", 1, true);
  concave_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("concave_edges", 1, true);
  depth_disc_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_disc_edges", 1, true);
  rgb_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("rgb_edges", 1, true);

  initColorMap(255);

  nh_private_.param("seg_canny_low_tresh", canny_low_tresh_, 50);
  nh_private_.param("seg_canny_high_tresh", canny_high_tresh_, 150);
  nh_private_.param("seg_canny_kernel_size", canny_kernel_size_, 3);
  nh_private_.param("seg_min_concavity", min_concavity_, 0.97f);
  nh_private_.param("seg_max_dist_step_", max_dist_step_, 0.005f);
}

void Segmenter::segmentPointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const pcl::PointCloud<int>& sub_cloud_indices,
                                  Labels& segments, LabelIndexMap& segment_map) {

  if (cloud->points.empty())
    return;

  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  segments.clear();
  segment_map.clear();

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  cv::UMat edge_img_concave(height, width, CV_8UC1, cv::Scalar(0));
  cv::UMat edge_img_depth_disc(height, width, CV_8UC1, cv::Scalar(0));
  cv::UMat edge_img_canny(height, width, CV_8UC1, cv::Scalar(0));

  timing::Timer seg_normal_estimation_timer("seg_normal_estimation");

  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
  normal_estimation.setNormalEstimationMethod(normal_estimation.AVERAGE_DEPTH_CHANGE);
  //normal_estimation.setMaxDepthChangeFactor(0.02f);
  //normal_estimation.setNormalSmoothingSize(5.0f);
  //normal_estimation.setBorderPolicy();
  normal_estimation.setInputCloud(cloud);
  normal_estimation.compute(*normals);

  seg_normal_estimation_timer.Stop();

  timing::Timer seg_concave_boundaries_timer("seg_concave_boundaries");
  detectConcaveBoundaries(cloud, normals, edge_img_concave);
  seg_concave_boundaries_timer.Stop();

  timing::Timer seg_depth_disc_boundaries_timer("seg_depth_disc_boundaries");
  detectGeometricalBoundaries(cloud, normals, edge_img_depth_disc);
  seg_depth_disc_boundaries_timer.Stop();

  timing::Timer seg_canny_boundaries_timer("seg_canny_boundaries");
  detectRgbBoundaries(cloud, edge_img_canny);
  seg_canny_boundaries_timer.Stop();

  timing::Timer seg_connected_components_timer("seg_connected_components");

  cv::UMat edge_img;
  cv::min(edge_img_concave, edge_img_depth_disc, edge_img);
  cv::min(edge_img, edge_img_canny, edge_img);

  // increase the borders a little bit before the segmentation
  cv::morphologyEx(edge_img, edge_img, cv::MORPH_OPEN, cv::UMat(), cv::Point(-1,-1), 1);

  cv::UMat labels;
  int num_labels = cv::connectedComponents(edge_img, labels, 8);

  ROS_INFO_STREAM("found " << num_labels << " labels!");

  seg_connected_components_timer.Stop();

  labels.convertTo(labels, CV_8U);

  segments.reserve(sub_cloud_indices.size());
  ImageIndexList segment_centroids(num_labels);

  for (size_t i = 0; i < sub_cloud_indices.size(); ++i) {

    int sub_cloud_index = sub_cloud_indices[i];
    const pcl::PointXYZRGB& p = cloud->points[sub_cloud_index];

    if (!std::isfinite(p.x) ||
        !std::isfinite(p.y) ||
        !std::isfinite(p.z)) {
      continue;
    }

    int col = sub_cloud_index % width;
    int row = sub_cloud_index / width;
    uchar label = labels.getMat(cv::ACCESS_READ).at<uchar>(row, col);

    segments.push_back(label);
    segment_map[label].emplace_back(i);
    segment_centroids[label] += ImageIndex(row, col);
  }

  cv::cvtColor(labels, labels, CV_GRAY2BGR);

  // Generate random colors
  std::vector<cv::Vec3b> colors;

  for (uint i = 0; i < 256; i++) {
    Color c = getSegmentColor(i);
    colors.push_back(cv::Vec3b(c.b, c.g, c.r));
  }

  cv::LUT(labels, colors, labels);
  enumerateSegments(segment_map, segment_centroids, labels);

  publishImg(edge_img, pcl_conversions::fromPCL(cloud->header), edge_img_pub_);
  publishImg(edge_img_concave, pcl_conversions::fromPCL(cloud->header), concave_edges_pub_);
  publishImg(edge_img_depth_disc, pcl_conversions::fromPCL(cloud->header), depth_disc_edges_pub_);
  publishImg(labels, pcl_conversions::fromPCL(cloud->header), segmentation_pub_);
  publishImg(edge_img_canny, pcl_conversions::fromPCL(cloud->header), rgb_edges_pub_);
}

void Segmenter::detectConcaveBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                         const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                         cv::UMat& edge_img) {

  pcl::PointIndices neighbors;

  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {

      getNeighbors(row, col, height, width, neighbors);

      float min_concavity = std::numeric_limits<float>::max();

      const Eigen::Vector4f& p = cloud->at(col, row).getVector4fMap();
      const Eigen::Vector4f& n = normals->at(col, row).getNormalVector4fMap();

      for (int i: neighbors.indices) {

        const Eigen::Vector4f& p_i = cloud->points[i].getVector4fMap();

        if ((p_i - p).dot(n) > 0) {
          min_concavity = std::min(1.0f, min_concavity);
        }
        else {
          const Eigen::Vector4f& n_i = normals->points[i].getNormalVector4fMap();
          min_concavity = std::min(n.dot(n_i), min_concavity);
        }
      }

      if (min_concavity >= min_concavity_)
      {
        edge_img.getMat(cv::ACCESS_WRITE).at<uchar>(row, col) = 255;
      }
    }
  }
}

void Segmenter::detectGeometricalBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                         const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                         cv::UMat& edge_img) {

  pcl::PointIndices neighbors;

  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {

      getNeighbors(row, col, height, width, neighbors);

      float max_dist = -std::numeric_limits<float>::max();

      const Eigen::Vector4f& p = cloud->at(col, row).getVector4fMap();
      const Eigen::Vector4f& n = normals->at(col, row).getNormalVector4fMap();

      for (int i: neighbors.indices) {

        const Eigen::Vector4f& p_i = cloud->points[i].getVector4fMap();
        max_dist = std::max(std::abs((p_i - p).dot(n)), max_dist);

      }

      if (max_dist <= max_dist_step_)
      {
        edge_img.getMat(cv::ACCESS_WRITE).at<uchar>(row, col) = 255;
      }
    }
  }
}

void Segmenter::detectRgbBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                    cv::UMat& edge_img) {
  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  // Create cv::UMat
  cv::UMat rgb_img = cv::UMat(height, width, CV_8UC3);
  cv::UMat gray_img = cv::UMat(height, width, CV_8UC1);

  // pcl::PointCloud to cv::UMat
  for(int y = 0; y < rgb_img.rows; y++) {
    for(int x = 0; x < rgb_img.cols; x++) {
      const pcl::PointXYZRGB& point = cloud->at(x, y);
      rgb_img.getMat(cv::ACCESS_WRITE).at<cv::Vec3b>(y, x)[0] = point.b;
      rgb_img.getMat(cv::ACCESS_WRITE).at<cv::Vec3b>(y, x)[1] = point.g;
      rgb_img.getMat(cv::ACCESS_WRITE).at<cv::Vec3b>(y, x)[2] = point.r;
    }
  }

  cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);

  // Reduce noise with blurring
  cv::blur(gray_img, edge_img, cv::Size(3,3));

  // Canny detector
  cv::Canny(edge_img, edge_img, canny_low_tresh_, canny_high_tresh_, canny_kernel_size_);
  cv::bitwise_not(edge_img, edge_img);
}

void Segmenter::getNeighbors(int row, int col, int height, int width, pcl::PointIndices& neighbors) {

  neighbors.indices.clear();
  int max_index = width * height -1;

  int north = (row - 1) * width + col;
  int south = (row + 1) * width + col;
  int east = row * width + (col + 1);
  int west = row * width + (col - 1);

  int north_west = (row + 1) * width + (col - 1);
  int north_east = (row + 1) * width + (col + 1);
  int south_west = (row - 1) * width + (col - 1);
  int south_east = (row - 1) * width + (col + 1);

  if (north >= 0 && north <= max_index) { neighbors.indices.push_back(north); }
  if (south >= 0 && south <= max_index) { neighbors.indices.push_back(south); }
  if (east >= 0 && east <= max_index) { neighbors.indices.push_back(east); }
  if (west >= 0 && west <= max_index) { neighbors.indices.push_back(west); }
  if (north_west >= 0 && north_west <= max_index) { neighbors.indices.push_back(north_west); }
  if (north_east >= 0 && north_east <= max_index) { neighbors.indices.push_back(north_east); }
  if (south_west >= 0 && south_west <= max_index) { neighbors.indices.push_back(south_west); }
  if (south_east >= 0 && south_east <= max_index) { neighbors.indices.push_back(south_east); }
}

void Segmenter::publishImg(const cv::UMat& img, const std_msgs::Header& header, ros::Publisher& pub) {

  sensor_msgs::ImagePtr msg;

  if (img.channels() > 1)
    msg = cv_bridge::CvImage(header, "bgr8", img.getMat(cv::ACCESS_READ)).toImageMsg();
  else
    msg = cv_bridge::CvImage(header, "mono8", img.getMat(cv::ACCESS_READ)).toImageMsg();

  pub.publish(msg);
}

Color Segmenter::getSegmentColor(uint segment) {

  //make sure the first segment is always black
  if (segment == 0)
    return Color::Black();

  auto it = segment_colors_.find(segment);

  if (it != segment_colors_.end()) {
    return it->second;
  } else {

    // seed is not set on purpose
    uint8_t r = static_cast<uint8_t>(rand() % 256);
    uint8_t g = static_cast<uint8_t>(rand() % 256);
    uint8_t b = static_cast<uint8_t>(rand() % 256);

    Color c(r, g, b);
    segment_colors_.emplace(segment, c);

    return c;
  }
}

// Adapted color map function for the PASCAL VOC data set.
void Segmenter::initColorMap(int num_entries) {
  segment_colors_.emplace(0, Color::Black());

  for (int i = 1; i < num_entries; i++) {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t c = static_cast<uint8_t>(i);

    for (int j = 0; j < 8; j++){
      r |= ((c >> 0) & 1) << (7-j);
      g |= ((c >> 1) & 1) << (7-j);
      b |= ((c >> 2) & 1) << (7-j);
      c = c >> 3;
    }

    Color color(r, g, b);
    segment_colors_.emplace(i, color);
  }
}

void Segmenter::enumerateSegments(const LabelIndexMap& segment_map, const ImageIndexList& segment_centroids, cv::UMat& img) {

  const auto font = cv::FONT_HERSHEY_PLAIN;
  const double font_scale = 2;
  const int font_thickness = 2;

  for (auto segment: segment_map) {

    if (segment.first == 0)
      continue;

    int num_pixel = static_cast<int>(segment.second.size());

    if (num_pixel < 20)
      continue;

    const std::string text = std::to_string(segment.first);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, font_thickness, &baseline);

    int avg_row = segment_centroids[segment.first](0)/num_pixel;
    int avg_col = segment_centroids[segment.first](1)/num_pixel;

    avg_row -= text_size.height / 2;
    avg_col -= text_size.width / 2;

    cv::putText(img, std::to_string(segment.first), cv::Point(avg_col, avg_row),
                font, font_scale, cvScalar(255, 255, 255), font_thickness);
  }
}
}  // namespace voxblox
