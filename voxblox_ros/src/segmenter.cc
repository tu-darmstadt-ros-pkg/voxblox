#include "voxblox_ros/segmenter.h"

namespace voxblox {

Segmenter::Segmenter(const ros::NodeHandle& nh_private) :
  nh_private_(nh_private) {

  edge_img_pub_ = nh_private_.advertise<sensor_msgs::Image>("all_edges", 1, true);
  segmentation_pub_ = nh_private_.advertise<sensor_msgs::Image>("segmentation", 1, true);
  concave_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("concave_edges", 1, true);
  depth_disc_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_disc_edges", 1, true);
  rgb_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("rgb_edges", 1, true);
  normals_pub_ = nh_private_.advertise<sensor_msgs::Image>("normals", 1, true);

  initColorMap(255);

  nh_private_.param("seg_canny_sigma", canny_sigma_, 0.4f);
  nh_private_.param("seg_canny_kernel_size", canny_kernel_size_, 3);
  nh_private_.param("seg_min_concavity", min_concavity_, 0.97f);
  nh_private_.param("seg_max_dist_step_", max_dist_step_, 0.005f);
}

void Segmenter::segmentPointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const pcl::PointCloud<int>& sub_cloud_indices, LabelIndexMap& segment_map) {

  if (cloud->points.empty())
    return;

  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  segment_map.clear();

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  cv::Mat edge_img_concave(height, width, CV_8UC1, cv::Scalar(0));
  cv::Mat edge_img_depth_disc(height, width, CV_8UC1, cv::Scalar(0));
  cv::Mat edge_img_canny(height, width, CV_8UC1, cv::Scalar(0));

  timing::Timer seg_normal_estimation_timer("seg_normal_estimation");

  pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
  normal_estimation.setNormalEstimationMethod(normal_estimation.COVARIANCE_MATRIX);
  normal_estimation.setMaxDepthChangeFactor(0.05f);
  normal_estimation.setNormalSmoothingSize(10.0f);
  normal_estimation.setBorderPolicy(normal_estimation.BORDER_POLICY_MIRROR);
  normal_estimation.setInputCloud(cloud);
  normal_estimation.setRectSize(5, 5);
  normal_estimation.setDepthDependentSmoothing(false);
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

  cv::Mat edge_img = cv::min(edge_img_concave, edge_img_depth_disc);
  edge_img = cv::min(edge_img, edge_img_canny);

  // increase the borders a little bit before the segmentation
  cv::morphologyEx(edge_img, edge_img, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 1);

  cv::Mat segmentation_img;
  int num_labels = cv::connectedComponents(edge_img, segmentation_img, 8, CV_16U);

  ROS_INFO_STREAM("found " << num_labels << " labels!");

  seg_connected_components_timer.Stop();

  ImageIndexList segment_centroids(static_cast<unsigned long>(num_labels));
  for (size_t i = 0; i < num_labels; num_labels++) {
    segment_centroids[i] += ImageIndex(0, 0);
  }

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
    ushort label = segmentation_img.at<ushort>(row, col);

    segment_map[label].emplace_back(i);
    segment_centroids[label] += ImageIndex(row, col);
  }

  if (segmentation_pub_.getNumSubscribers() > 0) {
    cv::Mat segmentation_img_color = colorizeSegmentationImg(segmentation_img, segment_map);
    enumerateSegments(segment_map, segment_centroids, segmentation_img_color);

    publishImg(segmentation_img_color, pcl_conversions::fromPCL(cloud->header), segmentation_pub_);
  }

  publishImg(edge_img, pcl_conversions::fromPCL(cloud->header), edge_img_pub_);
  publishImg(edge_img_concave, pcl_conversions::fromPCL(cloud->header), concave_edges_pub_);
  publishImg(edge_img_depth_disc, pcl_conversions::fromPCL(cloud->header), depth_disc_edges_pub_);
  publishImg(edge_img_canny, pcl_conversions::fromPCL(cloud->header), rgb_edges_pub_);
  publishNormalsImg(normals, pcl_conversions::fromPCL(cloud->header), normals_pub_);
}

cv::Mat Segmenter::colorizeSegmentationImg(const cv::Mat& seg_img, const LabelIndexMap& segment_map) {

  cv::Mat seg_img_color(seg_img);
  seg_img_color.convertTo(seg_img_color, CV_8U);
  cv::cvtColor(seg_img_color, seg_img_color, CV_GRAY2BGR);

  std::vector<cv::Vec3b> colors;
  colors.reserve(segment_map.size());

  // Prepare the colors
  for (uint i = 0; i < segment_map.size(); i++) {
    Color c = getSegmentColor(i);
    colors.push_back(cv::Vec3b(c.b, c.g, c.r));
  }

  cv::MatIterator_<cv::Vec3b> color_it = seg_img_color.begin<cv::Vec3b>();
  cv::MatConstIterator_<ushort> gray_it = seg_img.begin<ushort>();
  for(; color_it != seg_img_color.end<cv::Vec3b>() || gray_it != seg_img.end<ushort>(); ++color_it, ++gray_it )
  {
      (*color_it)[0] = colors[(*gray_it)][0];
      (*color_it)[1] = colors[(*gray_it)][1];
      (*color_it)[2] = colors[(*gray_it)][2];
  }

  return seg_img_color;
}

void Segmenter::detectConcaveBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                         const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                         cv::Mat& edge_img) {

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
        edge_img.at<uchar>(row, col) = 255;
      }
    }
  }
}

void Segmenter::detectGeometricalBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                         const pcl::PointCloud<pcl::Normal>::ConstPtr normals,
                         cv::Mat& edge_img) {

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
        edge_img.at<uchar>(row, col) = 255;
      }
    }
  }
}

uchar imageMedian(const cv::Mat& img) {
  std::vector<uchar> vec_from_mat(img.begin<uchar>(), img.end<uchar>());
  std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());
  return vec_from_mat[vec_from_mat.size() / 2];
}

void Segmenter::detectRgbBoundaries(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud,
                                    cv::Mat& edge_img) {
  int width = static_cast<int>(cloud->width);
  int height = static_cast<int>(cloud->height);

  // Create cv::Mat
  cv::Mat rgb_img = cv::Mat(height, width, CV_8UC3);
  cv::Mat gray_img = cv::Mat(height, width, CV_8UC1);

  // pcl::PointCloud to cv::Mat
  for(int y = 0; y < rgb_img.rows; y++) {
    for(int x = 0; x < rgb_img.cols; x++) {
      const pcl::PointXYZRGB& point = cloud->at(x, y);
      rgb_img.at<cv::Vec3b>(y, x)[0] = point.b;
      rgb_img.at<cv::Vec3b>(y, x)[1] = point.g;
      rgb_img.at<cv::Vec3b>(y, x)[2] = point.r;
    }
  }

  cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);

  // Reduce noise with blurring
  cv::blur(gray_img, gray_img, cv::Size(7,7));

  float median = static_cast<float>(imageMedian(gray_img));

  // apply automatic canny edge detection using the image median
  int lower_tresh = static_cast<int>(std::max(0, static_cast<int>((1.0f - canny_sigma_) * median)));
  int upper_tresh = static_cast<int>(std::min(255, static_cast<int>((1.0f + canny_sigma_) * median)));

  // Canny detector
  cv::Canny(gray_img, edge_img, lower_tresh, upper_tresh, canny_kernel_size_);
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

void Segmenter::publishImg(const cv::Mat& img, const std_msgs::Header& header, ros::Publisher& pub) {

  if (pub.getNumSubscribers() == 0)
    return;

  sensor_msgs::ImagePtr msg;

  if (img.channels() > 1)
    msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
  else
    msg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();

  pub.publish(msg);
}

void Segmenter::publishNormalsImg(pcl::PointCloud<pcl::Normal>::ConstPtr normals, const std_msgs::Header& header, ros::Publisher& pub) {
  if (pub.getNumSubscribers() == 0)
    return;

  pcl::io::PointCloudImageExtractorFromNormalField<pcl::Normal> img_extr;
  pcl::PCLImage image;
  sensor_msgs::Image ros_image;
  img_extr.extract(*normals, image);

  pcl_conversions::fromPCL(image, ros_image);
  ros_image.header = header;

  pub.publish(ros_image);
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

void Segmenter::enumerateSegments(const LabelIndexMap& segment_map, const ImageIndexList& segment_centroids, cv::Mat& img) {

  const auto font = cv::FONT_HERSHEY_PLAIN;
  const double font_scale = 1.5;
  const int font_thickness = 2;

  for (auto segment: segment_map) {

    Label segment_id = segment.first;

    if (segment_id == 0)
      continue;

    int num_pixel = static_cast<int>(segment.second.size());

    if (num_pixel < 20)
      continue;

    const std::string text = std::to_string(segment_id);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, font_thickness, &baseline);

    int avg_row = segment_centroids[segment.first](0)/num_pixel;
    int avg_col = segment_centroids[segment.first](1)/num_pixel;

    Color c = getSegmentColor(segment_id);
    cv::Point p1(avg_col - 0.5f * text_size.width, avg_row + 0.75f * text_size.height);
    cv::Point p2(avg_col + 0.5f * text_size.width, avg_row - 0.75f * text_size.height);
    cv::rectangle(img, p1, p2, cv::Scalar(255, 255, 255), CV_FILLED, 0);

    // offset the coordinates so that the text is inside the rectangle
    avg_row += text_size.height / 2;
    avg_col -= text_size.width / 2;

    cv::putText(img, std::to_string(segment_id), cv::Point(avg_col, avg_row),
                font, font_scale, cvScalar(c.b, c.g, c.r), font_thickness);
  }
}
}  // namespace voxblox
