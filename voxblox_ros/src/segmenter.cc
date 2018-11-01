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
  depth_inpainted_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_inpainted", 1, true);

  initColorMap(255);

  nh_private_.param("seg_canny_sigma", canny_sigma_, 0.4f);
  nh_private_.param("seg_canny_kernel_size", canny_kernel_size_, 3);
  nh_private_.param("seg_min_concavity", min_concavity_, 0.97f);
  nh_private_.param("seg_max_dist_step_", max_dist_step_, 0.005);

  std::string model_path = std::string(std::getenv("HOME")) + "/Downloads/model.yml.gz";
  nh_private_.param("structured_edges_model_path", model_path, model_path);

  structured_edges_ = cv::ximgproc::createStructuredEdgeDetection(model_path);

}

void Segmenter::segmentRgbdImage(const sensor_msgs::ImageConstPtr& color_img_msg, const sensor_msgs::CameraInfoConstPtr& color_cam_info_msg,
                                 const sensor_msgs::ImageConstPtr& depth_img_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info_msg,
                                 const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_msg, const pcl::PointCloud<int>& sub_cloud_indices, LabelIndexMap& segment_map)
 {

  if (cloud_msg->points.empty())
    return;

  int width = static_cast<int>(cloud_msg->width);

  image_geometry::PinholeCameraModel depth_camera_model_;
  depth_camera_model_.fromCameraInfo(depth_cam_info_msg);

  segment_map.clear();

  CvImageConstPtr depth_img, color_img;

  color_img = cv_bridge::toCvShare(color_img_msg, sensor_msgs::image_encodings::RGB8);

  // convert the unit to mm if needed
  if (depth_img_msg->encoding == "32FC1") {
    CvImagePtr depth_img_m = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_img_m->image *= 1000.0f;
    depth_img_m->image.convertTo(depth_img_m->image, CV_16U);
    depth_img = depth_img_m;
  } else {
    depth_img = cv_bridge::toCvShare(depth_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }

  cv::Mat depth_img_inpainted = inpaintDepth(depth_img->image);
  cv::Mat depth_img_smoothed = filterImage(depth_img_inpainted);

  cv::Mat points3d;
  cv::rgbd::depthTo3d(depth_img_smoothed, depth_camera_model_.fullIntrinsicMatrix(), points3d);
  cv::Mat normals = estimateNormals(points3d, depth_camera_model_.fullIntrinsicMatrix());

  cv::Mat edge_img_concave = detectConcaveBoundaries(points3d, normals);
  cv::Mat edge_img_depth_disc = detectGeometricalBoundaries(points3d, normals);
  cv::Mat edge_img_color = detectStructuredEdges(color_img->image);

  timing::Timer seg_connected_components_timer("seg_connected_components");

  cv::Mat edge_img = cv::min(edge_img_concave, edge_img_depth_disc);
  edge_img = cv::min(edge_img, edge_img_color);

  // increase the borders a little bit before the segmentation
  cv::Mat kernel = cv::Mat::ones(2, 2, CV_8U);
  cv::morphologyEx(edge_img, edge_img, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 1);

  cv::Mat segmentation_img;
  int num_labels = cv::connectedComponents(edge_img, segmentation_img, 8, CV_16U);

  ROS_INFO_STREAM("found " << num_labels << " labels!");

  seg_connected_components_timer.Stop();

  int radius = 5;
  double max_distance = 0.1;
  assignEdgePoints(radius, max_distance, points3d, segmentation_img);

  ImageIndexList segment_centroids(static_cast<unsigned long>(num_labels));
  for (size_t i = 0; i < num_labels; num_labels++) {
    segment_centroids[i] += ImageIndex(0, 0);
  }

  for (size_t i = 0; i < sub_cloud_indices.size(); ++i) {
    int sub_cloud_index = sub_cloud_indices[i];
    const pcl::PointXYZ& p = cloud_msg->points[sub_cloud_index];

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

    publishImg(segmentation_img_color, pcl_conversions::fromPCL(cloud_msg->header), segmentation_pub_);
  }

  publishImg(edge_img, pcl_conversions::fromPCL(cloud_msg->header), edge_img_pub_);
  publishImg(edge_img_concave, pcl_conversions::fromPCL(cloud_msg->header), concave_edges_pub_);
  publishImg(edge_img_depth_disc, pcl_conversions::fromPCL(cloud_msg->header), depth_disc_edges_pub_);
  publishImg(edge_img_color, pcl_conversions::fromPCL(cloud_msg->header), rgb_edges_pub_);
  publishNormalsImg(normals, pcl_conversions::fromPCL(cloud_msg->header), normals_pub_);
  publishImg(depth_img_inpainted, pcl_conversions::fromPCL(cloud_msg->header), depth_inpainted_pub_);
}

cv::Mat Segmenter::estimateNormals(const cv::Mat& points_3d, const cv::Matx33d& intrinsic_matrix) {

  timing::Timer seg_normal_estimation_timer("seg_normal_estimation");
  cv::Mat normals;

  int window_size = 5;
  cv::rgbd::RgbdNormals ocv_normals_estimation(points_3d.rows, points_3d.cols, CV_32F, intrinsic_matrix,
                                               window_size, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_FALS);
  ocv_normals_estimation(points_3d, normals);

  cv::GaussianBlur(normals, normals, cv::Size(5,5), 0);

  seg_normal_estimation_timer.Stop();

  return normals;
}

cv::Mat Segmenter::inpaintDepth(const cv::Mat& depth_img) {

  timing::Timer seg_inpaint_depth_timer("seg_inpaint_image");

  cv::Mat depth_img_inpainted(depth_img);

  //use a smaller version of the image
  cv::Mat small_depthf, small_depth_inpaint;
  cv::resize(depth_img, small_depthf, cv::Size(), 0.25, 0.25);
  cv::inpaint(small_depthf, (small_depthf == 0), small_depth_inpaint, 5.0, cv::INPAINT_TELEA);
  cv::resize(small_depth_inpaint, small_depth_inpaint, depth_img.size());
  small_depth_inpaint.copyTo(depth_img_inpainted, (depth_img == 0));

  seg_inpaint_depth_timer.Stop();

  return depth_img_inpainted;
}

cv::Mat Segmenter::filterImage(cv::Mat& depth_img) {

  timing::Timer seg_filter_image_timer("seg_filter_image");
  depth_img.convertTo(depth_img, CV_32F);
  cv::Mat depth_img_smoothed;

  int d = 5;
  double 	sigma_color = 50.0;
  double 	sigma_space = 50.0;
  cv::bilateralFilter(depth_img, depth_img_smoothed, d, sigma_color, sigma_space);
  depth_img_smoothed.convertTo(depth_img_smoothed, CV_16U);
  depth_img.convertTo(depth_img, CV_16U);

  seg_filter_image_timer.Stop();

  return depth_img_smoothed;
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

cv::Mat Segmenter::detectGeometricalBoundaries(const cv::Mat& points, const cv::Mat& normals) {
  timing::Timer seg_depth_disc_boundaries_timer("seg_depth_disc_boundaries");

  std::vector<cv::Point2i> neighbors;
  neighbors.reserve(8);

  const int width = points.cols;
  const int height = points.rows;

  cv::Mat edge_img(height, width, CV_8UC1, cv::Scalar(0));

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {

      getNeighbors(row, col, height, width, neighbors);

      double max_dist = -std::numeric_limits<double>::max();

      const cv::Point3f& p = points.at<cv::Point3f>(row, col);
      const cv::Point3f& n = normals.at<cv::Point3f>(row, col);

      for (const cv::Point2i& i: neighbors) {
        const cv::Point3f& p_i = points.at<cv::Point3f>(i);

        cv::Point3d diff = p_i - p;
        max_dist = std::max(std::abs(diff.dot(n)), max_dist);
      }

      if (max_dist <= max_dist_step_)
      {
        edge_img.at<uchar>(row, col) = 255;
      }
    }
  }
  seg_depth_disc_boundaries_timer.Stop();

  return edge_img;
}

cv::Mat Segmenter::detectConcaveBoundaries(const cv::Mat& points, const cv::Mat& normals) {
  timing::Timer seg_concave_boundaries_timer("seg_concave_boundaries");

  std::vector<cv::Point2i> neighbors;
  neighbors.reserve(8);

  int width = points.cols;
  int height = points.rows;

  cv::Mat edge_img(height, width, CV_8UC1, cv::Scalar(0));

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {

      getNeighbors(row, col, height, width, neighbors);

      float min_concavity = std::numeric_limits<float>::max();

      const cv::Point3f& p = points.at<cv::Point3f>(row, col);
      const cv::Point3f& n = normals.at<cv::Point3f>(row, col);

      for (const cv::Point2i& i: neighbors) {
        const cv::Point3f& p_i = points.at<cv::Point3f>(i);
        cv::Point3d diff = p_i - p;

        if (diff.dot(n) > 0) {
          min_concavity = std::min(1.0f, min_concavity);
        }
        else {
          const cv::Point3f& n_i = normals.at<cv::Point3f>(i);

          min_concavity = std::min(n.dot(n_i), min_concavity);
        }
      }

      if (min_concavity >= min_concavity_)
      {
        edge_img.at<uchar>(row, col) = 255;
      }
    }
  }
  seg_concave_boundaries_timer.Stop();

  return edge_img;
}

uchar imageMedian(const cv::Mat& img) {
  std::vector<uchar> vec_from_mat(img.begin<uchar>(), img.end<uchar>());
  std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());
  return vec_from_mat[vec_from_mat.size() / 2];
}

cv::Mat Segmenter::detectCannyEdgesMono(const cv::Mat& color_img) {
  timing::Timer seg_canny_boundaries_timer("seg_canny_mono_edges");
  int width = color_img.cols;
  int height = color_img.rows;

  // Create cv::Mat
  cv::Mat gray_img = cv::Mat(height, width, CV_8UC1);
  return applyCanny(gray_img);
}

cv::Mat Segmenter::detectCannyEdgesH1H2H3(const cv::Mat& color_img) {
  timing::Timer seg_canny_boundaries_timer("seg_canny_edges_h1h2h3");

  cv::Mat color_img_float;
  color_img.convertTo(color_img_float, CV_32FC3);

  cv::Mat bgr[3];   //destination array
  cv::split(color_img_float, bgr); //split source

  cv::Mat h1 = bgr[2] - bgr[1];
  cv::Mat h2 = bgr[1] - bgr[0];
  cv::Mat h3 = bgr[0] - bgr[2];

  cv::normalize(h1, h1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(h2, h2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(h3, h3, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  cv::Mat h1_edges = applyCanny(h1);
  cv::Mat h2_edges = applyCanny(h2);
  cv::Mat h3_edges = applyCanny(h3);

  cv::Mat rgb_edges = cv::min(h1_edges, h2_edges);
  rgb_edges = cv::min(rgb_edges, h3_edges);

  seg_canny_boundaries_timer.Stop();

  return rgb_edges;
}

cv::Mat Segmenter::detectStructuredEdges(const cv::Mat& color_img) {
  timing::Timer seg_canny_boundaries_timer("seg_structured_edges");

  cv::Mat3f fsrc;
  color_img.convertTo(fsrc, CV_32F, 1.0 / 255.0);

  cv::Mat edges;
  structured_edges_->detectEdges(fsrc, edges);
  // computes orientation from edge map
  //cv::Mat orientation_map;
  //structured_edges_->computeOrientation(edges, orientation_map);

  // suppress edges
  //structured_edges_->edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);
  edges.convertTo(edges, CV_8UC1, 255.0);
  cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
  cv::bitwise_not(edges, edges);

  seg_canny_boundaries_timer.Stop();

  return edges;
}

cv::Mat Segmenter::applyCanny(const cv::Mat& gray_img) {
  timing::Timer seg_canny_boundaries_timer("seg_canny_boundaries");
  int width = gray_img.cols;
  int height = gray_img.rows;

  cv::Mat edge_img(height, width, CV_8UC1, cv::Scalar(0));
  cv::Mat blur_img(gray_img);

  // Reduce noise with blurring
  cv::blur(gray_img, blur_img, cv::Size(7, 7));

  float median = static_cast<float>(imageMedian(blur_img));

  // apply automatic canny edge detection using the image median
  int lower_tresh = static_cast<int>(std::max(0, static_cast<int>((1.0f - canny_sigma_) * median)));
  int upper_tresh = static_cast<int>(std::min(255, static_cast<int>((1.0f + canny_sigma_) * median)));

  // Canny detector
  cv::Canny(blur_img, edge_img, lower_tresh, upper_tresh, canny_kernel_size_);
  cv::bitwise_not(edge_img, edge_img);

  seg_canny_boundaries_timer.Stop();

  return edge_img;
}

void Segmenter::getNeighbors(int row, int col, int height, int width, std::vector<cv::Point2i>& neighbors) {

  neighbors.clear();
  int max_rows = height -1;
  int max_cols = width -1;

  // north
  if (row > 0) { neighbors.emplace_back(cv::Point2i(col, row-1)); };

  // south
  if (row < max_rows) { neighbors.emplace_back(cv::Point2i(col, row+1)); };

  // east
  if (col < max_cols) { neighbors.emplace_back(cv::Point2i(col+1, row)); };

  // west
  if (col > 0) { neighbors.emplace_back(cv::Point2i(col-1,row)); };

  //  north west
  if (row > 0 && col > 0) { neighbors.emplace_back(cv::Point2i(col-1,row-1)); };

  //  north east
  if (row > 0 && col < max_cols) { neighbors.emplace_back(cv::Point2i(col+1,row-1)); };

  //  south west
  if (row < max_rows && col > 0) { neighbors.emplace_back(cv::Point2i(col-1,row+1)); };

  //  south east
  if (col < max_cols && row < max_rows) { neighbors.emplace_back(cv::Point2i(col+1,row+1)); };
}

void Segmenter::publishImg(const cv::Mat& img, const std_msgs::Header& header, ros::Publisher& pub) {

  if (pub.getNumSubscribers() == 0)
    return;

  sensor_msgs::ImagePtr msg;

  if (img.type() == CV_8UC3)
    msg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();
  else if (img.type() == CV_8U)
    msg = cv_bridge::CvImage(header, "mono8", img).toImageMsg();
  else if (img.type() == CV_16U)
    msg = cv_bridge::CvImage(header, "mono16", img).toImageMsg();
  else
    ROS_ERROR_STREAM("unknown img type to publish: " << img.type());

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

void Segmenter::publishNormalsImg(const cv::Mat& normals, const std_msgs::Header& header, ros::Publisher& pub) {
  if (pub.getNumSubscribers() == 0)
    return;

  pcl::PointCloud<pcl::Normal>::Ptr pcl_normals = boost::make_shared<pcl::PointCloud<pcl::Normal>>(normals.cols, normals.rows);

  for(int y = 0; y < normals.rows; y++) {
    for(int x = 0; x < normals.cols; x++) {
      pcl::Normal& pcl_normal = pcl_normals->at(x, y);
      const cv::Point3f& normal = normals.at<cv::Point3f>(y, x);

      pcl_normal.normal_x = static_cast<float>(normal.x);
      pcl_normal.normal_y = static_cast<float>(normal.y);
      pcl_normal.normal_z = static_cast<float>(normal.z);
    }
  }

  publishNormalsImg(pcl_normals, header, pub);
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

    if (num_pixel < 100)
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

void Segmenter::assignEdgePoints(int radius, double max_distance, const cv::Mat& points_3d, cv::Mat& img) {

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  cv::minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc );

  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      if (img.at<ushort>(row, col) == 0) {
        img.at<ushort>(row, col) = assignEdgePoint(row, col, radius, max_distance, points_3d, img);
      }
    }
  }
}

ushort Segmenter::assignEdgePoint(int row, int col, int radius, double max_distance, const cv::Mat& points_3d, const cv::Mat& img) {
  ushort segment_id = 0;

  int radius_sqr = radius * radius;
  double min_distance = max_distance;

  const cv::Point3f& point = points_3d.at<cv::Point3f>(row, col);

  for (int iy = -radius; iy <= radius; iy++) {
    int dx = static_cast<int>(sqrt(radius_sqr - iy * iy));
    for (int ix = - dx; ix <= dx; ix++) {

      int circle_row = row + ix;
      int circle_col = col + iy;

      if (circle_row == row && circle_col == col)
        continue;

      if (circle_row < 0 || circle_col < 0 || circle_row >= points_3d.rows || circle_col >= points_3d.cols)
        continue;

      const cv::Point3f& candidate = points_3d.at<cv::Point3f>(circle_row, circle_col);
      double dist = cv::norm(candidate-point);

      if (std::isfinite(dist) && dist <= min_distance) {
        ushort id = img.at<ushort>(circle_row, circle_col);

        if (id != 0) {
          segment_id = id;
          min_distance = dist;
        }
      }
    }
  }

  return segment_id;
}


}  // namespace voxblox
