#include "voxblox_ros/segmenter.h"

namespace voxblox {

Segmenter::Segmenter(const ros::NodeHandle& nh_private, float voxel_size) :
  nh_private_(nh_private), voxel_size_(voxel_size) {

  edge_img_pub_ = nh_private_.advertise<sensor_msgs::Image>("all_edges", 1, true);
  segmentation_pub_ = nh_private_.advertise<sensor_msgs::Image>("segmentation", 1, true);
  concave_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("concave_edges", 1, true);
  depth_disc_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_disc_edges", 1, true);
  rgb_edges_pub_ = nh_private_.advertise<sensor_msgs::Image>("rgb_edges", 1, true);
  normals_pub_ = nh_private_.advertise<sensor_msgs::Image>("normals", 1, true);
  depth_inpainted_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_inpainted", 1, true);
  depth_filtered_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_filtered", 1, true);
  depth_input_pub_ = nh_private_.advertise<sensor_msgs::Image>("depth_input", 1, true);

  initColorMap(255);

  nh_private_.param("seg_canny_sigma", canny_sigma_, 0.5f);
  nh_private_.param("seg_canny_kernel_size", canny_kernel_size_, 3);
  nh_private_.param("seg_min_concavity", min_concavity_, 0.97f);
  nh_private_.param("seg_max_dist_step", max_dist_step_, 0.005f);
  nh_private_.param("edges_window_size", edges_window_size_, 3);
  nh_private_.param("normals_window_size", normals_window_size_, 2);

  nh_private_.param("concave_weight", concave_weight_, 1.0);
  nh_private_.param("color_weight", color_weight_, 0.0);
  nh_private_.param("edge_treshold", edge_treshold_, 0.0);

  std::string model_path = std::string(std::getenv("HOME")) + "/Downloads/model.yml.gz";
  nh_private_.param("structured_edges_model_path", model_path, model_path);

  structured_edges_ = cv::ximgproc::createStructuredEdgeDetection(model_path);

}

void printMinMax(cv::Mat& mat, const std::string& mat_name) {
  float max = -std::numeric_limits<float>::max();
  float min = std::numeric_limits<float>::max();

  int n_rows = mat.rows;
  int n_cols = mat.cols * mat.channels();

  if (mat.isContinuous())
  {
    n_cols *= n_rows;
    n_rows = 1;
  }

  int i,j;
  float* p;
  for( i = 0; i < n_rows; ++i)
  {
    p = mat.ptr<float>(i);
    for ( j = 0; j < n_cols; ++j)
    {
      max = std::max(max, p[j]);
      min = std::min(min, p[j]);
    }
  }

  ROS_INFO_STREAM(mat_name << " min: " << min << " max: " << max);
}

void Segmenter::segmentRgbdImage(const cv::Mat& color_img, const sensor_msgs::CameraInfoConstPtr& /*color_cam_info_msg*/,
                                 const cv::Mat& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_cam_info_msg,
                                 const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, Pointcloud& cloud_out, LabelIndexMap& segment_map)
 {

  if (cloud_in->points.empty())
    return;

  image_geometry::PinholeCameraModel depth_camera_model_;
  depth_camera_model_.fromCameraInfo(depth_cam_info_msg);

  segment_map.clear();

  publishImg(depth_img, pcl_conversions::fromPCL(cloud_in->header), depth_input_pub_);

  cv::Mat depth_img_inpainted = inpaintDepth(depth_img);

  cv::Mat normals = estimateNormalsCrossProduct(depth_img_inpainted);

  // reduce the size of the depth img to the size of the normal estimation
  cv::Rect roi(normals_window_size_, normals_window_size_,
               normals.cols, normals.rows);
  depth_img_inpainted = depth_img_inpainted(roi);

  cv::Mat depth_img_smoothed = filterImage(depth_img_inpainted);

  cv::Mat points3d;
  cv::Matx33f intrinsic_matrix_f(depth_camera_model_.fullIntrinsicMatrix());
  cv::rgbd::depthTo3d(depth_img_smoothed, intrinsic_matrix_f, points3d);

  cv::Mat edge_img_concave = detectConcaveBoundaries(points3d, normals);
  cv::Mat edge_img_depth_disc = detectDepthDiscBoundaries(points3d, normals);
  cv::Mat edge_img_color = detectStructuredEdges(color_img(roi));

  printMinMax(edge_img_concave, std::string("edge_img_concave"));
  printMinMax(edge_img_depth_disc, std::string("edge_img_depth_disc"));

  timing::Timer seg_connected_components_timer("seg_connected_components");

  cv::Mat edge_img;
  cv::addWeighted(edge_img_concave, concave_weight_, edge_img_depth_disc, 1.0, 0.0, edge_img);
  printMinMax(edge_img, std::string("edge_img"));

  cv::threshold(edge_img, edge_img, edge_treshold_, 255.0, cv::THRESH_BINARY);
  edge_img.convertTo(edge_img, CV_8U);
  cv::bitwise_not(edge_img, edge_img);

  // increase the borders a little bit before the segmentation
  cv::Mat kernel = cv::Mat::ones(2, 2, CV_8U);
  cv::morphologyEx(edge_img, edge_img, cv::MORPH_OPEN, kernel, cv::Point(-1,-1), 1);

  cv::Mat segmentation_img;
  int num_labels = cv::connectedComponents(edge_img, segmentation_img, 8, CV_16U);

  ROS_INFO_STREAM("found " << num_labels << " labels!");

  seg_connected_components_timer.Stop();

  // TODO: add parameters
  int radius = 5;
  double max_distance = 0.05;
  timing::Timer assign_edge_points_timer("seg_assign_edge_points");
  assignEdgePoints(radius, max_distance, points3d, segmentation_img);
  assign_edge_points_timer.Stop();

  applyVoxelGridFilter(cloud_in, segmentation_img, cloud_out, segment_map);

 /* if (segmentation_pub_.getNumSubscribers() > 0) {
    ImageIndexList segment_centroids = computeSegmentCentroids(segment_map, segmentation_img);
    cv::Mat segmentation_img_color = colorizeSegmentationImg(segmentation_img, segment_map);
    enumerateSegments(segment_map, segment_centroids, segmentation_img_color);

    publishImg(segmentation_img_color, pcl_conversions::fromPCL(cloud_in->header), segmentation_pub_);
  }*/

  publishImg(edge_img, pcl_conversions::fromPCL(cloud_in->header), edge_img_pub_);
  publishImg(edge_img_concave, pcl_conversions::fromPCL(cloud_in->header), concave_edges_pub_);
  publishImg(edge_img_depth_disc, pcl_conversions::fromPCL(cloud_in->header), depth_disc_edges_pub_);
  //publishImg(edge_img_color, pcl_conversions::fromPCL(cloud_in->header), rgb_edges_pub_);
  publishImg(depth_img_inpainted, pcl_conversions::fromPCL(cloud_in->header), depth_inpainted_pub_);
  publishImg(depth_img_smoothed, pcl_conversions::fromPCL(cloud_in->header), depth_filtered_pub_);
  publishNormalsImg(normals, pcl_conversions::fromPCL(cloud_in->header), normals_pub_);
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

cv::Mat Segmenter::estimateNormalsCrossProduct(const cv::Mat& depth_img) {

  timing::Timer seg_normal_estimation_timer("seg_normal_estimation");

  cv::medianBlur(depth_img, depth_img, 5);
  cv::Mat normals(depth_img.rows-2*normals_window_size_, depth_img.cols-2*normals_window_size_, CV_32FC3);

  for (int row = normals_window_size_; row < depth_img.rows - normals_window_size_; row++) {
    for (int col = normals_window_size_; col < depth_img.cols - normals_window_size_; col++) {

      // cardinal directions relative to the current point
      int row_nw = row-normals_window_size_;
      int col_nw = col-normals_window_size_;
      cv::Vec3d nw(row_nw, col_nw, static_cast<double>(depth_img.at<uint16_t>(row_nw, col_nw)));

      int row_n = row-normals_window_size_;
      int col_n = col;
      cv::Vec3d n(row_n, col_n, static_cast<double>(depth_img.at<uint16_t>(row_n, col_n)));

      int row_ne = row-normals_window_size_;
      int col_ne = col+normals_window_size_;
      cv::Vec3d ne(row_ne, col_ne, static_cast<double>(depth_img.at<uint16_t>(row_ne, col_ne)));

      int row_e = row;
      int col_e = col+normals_window_size_;
      cv::Vec3d e(row_e, col_e, static_cast<double>(depth_img.at<uint16_t>(row_e, col_e)));

      int row_se = row+normals_window_size_;
      int col_se = col+normals_window_size_;
      cv::Vec3d se(row_se, col_se, static_cast<double>(depth_img.at<uint16_t>(row_se, col_se)));

      int row_s = row+normals_window_size_;
      int col_s = col;
      cv::Vec3d s(row_s, col_s, static_cast<double>(depth_img.at<uint16_t>(row_s, col_s)));

      int row_sw = row+normals_window_size_;
      int col_sw = col-normals_window_size_;
      cv::Vec3d sw(row_sw, col_sw, static_cast<double>(depth_img.at<uint16_t>(row_sw, col_sw)));

      int row_w = row;
      int col_w = col-normals_window_size_;
      cv::Vec3d w(row_w, col_w, static_cast<double>(depth_img.at<uint16_t>(row_w, col_w)));

      cv::Vec3d n1 = (sw-n).cross(se-n);
      cv::Vec3d n2 = (nw-e).cross(sw-e);
      cv::Vec3d n3 = (ne-s).cross(nw-s);
      cv::Vec3d n4 = (se-w).cross(ne-w);

      normals.at<cv::Vec3f>(row-normals_window_size_, col-normals_window_size_) = cv::normalize(0.25 * (n1 + n2 + n3 + n4));
    }
  }

  cv::GaussianBlur(normals, normals, cv::Size(5,5), 0);

  seg_normal_estimation_timer.Stop();

  return normals;
}

cv::Mat Segmenter::inpaintDepth(const cv::Mat& depth_img) {

  timing::Timer seg_inpaint_depth_timer("seg_inpaint_image");

  cv::Mat depth_img_inpainted;
  cv::inpaint(depth_img, (depth_img == 0), depth_img_inpainted, 5.0, cv::INPAINT_NS);

  seg_inpaint_depth_timer.Stop();

  return depth_img_inpainted;
}

cv::Mat Segmenter::filterImage(cv::Mat& depth_img) {

  timing::Timer seg_filter_image_timer("seg_filter_image");
  cv::Mat depth_img_smoothed;

  depth_img.convertTo(depth_img, CV_32F);

  int d = 5;
  double 	sigma_color = 250.0;
  double 	sigma_space = 250.0;
  cv::bilateralFilter(depth_img, depth_img_smoothed, d, sigma_color, sigma_space);

  depth_img.convertTo(depth_img, CV_16U);
  depth_img_smoothed.convertTo(depth_img_smoothed, CV_16U);

  seg_filter_image_timer.Stop();

  return depth_img_smoothed;
}

cv::Mat Segmenter::colorizeSegmentationImg(const cv::Mat& seg_img, const LabelIndexMap& segment_map) {

  cv::Mat seg_img_color(seg_img.rows, seg_img.cols, CV_8UC3);

  std::vector<cv::Vec3b> colors;
  colors.reserve(segment_map.size());

  // Prepare the colors
  for (uint i = 0; i < segment_map.size(); i++) {
    Color c = getSegmentColor(i);
    colors.push_back(cv::Vec3b(c.b, c.g, c.r));
  }

  cv::MatIterator_<cv::Vec3b> color_img_it = seg_img_color.begin<cv::Vec3b>();
  cv::MatConstIterator_<ushort> seg_img_it = seg_img.begin<ushort>();
  for(; color_img_it != seg_img_color.end<cv::Vec3b>() || seg_img_it != seg_img.end<ushort>(); ++color_img_it, ++seg_img_it )
  {
      (*color_img_it)[0] = colors[(*seg_img_it)][0];
      (*color_img_it)[1] = colors[(*seg_img_it)][1];
      (*color_img_it)[2] = colors[(*seg_img_it)][2];
  }

  return seg_img_color;
}

cv::Mat Segmenter::detectDepthDiscBoundaries(const cv::Mat& points, const cv::Mat& normals) {
  timing::Timer seg_depth_disc_boundaries_timer("seg_depth_disc_boundaries");

  std::vector<cv::Point2i> neighbors;
  neighbors.reserve(8);

  const int width = points.cols;
  const int height = points.rows;

  cv::Mat edge_img(height, width, CV_32F, cv::Scalar(0));
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int step_size = 1; step_size <= edges_window_size_; step_size++) {

        getNeighbors(row, col, height, width, step_size, neighbors);

        float max_dist = -std::numeric_limits<float>::max();

        const cv::Point3f& p = points.at<cv::Point3f>(row, col);
        const cv::Point3f& n = normals.at<cv::Point3f>(row, col);

        for (const cv::Point2i& i: neighbors) {
          const cv::Point3f& p_i = points.at<cv::Point3f>(i);

          cv::Point3f diff = p_i - p;
          max_dist = std::max(std::abs(diff.dot(n)), max_dist);
        }

        edge_img.at<float>(row, col) += max_dist/edges_window_size_;
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

  cv::Mat edge_img(height, width, CV_32F, cv::Scalar(0));

  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int step_size = 1; step_size <= edges_window_size_; step_size++) {

        getNeighbors(row, col, height, width, step_size, neighbors);

        float max_concavity = -std::numeric_limits<float>::max();

        const cv::Vec3f& p = points.at<cv::Vec3f>(row, col);
        const cv::Vec3f& n = normals.at<cv::Vec3f>(row, col);

        for (const cv::Point2i& i: neighbors) {
          const cv::Vec3f& p_i = points.at<cv::Vec3f>(i);

          cv::Vec3f diff = p_i - p;

          if (diff.dot(n) < 0) {
            max_concavity = std::max(0.0f, max_concavity);
          }
          else {
            const cv::Vec3f& n_i = normals.at<cv::Vec3f>(i);

            max_concavity = std::max(1-n_i.dot(n), max_concavity);
          }
        }

        edge_img.at<float>(row, col) += max_concavity / edges_window_size_;
      }
    }
  }
  seg_concave_boundaries_timer.Stop();

  return edge_img;
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

  seg_canny_boundaries_timer.Stop();

  return edges;
}

cv::Mat Segmenter::applyCanny(const cv::Mat& gray_img) {
  timing::Timer seg_canny_boundaries_timer("seg_canny_boundaries");
  int width = gray_img.cols;
  int height = gray_img.rows;

  cv::Mat edge_img(height, width, CV_8UC1, cv::Scalar(0));
  cv::Mat blur_img(height, width, CV_8UC1);

  // Reduce noise with blurring
  cv::blur(gray_img, blur_img, cv::Size(7, 7));

  float median = static_cast<float>(imageMedian<uint8_t>(blur_img));

  // apply automatic canny edge detection using the image median
  int lower_tresh = static_cast<int>(std::max(0, static_cast<int>((1.0f - canny_sigma_) * median)));
  int upper_tresh = static_cast<int>(std::min(255, static_cast<int>((1.0f + canny_sigma_) * median)));

  // Canny detector
  cv::Canny(blur_img, edge_img, lower_tresh, upper_tresh, canny_kernel_size_);

  seg_canny_boundaries_timer.Stop();

  return edge_img;
}

void Segmenter::getNeighbors(int row, int col, int height, int width, int step_size, std::vector<cv::Point2i>& neighbors) {

  neighbors.clear();

  // north
  if (row >= step_size) { neighbors.emplace_back(cv::Point2i(col, row-step_size)); };

  // south
  if (row < height-step_size) { neighbors.emplace_back(cv::Point2i(col, row+step_size)); };

  // east
  if (col < width-step_size) { neighbors.emplace_back(cv::Point2i(col+step_size, row)); };

  // west
  if (col >= step_size) { neighbors.emplace_back(cv::Point2i(col-step_size,row)); };

  //  north west
  if (row >= step_size && col >= step_size) { neighbors.emplace_back(cv::Point2i(col-step_size,row-step_size)); };

  //  north east
  if (row >= step_size && col < width-step_size) { neighbors.emplace_back(cv::Point2i(col+step_size,row-step_size)); };

  //  south west
  if (row < height-step_size && col >= step_size) { neighbors.emplace_back(cv::Point2i(col-step_size,row+step_size)); };

  //  south east
  if (col < width-step_size && row < height-step_size) { neighbors.emplace_back(cv::Point2i(col+step_size,row+step_size)); };
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
  else if (img.type() == CV_32F)
  {
    cv::Mat img_short;
    img.convertTo(img_short, CV_16U, 65536);
    msg = cv_bridge::CvImage(header, "mono16", img_short).toImageMsg();
  }
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

    int row = segment_centroids[segment.first](0);
    int col = segment_centroids[segment.first](1);

    Color c = getSegmentColor(segment_id);
    cv::Point p1(col - text_size.width/2, row + (3*text_size.height)/4);
    cv::Point p2(col + text_size.width/2, row - (3*text_size.height)/4);
    cv::rectangle(img, p1, p2, cv::Scalar(255, 255, 255), cv::FILLED, 0);

    // offset the coordinates so that the text is inside the rectangle
    row += text_size.height / 2;
    col -= text_size.width / 2;

    cv::putText(img, std::to_string(segment_id), cv::Point(col, row),
                font, font_scale, cvScalar(c.b, c.g, c.r), font_thickness);
  }
}

void Segmenter::assignEdgePoints(int radius, double max_distance, const cv::Mat& points_3d, cv::Mat& img) {
  // prevent the expansion of the segments by using the unassigned segmentation image for the process
  cv::Mat seg_img_orig = img.clone();
  for (int row = 0; row < img.rows; row++) {
    for (int col = 0; col < img.cols; col++) {
      if (img.at<ushort>(row, col) == 0) {
        img.at<ushort>(row, col) = assignEdgePoint(row, col, radius, max_distance, points_3d, seg_img_orig);
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

      if (circle_row < 0 || circle_col < 0 || circle_row >= img.rows || circle_col >= img.cols)
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

std::pair<ushort, int> getMostCommonLabel(const std::unordered_map<ushort, int>& x) {
  using pairtype=std::pair<ushort, int>;
  return *std::max_element(x.begin(), x.end(), [] (const pairtype& p1, const pairtype& p2) {
        return p1.second < p2.second;
  });
}

void Segmenter::applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, const cv::Mat& segmentation_img, Pointcloud& cloud_out, LabelIndexMap& segment_map) {
  Octree octree(static_cast<double>(voxel_size_));
  octree.setInputCloud(cloud_in);
  octree.addPointsFromInputCloud();

  int width = static_cast<int>(cloud_in->width);
  int height = static_cast<int>(cloud_in->height);

  cloud_out.reserve(octree.getLeafCount());

  Point point;
  int num_points;
  std::unordered_map<ushort, int> label_counts;

  int index = 0;
  for (auto leaf_it = octree.leaf_begin(); leaf_it != octree.leaf_end(); ++leaf_it) {
    auto& leaf_container = leaf_it.getLeafContainer();
    const auto& indices = leaf_container.getPointIndicesVector();

    point = Point::Zero();
    num_points = 0;
    label_counts.clear();

    for (int index: indices) {

      const pcl::PointXYZ& point_pcl = cloud_in->points[static_cast<size_t>(index)];

      int row = index / width;
      int col = index % width;

      if (row < normals_window_size_ || col < normals_window_size_ ||
          row >= height - normals_window_size_ || col >= width - normals_window_size_) {
        break;
      }

      ushort label = segmentation_img.at<ushort>(row-normals_window_size_, col-normals_window_size_);

      // keep track of the labels and centroid of the points in this voxel
      if (label_counts.count(label) == 0) {
        label_counts[label] = 1;
      } else {
        label_counts[label]++;
      }

      point.x() += point_pcl.x;
      point.y() += point_pcl.y;
      point.z() += point_pcl.z;

      num_points++;
    }

    if (num_points > 0) {
      // add the centroid of the points in this cell
      cloud_out.emplace_back(point/num_points);

      // select the most common label as label for this cell
      ushort label = getMostCommonLabel(label_counts).first;
      segment_map[label].emplace_back(index);

      index++;
    }
  }
}

ImageIndexList Segmenter::computeSegmentCentroids(const LabelIndexMap& segment_map, cv::Mat& segmentation_img) {
  ImageIndexList segment_centroids(static_cast<unsigned long>(segment_map.size()));
  std::vector<int> segment_sizes;
  for (size_t i = 0; i < static_cast<size_t>(segment_map.size()); i++) {
    segment_centroids[i] = ImageIndex(0, 0);
    segment_sizes.push_back(0);
  }

  for (int col = 0; col < segmentation_img.cols; ++col) {
    for (int row = 0; row < segmentation_img.rows; ++row) {

      ushort label = segmentation_img.at<ushort>(row, col);

      segment_centroids[label] += ImageIndex(row, col);
      segment_sizes[label]++;
    }
  }

  for (size_t i = 0; i < static_cast<size_t>(segment_map.size()); i++) {
    segment_centroids[i].x() /= segment_sizes[i];
    segment_centroids[i].y() /= segment_sizes[i];
  }

  return segment_centroids;
}
}  // namespace voxblox
