#ifndef SEG_TOOLBOX_H_
#define SEG_TOOLBOX_H_
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <voxblox/core/common.h>
#include <unordered_map>

#include <mask_rcnn_ros/Result.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl/common/common.h>

#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/utils/distance_utils.h"

#include "voxblox/integrator/label_map.h"
#include "voxblox/integrator/vertex_map.h"

#include <pcl_conversions/pcl_conversions.h>

#include "voxblox/utils/color_maps.h"

#include "voxblox/interpolator/interpolator.h"

#include <pcl/visualization/cloud_viewer.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/point_cloud_image_extractors.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <thread>

#include <voxblox_ros/image_operations.h>

//#include "voxblox_msgs/LabelLookup.h" TODO fix

namespace voxblox {

enum NormalCalculationMode {
  CROSSPRODUCT,
  CENTRALDIFFERENCES,
  CENTRALDIFFERENCES_DEPTH,
  CROSSPRODUCT_DEPTH,
  VOXBLOX
};

struct rawDataPointer {
  bool use_weight;
  std::string body_frame;
  Transformation cloud_body;
  Transformation label_map_body;

  sensor_msgs::ImageConstPtr color_image_ptr;
  sensor_msgs::CameraInfoConstPtr color_info_ptr;
  Transformation color_body;
  bool aligned_color;

  sensor_msgs::ImageConstPtr depth_image_ptr;
  sensor_msgs::CameraInfoConstPtr depth_info_ptr;
  Transformation depth_body;
  bool aligned_depth;

  sensor_msgs::PointCloud2ConstPtr pointcloud;
  Transformation pointcloud_body;
  bool aligned_pointcloud;

  sensor_msgs::CameraInfoConstPtr segmentation_info;

  // double scale_factor;

  rawDataPointer() {
    use_weight = true;
    body_frame = "";
    // cloud_body = NULL;
    // label_map_body = NULL;
    color_image_ptr = NULL;
    color_info_ptr = NULL;
    // color_body = NULL;
    depth_image_ptr = NULL;
    depth_info_ptr = NULL;
    // depth_body = NULL;
    pointcloud = NULL;
    // pointcloud_body = NULL;
    segmentation_info = NULL;

    aligned_color = false;
    aligned_depth = false;
    aligned_pointcloud = false;
    // scale_factor = 1.0;
    // TODO handle case downscaled but aligned depth image by transform checking
    // in data fuser
  }
};

struct LabelLookup {
  std::vector<std::string> names;
  std::vector<uint> ids;

  int last_id;

  LabelLookup() {
    names.push_back("BG");
    ids.push_back(0);
    last_id = 0;
  }

  void add(std::string name) {
    if (!exists(name)) {
      names.push_back(name);
      ids.push_back(++last_id);
    }
  }

  bool exists(std::string name) {
    std::vector<std::string>::iterator it =
        std::find(names.begin(), names.end(), name);

    return it != names.end();
  }

  bool exists(uint id) {
    std::vector<uint>::iterator it = std::find(ids.begin(), ids.end(), id);

    return it != ids.end();
  }

  std::string get(uint id) {
    std::vector<uint>::iterator it = std::find(ids.begin(), ids.end(), id);

    if (it == ids.end()) {
      return "None";
    }
    uint index = std::distance(ids.begin(), it);

    return names[index];
  }

  int get(std::string name) {
    std::vector<std::string>::iterator it =
        std::find(names.begin(), names.end(), name);

    if (it == names.end()) {
      return -1;
    }
    uint index = std::distance(names.begin(), it);

    return ids[index];
  }

  /*voxblox_msgs::LabelLookup getMessage(){
    voxblox_msgs::LabelLookup msg;
    msg.label_ids = ids;
    msg.label_names = names;
    return msg;
  }*/

  void debugPrint() {
    for (int i = 0; i < names.size(); i++) {
      std::cout << names[i] << std::endl;
    }
  }
};

struct ColorLookup {
  std::unordered_map<std::string, std::vector<int>> lookup;

  void add(std::string key, int r, int g, int b){
    
    if(lookup.find(key) == lookup.end()){
      std::vector<int> color;
      color.push_back(r);
      color.push_back(g);
      color.push_back(b);
      lookup[key] = color;
    }
    else{
      std::cout << "object was already added" << std::endl;
    }
  }

  std::vector<int> get(std::string key){
    if(lookup.find(key) != lookup.end()){
      
      return lookup[key];
    }
    else{
      std::vector<int> color;
      return color;
    }
  }
};

struct EdgeImage {
  std::vector<int> points;  // 0 for empty, 1 for used > 1 for invalid point < 0
                            // for unassigned
  std::vector<int> labels;
  std::vector<double> weights;
  int width;
  int height;

  void init(int width, int height) {
    if (points.size() == 0) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int p = -1;
          int l = 0;
          double w = 0.0;
          points.push_back(p);
          labels.push_back(l);
          weights.push_back(w);
        }
      }
    } else {
      std::cout << "already inited edge map" << std::endl;
    }
  }

  int getEdge(int x, int y) { return points[x + y * width]; }

  void setEdge(int val, int x, int y) { points[x + y * width] = val; }

  int getLabel(int x, int y) { return labels[x + y * width]; }

  void setLabel(int val, int x, int y) { labels[x + y * width] = val; }

  double getWeight(int x, int y) { return weights[x + y * width]; }

  void setWeight(double w, int x, int y) { weights[x + y * width] = w; }

  std::shared_ptr<labelMap> getLabelMap() {
    std::shared_ptr<labelMap> lmap(new labelMap(width, height));

    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        if(getEdge(x, y) == 0){
          int label = getLabel(x, y);
          if(label > 0)//leaving out 0 labels
            lmap->add(x, y, label, getWeight(x, y));
        }
      }
    }
    return lmap;
  }

  EdgeImage(int w, int h) {
    width = w;
    height = h;
    init(width, height);
  }

  sensor_msgs::Image createImage() {
    sensor_msgs::Image edge_img;
    edge_img.height = height;
    edge_img.width = width;
    edge_img.encoding = "8UC1";
    edge_img.is_bigendian = false;
    edge_img.step = width;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        uint8_t color;
        int val = getEdge(x, y);

        if (val == 0) color = 255;
        if (val < 0 || val > 1) color = 100;
        if (val == 1) {
          color = 0;
        }
        edge_img.data.push_back(color);
      }
    }
    return edge_img;
  }

  cv::Mat createCvImage() {
    cv::Mat edge_im(cv::Size(width, height), CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        uint8_t color;
        int val = getEdge(x, y);

        if (val == 0) color = 255;
        if (val < 0 || val > 1) color = 100;
        if (val == 1) {
          color = 0;
        }
        edge_im.at<uint8_t>(y, x) = color;
      }
    }
    return edge_im;
  }



  sensor_msgs::Image getLabeledImage() {
    sensor_msgs::Image edge_img;
    edge_img.height = height;
    edge_img.width = width;
    edge_img.encoding = "8UC1";
    edge_img.is_bigendian = false;
    edge_img.step = width;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        uint8_t color;
        int val = getLabel(x, y) * 20;

        if (val == -20) color = 0;
        if (val > 255) color = 255;
        if (val >= 0 && val <= 255) {
          color = val;
        }
        edge_img.data.push_back(color);
      }
    }
    return edge_img;
  }
};

inline cv::Mat convert_16u_32f(cv::Mat input){
  cv::Mat output;
  input.convertTo(output, CV_32F);
  return output;
}

inline cv::Mat convert_32f_16u(cv::Mat input){
  cv::Mat output;
  input.convertTo(output, CV_16U);
  return output;

}

inline std::shared_ptr<labelMap> convertMaskRCNNSegmentation(
    mask_rcnn_ros::Result* res, sensor_msgs::CameraInfoConstPtr color_info,
    std::shared_ptr<LabelLookup> lookup) {
  // init label map

  std::shared_ptr<labelMap> lmap(
      new labelMap(color_info->width, color_info->height));
  //std::cout << color_info->width << std::endl;
  //std::cout << color_info->height << std::endl;

  // int id_counter = label_lookup->size();
  // case of no labels, return null, TODO
  // if(!(res.masks.size() > 0)){

  //    return NULL;
  //}

  // convert masks to images

  std::vector<cv_bridge::CvImagePtr> image_vec;
  for (int i = 0; i < res->masks.size(); i++) {
    cv_bridge::CvImagePtr ptr;

    try {
      ptr = cv_bridge::toCvCopy(res->masks[i], res->masks[i].encoding);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("Cv_bridge Exception: %s", e.what());
    }
    image_vec.push_back(ptr);
  }

  // lookup label for every string (if ids don't work)
  // also push back new vectors for each label

  // TODO this is wrong fix it
  int cols = image_vec[0]->image.cols;
  int rows = image_vec[0]->image.rows;
  int type = image_vec[0]->image.type();

  for (int i = 0; i < res->class_names.size(); i++) {
    if (!lookup->exists(res->class_names[i])) {
      lookup->add(res->class_names[i]);
    }
  }

  for (int x = 0; x < cols; x++) {
    for (int y = 0; y < rows; y++) {
      for (int i = 0; i < image_vec.size(); i++) {
        // get label

        int val = (int)image_vec[i]->image.at<uchar>(y, x);

        // if point was not set in mask
        if (val == 0) {
          continue;
        }
        // lookup id
        uint id = lookup->get(res->class_names[i]);
        // adding label, ignoring depth information
        lmap->add(x, y, id, res->scores[i]);
      }
    }
  }

  return lmap;
}

inline Color hueToRgb(std::vector<uint> hue) {
  int h = std::round(hue[0] / 60.0);
  float f = (hue[0] / 60.0 - h);

  float v = hue[2] / 255.0;
  float s = hue[1] / 255.0;

  float p = v * (1 - s);
  float q = v * (1 - s * f);
  float t = v * (1 - s * (1 - f));

  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  if (h == 0 || h == 6) {
    r = v;
    g = t;
    b = p;
  } else if (h == 1) {
    r = q;
    g = v;
    b = p;
  } else if (h == 2) {
    r = p;
    g = v;
    b = t;
  } else if (h == 3) {
    r = p;
    g = q;
    b = v;
  } else if (h == 4) {
    r = t;
    g = p;
    b = v;
  } else if (h == 5) {
    r = v;
    g = p;
    b = q;
  }

  r = std::round(r * 255);
  g = std::round(g * 255);
  b = std::round(b * 255);
  Color res;

  if (r > 255)
    res.r = 255;
  else if (r < 0)
    res.r = 0;
  else
    res.r = r;

  if (g > 255)
    res.g = 255;
  else if (r < 0)
    res.g = 0;
  else
    res.g = g;

  if (b > 255)
    res.b = 255;
  else if (b < 0)
    res.b = 0;
  else
    res.b = b;

  return res;
}

inline std::vector<uint> rgbToHue(Color color) {
  // calc hue
  float r = color.r / 255.0;
  float g = color.g / 255.0;
  float b = color.b / 255.0;

  float max_c = std::max(r, std::max(g, b));
  float min_c = std::min(r, std::min(g, b));

  float h = 0;
  uint h_final;
  float diff = max_c - min_c;

  if (max_c == min_c)
    h = 0;
  else if (max_c == r)
    h = 60 * (0 + (g - b) / diff);
  else if (max_c == g)
    h = 60 * (2 + (b - r) / diff);
  else if (max_c == b)
    h = 60 * (4 + (r - g) / diff);

  if (h > 360.0)
    h_final = 360;
  else if (h < 0.0)
    h_final = 0;
  else
    h_final = uint(std::round(h));

  float v = std::round(max_c * 255);
  uint v_final = 0;

  if (v > 255.0)
    v_final = 255;
  else if (v < 0.0)
    v_final = 0;
  else
    v_final = uint(v);

  uint s_final = 0;

  float s = 0.0;
  if (max_c == 0.0)
    s = 0;
  else
    s = std::round((diff / max_c) * 255);

  if (s > 255.0)
    s_final = 255;
  else if (s < 0.0)
    s_final = 0;
  else
    s_final = uint(s);

  std::vector<uint> vec;
  vec.push_back(h_final);
  vec.push_back(s_final);
  vec.push_back(v_final);

  return vec;
}

// like marius work
inline pcl::PointCloud<pcl::PointXYZRGB> convertDepthImageToCloudColored(
    cv::Mat& depth_img, sensor_msgs::CameraInfoPtr depth_info,
    cv::Mat color_img) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud(static_cast<uint>(depth_img.cols),
                                          static_cast<uint>(depth_img.rows));

  float center_x = static_cast<float>(depth_info->K[2]);
  float center_y = static_cast<float>(depth_info->K[5]);

  // assuming depth image in mm
  float unit_scaling = 0.001f;

  float f_x = static_cast<float>(depth_info->K[0]);
  float f_y = static_cast<float>(depth_info->K[4]);

  for (int row = 0; row < depth_img.rows; row++) {
    for (int col = 0; col < depth_img.cols; col++) {
      const cv::Vec3b& rgb = color_img.at<cv::Vec3b>(row, col);
      uint16_t depth = depth_img.at<uint16_t>(row, col);
      float scaled_depth = unit_scaling * float(depth);

      pcl::PointXYZRGB& p = cloud.at(col, row);
      // pcl::PointXYZRGB p;

      // Check for invalid measurements
      // if (!isDepthValid(depth)) { //do depth valid check TODO
      if (!(depth != 0)) {
        p.x = p.y = p.z = NAN;
      } else {
        // Fill in XYZ
        p.x = (col - center_x) * scaled_depth / f_x;
        p.y = (row - center_y) * scaled_depth / f_y;
        p.z = scaled_depth;
      }

      // Fill in color //lookupColor TODO

      // use only 3rd hue channel
      Color gray;
      gray.r = 100;
      gray.g = 100;
      gray.b = 100;

      Color i_color;
      i_color.r = rgb[0];
      i_color.g = rgb[1];
      i_color.b = rgb[2];

      std::vector<uint> mesh_col_hue = rgbToHue(i_color);
      std::vector<uint> label_col_hue = rgbToHue(gray);

      // merge
      label_col_hue[2] = mesh_col_hue[2];

      // reconvert
      Color new_color = hueToRgb(label_col_hue);

      p.r = new_color.r;
      p.g = new_color.g;
      p.b = new_color.b;

      // cloud.push_back(p);
    }
  }

  return cloud;
}

// like marius
inline pcl::PointCloud<pcl::PointXYZRGB> convertDepthImageToCloudMono(
    const cv::Mat& depth_img,
    const sensor_msgs::CameraInfoConstPtr& depth_info) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud(static_cast<uint>(depth_img.cols),
                                          static_cast<uint>(depth_img.rows));

  // Use correct principal point from calibration
  float center_x = static_cast<float>(depth_info->K[2]);
  float center_y = static_cast<float>(depth_info->K[5]);

  float unit_scaling = 0.001f;

  // unit_scaling = 1.0f;

  if (std::is_same<uint16_t, float>::value) unit_scaling = 1.0f;

  float f_x = static_cast<float>(depth_info->K[0]);
  float f_y = static_cast<float>(depth_info->K[4]);

  for (int row = 0; row < depth_img.rows; row++) {
    for (int col = 0; col < depth_img.cols; col++) {
      // const cv::Vec3b& rgb = rgb_img.at<cv::Vec3b>(row, col);
      uint16_t depth = depth_img.at<uint16_t>(row, col);
      float scaled_depth = unit_scaling * float(depth);

      pcl::PointXYZRGB& p = cloud.at(col, row);
      // pcl::PointXYZRGB p;

      // Check for invalid measurements
      // if (!isDepthValid(depth)) { //do depth valid check TODO
      if (!(depth != 0)) {
        p.x = p.y = p.z = NAN;
      } else {
        // Fill in XYZ
        p.x = (col - center_x) * scaled_depth / f_x;
        p.y = (row - center_y) * scaled_depth / f_y;
        p.z = scaled_depth;
      }

      // Fill in color //lookupColor TODO
      p.r = 100;  // rgb[0];
      p.g = 100;  // rgb[1];
      p.b = 100;  // rgb[2];

      // cloud.push_back(p);
    }
  }

  return cloud;
}

// like marius/taken from him directly
inline cv_bridge::CvImageConstPtr convertImagePtr(
    const sensor_msgs::ImageConstPtr& depth_img) {
  cv_bridge::CvImageConstPtr depth_img_ptr;
  // from image_operations.h
  // convert the unit to mm if needed
  if (depth_img->encoding == "32FC1") {
    std::cout << "converting to mm" << std::endl;
    cv_bridge::CvImagePtr depth_img_mm = cv_bridge::toCvCopy(
        depth_img, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_img_mm->image.convertTo(depth_img_mm->image, CV_16U, 1000.0);
    depth_img_ptr = depth_img_mm;
  } else {
    depth_img_ptr = cv_bridge::toCvShare(
        depth_img, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  return depth_img_ptr;
}

// TODO maybe use pointer here????
// resuing stuff from marius and creating a vertex map from it
inline std::shared_ptr<vertexMap> vertices_from_depth(
    sensor_msgs::ImageConstPtr depth_img_ptr,
    sensor_msgs::CameraInfoConstPtr depth_info, Transformation T_f_d,
    double max_dist,
    int downsample_factor,
    int bilateral_filter_size,
    double bilateral_filter_diff) {
  // creating vertex map from depth image
  // resuing from marius
  

  

  cv::Mat depth_img = convertImagePtr(depth_img_ptr)->image;
  if(downsample_factor > 1){
    //use method from marius
    depth_img = downSampleNonZeroMedian(depth_img_ptr, downsample_factor);
    depth_info = downsampleCameraInfo(depth_info, downsample_factor);
  }

  if(bilateral_filter_size > 0){
    cv::Mat debug;
    cv::Mat debug2;
    debug = convert_16u_32f(depth_img);
    cv::bilateralFilter(debug, debug2, bilateral_filter_size, bilateral_filter_diff, 0);
    depth_img = convert_32f_16u(debug2);
  }

  std::shared_ptr<vertexMap> vertices(
      new vertexMap(depth_info->width, depth_info->height));

  float center_x = static_cast<float>(depth_info->K[2]);
  float center_y = static_cast<float>(depth_info->K[5]);
  float f_x = static_cast<float>(depth_info->K[0]);
  float f_y = static_cast<float>(depth_info->K[4]);
  

  for (int y = 0; y < depth_info->height; y++) {
    for (int x = 0; x < depth_info->width; x++) {
      // TODO check if this skips max range
      double depth;
      bool valid;
      // point_entry* vertex = vertices.get(x, y);
      if (depth_img.at<uint16_t>(y, x) == 0) {
        depth = max_dist;
        valid = false;
      } else {
        depth = 0.001f * static_cast<double>(depth_img.at<uint16_t>(y, x));
        valid = true;
      }

      // resuing from marius
      Point p((x - center_x) * depth / f_x, (y - center_y) * depth / f_y,
              depth);

      // convert to correct frame
      Point frame_point = T_f_d.transform(p);
      // std::cout << frame_point[0] << std::endl;
      vertices->set(frame_point, x, y, valid);
    }
  }

  return vertices;
}

// using the voxblox intensity projection strategy
inline std::shared_ptr<vertexMap> vertices_from_projection(
    sensor_msgs::CameraInfoConstPtr camera_info, Transformation T_w_c,
    Transformation T_f_w, const Layer<TsdfVoxel>& tsdf_layer, double max_dist) {
  double px = camera_info->K[2];
  double py = camera_info->K[5];
  double f_x = camera_info->K[0];
  double f_y = camera_info->K[4];  // using x focal length

  Transformation T_f_c = T_f_w * T_w_c;

  std::shared_ptr<vertexMap> vertices(
      new vertexMap(camera_info->width, camera_info->height));

  // from intensity integrator
  const FloatingPoint voxel_size = tsdf_layer.voxel_size();
  FloatingPoint max_distance = (float)max_dist;

  // for each pixel shoot a vector to look up tsdf data
  for (int y = 0; y < camera_info->height; y++) {
    for (int x = 0; x < camera_info->width; x++) {
      // subsample?
      Point bearing_vector = T_w_c.getRotation().toImplementation() *
                             Point(x - px, y - py, f_x).normalized();

      Point intersect = Point::Zero();

      // ROS_INFO("hope this works");

      bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(
          tsdf_layer, T_w_c.getPosition(), bearing_vector, max_distance,
          &intersect);
      // ROS_INFO("no crash");

      // point_entry* vertex = vertices.get(x, y);

      if (!success) {
        // calculate point at max range
        // resuing from marius
        Point p((x - px) * max_dist / f_x, (y - py) * max_dist / f_y, max_dist);
        // convert to correct frame
        Point frame_point = T_f_c.transform(p);

        // vertex->p = frame_point;
        // vertex->valid_point = false;
        vertices->set(p, x, y, false);
        // std::cout << p[0] << std::endl;
      } else {
        Point frame_point = T_f_w.transform(intersect);
        vertices->set(frame_point, x, y, true);
        // std::cout << frame_point[0] << std::endl;
      }
    }
  }

  return vertices;
}

inline void change_frame(std::shared_ptr<vertexMap> vertices,
                         Transformation T_n_o) {
  for (int y = 0; y < vertices->height; y++) {
    for (int x = 0; x < vertices->width; x++) {
      point_entry* vertex = vertices->get(x, y);
      vertex->p = T_n_o.transform(vertex->p);
    }
  }
}

inline void publish_debug_pointcloud(std::shared_ptr<vertexMap> vertices,
                                     std::shared_ptr<labelMap> lmap,
                                     ros::Publisher pub, std::string frame_id,
                                     int max_label) {
  pcl::PointCloud<pcl::PointXYZRGB> cloud(static_cast<uint>(vertices->width),
                                          static_cast<uint>(vertices->height));

  std::shared_ptr<ColorMap> colorm;
  colorm.reset(new RainbowColorMap());
  colorm->setMinValue(0.0f);
  colorm->setMaxValue(max_label);

  for (int x = 0; x < vertices->width; x++) {
    for (int y = 0; y < vertices->height; y++) {
      pcl::PointXYZRGB& p = cloud.at(x, y);
      Point pi = vertices->get(x, y)->p;
      p.x = pi[0];
      p.y = pi[1];
      p.z = pi[2];

      // assign color TODO
      // get entry with heighest weight (example)
      std::vector<std::pair<int, double>> labels = lmap->get_labels(x, y);
      if (labels.size() == 0) {
        continue;
      }
      int label = 0;
      double weight = 0.0;
      for (int i = 0; i < labels.size(); i++) {
        if (weight < labels[i].second) {
          weight = labels[i].second;
          label = labels[i].first;
        }
      }

      if (label >= max_label) {
        std::cout << "exceeding debug color limit" << std::endl;
        label = max_label;
      }
      Color c = colorm->colorLookup(label);
      p.r = c.r;
      p.g = c.g;
      p.b = c.b;
      // std::cout << label << std::endl;
      // std::cout << (int) c.r << std::endl; //also fix this
    }
  }

  // convert to msg
  sensor_msgs::PointCloud2::Ptr cloud_msg =
      boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header.frame_id = frame_id;
  pub.publish(cloud_msg);
}

////////normal methods
// like marius; test if normals are scaled correct; using now window_sizes
inline Point crossproduct_window_size(std::shared_ptr<vertexMap> vertices, int x, int y,
    uint window_size){
      int xl = x - window_size;
      int yu = y - window_size;
      int xr = x + window_size;
      int yd = y + window_size;

      // replicate on border case
      if (xl < 0) xl = 0;
      if (yu < 0) yu = 0;
      if (xr >= vertices->width) xr = vertices->width - 1;
      if (yd >= vertices->height) yd = vertices->height - 1;

      double z;
      // nw
      z = vertices->get(xl, yu)->p[2];
      cv::Vec3d nw(xl, yu, z);

      // n
      z = vertices->get(x, yu)->p[2];
      cv::Vec3d n(x, yu, z);

      // ne
      z = vertices->get(xr, yu)->p[2];
      cv::Vec3d ne(xr, yu, z);

      // w
      z = vertices->get(xl, y)->p[2];
      cv::Vec3d w(xl, y, z);

      // e
      z = vertices->get(xr, y)->p[2];
      cv::Vec3d e(xr, y, z);

      // sw
      z = vertices->get(xl, yd)->p[2];
      cv::Vec3d sw(xl, yd, z);

      // s
      z = vertices->get(x, yd)->p[2];
      cv::Vec3d s(x, yd, z);

      // se
      z = vertices->get(xr, yd)->p[2];
      cv::Vec3d se(xr, yd, z);

      cv::Vec3d n1 = (nw - e).cross(sw - e);
      cv::Vec3d n2 = (ne - s).cross(nw - s);
      cv::Vec3d n3 = (sw - n).cross(se - n);
      cv::Vec3d n4 = (se - w).cross(ne - w);

      cv::Vec3d normal_vec = cv::normalize(0.25 * (n1 + n2 + n3 + n4));

      // create normal
      Point normal(normal_vec[0], normal_vec[1], normal_vec[2]);

      return normal;
    }

inline std::shared_ptr<normalMap> calculateNormalsCrossproduct(
    std::shared_ptr<vertexMap> vertices,
    uint window_size) {
  std::shared_ptr<normalMap> normals(
      new normalMap(vertices->width, vertices->height));

  for (int y = 0; y < vertices->height; y++) {
    for (int x = 0; x < vertices->width; x++) {
      
      // if point is not valid, don't calculate a normal for it
      if (!vertices->get(x, y)->valid_point) {
        continue;
      }

      Point normal(0,0,0);
      int count = 0;
      for(uint z = 1; z <= window_size; window_size++){
        normal += crossproduct_window_size(vertices, x, y, z);
        count++;
      }

      normal /= count;
      
      normals->set(normal, x, y, true);
    }
  }

  return normals;
}


inline Point central_differences_window_size(std::shared_ptr<vertexMap> vertices, int x, int y, uint window_size){
  int xl = x - window_size;
      int yu = y - window_size;
      int xr = x + window_size;
      int yd = y + window_size;

      // replicate on border case
      if (xl < 0) xl = 0;
      if (yu < 0) yu = 0;
      if (xr >= vertices->width) xr = vertices->width - 1;
      if (yd >= vertices->height) yd = vertices->height - 1;

      // calculate x difference
      // if point is not valid, treat it as max range
      double xleft = vertices->get(xl, y)->p[2];
      double x1 = vertices->get(xl, y)->p[0];

      double xright = vertices->get(xr, y)->p[2];
      double x2 = vertices->get(xr, y)->p[0];

      double xdiff = xleft - xright;

      double xdiff2 = x2 - x1;  // std::abs(x2 - x1);

      // threshold normalization term
      if (std::abs(xdiff2) < 0.001) {
        if (xdiff2 < 0.0) {
          xdiff2 = -0.001;
        } else {
          xdiff2 = 0.001;
        }
      }
      // normalize by actual difference
      xdiff /= xdiff2;  // normalize the normal with the actual difference and
                        // not the difference in the image

      // calculate y difference
      double yup = vertices->get(x, yu)->p[2];
      double y1 = vertices->get(x, yu)->p[1];

      double ydown = vertices->get(x, yd)->p[2];
      double y2 = vertices->get(x, yd)->p[1];

      double ydiff = yup - ydown;

      double ydiff2 = y2 - y1;  // std::abs(y2 - y1);

      // threshold normalization term
      if (std::abs(ydiff2) < 0.001) {
        if (ydiff2 < 0.0) {
          ydiff2 = -0.001;
        } else {
          ydiff2 = 0.001;
        }
      }

      ydiff /= ydiff2;

      // calulate magnitude
      double magnitude = std::sqrt(xdiff * xdiff + ydiff * ydiff + 1.0);

      // create normal
      Point n(-xdiff / magnitude, -ydiff / magnitude, -1.0 / magnitude);

      return n;
}


//calculating central differences over multiple pixels
inline std::shared_ptr<normalMap> calculateNormalsCentralDifferences(
    std::shared_ptr<vertexMap> vertices, uint window_size) {
  std::shared_ptr<normalMap> normals(
      new normalMap(vertices->width, vertices->height));

  for (int y = 0; y < vertices->height; y++) {
    for (int x = 0; x < vertices->width; x++) {

      // if point is not valid, don't calculate a normal for it
      if (!vertices->get(x, y)->valid_point) {
        continue;
      }

      Point normal(0,0,0);
      uint count = 0;
      for(uint i = 1; i <= window_size; i++){
        normal += central_differences_window_size(vertices, x, y, window_size);
        count++;
      }

      normal /= count;
      
      normals->set(normal, x, y, true);
    }
  }

  return normals;
}

inline std::shared_ptr<normalMap> calculateNormalsVoxblox(
    std::shared_ptr<vertexMap> vertices, const Layer<TsdfVoxel>& tsdf_layer) {
  Interpolator<TsdfVoxel> interpolator(&tsdf_layer);

  std::shared_ptr<normalMap> normals(
      new normalMap(vertices->width, vertices->height));

  // iterate over vertex map
  for (int x = 0; x < vertices->width; x++) {
    for (int y = 0; y < vertices->height; y++) {
      // if point is not valid, don't calculate a normal for it
      if (!vertices->get(x, y)->valid_point) {
        continue;
      }

      Point p = (vertices->get(x, y))->p;

      Point gradient;

      // gradient from interpolator
      bool success =
          interpolator.getGradient(p.cast<FloatingPoint>(), &gradient, false);

      if (success) {
        normals->set(gradient, x, y, true);
      }
    }
  }

  return normals;
}

// tsdf layer only needed for voxblox
inline std::shared_ptr<normalMap> calculateNormals(
    std::shared_ptr<vertexMap> vertices,
    NormalCalculationMode normal_calculation,
    const Layer<TsdfVoxel>& tsdf_layer,
    uint window_size) {
  std::shared_ptr<normalMap> normals;

  if (normal_calculation == CROSSPRODUCT ||
      normal_calculation == CROSSPRODUCT_DEPTH) {
    normals = calculateNormalsCrossproduct(vertices, window_size);
  } else if (normal_calculation == CENTRALDIFFERENCES ||
             normal_calculation == CENTRALDIFFERENCES_DEPTH) {
    normals = calculateNormalsCentralDifferences(vertices, window_size);//TODO added window_size here, maybe do this somewhere else
  } else if (normal_calculation == VOXBLOX) {
    normals = calculateNormalsVoxblox(vertices, tsdf_layer);
  } else {
    std::cout << "no known normal method specified" << std::endl;
  }

  return normals;
}

inline void show_debug_normals(std::shared_ptr<vertexMap> vertices,
                               std::shared_ptr<normalMap> normals) {
  // convert vertices and normals to point clouds (color by z???)
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertex_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(
      new pcl::PointCloud<pcl::Normal>);

  ROS_INFO("in pcl debug");

  // only add point if normal is valid
  for (int x = 0; x < vertices->width; x++) {
    for (int y = 0; y < vertices->height; y++) {
      if (normals->get(x, y)->valid_point) {
        Point pi = vertices->get(x, y)->p;
        pcl::PointXYZ p(pi[0], pi[1], pi[2]);

        vertex_cloud->push_back(p);
      }
    }
  }

  for (int x = 0; x < normals->width; x++) {
    for (int y = 0; y < normals->height; y++) {
      if (normals->get(x, y)->valid_point) {
        Point pi = normals->get(x, y)->p;
        pcl::Normal n(pi[0], pi[1], pi[2]);

        normal_cloud->push_back(n);
      }
    }
  }

  ROS_INFO("before creating viewer");
  // add them to viewer
  // http://pointclouds.org/documentation/tutorials/pcl_visualizer.php
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->initCameraParameters();
  ROS_INFO("before color handler");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(
      vertex_cloud, 0, 255, 0);
  ROS_INFO("before adding cloud");
  viewer->addPointCloud<pcl::PointXYZ>(vertex_cloud, single_color,
                                       "vertex cloud");
  ROS_INFO("before adding normals");
  viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(
      vertex_cloud, normal_cloud, 10, 0.05, "normals");

  // show viewer
  ROS_INFO("before spinning");
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

// like suggested by marius
inline void publish_pcl_image(std::shared_ptr<vertexMap> vertices,
                              std::shared_ptr<normalMap> normals,
                              ros::Publisher pub) {
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud =
      boost::make_shared<pcl::PointCloud<pcl::Normal>>(normals->width,
                                                       normals->height);

  for (int x = 0; x < normals->width; x++) {
    for (int y = 0; y < normals->height; y++) {
      if (normals->get(x, y)->valid_point) {
        Point p = normals->get(x, y)->p;

        pcl::Normal& n = normal_cloud->at(x, y);  //(pi[0], pi[1], pi[2]);
        n.normal_x = p[0];
        n.normal_y = p[1];
        n.normal_z = p[2];

        // normal_cloud->push_back(n);
      }
    }
  }
  ROS_INFO("created pcl normals");
  pcl::io::PointCloudImageExtractorFromNormalField<pcl::Normal> extr;
  pcl::PCLImage img;
  sensor_msgs::Image img_msg;
  extr.extract(*normal_cloud, img);
  ROS_INFO("extracted");

  pcl_conversions::fromPCL(img, img_msg);
  ROS_INFO("sending result");

  pub.publish(img_msg);
}

inline void publish_atan_image(std::shared_ptr<normalMap> normals,
                               ros::Publisher pub) {
  sensor_msgs::Image atan_img;
  atan_img.height = normals->height;
  atan_img.width = normals->width;
  atan_img.encoding = "8UC1";
  atan_img.is_bigendian = false;
  atan_img.step = normals->width;

  for (int y = 0; y < normals->height; y++) {
    for (int x = 0; x < normals->width; x++) {
      uint8_t color;
      if (normals->get(x, y)->valid_point) {
        Point p = normals->get(x, y)->p;
        double val = std::atan2(p[1], p[0]);
        val += 3.1415927;
        val /= (3.1415927 * 2);
        val = std::round(val * 255);
        if (val > 255) {
          std::cout << "atan error" << std::endl;
          color = 255;
        } else if (color < 0) {
          std::cout << "atan error" << std::endl;
          color = 0;
        } else
          color = (uint8_t)val;

      } else {
        color = 0;
      }
      atan_img.data.push_back(color);
    }
  }

  pub.publish(atan_img);
}

inline std::vector<std::pair<int, int>> getNeighbours(int x, int y, int width,
                                                      int height) {
  // ROS_INFO("getting neighbours")
  std::vector<std::pair<int, int>> vec;

  for (int dx = -1; dx < 2; dx++) {
    if (x + dx < 0) {
      continue;
    } else if (x + dx >= width) {
      continue;
    }

    for (int dy = -1; dy < 2; dy++) {
      if (y + dy < 0) {
        continue;
      } else if (y + dy >= height) {
        continue;
      }

      // skip the point itself
      if (dx == 0 && dy == 0) {
        continue;
      }

      vec.push_back(std::make_pair(x + dx, y + dy));
    }
  }
  return vec;
}

inline std::shared_ptr<EdgeImage> create_distance_image(
    std::shared_ptr<vertexMap> vertices, std::shared_ptr<normalMap> normals,
    double thresh) {
  std::shared_ptr<EdgeImage> distance_im(
      new EdgeImage(vertices->width, vertices->height));

  for (int x = 0; x < vertices->width; x++) {
    for (int y = 0; y < vertices->height; y++) {
      std::vector<std::pair<int, int>> neighbours =
          getNeighbours(x, y, vertices->width, vertices->height);
      point_entry* p = vertices->get(x, y);
      point_entry* n = normals->get(x, y);

      if (!p->valid_point || !n->valid_point) {
        distance_im->setEdge(2, x, y);
        continue;
      }
      Point vertex = p->p;
      Point normal = n->p;

      double phi = 0.0;

      for (int i = 0; i < neighbours.size(); i++) {
        point_entry* pn =
            vertices->get(neighbours[i].first, neighbours[i].second);
        point_entry* nn =
            normals->get(neighbours[i].first, neighbours[i].second);

        if (pn->valid_point && nn->valid_point) {
          Point n_vertex = pn->p;
          Point n_normal = nn->p;

          double delta = std::abs((n_vertex - vertex).dot(normal));

          phi = std::max(phi, delta);
        }
        // else skip this neighbour
      }

      // add point to map
      if (phi > thresh) {
        distance_im->setEdge(1, x, y);
      } else {
        distance_im->setEdge(0, x, y);
      }
    }
    
  }

  return distance_im;
}

inline std::shared_ptr<EdgeImage> create_concave_image(
    std::shared_ptr<vertexMap> vertices, std::shared_ptr<normalMap> normals,
    double thresh) {
  std::shared_ptr<EdgeImage> concave_im(
      new EdgeImage(vertices->width, vertices->height));

  for (int x = 0; x < vertices->width; x++) {
    for (int y = 0; y < vertices->height; y++) {
      std::vector<std::pair<int, int>> neighbours =
          getNeighbours(x, y, vertices->width, vertices->height);

      point_entry* p = vertices->get(x, y);
      point_entry* n = normals->get(x, y);

      if (!p->valid_point || !n->valid_point){
          concave_im->setEdge(2, x, y);
          continue;
        }
      Point vertex = p->p;
      Point normal = n->p;

      double phi = 0.0;

      for (int i = 0; i < neighbours.size(); i++) {
        point_entry* pn =
            vertices->get(neighbours[i].first, neighbours[i].second);
        point_entry* nn =
            normals->get(neighbours[i].first, neighbours[i].second);
        if (pn->valid_point && nn->valid_point) {
          Point n_vertex = pn->p;
          Point n_normal = nn->p;

          double delta = (n_vertex - vertex).dot(normal);
          
          if (delta >= 0) {
            double val = 1.0 - (n_normal.dot(normal));
            phi = std::max(phi, val);
          }
          // else phi was set to 0 before
        }
        // else skip this neighbour
      }

      // add point to map
      if (phi > thresh) {
        concave_im->setEdge(1, x, y);
      } else {
        concave_im->setEdge(0, x, y);
      }
    }
    
  }

  return concave_im;
}

// maybe redo this later
inline void fuse_edge_image_label_map(std::shared_ptr<EdgeImage> edge_im,
                                      std::shared_ptr<labelMap> lmap,
                                      double percentage) {
  
  cv::Mat cv_image = edge_im->createCvImage();
  // using connected like marius
  cv::Mat connected;

  int num_connected = cv::connectedComponents(cv_image, connected, 4, CV_16U);
  ROS_INFO("connected comp");

  // now iterate over connected image to get highest label
  int highest = 0;
  for (int x = 0; x < edge_im->width; x++) {
    for (int y = 0; y < edge_im->height; y++) {
      // check if not edge
      if (edge_im->getEdge(x, y) == 0) {
        ushort pixel_label = connected.at<ushort>(y, x);
        if (pixel_label > highest) {
          highest = pixel_label;
        }
      }
    }
  }


  // create vector to track scores of obbject labels per segmentation label
  std::unordered_map<int, std::unordered_map<int, double>> tracking;
  std::unordered_map<int, int> sizes;
  for (int i = 1; i <= highest; i++) {
    std::unordered_map<int, double> labels;
    tracking[i] = labels;
    sizes[i] = 0;
  }

  // now iterate over label map
  for (int x = 0; x < lmap->width; x++) {
    for (int y = 0; y < lmap->height; y++) {
      
      if (edge_im->getEdge(x, y) == 0) {
        ushort pixel_label = connected.at<ushort>(y, x);

        sizes[pixel_label] += 1;

        std::vector<std::pair<int, double>> labels = lmap->get_labels(x, y);

        // iterate over labels
        for (int i = 0; i < labels.size(); i++) {
          int label = labels[i].first;
          std::unordered_map<int, double>::iterator it =
              tracking[pixel_label].find(label);

          if (it == tracking[pixel_label].end()) {
            tracking[pixel_label][label] = 0.0;
          }

          // TODO here the actual weight can be used!!!!!
          tracking[pixel_label][label] += 1.0;
        }
      }
    }
  }

  // now get the highest labels for each segment (and a weight)
  // TODO rethink the weighting
  std::unordered_map<int,std::pair<int, double>> winning_labels;
  for (auto t : tracking) {
    int label = -1;
    double weight = 0.0;

    for (auto l : t.second) {
      if (l.second > weight) {
        weight = l.second;
        label = l.first;
      }
    }
    //check if enough correspondences were found
    if(weight / ((double) sizes[t.first]) > percentage){
      winning_labels[t.first] = std::make_pair(label, weight);
    }else{
      winning_labels[t.first] = std::make_pair(-1, 0.0);
    }
  }

  // now give this segmentation to the edge image
  for (int x = 0; x < edge_im->width; x++) {
    for (int y = 0; y < edge_im->height; y++) {
      if (edge_im->getEdge(x, y) == 0) {
        ushort pixel_label = connected.at<ushort>(y, x);

        // assign label
        int label = winning_labels[pixel_label].first;
        double weight = winning_labels[pixel_label].second;
        edge_im->setLabel(label, x, y);
        edge_im->setWeight(weight, x, y);
      }
    }
  }
}

inline void refine_edge_image(std::shared_ptr<EdgeImage> edge_im, std::shared_ptr<labelMap> lmap, bool match_lmap) {
  // iterate over edge_im and get all points with edge = 1 (calculated edge)
  std::vector<std::pair<int, int>> to_refine;
  for (int x = 1; x < edge_im->width - 1; x++) {  // avoid border
    for (int y = 1; y < edge_im->height - 1; y++) {
      if (edge_im->getEdge(x, y) == 1) {
        to_refine.push_back(std::make_pair(x, y));
      }
    }
  }

  bool change = true;
  while (change) {
    std::vector<std::pair<std::pair<int, int>, std::pair<int, double>>>
        to_update;
    std::vector<std::pair<int, int>> refine_again;

    change = false;


    
    for (int i = 0; i < to_refine.size(); i++) {
      // check all neighbours
      int label = -2;
      
      double weight = 0.0;
      int count = 0;
      int x = to_refine[i].first;
      int y = to_refine[i].second;

      if (edge_im->getEdge(x + 1, y) == 0) {
        if (label == -2) {
          count++;
          label = edge_im->getLabel(x + 1, y);
          weight = edge_im->getWeight(x + 1, y);
        } else if (label == edge_im->getLabel(x + 1, y)) {
          count++;
          weight += edge_im->getWeight(x + 1, y);
        } else {
          // label not valid, point is real edge point
          continue;
        }
      }

      if (edge_im->getEdge(x - 1, y) == 0) {
        if (label == -2) {
          count++;
          label = edge_im->getLabel(x - 1, y);
          weight = edge_im->getWeight(x - 1, y);
        } else if (label == edge_im->getLabel(x - 1, y)) {
          count++;
          weight += edge_im->getWeight(x - 1, y);
        } else {
          // label not valid, point is real edge point
          continue;
        }
      }

      if (edge_im->getEdge(x, y + 1) == 0) {
        if (label == -2) {
          count++;
          label = edge_im->getLabel(x, y + 1);
          weight = edge_im->getWeight(x, y + 1);
        } else if (label == edge_im->getLabel(x, y + 1)) {
          count++;
          weight += edge_im->getWeight(x, y + 1);
        } else {
          // label not valid, point is real edge point
          continue;
        }
      }

      if (edge_im->getEdge(x, y - 1) == 0) {
        if (label == -2) {
          count++;
          label = edge_im->getLabel(x, y - 1);
          weight = edge_im->getWeight(x, y - 1);
        } else if (label == edge_im->getLabel(x, y - 1)) {
          count++;
          weight += edge_im->getWeight(x, y - 1);
        } else {
          // label not valid, point is real edge point
          continue;
        }
      }

      if (count == 0) {
        // re-add
        refine_again.push_back(std::make_pair(x, y));

      } else {
        weight /= count;
        change = true;

        to_update.push_back(std::make_pair(std::make_pair(x, y),
                                           std::make_pair(label, weight)));
      }
    }
    to_refine = refine_again;

    // now make changes to map

    for (int i = 0; i < to_update.size(); i++) {
      int x = to_update[i].first.first;
      int y = to_update[i].first.second;
      int label = to_update[i].second.first;
      //check if label is possible labels (only labeling points that wre previously labeled, else the boundaries could be blurred between objects)
      if(match_lmap){
        std::vector<std::pair<int, double>> pos_labels = lmap->get_labels(x, y);
        bool found = false;
        for(int z = 0; z < pos_labels.size(); z++){
          if(pos_labels[z].first == label){
            found = true;
            continue;
          }
        }

        //skipping if the point wasn't labeled before
        if(!found){
          continue;
        }
      }

      double weight = to_update[i].second.second;
      edge_im->setEdge(0, x, y);
      edge_im->setLabel(label, x, y);
      edge_im->setWeight(weight, x, y);
    }
  }
}



inline std::shared_ptr<EdgeImage> upsample_edge_image(std::shared_ptr<EdgeImage> edge_im, int factor){
  if(factor <= 1){
    return edge_im; //can't upsample
  }

  std::shared_ptr<EdgeImage> new_edge_im(new EdgeImage(edge_im->width * factor, edge_im->height * factor));

  for(int x = 0; x < edge_im->width; x++){

    for(int y = 0; y < edge_im->height; y++){

      for(int u = 0; u < factor; u++){

        for(int v = 0; v < factor; v++){
          int edge = edge_im->getEdge(x, y);
          new_edge_im->setEdge(edge, x * factor + u, y * factor + v);  
        }
      }
    }

  }
  return new_edge_im;
}

inline std::shared_ptr<labelMap> downsample_label_map(std::shared_ptr<labelMap> lmap, int factor){
  if(factor <= 1){
    return lmap;
  }

  std::shared_ptr<labelMap> new_lmap(new labelMap( (int) std::ceil(lmap->width / ((double) factor)), (int) std::ceil(lmap->height / ((double) factor))));
  for(int x = 0; x < new_lmap->width; x++){
    for(int y = 0; y < new_lmap->height; y++){
      for(int u = 0; u < factor; u++){
        for(int v = 0; v < factor; v++){
          if(x * factor + u >= lmap->width || y *factor + v >= lmap->height){
            continue;
          }

          //this may make problems with scales that end in floating points, 
          //iterate over label map
          std::vector<std::pair<int, double>> labels = lmap->get_labels(x * factor + u, y * factor + v);
          for(int i = 0; i < labels.size(); i++){
            new_lmap->add(x, y, labels[i].first, labels[i].second / ((double) factor * factor) );
          }
          
        }
      }
      
    }
  }

  return new_lmap;
}



}  // namespace voxblox
#endif