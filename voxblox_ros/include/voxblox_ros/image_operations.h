#ifndef VOXBLOX_ROS_IMAGE_OPERATIONS_H_
#define VOXBLOX_ROS_IMAGE_OPERATIONS_H_

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <algorithm>
#include <vector>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>

namespace voxblox {

template<typename T>
inline T imageMedian(const cv::Mat& img) {
  std::vector<T> vec_from_mat(img.begin<T>(), img.end<T>());
  std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());
  return vec_from_mat[vec_from_mat.size() / 2];
}

template<typename T>
inline T imageMedianNonZero(const cv::Mat& img) {
  std::vector<T> vec_from_mat(img.begin<T>(), img.end<T>());
  //filter all zeros
  vec_from_mat.erase(std::remove(vec_from_mat.begin(), vec_from_mat.end(), 0), vec_from_mat.end());
  std::nth_element(vec_from_mat.begin(), vec_from_mat.begin() + vec_from_mat.size() / 2, vec_from_mat.end());
  return vec_from_mat[vec_from_mat.size() / 2];
}

inline cv::Mat downSampleNonZeroMedian(const sensor_msgs::ImageConstPtr& depth_img_msg, int downsample_factor) {
  ROS_ASSERT(downsample_factor > 0);

  cv_bridge::CvImageConstPtr depth_img;

  // convert the unit to mm if needed
  if (depth_img_msg->encoding == "32FC1") {
    cv_bridge::CvImagePtr depth_img_mm = cv_bridge::toCvCopy(depth_img_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_img_mm->image.convertTo(depth_img_mm->image, CV_16U, 1000.0);
    depth_img = depth_img_mm;
  } else {
    depth_img = cv_bridge::toCvShare(depth_img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  }

  int width = static_cast<int>(depth_img_msg->width);
  int height = static_cast<int>(depth_img_msg->height);

  cv::Mat img_downsampled(height/downsample_factor, width/downsample_factor, CV_16U, cv::Scalar(0));

  for (int row = 0; row < (height - downsample_factor); row+=downsample_factor) {
    for (int col = 0; col < (width - downsample_factor); col+=downsample_factor) {
      cv::Rect roi(col, row, downsample_factor, downsample_factor);
      uint16_t median = imageMedianNonZero<uint16_t>(depth_img->image(roi));
      img_downsampled.at<uint16_t>(row/downsample_factor, col/downsample_factor) = median;
    }
  }

  cv::Mat depth_8bit, img_downsampled_8bit;
  cv::normalize(img_downsampled, img_downsampled_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  cv::normalize(depth_img->image, depth_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);

  return img_downsampled;
}

inline cv::Mat downSampleColorImg(const sensor_msgs::ImageConstPtr& img_msg, int downsample_factor) {
  cv::Mat img_downsampled;
  cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::TYPE_8UC3);
  cv::resize(img->image, img_downsampled, cv::Size(0, 0),
             1.0/downsample_factor, 1.0/downsample_factor,
             cv::INTER_AREA);

  return img_downsampled;
}

inline sensor_msgs::CameraInfoPtr downsampleCameraInfo(const sensor_msgs::CameraInfoConstPtr& cam_info, int downsample_factor) {
  sensor_msgs::CameraInfoPtr cam_info_downsampled = boost::make_shared<sensor_msgs::CameraInfo>(*cam_info);

  cam_info_downsampled->height /= static_cast<uint>(downsample_factor);
  cam_info_downsampled->width /= static_cast<uint>(downsample_factor);

  cam_info_downsampled->K[0] /= downsample_factor;  // fx
  cam_info_downsampled->K[2] /= downsample_factor;  // cx
  cam_info_downsampled->K[4] /= downsample_factor;  // fy
  cam_info_downsampled->K[5] /= downsample_factor;  // cy

  cam_info_downsampled->P[0] /= downsample_factor;  // fx
  cam_info_downsampled->P[2] /= downsample_factor;  // cx
  cam_info_downsampled->P[3] /= downsample_factor;  // T
  cam_info_downsampled->P[5] /= downsample_factor;  // fy
  cam_info_downsampled->P[6] /= downsample_factor;  // cy

  return cam_info_downsampled;
}
}  // namespace voxblox

#endif  // VOXBLOX_ROS_IMAGE_OPERATIONS_H_
