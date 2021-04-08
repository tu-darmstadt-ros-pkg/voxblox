#include "voxblox_ros/SegDebugReceiver.h"

#include <voxblox_ros/image_operations.h>
namespace voxblox {
template <class T>
SegDebugReceiver<T>::SegDebugReceiver(const ros::NodeHandle& nh,
                                      const ros::NodeHandle& nh_private,
                                      std::string color_image_topic,
                                      std::string depth_image_topic,
                                      std::string color_info_topic,
                                      std::string depth_info_topic,
                                      std::string cnn_result_topic)
    : SegReceiver(nh, nh_private),
      color_img_sub(nh_, color_image_topic, 1),
      color_info_sub(nh_, color_info_topic, 1),
      depth_img_sub(nh_, depth_image_topic, 1),
      depth_info_sub(nh_, depth_info_topic, 1),
      cnn_result_sub(nh_, cnn_result_topic, 1),
      sync(message_filters::sync_policies::ApproximateTime<
               sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
               sensor_msgs::CameraInfo, mask_rcnn_ros::Result>(10),
           color_img_sub, depth_img_sub, color_info_sub, depth_info_sub,
           cnn_result_sub) {
  sync.setAgePenalty(0.5);
  sync.setMaxIntervalDuration(ros::Duration(0.1));

  // sync.registerCallback(boost::bind(&DebugReceiver::callback, this, _1, _2,
  // _3, _4, _5));
  sync.registerCallback(
      boost::bind(&SegDebugReceiver<T>::callback, this, _1, _2, _3, _4, _5));

      highest_depth=0;

  id_counter = 1;

  nh.param<bool>("aligned_rgbd", aligned_rgbd, false);
  nh.param<bool>("use_weight", use_weight, true);
  nh.param<std::string>("body_frame", body_frame, "base_link");

  nh.param<int>("downsample_receiver_factor", downsample_receiver_factor, 0);

  nh.param<bool>("skip_tsdf", skip_tsdf, false);

  nh.param<int>("bilateral_filter_size", bilateral_filter_size, 0);
  nh.param<double>("bilateral_filter_diff", bilateral_filter_diff, 0.0);

  // TODO read from param server
  nh.param<double>("mrcnn_image_scale", scale, 1.0);
}

template <class T>
void SegDebugReceiver<T>::callback(
    const sensor_msgs::ImageConstPtr& color_img,
    const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info,
    const mask_rcnn_ros::ResultConstPtr& cnn_result) {
  // create raw pointer struct
  std::shared_ptr<rawDataPointer> data_ptr(new rawDataPointer());


  data_ptr->color_image_ptr = color_img;
  data_ptr->depth_image_ptr = depth_img;
  data_ptr->color_info_ptr = color_info;
  data_ptr->depth_info_ptr = depth_info;
  data_ptr->use_weight = use_weight;

  data_ptr->segmentation_info = color_info;

  data_ptr->aligned_color = true;
  data_ptr->aligned_depth = aligned_rgbd;

  data_ptr->body_frame = body_frame;

  // TODO use pointer?
  Transformation T_body_cloud;
  if (!transformer_.lookupTransform(depth_img->header.frame_id, body_frame,
                                    depth_img->header.stamp, &T_body_cloud)) {
    ROS_ERROR("Failed to get transform for depth image");
    return;
  }

  Transformation T_body_depth = T_body_cloud;

  Transformation T_body_color;
  if (!transformer_.lookupTransform(color_img->header.frame_id, body_frame,
                                    depth_img->header.stamp, &T_body_color)) {
    ROS_ERROR("Failed to get transform for color image");
    return;
  }


  Transformation T_body_label_map = T_body_color;

  data_ptr->cloud_body = T_body_cloud;
  data_ptr->depth_body = T_body_depth;
  data_ptr->color_body = T_body_color;
  data_ptr->label_map_body = T_body_label_map;

  // get colored pointcloud
  cv::Mat depth_img_mat = convertImagePtr(depth_img)->image;
  cv::Mat color_img_mat =
      cv_bridge::toCvShare(color_img, sensor_msgs::image_encodings::TYPE_8UC3)
          ->image;

  sensor_msgs::CameraInfoPtr depth_cam_info_ptr =
      boost::make_shared<sensor_msgs::CameraInfo>(*depth_info);


  //DEBUG, DELETE THIS LATER
  /*for (int y = 0; y < depth_info->height; y++) {
    for (int x = 0; x < depth_info->width; x++) {
      // TODO check if this skips max range
      
      if (depth_img_mat.at<uint16_t>(y, x) > highest_depth){
          highest_depth = depth_img_mat.at<uint16_t>(y, x);
      }}
  }*/

  //std::cout << "highest depth value: " << highest_depth <<"mm" << std::endl;


  // done similar to segmentation server
  // now convert image to coordinates/pointcloud and vector of labels? then add
  // them to map like intensity integrator?
  pcl::PointCloud<pcl::PointXYZRGB> cloud;

  if(downsample_receiver_factor > 1){
    aligned_rgbd = false;
    data_ptr->aligned_depth = false;
    //using marius method
    depth_img_mat = downSampleNonZeroMedian(depth_img, downsample_receiver_factor);
    depth_cam_info_ptr = downsampleCameraInfo(depth_info, downsample_receiver_factor);
  }

  if(bilateral_filter_size > 0){
    cv::Mat debug;
    cv::Mat debug2;
    debug = convert_16u_32f(depth_img_mat);
    cv::bilateralFilter(debug, debug2, bilateral_filter_size, bilateral_filter_diff, 0);
    depth_img_mat = convert_32f_16u(debug2);
  }

  if (aligned_rgbd) {
    cloud = convertDepthImageToCloudColored(depth_img_mat, depth_cam_info_ptr,
                                            color_img_mat);
  } else {
    cloud = convertDepthImageToCloudMono(depth_img_mat, depth_cam_info_ptr);
  }

  // give cloud to server
  sensor_msgs::PointCloud2::Ptr cloud_msg =
      boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header = depth_img->header;

  std::cout << "cloud size" << std::endl;
  std::cout << cloud.size() << std::endl;


  if(!server->receivePointCloud(cloud_msg)){
    return;
  }

  // if problems exist, change scale for image for cnn here TODO
  /*
   * Do upscaling of results
   */

  //std::cout << "a" << std::endl;
  mask_rcnn_ros::Result res = *cnn_result;
  if (scale != 1.0) {
    // DEBUG rescale images

    double downsample_factor = 1.0 / scale;

    mask_rcnn_ros::Result res_new;
    res_new.class_ids = cnn_result->class_ids;
    res_new.class_names = cnn_result->class_names;
    res_new.scores = cnn_result->scores;

    //std::cout << "b" << std::endl;

    if ((cnn_result->masks).size() == 0) {
      return;
      // nothing to do without masks
    }

    //std::cout << "c" << std::endl;
    for (int i = 0; i < (cnn_result->masks).size(); i++) {
      // ripped from downsample
      cv::Mat img_downsampled;

      sensor_msgs::ImageConstPtr ptr(
          new sensor_msgs::Image(cnn_result->masks[i]));
      cv_bridge::CvImageConstPtr img =
          cv_bridge::toCvShare(ptr, sensor_msgs::image_encodings::TYPE_8UC1);
      cv::resize(img->image, img_downsampled, cv::Size(0, 0),
                 1.0 / downsample_factor, 1.0 / downsample_factor,
                 cv::INTER_AREA);

      // small changes to make it work in this context
      cv_bridge::CvImage im = *img;
      im.image = img_downsampled;
      im.header = cnn_result->masks[i].header;
      im.encoding = cnn_result->masks[i].encoding;
      sensor_msgs::ImagePtr color_img_new = im.toImageMsg();

      res_new.masks.push_back(*color_img_new);
    }
    //std::cout << "d" << std::endl;
    // using original camera info, hoping it is correctly rescaled

    // setting result, TODO test it
    res = res_new;
  }
  data = data_ptr;

  if (!initedLookup) {
    // initing private lookup
    lookup.reset(new LabelLookup());
    initedLookup = true;
  }

  //std::cout << "e" << std::endl;

  std::shared_ptr<labelMap> label_map =
      convertMaskRCNNSegmentation(&res, data->color_info_ptr, lookup);

  //std::cout << "f" << std::endl;

  server->receiveInitialSegmentation(label_map, data);
  std::cout << "finished" << std::endl;
}

template <class T>
void SegDebugReceiver<T>::setServerPointer(
    std::shared_ptr<SegServer<T>> server_ptr) {
  server = server_ptr;
}

template class SegDebugReceiver<LabelVoxel>;
template class SegDebugReceiver<MultiLabelVoxel>;
}  // namespace voxblox