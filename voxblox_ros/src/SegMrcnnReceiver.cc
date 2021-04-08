#include "voxblox_ros/SegMrcnnReceiver.h"

#include <voxblox_ros/image_operations.h>

namespace voxblox {
template <class T>
SegMrcnnReceiver<T>::SegMrcnnReceiver(const ros::NodeHandle& nh,
                                      const ros::NodeHandle& nh_private,
                                      std::string color_image_topic,
                                      std::string depth_image_topic,
                                      std::string color_info_topic,
                                      std::string depth_info_topic)
    : SegReceiver(nh, nh_private),
      color_img_sub(nh_, color_image_topic, 1),
      color_info_sub(nh_, color_info_topic, 1),
      depth_img_sub(nh_, depth_image_topic, 1),
      depth_info_sub(nh_, depth_info_topic, 1),
      sync(message_filters::sync_policies::ApproximateTime<
               sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,
               sensor_msgs::CameraInfo>(10),
           color_img_sub, depth_img_sub, color_info_sub, depth_info_sub) {
  id_counter = 1;
  nh.param<double>("mrcnn_image_scale", scale, 1.0);
  nh.param<std::string>("body_frame", body_frame, "base_link");
  published_to_cnn = false;
  got_callback = false;
  nh.param<bool>("aligned_rgbd", aligned_rgbd, false);
  nh.param<bool>("use_weight", use_weight, true);

  nh.param<int>("downsample_receiver_factor", downsample_receiver_factor, 0);

  nh.param<int>("bilateral_filter_size", bilateral_filter_size, 0);
  nh.param<double>("bilateral_filter_diff", bilateral_filter_diff, 0.0);

  sync.setAgePenalty(0.5);
  sync.setMaxIntervalDuration(ros::Duration(0.1));
  sync.registerCallback(
      boost::bind(&SegMrcnnReceiver<T>::callback, this, _1, _2, _3, _4));
  seg_img_pub = nh_.advertise<sensor_msgs::Image>("/segmented_image", 2);
  debug_seg_cloud_pub =
      nh_.advertise<sensor_msgs::PointCloud2>("/segmented/debugcloud", 2);
}

template <class T>
void SegMrcnnReceiver<T>::callback(
    const sensor_msgs::ImageConstPtr& color_img,
    const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info) {
  sensor_msgs::Image c_img = *color_img;
  // TODO make a way to escape from this, if cnn fails
  if (published_to_cnn) return;

  // create raw pointer struct
  std::shared_ptr<rawDataPointer> data_ptr(new rawDataPointer());

  // TODO share this smhw between both files
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
    ROS_ERROR("Failed to get transform for segmented image");
    return;
  }

  Transformation T_body_depth = T_body_cloud;

  Transformation T_body_color;
  if (!transformer_.lookupTransform(color_img->header.frame_id, body_frame,
                                    depth_img->header.stamp, &T_body_color)) {
    ROS_ERROR("Failed to get transform for segmented image");
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

  sensor_msgs::PointCloud2::Ptr cloud_msg =
      boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header = depth_img->header;
  // give cloud to server
  if(!server->receivePointCloud(cloud_msg)){
    return;
  }

  // if problems exist, change scale for image for cnn here TODO
  /*
   * Do downscaling
   */

  // if color image hasnt right scale, rescale it || using marius versions to
  // avoid mask rccn error || upsample later? discuss this
  if (color_img->height > 1024 || color_img->width > 1024) {
    // cv::Mat color_img_downsampled = downSampleColorImg(color_img_msg,
    // downsampling_factor);
    double downsample_factorx =
        std::max(1.0, double(color_img->width / 1024.0));
    double downsample_factory =
        std::max(1.0, double(color_img->height / 1024.0));
    double downsample_factor = std::max(downsample_factorx, downsample_factory);

    // ripped from downsample
    cv::Mat img_downsampled;
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(
        color_img, sensor_msgs::image_encodings::TYPE_8UC3);
    cv::resize(img->image, img_downsampled, cv::Size(0, 0),
               1.0 / downsample_factor, 1.0 / downsample_factor,
               cv::INTER_AREA);

    // small changes to make it work in this context
    cv_bridge::CvImage i = *img;
    i.image = img_downsampled;
    i.header = color_img->header;
    i.encoding = color_img->encoding;
    sensor_msgs::ImagePtr color_img_new = i.toImageMsg();

    // not rescaling cam info, as we upsample later
    c_img = *color_img_new;

    scale = downsample_factor;
  }

  // publish to mrcnn
  cnn_image_pub.publish(c_img);
  published_to_cnn = true;

  data = data_ptr;
}
template <class T>
void SegMrcnnReceiver<T>::cnn_callback(const mask_rcnn_ros::Result cnn_result) {
  // if scaled before, do upscaling! TODO
  mask_rcnn_ros::Result res = cnn_result;
  if (scale != 1.0) {
    // DEBUG rescale images

    double downsample_factor = 1.0 / scale;

    mask_rcnn_ros::Result res_new;
    res_new.class_ids = cnn_result.class_ids;
    res_new.class_names = cnn_result.class_names;
    res_new.scores = cnn_result.scores;

    ROS_INFO("2");
    if ((cnn_result.masks).size() == 0) {
      return;
      // nothing to do without masks
    }

    for (int i = 0; i < (cnn_result.masks).size(); i++) {
      // ripped from downsample
      cv::Mat img_downsampled;

      sensor_msgs::ImageConstPtr ptr(
          new sensor_msgs::Image(cnn_result.masks[i]));
      cv_bridge::CvImageConstPtr img =
          cv_bridge::toCvShare(ptr, sensor_msgs::image_encodings::TYPE_8UC1);
      cv::resize(img->image, img_downsampled, cv::Size(0, 0),
                 1.0 / downsample_factor, 1.0 / downsample_factor,
                 cv::INTER_AREA);

      // small changes to make it work in this context
      cv_bridge::CvImage im = *img;
      im.image = img_downsampled;
      im.header = cnn_result.masks[i].header;
      im.encoding = cnn_result.masks[i].encoding;
      sensor_msgs::ImagePtr color_img_new = im.toImageMsg();

      res_new.masks.push_back(*color_img_new);
      ROS_INFO("3");
    }

    // using original camera info, hoping it is correctly rescaled

    // setting result, TODO test it
    res = res_new;
  }
  if (!initedLookup) {
    // initing private lookup
    lookup.reset(new LabelLookup());
    initedLookup = true;
  }

  std::shared_ptr<labelMap> label_map =
      convertMaskRCNNSegmentation(&res, data->color_info_ptr, lookup);

  // TODO maybe check if label map has segmentation?/or do it in data fuser?

  server->receiveInitialSegmentation(label_map, data);

  published_to_cnn = false;
}

template <class T>
void SegMrcnnReceiver<T>::setServerPointer(
    std::shared_ptr<SegServer<T>> server_ptr) {
  server = server_ptr;
}

template class SegMrcnnReceiver<LabelVoxel>;
template class SegMrcnnReceiver<MultiLabelVoxel>;

}  // namespace voxblox