#include "voxblox_ros/SegEdgeFuser.h"

#include <voxblox_ros/image_operations.h>

namespace voxblox {

template <class T>
SegEdgeFuser<T>::SegEdgeFuser(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf,
                              std::shared_ptr<Layer<T>> label)
    : SegDataFuser<T>(nh, tsdf, label), tsdf_layer(tsdf) {
  nh.param<double>("threshold_concave", thresh_concave, 0.94);
  nh.param<double>("threshold_distance", thresh_distance, 0.94);
  nh.param<bool>("aligned_rgbd", aligned_rgbd, false);
  nh.param<bool>("pcl_debug", show_pcl_debug, false);
  nh.param<bool>("pcl_normal_image", show_pcl_image, false);
  nh.param<bool>("atan_image", show_atan_image, false);
  nh.param<double>("max_distance", max_dist_, 20.0);
  nh.param<bool>("edge_debug_images", edge_debug_images, false);
  nh.param<bool>("refine_edge_image", to_refine_edge_image, false);
  nh.param<bool>("no_blur_refine", no_blur_refine, false);
  nh.param<int>("window_size", window_size, 3);
  nh.param<double>("enforce_percentage", enforce_percentage, 0);

  nh.param<int>("bilateral_filter_size", bilateral_filter_size, 0);
  nh.param<double>("bilateral_filter_diff", bilateral_filter_diff, 0.0);

  enforce_percentage /= 100;

  concave_im_pub =
      nh.advertise<sensor_msgs::Image>("/debug/concave_image", 1, true);

  distance_im_pub =
      nh.advertise<sensor_msgs::Image>("/debug/distance_image", 1, true);

  merged_im_pub =
      nh.advertise<sensor_msgs::Image>("/debug/merged_image", 1, true);

  labeled_im_pub = 
      nh.advertise<sensor_msgs::Image>("/debug/labeled_image", 1, true);

  std::string normal_mode;
  nh.param<std::string>("normal_calculation_mode", normal_mode,
                        "CENTRALDIFFERENCES");

  if (normal_mode.compare("CENTRALDIFFERENCES") == 0) {
    normal_calculation = CENTRALDIFFERENCES;
  } else if (normal_mode.compare("CROSSPRODUCT") == 0) {
    normal_calculation = CROSSPRODUCT;
  } else if (normal_mode.compare("CENTRALDIFFERENCES_DEPTH") == 0) {
    normal_calculation = CENTRALDIFFERENCES_DEPTH;
  } else if (normal_mode.compare("CROSSPRODUCT_DEPTH") == 0) {
    normal_calculation = CROSSPRODUCT_DEPTH;
  } else if (normal_mode.compare("VOXBLOX") == 0) {
    normal_calculation = VOXBLOX;
  } else {
    ROS_INFO(
        "Wrong parameter for normal_calculation_mode, using central "
        "differences, using default");
    normal_calculation = CENTRALDIFFERENCES;
  }
  if (show_pcl_image)
    pcl_normal_pub =
        nh.advertise<sensor_msgs::Image>("pcl_normal_image", 1, true);

  if (show_atan_image)
    atan_img_pub = nh.advertise<sensor_msgs::Image>("atan_image", 1, true);

  nh.param<int>("downsample_datafuser_factor", downsample_datafuser_factor, 0);
}

template <class T>
void SegEdgeFuser<T>::set_integrator(std::shared_ptr<SegTsdfIntegrator<T>> i) {
  integrator = i;
}

template <class T>
void SegEdgeFuser<T>::fuse(Transformation T_G_C, std::shared_ptr<labelMap> lmap,
                           std::shared_ptr<rawDataPointer> data) {
  //ROS_INFO("in fuse");
  // claculate rotation T_w_body
  Transformation T_w_b = T_G_C * data->cloud_body;

  // here calculations should be done later
  fuseEdgeMap(T_w_b, lmap, data);
}

template <class T>
void SegEdgeFuser<T>::fuseEdgeMap(Transformation T_w_b,
                                  std::shared_ptr<labelMap> lmap,
                                  std::shared_ptr<rawDataPointer> data) {
  // calculation of vertices
  std::shared_ptr<vertexMap> vertices;
  std::shared_ptr<vertexMap> original_vertices;
  // Transformation T_w_lm = T_w_b * data->label_map_body.inverse();
  Transformation T_w_im;
  //if(aligned_depth)
  
  if (normal_calculation == CROSSPRODUCT_DEPTH ||
      normal_calculation == CENTRALDIFFERENCES_DEPTH) {
    Transformation T_w_d = T_w_b * data->depth_body.inverse();
    T_w_im = T_w_d;
    Transformation i;
    i.setIdentity();

    if(downsample_datafuser_factor > 1){
      vertices = vertices_from_depth(data->depth_image_ptr, data->depth_info_ptr,
                                   T_w_d, max_dist_, downsample_datafuser_factor, bilateral_filter_size, bilateral_filter_diff);
      original_vertices = vertices_from_depth(data->depth_image_ptr, data->depth_info_ptr,
                                   T_w_d, max_dist_, 1, bilateral_filter_size, bilateral_filter_diff);

      
      //original_vertices = vertices_from_projection(data->depth_info_ptr, T_w_d, i,
        //                                tsdf_layer, max_dist_);

      //TODO downsample lmap here DEBUG IMPORTANT
    }else{
    vertices = vertices_from_depth(data->depth_image_ptr, data->depth_info_ptr,
                                   T_w_d, max_dist_, 1, bilateral_filter_size, bilateral_filter_diff);
    original_vertices = vertices_from_depth(data->depth_image_ptr, data->depth_info_ptr,
                                   T_w_d, max_dist_, 1, bilateral_filter_size, bilateral_filter_diff);

    //original_vertices = vertices_from_projection(data->depth_info_ptr, T_w_d, i,
      //                                  tsdf_layer, max_dist_);
    }
  } else {
    Transformation T_w_l = T_w_b * data->label_map_body.inverse();
    T_w_im = T_w_l;
    Transformation i;
    i.setIdentity();

    if(downsample_datafuser_factor > 1){
      sensor_msgs::CameraInfoConstPtr depth_info = downsampleCameraInfo(data->segmentation_info, downsample_datafuser_factor);

      vertices = vertices_from_projection(depth_info, T_w_l, i,
                                        tsdf_layer, max_dist_);
    original_vertices = vertices_from_projection(depth_info, T_w_l, i,
                                        tsdf_layer, max_dist_);
    }else{
    vertices = vertices_from_projection(data->segmentation_info, T_w_l, i,
                                        tsdf_layer, max_dist_);
    original_vertices = vertices_from_projection(data->segmentation_info, T_w_l, i,
                                        tsdf_layer, max_dist_);
    }
  }

  // bilinear filter ? TODO? (like tateno)

  // calculate normals
  //ROS_INFO("calculating normals");

  // change vertices to label_map frame
  if (normal_calculation != VOXBLOX) {
    // transform into camera frame
    change_frame(vertices, T_w_im.inverse());
  }

  std::shared_ptr<normalMap> normals =
      calculateNormals(vertices, normal_calculation, tsdf_layer, window_size);

  // show debug normals if set to true
  if (show_pcl_debug) {
    //ROS_INFO("showing debug pcl");
    show_debug_normals(vertices, normals);
  }

  if (show_pcl_image) {
    //ROS_INFO("sending pcl image");
    publish_pcl_image(vertices, normals, pcl_normal_pub);
  }

  if (show_atan_image) {
    //ROS_INFO("sending atan image");
    publish_atan_image(normals, atan_img_pub);
  }

  // create edge images
  std::shared_ptr<EdgeImage> distance_im =
      create_distance_image(vertices, normals, thresh_distance);

  std::shared_ptr<EdgeImage> concave_im =
      create_concave_image(vertices, normals, thresh_concave);

  // merge
  std::shared_ptr<EdgeImage> merged_im(
      new EdgeImage(vertices->width, vertices->height));

  for (int x = 0; x < distance_im->width; x++) {
    for (int y = 0; y < distance_im->height; y++) {
      merged_im->setEdge(
          std::max(distance_im->getEdge(x, y), concave_im->getEdge(x, y)), x,
          y);
    }
  }


  //maybe check also if downsampled in receiver??? TODO
  if(downsample_datafuser_factor > 1){
    //resample
    if (normal_calculation == CROSSPRODUCT_DEPTH ||
      normal_calculation == CENTRALDIFFERENCES_DEPTH){
        //upsample edge image
        merged_im = upsample_edge_image(merged_im, downsample_datafuser_factor);
      }
      else{
        //downsample initial segmentation
        lmap = downsample_label_map(lmap, downsample_datafuser_factor);
      }
  }

  // debug prints
  if (edge_debug_images) {
    sensor_msgs::Image distance_msg = distance_im->createImage();
    sensor_msgs::Image concave_msg = concave_im->createImage();
    sensor_msgs::Image merged_msg = merged_im->createImage();

    distance_im_pub.publish(distance_msg);
    concave_im_pub.publish(concave_msg);
    merged_im_pub.publish(merged_msg);
  }

  // fuse edge images
  fuse_edge_image_label_map(merged_im, lmap, enforce_percentage);

  // refine
  if (to_refine_edge_image) {
    refine_edge_image(merged_im, lmap, no_blur_refine);
  }

  if (edge_debug_images) {
    sensor_msgs::Image labeled_msg = merged_im->getLabeledImage();

    labeled_im_pub.publish(labeled_msg);
  }
  

  // retransform vertices
  if (normal_calculation != VOXBLOX) {
    // transform into camera frame
    vertices = original_vertices;
  }

  // integrate new labelmap
  std::shared_ptr<labelMap> lm = merged_im->getLabelMap();
  //ROS_INFO("before integration");

  integrator->direct_integration(vertices, lm);


  //ROS_INFO("finished edge stuff");
}

template class SegEdgeFuser<LabelVoxel>;
template class SegEdgeFuser<MultiLabelVoxel>;

}  // namespace voxblox