
#include "voxblox_ros/SegDataFuser.h"

namespace voxblox {

template <class T>
SegDataFuser<T>::SegDataFuser(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf,
                              std::shared_ptr<Layer<T>> label)
    : tsdf_layer(tsdf), label_layer(label), nh_(nh) {
  nh.param<double>("max_distance", max_dist, 20.0);
  nh.param<bool>("aligned_rgbd", aligned_rgbd, 20.0);
  nh.param<bool>("direct_integration", direct_integration, false);  // TODO test

  nh.param<int>("bilateral_filter_size", bilateral_filter_size, 0);
  nh.param<double>("bilateral_filter_diff", bilateral_filter_diff, 0.0);

  fused_pub = nh.advertise<sensor_msgs::PointCloud2>("/debug/fused_cloud", 1);
}

template <class T>
void SegDataFuser<T>::fuse(Transformation T_G_C, std::shared_ptr<labelMap> lmap,
                           std::shared_ptr<rawDataPointer> data) {
  //ROS_INFO("in fuse");
  // claculate rotation T_w_body
  Transformation T_w_b = T_G_C * data->cloud_body;

  // here calculations should be done later

  // prepare integration (in case of already known vertices adapt this)
  //ROS_INFO("preparing integration");
  prepare_integration(lmap, T_w_b, data);
}
template <class T>
void SegDataFuser<T>::set_integrator(std::shared_ptr<SegTsdfIntegrator<T>> i) {
  integrator = i;
}

template <class T>
void SegDataFuser<T>::prepare_integration(
    std::shared_ptr<labelMap> lmap, Transformation T_w_b,
    std::shared_ptr<rawDataPointer> data) {
  // vertexMap vertices(0, 0);
  // Transformation T_w_lm = T_w_b * data->label_map_body.inverse();
  // if aligned and wished, directly integrate

  //ROS_INFO("checking for alignment");
  if (aligned_rgbd && direct_integration && data->aligned_depth) {
    Transformation T_w_d = T_w_b * data->depth_body.inverse();
    //ROS_INFO("getting vertices from depth image");
    std::shared_ptr<vertexMap> vertices = vertices_from_depth(
        data->depth_image_ptr, data->depth_info_ptr, T_w_d, max_dist, 0, bilateral_filter_size, bilateral_filter_diff);

    std::string world = "world";
    publish_debug_pointcloud(vertices, lmap, fused_pub, world, 25);

    integrator->direct_integration(vertices, lmap);
    //ROS_INFO("integration aligned");
  } else {
    Transformation i;
    i.setIdentity();
    Transformation T_w_c = T_w_b * data->cloud_body.inverse();
    //ROS_INFO("getting vertices by projection");
    std::shared_ptr<vertexMap> vertices = vertices_from_projection(
        data->segmentation_info, T_w_c, i, tsdf_layer, max_dist);

    std::string world = "world";
    publish_debug_pointcloud(vertices, lmap, fused_pub, world, 25);

    std::cout << "debug " << (vertices->get(5,5))->p[0] << std::endl;
    ///ROS_INFO("integrating with projection");
    integrator->direct_integration(vertices, lmap);
    /*for(int x=0; x < (data->segmentation_info)->width; x++){
        for(int y=0; y < (data->segmentation_info)->height; y++){
            std::cout << vertices.get(x, y)->p[0] << std::endl;
        }

    }*/
  }
}

template class SegDataFuser<LabelVoxel>;
template class SegDataFuser<MultiLabelVoxel>;
}  // namespace voxblox