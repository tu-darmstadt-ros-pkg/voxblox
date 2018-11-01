#include "voxblox_ros/segmentation_server.h"

namespace voxblox {

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private)
  : SegmentationServer(nh, nh_private, getSegTsdfIntegratorConfigFromRosParam(nh_private)) {}

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private,
                                       const SegmentedTsdfIntegrator::Config& seg_integrator_config)
    : TsdfServer(nh, nh_private), segmenter_(nh_private), color_image_sub_(nh_, "color_image", 1), color_info_sub_(nh_, "color_camera_info", 1), depth_image_sub_(nh_, "depth_image", 1),
      depth_info_sub_(nh_, "depth_camera_info", 1), msg_sync_(RgbdSyncPolicy(10), color_image_sub_, depth_image_sub_, color_info_sub_, depth_info_sub_) {
  cache_mesh_ = true;

  msg_sync_.setAgePenalty(0.5);
  msg_sync_.setMaxIntervalDuration(ros::Duration(0.1));
  msg_sync_.registerCallback(boost::bind(&SegmentationServer::rgbdCallback, this, _1, _2, _3, _4));

  // TODO: why is this needed in order to work?
  for (int i = 0; i < 9; i++)
    msg_sync_.setInterMessageLowerBound(i, ros::Duration(1.0));

  // Publishers for output.
  segment_pointclouds_pub_ = nh_private_.advertise<voxblox_msgs::PointCloudList>("segment_pointclouds", 1, true);
  segmentation_mesh_pub_ =
      nh_private_.advertise<voxblox_msgs::Mesh>("segmentation_mesh", 1, true);
  extracted_seg_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("extracted_segment", 1, true);

  seg_tsdf_map_.reset(new SegmentedTsdfMap(tsdf_map_->voxel_size(), tsdf_map_->getTsdfLayer().voxels_per_side()));

  seg_tsdf_integrator_.reset(new SegmentedTsdfIntegrator(
      seg_integrator_config, tsdf_map_->getTsdfLayerPtr(),
      seg_tsdf_map_->getTsdfLayerPtr()));

  segment_tool_.reset(new SegmentTools(tsdf_map_->getTsdfLayerPtr(), seg_tsdf_map_->getTsdfLayerPtr()));

  segment_cloud_service_ = nh_private_.advertiseService("extract_segment_cloud",
                                                        &SegmentationServer::extractSegmentCloud,
                                                        this);
  segment_cloud_from_ray_service_ = nh_private_.advertiseService("extract_segment_cloud_from_ray",
                                                                 &SegmentationServer::extractSegmentCloudFromRay,
                                                                 this);
  // we don't need this subscriber anymore
  pointcloud_sub_.shutdown();
}

void SegmentationServer::updateMesh() {
  TsdfServer::updateMesh();

  // Now recolor the mesh...
  timing::Timer publish_mesh_timer("segmented_mesh/publish");
  recolorVoxbloxMeshMsgBySegmentation(&cached_mesh_msg_);
  segmentation_mesh_pub_.publish(cached_mesh_msg_);
  publish_mesh_timer.Stop();
}

void SegmentationServer::integrateSegmentation(const sensor_msgs::PointCloud2ConstPtr& pointcloud, const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img,
                                               const sensor_msgs::CameraInfoConstPtr& color_cam_info, const sensor_msgs::CameraInfoConstPtr& depth_cam_info) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::fromROSMsg(*pointcloud, *cloud_pcl);

  pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
  uniform_sampling.setInputCloud(cloud_pcl);
  uniform_sampling.setRadiusSearch(tsdf_map_->voxel_size());
  pcl::PointCloud<int> sub_cloud_indices;
  uniform_sampling.compute(sub_cloud_indices);
  std::cout << "Total points: " << cloud_pcl->points.size() << " Selected Points: " << sub_cloud_indices.points.size() << " "<< std::endl;

  Pointcloud points_C;
  points_C.reserve(sub_cloud_indices.size());
  for (size_t i = 0; i < sub_cloud_indices.size(); ++i) {

    const pcl::PointXYZ& p = cloud_pcl->points[sub_cloud_indices[i]];

    if (!std::isfinite(p.x) ||
        !std::isfinite(p.y) ||
        !std::isfinite(p.z)) {
      continue;
    }

    points_C.push_back(Point(p.x, p.y, p.z));
  }

  LabelIndexMap segment_map;
  segmenter_.segmentRgbdImage(color_img, color_cam_info, depth_img, depth_cam_info, cloud_pcl, sub_cloud_indices, segment_map);

  timing::Timer seg_integrate_timer("seg_integrate_segmentation");
  seg_tsdf_integrator_->integrateSegmentedPointCloud(T_G_C_current_, points_C, segment_map, segmenter_.getColorMap());
  seg_integrate_timer.Stop();

  seg_tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C_current_.getPosition(), max_block_distance_from_body_);
}

void SegmentationServer::recolorVoxbloxMeshMsgBySegmentation(voxblox_msgs::Mesh* mesh_msg) {

  const Layer<SegmentedVoxel>& segment_layer = seg_tsdf_map_->getTsdfLayer();

  // Go over all the blocks in the mesh.
  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg->mesh_blocks) {
    // Go over all the triangles in the mesh.

    for (voxblox_msgs::Triangle& triangle : mesh_block.triangles) {
      // Look up triangles in the thermal layer.
      for (size_t local_vert_idx = 0u; local_vert_idx < 3; ++local_vert_idx) {
        const SegmentedVoxel* voxel = segment_layer.getVoxelPtrByCoordinates(
            Point(triangle.x[local_vert_idx], triangle.y[local_vert_idx],
                  triangle.z[local_vert_idx]));
        if (voxel != nullptr) {
          Color segment_color = segmenter_.getSegmentColor(voxel->segment_id);
          triangle.r[local_vert_idx] = segment_color.r;
          triangle.g[local_vert_idx] = segment_color.g;
          triangle.b[local_vert_idx] = segment_color.b;
          triangle.a[local_vert_idx] = 255;
        } else {
          triangle.r[local_vert_idx] = 0;
          triangle.g[local_vert_idx] = 0;
          triangle.b[local_vert_idx] = 0;
          triangle.a[local_vert_idx] = 0;
        }
      }
    }
  }
}

void SegmentationServer::newPoseCallback(const Transformation& T_G_C) {
  // remember the latest transform for integrating the segmentation
  T_G_C_current_ = T_G_C;
}

inline void SegmentationServer::fillPointcloudWithMesh(const MeshLayer::ConstPtr& mesh_layer, pcl::PointCloud<pcl::PointNormal>& pointcloud) {
  pointcloud.clear();

  BlockIndexList mesh_indices;
  mesh_layer->getAllAllocatedMeshes(&mesh_indices);

  for (const BlockIndex& block_index : mesh_indices) {
    Mesh::ConstPtr mesh = mesh_layer->getMeshPtrByIndex(block_index);

    if (!mesh->hasVertices()) {
      continue;
    }

    CHECK(mesh->hasNormals());

    for (size_t i = 0u; i < mesh->vertices.size(); i++) {
      pcl::PointNormal point;
      point.x = mesh->vertices[i].x();
      point.y = mesh->vertices[i].y();
      point.z = mesh->vertices[i].z();

      point.normal_x = mesh->normals[i].x();
      point.normal_y = mesh->normals[i].y();
      point.normal_z = mesh->normals[i].z();
      point.curvature = 1.0f;

      pointcloud.push_back(point);
    }
  }
}

void SegmentationServer::publishPointclouds() {
  voxblox_msgs::PointCloudList pointcloud_list;

  for (auto segment_id: seg_tsdf_integrator_->getUpdatedSegments()) {
    if (segment_id == 0)
      continue;

    sensor_msgs::PointCloud2 cloud;
    extractSegmentCloud(segment_id, cloud);
    pointcloud_list.clouds.push_back(cloud);
  }

  segment_pointclouds_pub_.publish(pointcloud_list);
}

void SegmentationServer::rgbdCallback(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img,
                                      const sensor_msgs::CameraInfoConstPtr& color_cam_info, const sensor_msgs::CameraInfoConstPtr& depth_cam_info) {

  sensor_msgs::PointCloud2::Ptr cloud = boost::make_shared<sensor_msgs::PointCloud2>();
  cloud->header = depth_img->header;
  cloud->height = depth_img->height;
  cloud->width  = depth_img->width;
  cloud->is_dense = false;
  cloud->is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud);
  pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  if (depth_img->encoding == "32FC1") {
    convertToCloud<float>(depth_img, color_img, depth_cam_info, cloud);
  }
  else if (depth_img->encoding == "16UC1") {
    convertToCloud<uint16_t>(depth_img, color_img, depth_cam_info, cloud);
  }
  else {
    ROS_ERROR("unsupported depth image encoding: %s", depth_img->encoding.c_str());
  }

  insertPointcloud(cloud);
  integrateSegmentation(cloud, color_img, depth_img, color_cam_info, depth_cam_info);
}

template <typename T>
void SegmentationServer::convertToCloud(const sensor_msgs::ImageConstPtr& depth_msg,
                                        const sensor_msgs::ImageConstPtr& rgb_msg,
                                        const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
                                        const sensor_msgs::PointCloud2::Ptr& cloud) {
  // Use correct principal point from calibration
  float center_x = static_cast<float>(depth_cam_info->K[2]);
  float center_y = static_cast<float>(depth_cam_info->K[5]);

  float unit_scaling = 0.001f;

  if (std::is_same<T, float>::value)
    unit_scaling = 1.0f;

  float f_x = static_cast<float>(depth_cam_info->K[0]);
  float f_y = static_cast<float>(depth_cam_info->K[4]);

  const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  const uint8_t* rgb = &rgb_msg->data[0];
  int rgb_skip = rgb_msg->step - rgb_msg->width * 3;

  sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud, "b");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud, "a");

  for (int v = 0; v < int(cloud->height); ++v, depth_row += row_step, rgb += rgb_skip)
  {
    for (int u = 0; u < int(cloud->width); ++u, rgb += 3, ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b)
    {
      T depth = depth_row[u];
      float scaled_depth = unit_scaling * float(depth);

      // Check for invalid measurements
      if (depth == 0) {
        *iter_x = *iter_y = *iter_z = NAN;
      } else {
        // Fill in XYZ
        *iter_x = (u - center_x) * scaled_depth / f_x;
        *iter_y = (v - center_y) * scaled_depth / f_y;
        *iter_z = scaled_depth;
      }

      // Fill in color
      *iter_a = 255;
      *iter_r = rgb[0];
      *iter_g = rgb[1];
      *iter_b = rgb[2];
    }
  }
}

bool SegmentationServer::extractSegmentCloud(voxblox_msgs::ExtractSegmentRequest& req, voxblox_msgs::ExtractSegmentResponse& res) {
  extractSegmentCloud(req.segment_id, res.segment_cloud);
  extracted_seg_pub_.publish(res.segment_cloud);

  return true;
}

bool SegmentationServer::extractSegmentCloudFromRay(voxblox_msgs::ExtractSegmentFromRayRequest& req, voxblox_msgs::ExtractSegmentFromRayResponse& res) {
  Point origin(req.ray_origin.x, req.ray_origin.y, req.ray_origin.z);
  Point direction(req.ray_direction.x, req.ray_direction.y, req.ray_direction.z);
  Label segment_id = segment_tool_->getSegmentIdFromRay(origin, direction);

  if (segment_id == 0)
    return false;

  extractSegmentCloud(segment_id, res.segment_cloud);
  extracted_seg_pub_.publish(res.segment_cloud);

  ROS_INFO_STREAM("cloud size: " << res.segment_cloud.height * res.segment_cloud.width);

  return true;
}

bool SegmentationServer::extractSegmentCloud(Label segment_id, sensor_msgs::PointCloud2& cloud) {
  pcl::PointCloud<pcl::PointNormal> pointcloud_pcl;

  MeshLayer::ConstPtr mesh = segment_tool_->meshSegment(seg_tsdf_integrator_->getSegmentBlocksMap(), segment_id);
  fillPointcloudWithMesh(mesh, pointcloud_pcl);

  pcl::toROSMsg(pointcloud_pcl, cloud);
  cloud.header.frame_id = world_frame_;

  // TODO: time stamp of the pointcloud input msg
  cloud.header.stamp = ros::Time::now();

  return true;
}
}  // namespace voxblox
