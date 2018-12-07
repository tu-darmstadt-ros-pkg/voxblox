#include "voxblox_ros/segmentation_server.h"

namespace voxblox {

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private)
  : SegmentationServer(nh, nh_private, getSegTsdfIntegratorConfigFromRosParam(nh_private)) {}

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private,
                                       const SegmentedTsdfIntegrator::Config& seg_integrator_config)
    : TsdfServer(nh, nh_private), segmenter_(nh_private, tsdf_map_->voxel_size()), color_image_sub_(nh_, "color_image", 1), color_info_sub_(nh_, "color_camera_info", 1), depth_image_sub_(nh_, "depth_image", 1),
      depth_info_sub_(nh_, "depth_camera_info", 1), msg_sync_(RgbdSyncPolicy(10), color_image_sub_, depth_image_sub_, color_info_sub_, depth_info_sub_) {
  cache_mesh_ = true;

  msg_sync_.setAgePenalty(0.5);
  msg_sync_.setMaxIntervalDuration(ros::Duration(0.1));
  msg_sync_.registerCallback(boost::bind(&SegmentationServer::rgbdCallback, this, _1, _2, _3, _4));

  // TODO: why is this needed in order to work?
  for (int i = 0; i < 9; i++)
    msg_sync_.setInterMessageLowerBound(i, ros::Duration(1.0));

  // Publishers for output.
  segment_pointclouds_pub_ = nh_private_.advertise<voxblox_msgs::PointCloudList>("segment_pointclouds", 1, false);
  segmentation_mesh_pub_ =
      nh_private_.advertise<voxblox_msgs::Mesh>("segmentation_mesh", 1, true);
  extracted_seg_cloud_pub_ = nh_private_.advertise<sensor_msgs::PointCloud2>("extracted_segment_cloud", 1, false);
  extracted_seg_mesh_pub_ = nh_private_.advertise<shape_msgs::Mesh>("extracted_segment_mesh", 1, false);

  seg_tsdf_map_.reset(new SegmentedTsdfMap(tsdf_map_->voxel_size(), tsdf_map_->getTsdfLayer().voxels_per_side()));

  seg_tsdf_integrator_.reset(new SegmentedTsdfIntegrator(
      seg_integrator_config, tsdf_map_->getTsdfLayerPtr(),
      seg_tsdf_map_->getTsdfLayerPtr()));

  segment_tool_.reset(new SegmentTools(tsdf_map_->getTsdfLayerPtr(), seg_tsdf_map_->getTsdfLayerPtr()));

  segment_mesh_service_ = nh_private_.advertiseService("extract_segment_mesh",
                                                        &SegmentationServer::extractSegmentMesh,
                                                        this);
  segment_id_from_ray_service_ = nh_private_.advertiseService("extract_segment_id_from_ray",
                                                              &SegmentationServer::extractSegmentIDFromRay,
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

void SegmentationServer::integrateSegmentation(const sensor_msgs::PointCloud2ConstPtr& pointcloud, const cv::Mat& color_img, const cv::Mat& depth_img,
                                               const sensor_msgs::CameraInfoConstPtr& color_cam_info, const sensor_msgs::CameraInfoConstPtr& depth_cam_info) {

  ROS_ERROR_STREAM("Seg TSDF server timestamp: " << pointcloud->header.stamp);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::fromROSMsg(*pointcloud, *cloud_pcl);

  Pointcloud points_C;
  LabelIndexMap segment_map;
  segmenter_.segmentRgbdImage(color_img, color_cam_info, depth_img, depth_cam_info, cloud_pcl, points_C, segment_map);

  std::cout << "points size: " << points_C.size() << std::endl;

  timing::Timer seg_integrate_timer("seg_integrate_segmentation");

  seg_tsdf_integrator_->integrateSegmentedPointCloud(T_G_C_current_, points_C, segment_map, segmenter_.getColorMap());
  seg_integrate_timer.Stop();

  timing::Timer seg_remove_distant_blocks_timer("seg_remove_distant_blocks");
  seg_tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C_current_.getPosition(), max_block_distance_from_body_);
  seg_remove_distant_blocks_timer.Stop();
}

void SegmentationServer::recolorVoxbloxMeshMsgBySegmentation(voxblox_msgs::Mesh* mesh_msg) {

  const Layer<SegmentedVoxel>& segment_layer = seg_tsdf_map_->getTsdfLayer();

  // Go over all the blocks in the mesh.
  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg->mesh_blocks) {

    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {

      constexpr float point_conv_factor = 2.0f / std::numeric_limits<uint16_t>::max();
      const float mesh_x =
          (static_cast<float>(mesh_block.x[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[0])) * mesh_msg->block_edge_length;
      const float mesh_y =
          (static_cast<float>(mesh_block.y[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[1])) * mesh_msg->block_edge_length;
      const float mesh_z =
          (static_cast<float>(mesh_block.z[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[2])) * mesh_msg->block_edge_length;

      const SegmentedVoxel* voxel = segment_layer.getVoxelPtrByCoordinates(
          Point(mesh_x, mesh_y, mesh_z));

      if (voxel != nullptr) {
        Color segment_color = segmenter_.getSegmentColor(voxel->segment_id);
        mesh_block.r[vert_idx] = segment_color.r;
        mesh_block.g[vert_idx] = segment_color.g;
        mesh_block.b[vert_idx] = segment_color.b;
      } else {
        mesh_block.r[vert_idx] = 0;
        mesh_block.g[vert_idx] = 0;
        mesh_block.b[vert_idx] = 0;
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

inline void SegmentationServer::fillMeshMsgWithMesh(const MeshLayer::ConstPtr& mesh_layer, shape_msgs::Mesh& mesh_msg) {
  mesh_msg.triangles.clear();
  mesh_msg.vertices.clear();
  shape_msgs::MeshTriangle triangle;
  geometry_msgs::Point vertex;

  Mesh connected_mesh;
  mesh_layer->getConnectedMesh(&connected_mesh);

  if (!connected_mesh.hasVertices() || !connected_mesh.hasTriangles()) {
    return;
  }

  for (size_t i = 0u; i < connected_mesh.vertices.size(); i+=3) {
    for (size_t j = 0u; j < 3; j++) {
      vertex.x = static_cast<double>(connected_mesh.vertices[i+j].x());
      vertex.y = static_cast<double>(connected_mesh.vertices[i+j].y());
      vertex.z = static_cast<double>(connected_mesh.vertices[i+j].z());

      mesh_msg.vertices.push_back(vertex);

      triangle.vertex_indices[j] = static_cast<uint>(connected_mesh.indices[i+j]);
    }
    mesh_msg.triangles.push_back(triangle);
  }
}

void SegmentationServer::publishPointclouds() {
  voxblox_msgs::PointCloudList pointcloud_list;

  for (auto segment_id: seg_tsdf_integrator_->getUpdatedSegments()) {
    if (segment_id == 0)
      continue;

    sensor_msgs::PointCloud2 cloud;
    shape_msgs::Mesh mesh;
    extractSegmentMesh(segment_id, cloud, mesh);
    pointcloud_list.clouds.push_back(cloud);
  }

  segment_pointclouds_pub_.publish(pointcloud_list);
}

void SegmentationServer::rgbdCallback(const sensor_msgs::ImageConstPtr& color_img_msg, const sensor_msgs::ImageConstPtr& depth_img_msg,
                                      const sensor_msgs::CameraInfoConstPtr& color_cam_info_msg, const sensor_msgs::CameraInfoConstPtr& depth_cam_info_msg) {

  int downsampling_factor = 2;
  cv::Mat depth_img_downsampled = downSampleNonZeroMedian(depth_img_msg, downsampling_factor);
  cv::Mat color_img_downsampled = downSampleColorImg(color_img_msg, downsampling_factor);

  sensor_msgs::CameraInfoPtr color_cam_info = downsampleCameraInfo(color_cam_info_msg, downsampling_factor);
  sensor_msgs::CameraInfoPtr depth_cam_info = downsampleCameraInfo(depth_cam_info_msg, downsampling_factor);

  sensor_msgs::PointCloud2::Ptr cloud_msg = boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::PointCloud<pcl::PointXYZRGB> cloud(depth_img_downsampled.cols, depth_img_downsampled.rows);

  if (depth_img_msg->encoding == "32FC1") {
    convertToCloud<float>(depth_img_downsampled, color_img_downsampled, depth_cam_info, cloud);
  }
  else if (depth_img_msg->encoding == "16UC1") {
    convertToCloud<uint16_t>(depth_img_downsampled, color_img_downsampled, depth_cam_info, cloud);
  }
  else {
    ROS_ERROR("unsupported depth image encoding: %s", depth_img_msg->encoding.c_str());
  }

  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header = depth_img_msg->header;

  insertPointcloud(cloud_msg);
  integrateSegmentation(cloud_msg, color_img_downsampled, depth_img_downsampled, color_cam_info, depth_cam_info);
}

template <typename T>
void SegmentationServer::convertToCloud(const cv::Mat& depth_img,
                                        const cv::Mat& rgb_img,
                                        const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
                                        pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  // Use correct principal point from calibration
  float center_x = static_cast<float>(depth_cam_info->K[2]);
  float center_y = static_cast<float>(depth_cam_info->K[5]);

  float unit_scaling = 0.001f;

  if (std::is_same<T, float>::value)
    unit_scaling = 1.0f;

  float f_x = static_cast<float>(depth_cam_info->K[0]);
  float f_y = static_cast<float>(depth_cam_info->K[4]);

  for (int row = 0; row < depth_img.rows; row++) {
    for (int col = 0; col < depth_img.cols; col++) {

      const cv::Vec3b& rgb = rgb_img.at<cv::Vec3b>(row, col);
      T depth = depth_img.at<T>(row, col);
      float scaled_depth = unit_scaling * float(depth);

      pcl::PointXYZRGB& p = cloud.at(col, row);

      // Check for invalid measurements
      if (!isDepthValid(depth)) {
        p.x = p.y = p.z = NAN;
      } else {
        // Fill in XYZ
        p.x = (col - center_x) * scaled_depth / f_x;
        p.y = (row - center_y) * scaled_depth / f_y;
        p.z = scaled_depth;
      }

      // Fill in color
      p.r = rgb[0];
      p.g = rgb[1];
      p.b = rgb[2];
    }
  }
}

bool SegmentationServer::extractSegmentMesh(voxblox_msgs::ExtractSegmentMeshRequest& req, voxblox_msgs::ExtractSegmentMeshResponse& res) {
  extractSegmentMesh(req.segment_id, res.segment_cloud, res.segment_mesh);
  extracted_seg_cloud_pub_.publish(res.segment_cloud);
  extracted_seg_mesh_pub_.publish(res.segment_mesh);

  ROS_INFO_STREAM("cloud size: " << res.segment_cloud.height * res.segment_cloud.width);
  ROS_INFO_STREAM("Created a mesh consisting of " << res.segment_mesh.triangles.size() << " triangles and " << res.segment_mesh.vertices.size() << " vertices");

  return true;
}

bool SegmentationServer::extractSegmentIDFromRay(voxblox_msgs::ExtractSegmentIdFromRayRequest& req, voxblox_msgs::ExtractSegmentIdFromRayResponse& res) {
  Point origin(static_cast<float>(req.ray_origin.x), static_cast<float>(req.ray_origin.y), static_cast<float>(req.ray_origin.z));
  Point direction(static_cast<float>(req.ray_direction.x), static_cast<float>(req.ray_direction.y), static_cast<float>(req.ray_direction.z));
  Label segment_id = segment_tool_->getSegmentIdFromRay(origin, direction);

  ROS_INFO_STREAM("ray hit segment " << segment_id);

  res.segment_id = segment_id;

  if (segment_id == 0)
    return false;

  return true;
}

bool SegmentationServer::extractSegmentMesh(Label segment_id, sensor_msgs::PointCloud2& cloud_msg, shape_msgs::Mesh& mesh_msg) {
  pcl::PointCloud<pcl::PointNormal> pointcloud_pcl;

  MeshLayer::ConstPtr mesh = segment_tool_->meshSegment(seg_tsdf_integrator_->getSegmentBlocksMap(), segment_id);
  fillPointcloudWithMesh(mesh, pointcloud_pcl);
  fillMeshMsgWithMesh(mesh, mesh_msg);

  pcl::toROSMsg(pointcloud_pcl, cloud_msg);
  cloud_msg.header.frame_id = world_frame_;

  // TODO: time stamp of the pointcloud input msg
  cloud_msg.header.stamp = ros::Time::now();

  return true;
}

template<>
bool SegmentationServer::isDepthValid(float depth) {
  return depth > 0.0001f;
}

template<>
bool SegmentationServer::isDepthValid(uint16_t depth) {
  return depth != 0;
}
}  // namespace voxblox
