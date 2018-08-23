#include "voxblox_ros/segmentation_server.h"

namespace voxblox {

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private)
  : SegmentationServer(nh, nh_private, getSegTsdfIntegratorConfigFromRosParam(nh_private)) {}

SegmentationServer::SegmentationServer(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private,
                                       const SegmentedTsdfIntegrator::Config& seg_integrator_config)
    : TsdfServer(nh, nh_private), segmenter_(nh_private) {
  cache_mesh_ = true;

  // Publishers for output.
  segmentation_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZI> >(
          "segmentation_pointcloud", 1, true);
  segmentation_mesh_pub_ =
      nh_private_.advertise<voxblox_msgs::Mesh>("segmentation_mesh", 1, true);

  seg_tsdf_map_.reset(new SegmentedTsdfMap(tsdf_map_->voxel_size(), tsdf_map_->getTsdfLayer().voxels_per_side()));

  seg_tsdf_integrator_.reset(new SegmentedTsdfIntegrator(
      seg_integrator_config, tsdf_map_->getTsdfLayerPtr(),
      seg_tsdf_map_->getTsdfLayerPtr()));

  segment_tool_.reset(new SegmentTools(tsdf_map_->getTsdfLayerPtr(), seg_tsdf_map_->getTsdfLayerPtr()));
}

void SegmentationServer::updateMesh() {
  TsdfServer::updateMesh();

  // Now recolor the mesh...
  timing::Timer publish_mesh_timer("segmented_mesh/publish");
  recolorVoxbloxMeshMsgBySegmentation(&cached_mesh_msg_);
  segmentation_mesh_pub_.publish(cached_mesh_msg_);
  publish_mesh_timer.Stop();
}

void SegmentationServer::publishPointclouds() {
  // Create a pointcloud with segmented points.
  pcl::PointCloud<pcl::PointXYZI> pointcloud;

  //createIntensityPointcloudFromIntensityLayer(*intensity_layer_, &pointcloud);

  pointcloud.header.frame_id = world_frame_;
  segmentation_pointcloud_pub_.publish(pointcloud);

  TsdfServer::publishPointclouds();
}

void SegmentationServer::integrateSegmentation(const sensor_msgs::PointCloud2::Ptr pointcloud_msg, const Transformation& T_G_C) {
  // Convert the PCL pointcloud into our awesome format.
  // TODO(helenol): improve...
  // Horrible hack fix to fix color parsing colors in PCL.
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
    }
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::fromROSMsg(*pointcloud_msg, *cloud_pcl);

  pcl::UniformSampling<pcl::PointXYZRGB> uniform_sampling;
  uniform_sampling.setInputCloud(cloud_pcl);
  uniform_sampling.setRadiusSearch(tsdf_map_->voxel_size());
  pcl::PointCloud<int> sub_cloud_indices;
  uniform_sampling.compute(sub_cloud_indices);
  std::cout << "Total points: " << cloud_pcl->points.size() << " Selected Points: " << sub_cloud_indices.points.size() << " "<< std::endl;

  Pointcloud points_C;
  points_C.reserve(sub_cloud_indices.size());
  for (size_t i = 0; i < sub_cloud_indices.size(); ++i) {

    const pcl::PointXYZRGB& p = cloud_pcl->points[sub_cloud_indices[i]];

    if (!std::isfinite(p.x) ||
        !std::isfinite(p.y) ||
        !std::isfinite(p.z)) {
      continue;
    }

    points_C.push_back(Point(p.x, p.y, p.z));
  }

  Labels segmentation;
  LabelIndexMap segment_map;
  segmenter_.segmentPointcloud(cloud_pcl, sub_cloud_indices, segmentation, segment_map);

  timing::Timer seg_integrate_timer("seg_integrate_segmentation");
  seg_tsdf_integrator_->integrateSegmentedPointCloud(T_G_C, points_C, segmentation, segment_map, segmenter_.getColorMap());
  seg_integrate_timer.Stop();

  seg_tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
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

void SegmentationServer::processPointCloudMessageAndInsert(const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
                                                           const Transformation& T_G_C,
                                                           const bool is_freespace_pointcloud) {
  TsdfServer::processPointCloudMessageAndInsert(pointcloud_msg, T_G_C, is_freespace_pointcloud);
  integrateSegmentation(pointcloud_msg, T_G_C);

  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;

  for (auto segment_id: seg_tsdf_integrator_->getUpdatedSegments()) {
    if (segment_id == 0)
      continue;
    pointcloud.clear();
    MeshLayer::ConstPtr mesh = segment_tool_->meshSegment(seg_tsdf_integrator_->getSegmentBlocksMap(), segment_id);
    fillPointcloudWithMesh(mesh, ColorMode::kLambertColor, &pointcloud);

    pointcloud.width = 1;
    pointcloud.height = pointcloud.points.size();

    if (!pointcloud.points.empty())
      pcl::io::savePCDFile("/home/marius/pcds/segment_" + std::to_string(segment_id) + ".pcd", pointcloud);
  }

}

}  // namespace voxblox
