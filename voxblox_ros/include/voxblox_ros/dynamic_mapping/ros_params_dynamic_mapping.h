#ifndef VOXBLOX_ROS_ROS_PARAMS_DYNAMIC_MAPPING_H_
#define VOXBLOX_ROS_ROS_PARAMS_DYNAMIC_MAPPING_H_

#include <ros/node_handle.h>

#include <voxblox_ros/dynamic_mapping/pcl_icp.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>

namespace voxblox {

inline TsdfMap::Config getDMTsdfMapConfigFromRosParam(
    const ros::NodeHandle& nh_private,
    const std::string prefix) {
  TsdfMap::Config tsdf_config;

  /**
   * Workaround for OS X on mac mini not having specializations for float
   * for some reason.
   */
  double voxel_size = tsdf_config.tsdf_voxel_size;
  int voxels_per_side = tsdf_config.tsdf_voxels_per_side;
  nh_private.param(prefix + "/map/voxel_size", voxel_size, voxel_size);
  nh_private.param(prefix + "/map/voxels_per_side", voxels_per_side, voxels_per_side);
  if (!isPowerOfTwo(voxels_per_side)) {
    ROS_ERROR("voxels_per_side must be a power of 2, setting to default value");
    voxels_per_side = tsdf_config.tsdf_voxels_per_side;
  }

  tsdf_config.tsdf_voxel_size = static_cast<FloatingPoint>(voxel_size);
  tsdf_config.tsdf_voxels_per_side = voxels_per_side;

  return tsdf_config;
}

inline PCL_ICP::Config getPCLICPConfigFromRosParam(const ros::NodeHandle& nh_private) {
  PCL_ICP::Config icp_config;
  nh_private.param("icp/icp_use_reciprocal_correspondences",
                   icp_config.use_reciprocal_correspondences,
                   icp_config.use_reciprocal_correspondences);
  nh_private.param("icp/icp_max_correspondence_distance",
                   icp_config.max_correspondence_distance,
                   icp_config.max_correspondence_distance);
  nh_private.param("icp/icp_use_symmetric_objective",
                   icp_config.use_symmetric_objective,
                   icp_config.use_symmetric_objective);
  nh_private.param("icp/icp_max_iterations",
                   icp_config.max_iterations, icp_config.max_iterations);
  nh_private.param("icp/icp_transformation_epsilon",
                   icp_config.transformation_epsilon,
                   icp_config.transformation_epsilon);
  nh_private.param("icp/icp_absolute_mse", icp_config.absolute_mse,
                   icp_config.absolute_mse);
  nh_private.param("icp/icp_euclidean_fitness_epsilon",
                   icp_config.euclidean_fitness_epsilon,
                   icp_config.euclidean_fitness_epsilon);

  return icp_config;
}

inline TsdfIntegratorBase::Config getDMTsdfIntegratorConfigFromRosParam(
    const ros::NodeHandle& nh_private,
    const std::string prefix) {
  TsdfIntegratorBase::Config integrator_config;

  integrator_config.voxel_carving_enabled = true;

  const TsdfMap::Config tsdf_config = getDMTsdfMapConfigFromRosParam(nh_private, prefix);
  integrator_config.default_truncation_distance =
      tsdf_config.tsdf_voxel_size * 4;

  double truncation_distance = integrator_config.default_truncation_distance;
  double max_weight = integrator_config.max_weight;
  nh_private.param(prefix + "/integrator/voxel_carving_enabled",
                   integrator_config.voxel_carving_enabled,
                   integrator_config.voxel_carving_enabled);
  nh_private.param(prefix + "/integrator/truncation_distance", truncation_distance,
                   truncation_distance);
  nh_private.param(prefix + "/integrator/max_ray_length_m", integrator_config.max_ray_length_m,
                   integrator_config.max_ray_length_m);
  nh_private.param(prefix + "/integrator/min_ray_length_m", integrator_config.min_ray_length_m,
                   integrator_config.min_ray_length_m);
  nh_private.param(prefix + "/integrator/max_weight", max_weight, max_weight);
  nh_private.param(prefix + "/integrator/use_const_weight", integrator_config.use_const_weight,
                   integrator_config.use_const_weight);
  nh_private.param(prefix + "/integrator/use_weight_dropoff", integrator_config.use_weight_dropoff,
                   integrator_config.use_weight_dropoff);
  nh_private.param(prefix + "/integrator/allow_clear", integrator_config.allow_clear,
                   integrator_config.allow_clear);
  nh_private.param(prefix + "/integrator/start_voxel_subsampling_factor",
                   integrator_config.start_voxel_subsampling_factor,
                   integrator_config.start_voxel_subsampling_factor);
  nh_private.param(prefix + "/integrator/max_consecutive_ray_collisions",
                   integrator_config.max_consecutive_ray_collisions,
                   integrator_config.max_consecutive_ray_collisions);
  nh_private.param(prefix + "/integrator/clear_checks_every_n_frames",
                   integrator_config.clear_checks_every_n_frames,
                   integrator_config.clear_checks_every_n_frames);
  nh_private.param(prefix + "/integrator/max_integration_time_s",
                   integrator_config.max_integration_time_s,
                   integrator_config.max_integration_time_s);
  nh_private.param(prefix + "/integrator/anti_grazing", integrator_config.enable_anti_grazing,
                   integrator_config.enable_anti_grazing);
  nh_private.param(prefix + "/integrator/use_sparsity_compensation_factor",
                   integrator_config.use_sparsity_compensation_factor,
                   integrator_config.use_sparsity_compensation_factor);
  nh_private.param(prefix + "/integrator/sparsity_compensation_factor",
                   integrator_config.sparsity_compensation_factor,
                   integrator_config.sparsity_compensation_factor);
  nh_private.param(prefix + "/integrator/integration_order_mode",
                   integrator_config.integration_order_mode,
                   integrator_config.integration_order_mode);

  integrator_config.default_truncation_distance =
      static_cast<float>(truncation_distance);
  integrator_config.max_weight = static_cast<float>(max_weight);

  return integrator_config;
}

inline MeshIntegratorConfig getDMMeshIntegratorConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  MeshIntegratorConfig mesh_integrator_config;

  nh_private.param("meshing/mesh_min_weight", mesh_integrator_config.min_weight,
                   mesh_integrator_config.min_weight);
  nh_private.param("meshing/mesh_use_color", mesh_integrator_config.use_color,
                   mesh_integrator_config.use_color);

  return mesh_integrator_config;
}

inline DynamicMapper::Config getDynamicMapperConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  DynamicMapper::Config dynamic_mapper_config;

  dynamic_mapper_config.background_map_config =
                       getDMTsdfMapConfigFromRosParam(nh_private, "background");
  dynamic_mapper_config.object_map_config =
                       getDMTsdfMapConfigFromRosParam(nh_private, "object");
  dynamic_mapper_config.icp_config = getPCLICPConfigFromRosParam(nh_private);
  dynamic_mapper_config.background_integrator_config =
                getDMTsdfIntegratorConfigFromRosParam(nh_private, "background");
  dynamic_mapper_config.object_integrator_config =
                    getDMTsdfIntegratorConfigFromRosParam(nh_private, "object");
  dynamic_mapper_config.mesh_config =
                              getDMMeshIntegratorConfigFromRosParam(nh_private);
  nh_private.param("background/integrator/method",
                   dynamic_mapper_config.background_integrator_method,
                   dynamic_mapper_config.background_integrator_method);
  nh_private.param("object/integrator/method",
                   dynamic_mapper_config.object_integrator_method,
                   dynamic_mapper_config.object_integrator_method);
  nh_private.param("object_map/dynamic",
                   dynamic_mapper_config.dynamic_object_voxel_size,
                   dynamic_mapper_config.dynamic_object_voxel_size);


  return dynamic_mapper_config;
}

}  // namespace voxblox

#endif  // VOXBLOX_ROS_ROS_PARAMS_DYNAMIC_MAPPING_H_
