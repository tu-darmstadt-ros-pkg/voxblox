#ifndef VOXBLOX_ROS_DYNAMIC_MAPPER_H_
#define VOXBLOX_ROS_DYNAMIC_MAPPER_H_

#include <vector>
#include <list>

#include <pcl/common/common.h>

#include <voxblox_ros/dynamic_mapping/pcl_icp.h>


#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>

#include "voxblox_ros/dynamic_mapping/common.h"
#include <voxblox_ros/dynamic_mapping/dynamic_object.h>

namespace voxblox {

class DynamicMapper {
 public:
 struct Config {

   TsdfMap::Config background_map_config;
   TsdfMap::Config object_map_config;
   TsdfIntegratorBase::Config background_integrator_config;
   TsdfIntegratorBase::Config object_integrator_config;
   MeshIntegratorConfig mesh_config;
   PCL_ICP::Config icp_config;

   std::string background_integrator_method;
   std::string object_integrator_method;
   bool dynamic_object_voxel_size;

 };

  DynamicMapper(Config config);

  void setInputCloud(pcl::PointCloud<InputPointType>::Ptr input_cloud, const Transformation& T_G_C);

  void align();

  void backgroundICP();

  void clearDistant(const Transformation& T_G_C, float max_distance);

  void integrate(const Transformation& T_G_C);

  void generateMesh();

  void updateObjectStates(const Transformation& T_G_C);

  bool checkPointInObject(const InputPointType point);

  void reset();

  int getNumObjects() const { return objects_.size(); }

  // std::shared_ptr<TsdfMap> getBackgroundMap() const { return background_map_; }
  std::shared_ptr<MeshLayer> getBackgroundMeshLayer() const { return background_object_.getMeshLayer(); } //TODO refactor everything using getObjects
  std::shared_ptr<MeshLayer> getObjectMeshLayer(int i) const { return objects_[i].getMeshLayer(); }
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getObjectMeshCloud(int i) const { return objects_[i].getMeshCloud(); }
  int getObjectID(int i) const { return objects_[i].getID(); }

  Eigen::Matrix4f getTransformation(int i) const { return objects_[i].getTransformation();}

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getTransformedCloud(int i) const { return objects_[i].getTransformedCloud(); }

  std::vector<DynamicObject> getObjects() const { return objects_; }

private:

  void convertPointclouds(const Transformation& T_G_C);


  Config config_;
  // std::unique_ptr<TsdfIntegratorBase> background_integrator_;

  // std::map<int, DynamicObject> objects_;
  DynamicObject background_object_;
  std::vector<DynamicObject> objects_;

  std::shared_ptr<ColorMap> color_map_;

  // pcl::PointCloud<pcl::PointXYZRGBNormal> icp_bg_cloud_;
  // pcl::PointCloud<pcl::PointXYZRGBNormal> icp_bg_cloud_last_;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr background_cloud_;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr background_cloud_last_;


  std::shared_ptr<PCL_ICP> icp_;


  // Maps and integrators.
  std::shared_ptr<TsdfMap> background_map_;
  std::unique_ptr<TsdfIntegratorBase> background_integrator_;
  std::shared_ptr<MeshLayer> background_mesh_layer_;
  std::shared_ptr<MeshIntegrator<TsdfVoxel>> background_mesh_integrator_;

};

} // namespace voxblox

#endif  // VOXBLOX_ROS_DYNAMIC_MAPPER_H_
