#ifndef VOXBLOX_ROS_DYNAMIC_MAPPER_H_
#define VOXBLOX_ROS_DYNAMIC_MAPPER_H_

#include <vector>
#include <list>

#include <pcl/common/common.h>

#include <voxblox/core/common.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/core/voxel.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>

#include <voxblox_ros/dynamic_mapping/common.h>
#include <voxblox_ros/dynamic_mapping/dynamic_object.h>
#include <voxblox_ros/dynamic_mapping/pcl_icp.h>

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
   float dynamic_object_min_voxel_size;
   float dynamic_object_max_voxel_size;
   int max_steps_since_last_occurence;
   int max_consecutive_alignment_failures;
   int max_num_resets_before_inactive;


 };

  DynamicMapper(Config config);

  void distributeInputCloud(pcl::PointCloud<InputPointType>::Ptr input_cloud);

  void align(const Transformation& T_G_C);

  void backgroundICP();

  void clearDistant(const Transformation& T_G_C, float max_distance);

  void integrate(const Transformation& T_G_C);

  void generateMesh();

  void updateObjectStates(const Transformation& T_G_C,
                          const sensor_msgs::PointCloud2::Ptr& pointcloud_msg);

  bool checkPointInObject(const InputPointType point);

  void reset();

  int getNumObjects() const { return objects_.size(); }

  std::shared_ptr<MeshLayer> getBackgroundMeshLayer() const {
                  return background_object_.getMeshLayer(); }
  std::shared_ptr<MeshLayer> getObjectMeshLayer(int i) const {
                          return objects_[i].getMeshLayer(); }
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getObjectMeshCloud(int i)
                          const { return objects_[i].getMeshCloud(); }
  int getObjectID(int i) const { return objects_[i].getID(); }

  Eigen::Matrix4f getTransformation(int i) const { return objects_[i].getTransformation();}
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getTransformedCloud(int i) const { return objects_[i].getTransformedCloud(); }
  std::vector<DynamicObject> getObjects() const { return objects_; }
  void setNonRigidClasses(const std::vector<int> non_rigid_classes) {non_rigid_classes_ = non_rigid_classes; }

private:

  void convertPointclouds(const Transformation& T_G_C);


  Config config_;

  DynamicObject background_object_;
  std::vector<DynamicObject> objects_;
  std::shared_ptr<ColorMap> color_map_;
  std::vector<int> non_rigid_classes_;
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
