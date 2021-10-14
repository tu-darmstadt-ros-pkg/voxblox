#include "voxblox_ros/dynamic_mapping/dynamic_mapper.h"
#include "voxblox_ros/conversions.h"
#include <iostream>
#include <pcl/common/time.h>

namespace voxblox {

DynamicMapper::DynamicMapper(Config config)
    : config_(config),
      color_map_(new RainbowColorMap())
{

  icp_.reset(new PCL_ICP(config_.icp_config));

  background_object_ = DynamicObject(config_.background_map_config,
                                   config_.background_integrator_config,
                                   config_.mesh_config, icp_,
                                   config_.background_integrator_method,
                                   0, -1, Color(100u, 100u, 100u), 1e3);
  background_object_.reset();

}

void DynamicMapper::distributeInputCloud(
        pcl::PointCloud<InputPointType>::Ptr input_cloud){

  background_object_.initNextStep();

  std::set<int> occuring_objects;
  for (const auto& point : input_cloud->points) occuring_objects.insert(point.id);

  // if (occuring_objects.find(7) == occuring_objects.end()) return;
  for (auto & object : objects_) {

    object.increaseOccurenceCounter();
    object.setFrameOccurence(false);
    // Reset only objects which are already stored
    if (occuring_objects.find(object.getID()) != occuring_objects.end()) object.initNextStep();

    std::vector<int>::iterator it = std::find(non_rigid_classes_.begin(),
                      non_rigid_classes_.end(), object.getSemanticClass() + 1);
    if (it != non_rigid_classes_.end()){
      object.reset();
    }

  }

  for (const auto& point : input_cloud->points) {

    pcl::PointXYZRGBNormal new_point;

    int id = point.id;
    int semantic_class = point.semantic_class;

    new_point.x = point.x;
    new_point.y = point.y;
    new_point.z = point.z;
    new_point.normal_x = point.normal_x;
    new_point.normal_y = point.normal_y;
    new_point.normal_z = point.normal_z;
    new_point.curvature = point.curvature;
    new_point.r = point.r;
    new_point.g = point.g;
    new_point.b = point.b;

    if (id == 0){ // Point belongs to background
      // Check if point wrongly assigned as background
      if (!checkPointInObject(point)) {
        background_object_.cloud_current_->points.push_back(new_point);
      }
    } else { // Point belongs to object

      bool object_is_known =  std::any_of(objects_.begin(), objects_.end(),
                              [&id](const DynamicObject& obj)
                                            { return obj.getID() == id; });

      if (!object_is_known) {
          // Color c;
          // if (id == 7){
          //   c = Color(190, 45, 6);
          // } else{
          //   c = randomColor();
          // }
          Color c = randomColor();

          DynamicObject new_object = DynamicObject(config_.object_map_config,
                                       config_.object_integrator_config,
                                       config_.mesh_config, icp_,
                                       config_.object_integrator_method, id,
                                       std::max(semantic_class, -1),
                                       c,
                                       config_.max_num_resets_before_inactive);
          new_object.cloud_current_->points.push_back(new_point);
          objects_.push_back(new_object);

      } else {

        for (auto & object : objects_){
          if (object.getID() == id){
              object.cloud_current_->points.push_back(new_point);
              object.setSemanticClass(semantic_class);
          }
        }
      }
    }
  }

  for (auto & object : objects_)
  {
    object.updateMeshMinMax();
    object.setVoxelSize(config_.dynamic_object_min_voxel_size,
                              config_.dynamic_object_max_voxel_size);
    if (object.getTimeSinceLastOccurence() >=
                                    config_.max_steps_since_last_occurence) {
      object.setDelete();
      object.reset();
    }
  }

  objects_.erase(
    std::remove_if(
        objects_.begin(),
        objects_.end(),
        [](DynamicObject const & o) { return o.getDelete(); }
    ),
    objects_.end()
);

}

bool DynamicMapper::checkPointInObject(const InputPointType point){

  for (auto & object : objects_)
  {
    if (object.isPointInside(point)) return true;
  }
  return false;
}

void DynamicMapper::align(const Transformation& T_G_C){

  for (auto & object : objects_)
  {
    timing::Timer object_timer("objectalign");
    int frames_not_aligned = object.align(T_G_C);
    object_timer.Stop();

    if (frames_not_aligned > config_.max_consecutive_alignment_failures) {
      object.reset();
    }
  }
}

void DynamicMapper::updateObjectStates(const Transformation& T_G_C,
                          const sensor_msgs::PointCloud2::Ptr& pointcloud_msg){

  for (auto & object : objects_)
  {
    object.updateState(T_G_C, pointcloud_msg);
  }
}

void DynamicMapper::clearDistant(const Transformation& T_G_C,
                                          float max_distance){

  background_object_.getTsdfMap()->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_distance);
  background_object_.getMeshLayer()->clearDistantMesh(T_G_C.getPosition(),
                                max_distance);
}

void DynamicMapper::generateMesh(){

    background_object_.generateMesh();
    for(auto & object : objects_){
         object.generateMesh();
    }
}

void DynamicMapper::integrate(const Transformation& T_G_C){

   timing::Timer integrate_timer("background_integrate");
   background_object_.integrate(T_G_C, true);
   integrate_timer.Stop();

   Transformation identity = Transformation();
   for(auto & object : objects_){
        timing::Timer objintegrate_timer("object_integrate");
        object.integrate(T_G_C, false);
        objintegrate_timer.Stop();
   }
}

} // namespace voxblox
