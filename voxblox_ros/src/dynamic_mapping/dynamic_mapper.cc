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
                                   false, 0, -1, Color(200u, 200u, 200u));

}

void DynamicMapper::reset(){

  *background_cloud_last_ = *background_cloud_;
  background_cloud_->clear();

}

void DynamicMapper::setInputCloud(
        pcl::PointCloud<InputPointType>::Ptr input_cloud, const Transformation& T_G_C){

  background_object_.reset();

  std::set<int> occuring_objects;
  for (const auto& point : input_cloud->points) occuring_objects.insert(point.id);

  for (auto & object : objects_) {
    // TODO: Inefficient to set all objects on "not seen" and later reset ?
    object.increaseOccurenceCounter();
    object.setFrameOccurence(false);
    // Reset only objects which are already stored
    if (occuring_objects.find(object.getID()) != occuring_objects.end()) object.reset();
  }

  for (const auto& point : input_cloud->points) {


    pcl::PointXYZRGBNormal new_point;

    int id = point.id;
    int semantic_class = point.semantic_class; //TODO use correct data types

    new_point.x = point.x;
    new_point.y = point.y;
    new_point.z = point.z;
    new_point.normal_x = point.normal_x;
    new_point.normal_y = point.normal_y;
    new_point.normal_z = point.normal_z;
    new_point.r = point.r;
    new_point.g = point.g;
    new_point.b = point.b;

    if (id == 0){ // Background
      // Check if point wrongly assigned as background
      if (!checkPointInObject(point)) background_object_.cloud_current_->points.push_back(new_point);
    } else {

      bool object_is_known =  std::any_of(objects_.begin(), objects_.end(),
                              [&id](const DynamicObject& obj)
                                            { return obj.getID() == id; });

      if (!object_is_known) {
        DynamicObject new_object = DynamicObject(config_.object_map_config,
                                         config_.object_integrator_config,
                                         config_.mesh_config, icp_,
                                         config_.object_integrator_method,
                                         config_.dynamic_object_voxel_size, id,
                                         std::max(semantic_class, -1), // In case semantic class is -2
                                         randomColor());
        new_object.cloud_current_->points.push_back(new_point);
        objects_.push_back(new_object);

      } else {

        for (auto & object : objects_){
          if (object.getID() == id){
              object.cloud_current_->points.push_back(new_point);
              object.setSemanticClass(semantic_class); //TODO majority voting?
          }
        }
      }
    }
  }

  for (std::vector<DynamicObject>::iterator it=objects_.begin();
                              it!=objects_.end();)
  {

    if (it->getTimeSinceLastOccurence() >= 5) //TODO this should be parameterized
    {
      std::cout<<"DELETED OBJECT "<<it->getID()<<std::endl;
      it = objects_.erase(it);
    }
    else
    {
      it->updatePosition(T_G_C);
      ++it;
    }
  }

}

bool DynamicMapper::checkPointInObject(const InputPointType point){

  float padding = 0.1; //TODO parameter
  for (auto & object : objects_)
  {

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_mesh = object.getMeshCloud();
    pcl::PointXYZRGBNormal p_min, p_max;
    pcl::getMinMax3D(*object_mesh, p_min, p_max);

    if ((point.x < p_min.x - padding || point.x > p_max.x + padding) ||
        (point.y < p_min.y - padding || point.y > p_max.y + padding) ||
        (point.z < p_min.z - padding || point.z > p_max.z + padding)) continue;
    else return true;
  }
  return false;
}

void DynamicMapper::align(){

  for (auto & object : objects_)
  {
    object.align();
  }

}

void DynamicMapper::updateObjectStates(const Transformation& T_G_C){

  for (auto & object : objects_)
  {
    object.updateState(T_G_C);
  }

}

void DynamicMapper::clearDistant(const Transformation& T_G_C,
                                          float max_distance){

  background_object_.getTsdfMap()->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_distance);

  background_object_.getMeshLayer()->clearDistantMesh(T_G_C.getPosition(),
                                max_distance);

}

void DynamicMapper::backgroundICP(){

  // Eigen::Matrix4f G_T_S_O = Eigen::Matrix4f::Identity();
  //
  // if (icp_bg_cloud_last_.empty()) return;
  //
  //
  // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bg_ptr (
  //                         new pcl::PointCloud<pcl::PointXYZRGBNormal> (icp_bg_cloud_));
  // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr bg_ptr_last (
  //                    new pcl::PointCloud<pcl::PointXYZRGBNormal> (icp_bg_cloud_last_));
  // // Align the source: segment point cloud to the target: object model.
  // pcl::StopWatch test;
  // double fitness_score;
  // bool success = icp_->align(bg_ptr, bg_ptr_last,
  //                            Eigen::Matrix4f::Identity(), &G_T_S_O, &fitness_score);
  // std::cout<<test.getTimeSeconds()<<std::endl;

}

void DynamicMapper::generateMesh(){


    background_object_.generateMesh();

    for(auto & object : objects_){
         object.generateMesh();
    }
}

void DynamicMapper::integrate(const Transformation& T_G_C){


   background_object_.integrate(T_G_C, true);

   Transformation identity = Transformation();
   for(auto & object : objects_){
        object.integrate(identity, false);
        // std::cout<<"class: " << object.getSemanticClass()<<std::endl;
   }

}


} // namespace voxblox
