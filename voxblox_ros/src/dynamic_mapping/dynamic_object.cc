#include "voxblox_ros/dynamic_mapping/dynamic_object.h"

#include <iostream>
#include <pcl/common/time.h>

#include "voxblox_ros/dynamic_mapping/obb_utils.h"

namespace voxblox {

DynamicObject::DynamicObject(const TsdfMap::Config& tsdf_config,
                            TsdfIntegratorBase::Config& integrator_config,
                            const MeshIntegratorConfig& mesh_config,
                            const std::shared_ptr<PCL_ICP>& icp,
                            std::string method,
                            const int id, const int semantic_class,
                            Color mesh_color,
                            const int max_num_resets_before_inactive)
  : cloud_current_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    T_O_accumulated_(Eigen::Matrix4f::Identity()),
    T_O_last_(Eigen::Matrix4f::Identity()),
    id_(id),
    semantic_class_(semantic_class),
    semantic_class_confidence_(0),
    semantic_class_set_(true),
    occured_in_current_frame_(true),
    time_since_last_occurence_(0),
    frames_not_aligned_(0),
    mesh_color_(mesh_color),
    icp_(icp),
    cloud_last_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_current_transformed_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_last_transformed_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_transformed_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_accumulated_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    mesh_cloud_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    color_map_(new RainbowColorMap()),
    tsdf_config_(tsdf_config),
    integrator_config_(integrator_config),
    mesh_config_(mesh_config),
    integrator_method_(method),
    reset_counter_(-1),
    active_(true),
    delete_(false),
    static_(false),
    voxel_size_set_(false),
    xy_in_object_padding_(0.05),
    max_num_resets_before_inactive_(max_num_resets_before_inactive)
{}

void DynamicObject::setVoxelSize(float min_size, float max_size){

  if(voxel_size_set_) return;
  voxel_size_set_ = true;

  int num_bins = 6;

  std::vector<float> curvature_values;
  for (const auto& point : cloud_current_->points)
                            curvature_values.push_back(point.curvature);
  float max_curvature = *std::max_element(
                              curvature_values.begin(), curvature_values.end());

  if (max_curvature <= 1e-6){
    tsdf_config_.tsdf_voxel_size = max_size;
    integrator_config_.default_truncation_distance = 3 * max_size;
    reset();
    return;
  }
  std::vector<float> hist(num_bins, 0);
  float bin_size = max_curvature / num_bins;
  float num_pts_inv = 1.0 / curvature_values.size();
  float bin_size_inv = 1.0 / bin_size;

  for (const auto& value : curvature_values)
  {
    // Last bin includes the right bound
    int bin = std::min(static_cast<int>(value * bin_size_inv), num_bins - 1);
    hist[bin] += num_pts_inv;
  }


  float max_height = 0.0;
  int max_idx = 0;
  for (int idx=0; idx < hist.size(); idx++)
  {
    float value = hist[idx];
    if(value > max_height){
      max_height = value;
      max_idx = idx;
    }
  }

  int diff = num_bins - 1;
  float prop = max_idx*1.0/diff;
  float size = max_size - prop * (max_size - min_size);

  tsdf_config_.tsdf_voxel_size = size;
  integrator_config_.default_truncation_distance = 3 * size;
  reset();

}

void DynamicObject::reset(){

  if(!active_) return;

  reset_counter_++;

  if (reset_counter_ >= max_num_resets_before_inactive_){
    active_ = false;
  }

  frames_not_aligned_ = 0;

  T_O_accumulated_ = Eigen::Matrix4f::Identity();
  T_O_last_ = Eigen::Matrix4f::Identity();
  cloud_transformed_->clear();

  map_.reset(new TsdfMap(tsdf_config_));

  if (integrator_method_.compare("simple") == 0) {
    integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config_, map_->getTsdfLayerPtr()));
  } else if (integrator_method_.compare("merged") == 0) {
    integrator_.reset(new MergedTsdfIntegrator(
        integrator_config_, map_->getTsdfLayerPtr()));
  } else if (integrator_method_.compare("fast") == 0) {
    integrator_.reset(new FastTsdfIntegrator(
        integrator_config_, map_->getTsdfLayerPtr()));
  } else {
    integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config_, map_->getTsdfLayerPtr()));
  }

  mesh_layer_.reset(new MeshLayer(map_->block_size()));

  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config_, map_->getTsdfLayerPtr(), mesh_layer_.get()));

}

void DynamicObject::initNextStep() {

  // Update last cloud and prepare current cloud to be filled
  *cloud_last_ = *cloud_current_;
  *cloud_last_transformed_ = *cloud_current_transformed_;
  cloud_current_->clear();
  cloud_current_transformed_->clear();

  // Update occurence flage and reset counter
  occured_in_current_frame_ = true;
  time_since_last_occurence_ = 0;
  semantic_class_set_ = false;

}

void DynamicObject::setSemanticClass(const int semantic_class){

  if (!active_) return;

  // Semantic class already set in this frame or object did not occur
  if (semantic_class_set_ || semantic_class == -1) return;

  if (semantic_class == -2){ // Object was seen but not recognized
    semantic_class_confidence_ = std::max(semantic_class_confidence_ - 1, 0);
  } else {
    if (semantic_class_ == -1){
      semantic_class_ = semantic_class;
    } else {
      if (semantic_class_ == semantic_class){
        semantic_class_confidence_ = std::min(
                                    semantic_class_confidence_ + 1, 10);
      } else {
        semantic_class_confidence_ = std::max(
                                    semantic_class_confidence_ - 1, 0);
        if (semantic_class_confidence_ == 0){
          semantic_class_ = semantic_class;
        }
      }
    }
  }

  semantic_class_set_ = true;
}

void DynamicObject::integrate(
  const Transformation& T_G_C, const bool is_background) {

  if (!active_) return;

  if (!occured_in_current_frame_) return;

  Pointcloud points;
  Colors colors;

  Transformation T_ACC;
  if (is_background){
    T_ACC = Transformation();
    convertPointcloud(*cloud_current_, color_map_, &points, &colors);
    integrator_->integratePointCloud(T_G_C, points, colors, false, T_ACC);
  } else {

    Eigen::Quaternionf quat = Eigen::Quaternionf(
                    T_O_accumulated_.block<3,3>(0,0));
    quat.normalize();
    T_ACC = Transformation(quat, T_O_accumulated_.block<3,1>(0,3));

    convertPointcloud(*cloud_transformed_, color_map_, &points, &colors);

    integrator_->integratePointCloud(Transformation(), points, colors,
                                                        false, T_ACC * T_G_C);
  }
}

int DynamicObject::align(const Transformation& T_G_C){

  not_aligned_ = false;
  if (!active_) return -1;
  if (!occured_in_current_frame_) return -1;
  mesh_cloud_->clear();

  Mesh connected_mesh;
  mesh_layer_->getConnectedMesh(&connected_mesh);

  pcl::transformPointCloud (
   *cloud_current_, *cloud_current_transformed_,
                      T_G_C.getTransformationMatrix());

  if (!(cloud_current_->empty() ||
        cloud_last_->empty() ||
        connected_mesh.vertices.empty()))
  {

    for (size_t idx = 0;  idx <  connected_mesh.vertices.size(); idx++){
      pcl::PointXYZRGBNormal point;
      point.x = connected_mesh.vertices[idx](0,0);
      point.y = connected_mesh.vertices[idx](1,0);
      point.z = connected_mesh.vertices[idx](2,0);
      point.normal_x = connected_mesh.normals[idx](0,0);
      point.normal_y = connected_mesh.normals[idx](1,0);
      point.normal_z = connected_mesh.normals[idx](2,0);
      mesh_cloud_->push_back(point);
    }

    pcl::Indices indices;
    pcl::removeNaNFromPointCloud(*mesh_cloud_, *mesh_cloud_, indices);

    Eigen::Matrix4f T_O = Eigen::Matrix4f::Identity();
    double fitness_score;
    bool s1 = icp_->align(cloud_current_transformed_, cloud_last_transformed_,
                               T_O_last_.cast<float>(), &T_O, &fitness_score);

    if (T_O.block<3,1>(0,3).norm() < 0.0001){
      T_O_accumulated_ = Eigen::Matrix4f::Identity();
      static_ = true;
      frames_not_aligned_ = 0;
    }

    if (!static_){

       if (!s1){

         not_aligned_=true;
         T_O = T_O_last_;
       } else {
         T_O_last_ = T_O;
       }

      Eigen::Matrix4f T_O_accumulated_est = T_O_accumulated_ * T_O;
      Eigen::Matrix4f T_O_accumulated_new = Eigen::Matrix4f::Identity();

      bool s2 = icp_->align(cloud_current_transformed_, mesh_cloud_,
                               T_O_accumulated_est.cast<float>(),
                                    &T_O_accumulated_new, &fitness_score);

      if (!s2){
          T_O_accumulated_ = T_O_accumulated_est;
      } else {
          T_O_accumulated_ = T_O_accumulated_new;
      }

      if (!s2){
        frames_not_aligned_++;
      } else {
        frames_not_aligned_=0;
      }

    }
  }

  pcl::transformPointCloud (
            *cloud_current_transformed_, *cloud_transformed_, T_O_accumulated_);

  return frames_not_aligned_;

}


bool DynamicObject::isPointInside(const InputPointType point){

  return (point.x > p_mesh_min_.x - xy_in_object_padding_ &&
          point.y > p_mesh_min_.y - xy_in_object_padding_ &&
          point.z > p_mesh_min_.z - xy_in_object_padding_&&
          point.x < p_mesh_max_.x + xy_in_object_padding_ &&
          point.y < p_mesh_max_.y + xy_in_object_padding_ &&
          point.z < p_mesh_max_.z + xy_in_object_padding_ );
}

void DynamicObject::updateMeshMinMax(){



  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed_cloud(
                                  new pcl::PointCloud<pcl::PointXYZRGBNormal>());
  pcl::transformPointCloud (
        *cloud_current_, *transformed_cloud, Eigen::Matrix4f::Identity());
  pcl::getMinMax3D(*transformed_cloud, p_mesh_min_, p_mesh_max_);

}

void DynamicObject::updateState(
  const Transformation& T_G_C,
            const sensor_msgs::PointCloud2::Ptr& pointcloud_msg){

  Eigen::Matrix4f result_transform;
  if (!active_){
    result_transform = T_G_C.getTransformationMatrix();
    state_ = getOBBDetection(cloud_current_, result_transform);
  } else {
    result_transform = T_O_accumulated_.inverse();
    state_ = getOBBDetection(mesh_cloud_, result_transform);
  }

  geometry_msgs::PoseStamped obj_pose;
  obj_pose.header.stamp = pointcloud_msg->header.stamp;
  obj_pose.pose.position.x = state_(0);
  obj_pose.pose.position.y = state_(1);
  obj_pose.pose.position.z = state_(2);
  obj_pose.pose.orientation.x = 0;
  obj_pose.pose.orientation.y = 0;
  obj_pose.pose.orientation.z = std::sin(state_(3)/2.0);
  obj_pose.pose.orientation.w = std::cos(state_(3)/2.0);

  stamped_trajectory_.push_back(obj_pose);

}

void DynamicObject::generateMesh(){

  if (!active_) return;

  mesh_integrator_->generateMesh(true, true);

  BlockIndexList mesh_indices;
  mesh_layer_->getAllAllocatedMeshes(&mesh_indices);
  for (const BlockIndex& block_index : mesh_indices) {
    Mesh::Ptr mesh = mesh_layer_->getMeshPtrByIndex(block_index);
    mesh->colors.clear();
    mesh->colors.resize(mesh->indices.size());
    for (size_t i = 0; i < mesh->vertices.size(); i++) {
      mesh->colors[i] = mesh_color_;
    }
  }
}

}  // namespace voxblox
