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
                            bool dynamic_voxel_size,
                            const int id, const int semantic_class,
                            Color mesh_color)
  : id_(id),
    semantic_class_(semantic_class),
    semantic_class_confidence_(0),
    semantic_class_set_(true),
    occured_in_current_frame_(true),
    time_since_last_occurence_(0),
    mesh_color_(mesh_color),
    icp_(icp),
    cloud_current_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_last_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    cloud_transformed_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    mesh_cloud_(new pcl::PointCloud<pcl::PointXYZRGBNormal>),
    T_O_accumulated_(Eigen::Matrix4f::Identity()),
    T_O_last_(Eigen::Matrix4f::Identity()),
    color_map_(new RainbowColorMap())
{

  map_.reset(new TsdfMap(tsdf_config));

  if (method.compare("simple") == 0) {
    integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, map_->getTsdfLayerPtr()));
  } else if (method.compare("merged") == 0) {
    integrator_.reset(new MergedTsdfIntegrator(
        integrator_config, map_->getTsdfLayerPtr()));
  } else if (method.compare("fast") == 0) {
    integrator_.reset(new FastTsdfIntegrator(
        integrator_config, map_->getTsdfLayerPtr()));
  } else {
    integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, map_->getTsdfLayerPtr()));
  }

  mesh_layer_.reset(new MeshLayer(map_->block_size()));

  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config, map_->getTsdfLayerPtr(), mesh_layer_.get()));

}

void DynamicObject::reset() {

  // Update last cloud and prepare current cloud to be filled
  *cloud_last_ = *cloud_current_;
  cloud_current_->clear();
  mesh_cloud_->clear();
  // Update occurence flage and reset counter
  occured_in_current_frame_ = true;
  time_since_last_occurence_ = 0;
  semantic_class_set_ = false;

}

void DynamicObject::setSemanticClass(const int semantic_class){

  // Semantic class already set in this frame or object did not occur
  if (semantic_class_set_ || semantic_class == -1) return;

  // std::cout << "Set class for Object: "<< id_ <<std::endl;
  // std::cout << "Class: "<< semantic_class << std::endl;
  // std::cout << "Conf: "<< semantic_class_confidence_ << std::endl;

  if (semantic_class == -2){ // Object was seen but not recognized
    semantic_class_confidence_ = std::max(semantic_class_confidence_ - 1, 0);
  } else {
    if (semantic_class_ == -1){
      semantic_class_ = semantic_class;
    } else {
      if (semantic_class_ == semantic_class){
        semantic_class_confidence_ = std::min(semantic_class_confidence_ + 1, 10);
      } else {
        semantic_class_confidence_ = std::max(semantic_class_confidence_ - 1, 0);
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

  if (!occured_in_current_frame_) return;

  Pointcloud points;
  Colors colors;

  Transformation T_O;
  if (is_background){
    T_O = Transformation();
    convertPointcloud(*cloud_current_, color_map_, &points, &colors);

  } else {
    // TODO SOMETIMES ERROR HERE with isValidRotationMatrix from minkindr
    // This detour prevents it
    Eigen::Quaternionf quat = Eigen::Quaternionf(T_O_accumulated_.block<3,3>(0,0));
    quat.normalize();
    T_O = Transformation(quat, T_O_accumulated_.block<3,1>(0,3));
    convertPointcloud(*cloud_transformed_, color_map_, &points, &colors);
  }

  integrator_->integratePointCloud(T_G_C, points, colors, false, T_O);

}

void DynamicObject::align(){

  if (!occured_in_current_frame_) return;

  Eigen::Matrix4f T_O = Eigen::Matrix4f::Identity();

  Mesh connected_mesh;
  mesh_layer_->getConnectedMesh(&connected_mesh);

  if (!(cloud_current_->empty() ||
        cloud_last_->empty() ||
        connected_mesh.vertices.empty()))
  { //TODO CLEAN up intendation with visual studio code

    // std::cout<<"Align Object "<<id_<<std::endl;

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
    pcl::StopWatch test;

    double fitness_score;
    // std::cout<<"Align Object "<<id_<<" "<<semantic_class_<<std::endl;
    // std::cout << cloud_current_->points.size() <<std::endl;
    // std::cout << cloud_last_->points.size() <<std::endl;
    // std::cout << mesh_cloud_->points.size() <<std::endl;
    // std::cout << T_O_accumulated_ << std::endl;
    bool SuccessFirst = icp_->align(cloud_current_, cloud_last_,
                               T_O_last_.cast<float>(), &T_O, &fitness_score);

    // std::cout <<"Fitness first "<< fitness_score <<std::endl;

    if (!SuccessFirst){
      std::cout<<"Align Object "<<id_<<" NOT CONVERGED 1"<<std::endl;
    }

     if (fitness_score > 0.01){ //TODO Do this correctly
       std::cout<<"LOW FITNESS"<<std::endl;
       T_O = T_O_last_;
     } else{
       T_O_last_ = T_O;
     }
    //T_O_last_ =  T_O;

    Eigen::Matrix4f PRODUCT = T_O_accumulated_ * T_O;

    Eigen::Matrix4f T_TEST = Eigen::Matrix4f::Identity();
    bool SuccessSecond = icp_->align(cloud_current_, mesh_cloud_,
                               PRODUCT.cast<float>(), &T_TEST, &fitness_score);

    if (!SuccessSecond){
      std::cout<<"Align Object "<<id_<<" NOT CONVERGED 2"<<std::endl;
    }

    // std::cout <<"Current Translate "<<id_<<" "<< T_O.block<3,1>(0,3).norm() <<std::endl;
    // std::cout <<"Last Translate "<<id_<<" "<< T_O_last_.block<3,1>(0,3).norm() <<std::endl;
    // std::cout <<"Last Translate "<<id_<<" "<< T_TEST.block<3,1>(0,3).norm() <<std::endl;

    // std::cout <<"Fitness second "<< fitness_score <<std::endl;

    if (fitness_score > 0.01){
      std::cout<<"LOW FITNESS"<<std::endl;
      T_O_accumulated_ = T_O_accumulated_ * T_O;
    } else{
      T_O_accumulated_ = T_TEST;
    }


    // std::cout << "icp took  " << std::fixed << test.getTimeSeconds()
    //                                                 << " seconds." << std::endl;
  }

  pcl::transformPointCloud (*cloud_current_, *cloud_transformed_, T_O_accumulated_);

}

// void DynamicObject::align2(){
//
//   if (!occured_in_current_frame_) return;
//
//   Eigen::Matrix4f T_O = Eigen::Matrix4f::Identity();
//
//   Mesh connected_mesh;
//   mesh_layer_->getConnectedMesh(&connected_mesh);
//
//   if (!(cloud_current_->empty() ||
//         cloud_last_->empty() ||
//         connected_mesh.vertices.empty()))
//   { //TODO CLEAN up intendation with visual studio code
//
//     // std::cout<<"Align Object "<<id_<<std::endl;
//
//     for (size_t idx = 0;  idx <  connected_mesh.vertices.size(); idx++){
//       pcl::PointXYZRGBNormal point;
//       point.x = connected_mesh.vertices[idx](0,0);
//       point.y = connected_mesh.vertices[idx](1,0);
//       point.z = connected_mesh.vertices[idx](2,0);
//       point.normal_x = connected_mesh.normals[idx](0,0);
//       point.normal_y = connected_mesh.normals[idx](1,0);
//       point.normal_z = connected_mesh.normals[idx](2,0);
//       mesh_cloud_->push_back(point);
//     }
//     pcl::StopWatch test;
//
//     // Calculate object centroid
//     float c_x = 0;
//     float c_y = 0;
//     float c_z = 0;
//     for (size_t idx = 0u;  idx < (*cloud_current_).size(); ++idx)
//     {
//
//       c_x += cloud_current_->points[idx].x;
//       c_y += cloud_current_->points[idx].y;
//       c_z += cloud_current_->points[idx].z;
//
//     }
//     c_x /= (*cloud_current_).size();
//     c_y /= (*cloud_current_).size();
//     c_z /= (*cloud_current_).size();
//
//
//     Eigen::Matrix4f T_O_0 = Eigen::Matrix4f::Identity();
//     T_O_0(0,3) = - c_x;
//     T_O_0(1,3) = - c_y;
//     T_O_0(2,3) = - c_z;
//
//     // Shift object to zero
//     pcl::transformPointCloud (*cloud_current_, *cloud_transformed_, T_O_0);
//
//
//     double fitness_score;
//     bool SuccessFirst = icp_->align(cloud_transformed_, mesh_cloud_,
//                             Eigen::Matrix4f::Identity(), &T_O, &fitness_score);
//
//
//     T_O_accumulated_ =  T_O_0 * T_O;
//
//   }
//
//   pcl::transformPointCloud (*cloud_current_, *cloud_transformed_, T_O_accumulated_);
//
// }

void DynamicObject::updateState(const Transformation& T_G_C){

  Eigen::Matrix4f result_transform = T_G_C.getTransformationMatrix()
                                                  * T_O_accumulated_.inverse();

  state_ = getOBBDetection(getMeshCloud(), result_transform);

}

void DynamicObject::generateMesh(){

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

void DynamicObject::updatePosition(const Transformation& T_G_C){

    Eigen::Matrix4f tfmatrix = T_G_C.getTransformationMatrix();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr obj_cloud_temp;
    obj_cloud_temp.reset(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::transformPointCloud (*cloud_current_,
                        *obj_cloud_temp,
                        tfmatrix);

    // Almost same as used in Lidar MODT
    pcl::PointXYZRGBNormal origMinPoint, origMaxPoint;
    pcl::getMinMax3D(*obj_cloud_temp, origMinPoint, origMaxPoint);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud_transformed(
                                            new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t idx = 0u;  idx < (*obj_cloud_temp).size(); ++idx)
    {
      pcl::PointXYZ point;

      float x = obj_cloud_temp->points[idx].x;
      float y = obj_cloud_temp->points[idx].y;

      point.x = x;
      point.y = y;
      point.z = 0.0;

      cluster_cloud_transformed->points.push_back(point);
    }

      // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cluster_cloud_transformed, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(
                           *cluster_cloud_transformed, pcaCentroid, covariance);
    Eigen::Matrix2f smallcovariance = covariance.block(0,0,2,2);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigen_solver(
                                   smallcovariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix2f eigenVectorsPCA = eigen_solver.eigenvectors();

    // Transform the original cloud to the origin where the principal
    // components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<2,2>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<2,1>(0,3) = -1.f *
                  (projectionTransform.block<2,2>(0,0) * pcaCentroid.head<2>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(
                                            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(
        *cluster_cloud_transformed, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector2f meanXY = 0.5f * (maxPoint.getVector3fMap().head<2>() +
                                           minPoint.getVector3fMap().head<2>());
    const float meanZ = 0.5f * (origMaxPoint.z + origMinPoint.z);

    Eigen::Vector2f bboxTransform = eigenVectorsPCA * meanXY +
                                                          pcaCentroid.head<2>();
    Point position(bboxTransform[0], bboxTransform[1] , meanZ);

    trajectory_.push_back(position);
}


}  // namespace voxblox
