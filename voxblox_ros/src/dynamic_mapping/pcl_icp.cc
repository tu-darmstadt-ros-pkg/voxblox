// Taken from tsdf-plusplus

// MIT License
//
// Copyright (c) 2021 Margarita Grinvald, Autonomous Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "voxblox_ros/dynamic_mapping/pcl_icp.h"

#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

#include "voxblox_ros/dynamic_mapping/icp_utils.h"

namespace voxblox {

PCL_ICP::PCL_ICP(Config config) : config_(config) {}

pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>
PCL_ICP::init() {
  pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp;

  icp.setUseReciprocalCorrespondences(config_.use_reciprocal_correspondences);
  icp.setMaxCorrespondenceDistance(config_.max_correspondence_distance);

  pcl::registration::TransformationEstimationPointToPlane<pcl::PointXYZRGBNormal,
                                                          pcl::PointXYZRGBNormal>::Ptr
      transformation_estimation(
          new pcl::registration::TransformationEstimationPointToPlane<
              pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
  icp.setTransformationEstimation(transformation_estimation);

  icp.setUseSymmetricObjective(config_.use_symmetric_objective);

  icp.setMaximumIterations(config_.max_iterations);
  icp.getConvergeCriteria()->setAbsoluteMSE(config_.absolute_mse);
  icp.setEuclideanFitnessEpsilon(config_.euclidean_fitness_epsilon);
  icp.setTransformationEpsilon(config_.transformation_epsilon);

  return icp;
}

// Point-to-plane ICP alignment with normals.
bool PCL_ICP::align(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr source_cloud,
                const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target_cloud,
                const Eigen::Matrix4f& guess,
                Eigen::Matrix4f* transformation_matrix_float,
                double* fitness_score) {
  Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

  pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_ =
      init();

  // std::cout << "ICP: " <<std::endl;
  // std::cout << source_cloud->points.size() <<std::endl;
  // std::cout << target_cloud->points.size() <<std::endl;
  icp_.setMaximumIterations(config_.max_iterations);
  icp_.setInputSource(source_cloud);
  icp_.setInputTarget(target_cloud);

  int iterations = config_.max_iterations;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr aligned_source(
      new pcl::PointCloud<pcl::PointXYZRGBNormal>);

  bool success = false;

  icp_.align(*aligned_source, guess);
  success =
      checkConvergenceState(icp_.getConvergeCriteria()->getConvergenceState());
  *fitness_score = icp_.getFitnessScore();

  transformation_matrix = icp_.getFinalTransformation().cast<double>();

  *transformation_matrix_float = transformation_matrix.cast<float>();

  return success;
}

}
