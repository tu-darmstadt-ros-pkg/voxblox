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

#ifndef VOXBLOX_ROS_ALIGNMENT_PCL_ICP_H_
#define VOXBLOX_ROS_ALIGNMENT_PCL_ICP_H_

#include <pcl/registration/gicp.h>

namespace voxblox {

class PCL_ICP {
 public:
  struct Config {
    bool point_to_plane = false;

    bool use_reciprocal_correspondences = false;
    double max_correspondence_distance =
        std::sqrt(std::numeric_limits<double>::max());

    bool use_symmetric_objective = false;

    int max_iterations = 10;
    double absolute_mse = 1e-12;
    double euclidean_fitness_epsilon = -std::numeric_limits<double>::max();
    double transformation_epsilon = 0.0;
  };

  PCL_ICP(Config config);

  pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal,
                                                        pcl::PointXYZRGBNormal>
  init();

  void setMaximumIterations(int max_iterations);

  bool align(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr source_cloud,
             const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target_cloud,
             const Eigen::Matrix4f& guess,
             Eigen::Matrix4f* transformation_matrix,
             double* fitness_score);

 protected:
  Config config_;
};

} // namespace voxblox

#endif  // VOXBLOX_ROS_ALIGNMENT_PCL_ICP_H_
