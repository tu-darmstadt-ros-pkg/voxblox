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

#ifndef VOXBLOX_ROS_ALIGNMENT_ICP_UTILS_H_
#define VOXBLOX_ROS_ALIGNMENT_ICP_UTILS_H_

#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

using ConvergenceState =
    pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState;

inline bool checkConvergenceState(ConvergenceState state) {
  bool success = false;

  if (state == ConvergenceState::CONVERGENCE_CRITERIA_NOT_CONVERGED ||
      state == ConvergenceState::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES ||
      state == ConvergenceState::CONVERGENCE_CRITERIA_ITERATIONS) {
    if (state == ConvergenceState::CONVERGENCE_CRITERIA_NOT_CONVERGED)std::cout<<"1"<<std::endl;
    if (state == ConvergenceState::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES)std::cout<<"2"<<std::endl;
    if (state == ConvergenceState::CONVERGENCE_CRITERIA_ITERATIONS) std::cout<<"3"<<std::endl;
    LOG(INFO) << "\nICP has NOT CONVERGED. ";
  } else {
    success = true;
  }

  return success;
}

#endif  // VOXBLOX_ROS_ALIGNMENT_ICP_UTILS_H_
