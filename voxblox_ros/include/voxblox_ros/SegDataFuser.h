#ifndef SEG_DATA_FUSER_H_
#define SEG_DATA_FUSER_H_

#include <ros/ros.h>
#include "voxblox_ros/SegToolbox.h"

#include <voxblox/core/layer.h>

#include <voxblox/integrator/SegTsdfIntegrator.h>

namespace voxblox {

template <class T>
class SegDataFuser {
 public:
  SegDataFuser(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf_layer,
               std::shared_ptr<Layer<T>> label_layer);

  virtual void fuse(Transformation T_G_C, std::shared_ptr<labelMap> lmap,
                    std::shared_ptr<rawDataPointer> data);

  virtual void prepare_integration(std::shared_ptr<labelMap> lmap,
                                   Transformation T_w_lm,
                                   std::shared_ptr<rawDataPointer> data);

  virtual void set_integrator(std::shared_ptr<SegTsdfIntegrator<T>> i);

 protected:
  Transformation T_w_b;
  bool aligned_rgbd;
  double max_dist;
  ros::NodeHandle nh_;

  std::shared_ptr<Layer<T>> label_layer;

  bool direct_integration;

  // integrator
  std::shared_ptr<SegTsdfIntegrator<T>> integrator;
  const Layer<TsdfVoxel>& tsdf_layer;

  ros::Publisher fused_pub;

  int bilateral_filter_size;
  double bilateral_filter_diff;

 
};

}  // namespace voxblox

#endif