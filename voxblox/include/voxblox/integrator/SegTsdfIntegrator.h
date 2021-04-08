#ifndef SEG_TSDF_INTEGRATOR_H_
#define SEG_TSDF_INTEGRATOR_H_

#include <ros/ros.h>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/integrator/vertex_map.h"
#include "voxblox/utils/distance_utils.h"
#include "voxblox/utils/timing.h"

#include "voxblox/integrator/label_map.h"
#include "voxblox/integrator/vertex_map.h"

namespace voxblox {

enum IntegrationMode {
  DEFAULT,
  CONFIDENCE,
  CONFIDENCE_WEIGHT,
  CONFIDENCE_HIGHEST
};
// enum NormalCalculationMode {CROSSPRODUCT, CENTRALDIFFERENCES,
// CENTRALDIFFERENCES_NORMALIZED, VOXBLOX};

template <class T>
class SegTsdfIntegrator {
 public:
  SegTsdfIntegrator(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf,
                    std::shared_ptr<Layer<T>> label);

  void direct_integration(std::shared_ptr<vertexMap> vertices,
                          std::shared_ptr<labelMap> lmap);

  void integrateHighest(Point p, std::vector<std::pair<int, double>> labels);

  void integrateConfidenceHighest(Point p,
                                  std::vector<std::pair<int, double>> labels);

  void integrateConfidence(Point p, std::vector<std::pair<int, double>> labels);

  void integrateConfidenceWeight(Point p,
                                 std::vector<std::pair<int, double>> labels);

 protected:
  const Layer<TsdfVoxel>& tsdf_layer;
  std::shared_ptr<Layer<T>> label_layer;

  IntegrationMode mode;
  // NormalCalculationMode normal_calculation;

  ros::NodeHandle nh_;
};

}  // namespace voxblox
#endif