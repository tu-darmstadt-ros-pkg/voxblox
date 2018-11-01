#ifndef VOXBLOX_UTILS_SEGMENT_UTILS_H_
#define VOXBLOX_UTILS_SEGMENT_UTILS_H_

#include "voxblox/core/common.h"
#include "voxblox/core/voxel.h"
#include "voxblox/core/layer.h"
#include "voxblox/mesh/mesh_integrator.h"
#include "voxblox/utils/distance_utils.h"

namespace voxblox {

class SegmentTools {

public:

  typedef std::shared_ptr<SegmentTools> Ptr;

  SegmentTools(Layer<TsdfVoxel>* tsdf_layer, Layer<SegmentedVoxel>* seg_layer);

  MeshLayer::ConstPtr meshSegment(const LabelBlockIndexesMap& segment_blocks_map, Label segment);
  Label getSegmentIdFromRay(const Point& origin, const Point& direction);

private:

  Layer<TsdfVoxel>::Ptr extractSegmentTsdfLayer(const LabelBlockIndexesMap& segment_blocks_map, Label segment);
  void generateSegmentMesh(Layer<TsdfVoxel>::Ptr segment_layer);

  Layer<TsdfVoxel>* tsdf_layer_;
  Layer<SegmentedVoxel>* seg_layer_;

  MeshIntegratorConfig mesh_config_;
  MeshLayer::Ptr mesh_layer_;
  //std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;
};


}  // namespace voxblox

#endif  // VOXBLOX_UTILS_SEGMENT_UTILS_H_
