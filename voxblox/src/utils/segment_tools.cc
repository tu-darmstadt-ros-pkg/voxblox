#include "voxblox/utils/segment_tools.h"

#include "voxblox/core/color.h"
#include "voxblox/core/common.h"
#include "voxblox/core/voxel.h"
#include "voxblox/mesh/mesh_integrator.h"

namespace voxblox {

SegmentTools::SegmentTools(Layer<TsdfVoxel>* tsdf_layer, Layer<SegmentedVoxel>* seg_layer)
  : tsdf_layer_(tsdf_layer), seg_layer_(seg_layer) {
  mesh_layer_= std::make_shared<MeshLayer>(tsdf_layer->block_size());
}

Layer<TsdfVoxel>::Ptr SegmentTools::extractSegmentTsdfLayer(const LabelBlockIndexesMap& segment_blocks_map, Label segment) {

  timing::Timer extract_segment_tsdf_layer_timer("segmen_tools/extract_segment_tsdf_layer");

  Layer<TsdfVoxel>::Ptr seg_tsdf_layer = std::make_shared<Layer<TsdfVoxel>>(tsdf_layer_->voxel_size(), tsdf_layer_->voxels_per_side());

  auto it = segment_blocks_map.find(segment);
  if (it == segment_blocks_map.end()) {
    return seg_tsdf_layer;
  }

  const BlockIndexSet& blocks = it->second;

  // iterate blocks assigned to the chosen segment_id
  for (auto block_idx: blocks) {
    Block<TsdfVoxel>::Ptr target_tsdf_block = seg_tsdf_layer->allocateNewBlock(block_idx);
    Block<TsdfVoxel>::Ptr source_tsdf_block = tsdf_layer_->getBlockPtrByIndex(block_idx);
    Block<SegmentedVoxel>::Ptr seg_block = seg_layer_->getBlockPtrByIndex(block_idx);

    if (!target_tsdf_block || ! source_tsdf_block || !seg_block)
      continue;

    // determine the voxels belonging to the chosen segment and transfer these to the new tsdf layer
    for (size_t voxel_idx = 0; voxel_idx < target_tsdf_block->num_voxels(); voxel_idx++) {
      const SegmentedVoxel& seg_voxel = seg_block->getVoxelByLinearIndex(voxel_idx);

      if(seg_voxel.segment_id == segment /*&& seg_voxel.confidence > 0*/) {
        TsdfVoxel& tgt_voxel = target_tsdf_block->getVoxelByLinearIndex(voxel_idx);
        const TsdfVoxel& src_voxel = source_tsdf_block->getVoxelByLinearIndex(voxel_idx);

        tgt_voxel.distance = src_voxel.distance;
        tgt_voxel.weight = src_voxel.weight;
        tgt_voxel.color = src_voxel.color;
      }
    }
  }

  extract_segment_tsdf_layer_timer.Stop();

  return seg_tsdf_layer;
}

void SegmentTools::generateSegmentMesh(Layer<TsdfVoxel>::Ptr segment_layer) {

  timing::Timer generate_segment_mesh_timer("segmen_tools/generate_segment_mesh");

  // Generate mesh layer.
  constexpr bool only_mesh_updated_blocks = false;
  constexpr bool clear_updated_flag = false;
  MeshIntegrator<TsdfVoxel> mesh_integrator(mesh_config_, segment_layer.get(), mesh_layer_.get());
  mesh_integrator.generateMesh(only_mesh_updated_blocks, clear_updated_flag);

  generate_segment_mesh_timer.Stop();
}

MeshLayer::ConstPtr SegmentTools::meshSegment(const LabelBlockIndexesMap& segment_blocks_map, Label segment) {

  mesh_layer_->clear();
  Layer<TsdfVoxel>::Ptr tsdf_layer = extractSegmentTsdfLayer(segment_blocks_map, segment);
  generateSegmentMesh(tsdf_layer);

  return mesh_layer_;
}

Label SegmentTools::getSegmentIdFromRay(const Point& origin, const Point& direction) {

  Point surface_intersection;
  float max_distance = 2.0;
  bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(*tsdf_layer_, origin, direction,
                                                       max_distance, &surface_intersection);
  if (success) {
    SegmentedVoxel* voxel = seg_layer_->getVoxelPtrByCoordinates(surface_intersection);
    return voxel->segment_id;
  } else {
    return 0;
  }
}
}  // namespace voxblox
