#ifndef VOXBLOX_INTEGRATOR_SEGMENTED_TSDF_INTEGRATOR_H_
#define VOXBLOX_INTEGRATOR_SEGMENTED_TSDF_INTEGRATOR_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Core>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/utils/approx_hash_array.h"
#include "voxblox/utils/timing.h"
#include <voxblox/utils/camera_model.h>
#include <voxblox/utils/distance_utils.h>

#include "voxblox/core/block_hash.h"

// TODO: remove after debugging
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

namespace voxblox {

// Note most functions state if they are thread safe. Unless explicitly stated
// otherwise, this thread safety is based on the assumption that any pointers
// passed to the functions point to objects that are guaranteed to not be
// accessed by other threads.
class SegmentedTsdfIntegrator {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FloatingPoint min_ray_length_m = 0.1;
    FloatingPoint max_ray_length_m = 5.0;
    size_t integrator_threads = 1; //std::thread::hardware_concurrency();

    // segmentation specific
    LabelConfidence min_merge_confidence = 3;
    float min_segment_overlap = 0.3f;
    size_t min_segment_pixel_size = 50;
    float min_segment_merge_overlap = 0.2f;
    int voxel_prop_radius = 2;
    bool write_debug_data_ = false;
  };

  SegmentedTsdfIntegrator(const Config& config, Layer<TsdfVoxel>* tsdf_layer,
                          Layer<SegmentedVoxel>* segmentation_layer);

  // NOT thread safe.
  void integrateSegmentedPointCloud(const Transformation& T_G_C,
                           const Pointcloud& points_C,
                           LabelIndexMap& segment_map,
                           const std::map<uint, Color>& color_map);

  // Returns a CONST ref of the config.
  const Config& getConfig() const { return config_; }

  const LabelBlockIndexesMap& getSegmentBlocksMap() const { return segment_blocks_map_; }
  const Labels& getUpdatedSegments() const { return updated_segments_; }

 private:
  // Thread safe.
  inline bool isPointValid(const Point& point_C, const Label &segment) const;
  inline bool isPointValid(const Point& point_C) const;

  // Updates seg_voxel. Thread safe.
  void updateSegmentedVoxel(const GlobalIndex& global_voxel_index, const Label& segment);

  // Thread safe.
  float computeDistance(const Point& origin, const Point& point_G,
                        const Point& voxel_center) const;

  // Thread safe.
  float getVoxelSegment(const Point& point_C) const;

  void getVisibleVoxels(const Transformation& T_G_C,
                                  const Pointcloud& points_C,
                                  ThreadSafeIndex* index_getter);

  LabelIndexMap propagateSegmentLabels(LabelIndexMap& segment_map);

  void updateGlobalSegments(const LabelIndexMap& propagated_labels);

  float computeSegmentOverlap(Label segment_1_id, Label segment_2_id, Labels& segment1_idxs, Labels& segment2_idxs);

  void checkMergeCandidates(LabelIndexMap& propagated_labels);
  void mergeSegmentLabels(LabelIndexMap& propagated_labels, const LabelPair& label_pair);

  void applyLabelToVoxels(const BlockIndex& block_idx, Label old_segment, Label new_segment);

  Config config_;

  Layer<TsdfVoxel>* tsdf_layer_;
  Layer<SegmentedVoxel>* segmentation_layer_;

  // Cached map config.
  FloatingPoint voxel_size_;
  size_t voxels_per_side_;
  FloatingPoint block_size_;

  // Derived types.
  FloatingPoint voxel_size_inv_;
  FloatingPoint voxels_per_side_inv_;
  FloatingPoint block_size_inv_;

  VoxelIndexMap visible_voxels_;
  LabelIndexMap segment_map_;
  LabelIndexMap segment_merge_candidates_;
  LabelPairConfidenceMap label_pair_confidences_;
  LabelBlockIndexesMap  segment_blocks_map_;

  Labels updated_segments_;

  Label max_label_;
  uint num_frames_;
};
}  // namespace voxblox

#endif  // VOXBLOX_INTEGRATOR_SEGMENTED_TSDF_INTEGRATOR_H_
