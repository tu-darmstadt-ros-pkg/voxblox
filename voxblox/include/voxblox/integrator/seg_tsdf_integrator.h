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

#include "voxblox/core/block_hash.h"

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

    float default_truncation_distance = 0.05;
    FloatingPoint min_ray_length_m = 0.1;
    FloatingPoint max_ray_length_m = 5.0;
    size_t integrator_threads = 1; //std::thread::hardware_concurrency();

    // fast integrator specific
    float start_voxel_subsampling_factor = 2.0f;
    int max_consecutive_ray_collisions = 2;
    int clear_checks_every_n_frames = 1;
    float max_integration_time_s = std::numeric_limits<float>::max();

    // segmentation specific
    LabelConfidence min_merge_confidence = 3;
    float min_segment_overlap = 0.3f;
    size_t min_segment_pixel_size = 50;
    float min_segment_merge_overlap = 0.2f;
  };

  SegmentedTsdfIntegrator(const Config& config, Layer<TsdfVoxel>* tsdf_layer,
                          Layer<SegmentedVoxel>* segmentation_layer);

  // NOT thread safe.
  void integrateSegmentedPointCloud(const Transformation& T_G_C,
                           const Pointcloud& points_C,
                           const Labels& segmentation,
                           const LabelIndexMap& segment_map,
                           const std::map<uint, Color>& color_map);

  // Returns a CONST ref of the config.
  const Config& getConfig() const { return config_; }

  void updateCameraModel(const Eigen::Matrix<FloatingPoint, 2, 1>& resolution, double focal_length);


 protected:
  // Thread safe.
  inline bool isPointValid(const Point& point_C, const Label &segment) const;
  inline bool isPointValid(const Point& point_C) const;

  // Will return a pointer to a voxel located at global_voxel_idx in the seg
  // layer. Thread safe.
  // Takes in the last_block_idx and last_block to prevent unneeded map lookups.
  // If this voxel belongs to a block that has not been allocated, a block in
  // temp_block_map_ is created/accessed and a voxel from this map is returned
  // instead. Unlike the layer, accessing temp_block_map_ is controlled via a
  // mutex allowing it to grow during integration.
  // These temporary blocks can be merged into the layer later by calling
  // updateLayerWithStoredBlocks()
  SegmentedVoxel* allocateStorageAndGetVoxelPtr(const VoxelIndex& global_voxel_idx,
                                           Block<SegmentedVoxel>::Ptr* last_block,
                                           BlockIndex* last_block_idx);

  // NOT thread safe
  void updateLayerWithStoredBlocks();

  // Updates seg_voxel. Thread safe.
  void updateSegmentedVoxel(const VoxelIndex& global_voxel_index, const Label& segment);

  // Thread safe.
  float computeDistance(const Point& origin, const Point& point_G,
                        const Point& voxel_center) const;

  // Thread safe.
  float getVoxelSegment(const Point& point_C) const;

  void getVisibleVoxels(const Transformation& T_G_C,
                                  const Pointcloud& points_C,
                                  ThreadSafeIndex* index_getter);

  LabelIndexMap propagateSegmentLabels(const Labels& segmentation,
                              const LabelIndexMap& segment_map);

  void updateGlobalSegments(const LabelIndexMap& propagated_labels);

  float computeSegmentOverlap(Labels& segment1, Labels& segment2);

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

  // Temporary block storage, used to hold blocks that need to be created while
  // integrating a new pointcloud
  std::mutex temp_block_mutex_;
  Layer<SegmentedVoxel>::BlockHashMap temp_block_map_;

  // We need to prevent simultaneous access to the voxels in the map. We could
  // put a single mutex on the map or on the blocks, but as voxel updating is
  // the most expensive operation in integration and most voxels are close
  // together, both strategies would bottleneck the system. We could make a
  // mutex per voxel, but this is too ram heavy as one mutex = 40 bytes.
  // Because of this we create an array that is indexed by the first n bits of
  // the voxels hash. Assuming a uniform hash distribution, this means the
  // chance of two threads needing the same lock for unrelated voxels is
  // (num_threads / (2^n)). For 8 threads and 12 bits this gives 0.2%.
  ApproxHashArray<12, std::mutex> mutexes_;

  CameraModel depth_cam_model_;

  VoxelIndexMap visible_voxels_;
  LabelIndexMap segment_map_;
  LabelIndexMap segment_merge_candidates_;
  LabelPairConfidenceMap label_pair_confidences_;
  LabelBlockIndexesMap  segment_blocks_map_;

  Label max_label_;

  uint num_frames_;


private:
 // Two approximate sets are used below. The limitations of these sets are
 // outlined in approx_hash_array.h, but in brief they are thread safe and very
 // fast, but have a small chance of returning false positives and false
 // negatives. As rejecting a ray or integrating an uninformative ray are not
 // very harmful operations this trade-off works well in this integrator.

 // uses 2^20 bytes (8 megabytes) of ram per tester
 // A testers false negative rate is inversely proportional to its size
 static constexpr size_t masked_bits_ = 20;
 // only needs to zero the above 8mb of memory once every 10,000 scans
 // (uses an additional 80,000 bytes)
 static constexpr size_t full_reset_threshold = 10000;

 // Voxel start locations are added to this set before ray casting. The ray
 // casting only occurs if no ray has been cast from this location for this
 // scan.
 ApproxHashSet<masked_bits_, full_reset_threshold> start_voxel_approx_set_;
 // This set records which voxels a scans rays have passed through. If a ray
 // moves through max_consecutive_ray_collisions voxels in a row that have
 // already been seen this scan, it is deemed to be adding no new information
 // and the casting stops.
 ApproxHashSet<masked_bits_, full_reset_threshold> voxel_observed_approx_set_;

 // Used in terminating the integration early if it exceeds a time limit.
 std::chrono::time_point<std::chrono::steady_clock> integration_start_time_;

};
}  // namespace voxblox

#endif  // VOXBLOX_INTEGRATOR_SEGMENTED_TSDF_INTEGRATOR_H_
