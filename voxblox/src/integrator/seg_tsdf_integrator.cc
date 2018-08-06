#include "voxblox/integrator/seg_tsdf_integrator.h"
#include <iostream>
#include <list>

namespace voxblox {

// Note many functions state if they are thread safe. Unless explicitly stated
// otherwise, this thread safety is based on the assumption that any pointers
// passed to the functions point to objects that are guaranteed to not be
// accessed by other threads.

SegmentedTsdfIntegrator::SegmentedTsdfIntegrator(const Config& config,
                                       Layer<TsdfVoxel>* tsdf_layer, Layer<SegmentedVoxel>* segmentation_layer)
    : config_(config), tsdf_layer_(tsdf_layer), segmentation_layer_(segmentation_layer) {
  DCHECK(tsdf_layer_);
  DCHECK(segmentation_layer_);

  voxel_size_ = tsdf_layer_->voxel_size();
  block_size_ = tsdf_layer_->block_size();
  voxels_per_side_ = tsdf_layer_->voxels_per_side();

  voxel_size_inv_ = 1.0f / voxel_size_;
  block_size_inv_ = 1.0f / block_size_;
  voxels_per_side_inv_ = 1.0f / voxels_per_side_;

  if (config_.integrator_threads == 0) {
    LOG(WARNING) << "Automatic core count failed, defaulting to 1 threads";
    config_.integrator_threads = 1;
  }
}

// Thread safe.
inline bool SegmentedTsdfIntegrator::isPointValid(const Point& point_C, const Label& segment) const {
  return segment > 0 && isPointValid(point_C);
}

// Thread safe.
inline bool SegmentedTsdfIntegrator::isPointValid(const Point& point_C) const {

  const FloatingPoint ray_distance = point_C.norm();
  if (ray_distance < config_.min_ray_length_m ||
      ray_distance > config_.max_ray_length_m) {
    return false;
  } else {
    return true;
  }
}

// Will return a pointer to a voxel located at global_voxel_idx in the seg_tsdf
// layer. Thread safe.
// Takes in the last_block_idx and last_block to prevent unneeded map lookups.
// If the block this voxel would be in has not been allocated, a block in
// temp_block_map_ is created/accessed and a voxel from this map is returned
// instead. Unlike the layer, accessing temp_block_map_ is controlled via a
// mutex allowing it to grow during integration.
// These temporary blocks can be merged into the layer later by calling
// updateLayerWithStoredBlocks()
SegmentedVoxel* SegmentedTsdfIntegrator::allocateStorageAndGetVoxelPtr(
    const VoxelIndex& global_voxel_idx, Block<SegmentedVoxel>::Ptr* last_block,
    BlockIndex* last_block_idx) {
  DCHECK(last_block != nullptr);
  DCHECK(last_block_idx != nullptr);

  const BlockIndex block_idx =
      getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_inv_);

  if ((block_idx != *last_block_idx) || (*last_block == nullptr)) {
    *last_block = segmentation_layer_->getBlockPtrByIndex(block_idx);
    *last_block_idx = block_idx;
  }

  // If no block at this location currently exists, we allocate a temporary
  // voxel that will be merged into the map later
  if (*last_block == nullptr) {
    // To allow temp_block_map_ to grow we can only let one thread in at once
    std::lock_guard<std::mutex> lock(temp_block_mutex_);

    typename Layer<SegmentedVoxel>::BlockHashMap::iterator it =
        temp_block_map_.find(block_idx);
    if (it != temp_block_map_.end()) {
      *last_block = it->second;
    } else {
      auto insert_status = temp_block_map_.emplace(
          block_idx, std::make_shared<Block<SegmentedVoxel>>(
                         voxels_per_side_, voxel_size_,
                         getOriginPointFromGridIndex(block_idx, block_size_)));

      DCHECK(insert_status.second) << "Block already exists when allocating at "
                                   << block_idx.transpose();

      *last_block = insert_status.first->second;
    }
  }

  (*last_block)->updated() = true;

  const VoxelIndex local_voxel_idx =
      getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));
}

// NOT thread safe
void SegmentedTsdfIntegrator::updateLayerWithStoredBlocks() {

  for (const std::pair<const BlockIndex, Block<SegmentedVoxel>::Ptr>&
           temp_block_pair : temp_block_map_) {
    segmentation_layer_->insertBlock(temp_block_pair);
  }

  temp_block_map_.clear();
}

// Updates segmented voxel. Thread safe.
void SegmentedTsdfIntegrator::updateSegmentedVoxel(const VoxelIndex& global_voxel_idx,
                                                   const Label& segment) {

  SegmentedVoxel* seg_voxel = segmentation_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);

  if (seg_voxel == nullptr)
    return;

  // Lookup the mutex that is responsible for this voxel and lock it
  std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  if (seg_voxel->segment_id == 0) {
    seg_voxel->segment_id = segment;
    seg_voxel->confidence = 1;
  } else if (seg_voxel->segment_id == segment) {
    seg_voxel->confidence++;
  } else {
    if (--seg_voxel->confidence == 0) {
      seg_voxel->segment_id = segment;
    }
  }
}

// Thread safe.
// Figure out whether the voxel is behind or in front of the surface.
// To do this, project the voxel_center onto the ray from origin to point G.
// Then check if the the magnitude of the vector is smaller or greater than
// the original distance...
float SegmentedTsdfIntegrator::computeDistance(const Point& origin,
                                          const Point& point_G,
                                          const Point& voxel_center) const {
  const Point v_voxel_origin = voxel_center - origin;
  const Point v_point_origin = point_G - origin;

  const FloatingPoint dist_G = v_point_origin.norm();
  // projection of a (v_voxel_origin) onto b (v_point_origin)
  const FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

  const float sdf = static_cast<float>(dist_G - dist_G_V);
  return sdf;
}

void SegmentedTsdfIntegrator::integrateSegmentedPointCloud(const Transformation& T_G_C,
                                                  const Pointcloud& points_C,
                                                  const Labels& segmentation,
                                                  const LabelIndexMap& segment_map) {

  timing::Timer seg_get_visible_voxels_timer("seg_get_visible_voxels");

  CHECK_EQ(points_C.size(), segmentation.size());

  visible_voxels_.clear();
  segment_map_.clear();

  integration_start_time_ = std::chrono::steady_clock::now();

  static int64_t reset_counter = 0;
  if ((++reset_counter) >= config_.clear_checks_every_n_frames) {
    reset_counter = 0;
    start_voxel_approx_set_.resetApproxSet();
    voxel_observed_approx_set_.resetApproxSet();
  }

  ThreadSafeIndex index_getter(points_C.size());

  std::list<std::thread> integration_threads;

  for (size_t i = 0; i < config_.integrator_threads; ++i) {
    integration_threads.emplace_back(&SegmentedTsdfIntegrator::getVisibleVoxels,
                                     this, T_G_C, points_C,
                                     &index_getter);
  }

  for (std::thread& thread : integration_threads) {
    thread.join();
  }

  seg_get_visible_voxels_timer.Stop();

  std::cout << "visible voxels size: " << visible_voxels_.size() << std::endl;

  int seg_map_size = 0;

  for (auto item : segment_map_)
    seg_map_size += item.second.size();
  std::cout << "segment map size: " << seg_map_size << std::endl;

  timing::Timer insertion_timer("seg_inserting_missed_blocks");
  updateLayerWithStoredBlocks();
  insertion_timer.Stop();

  timing::Timer seg_propagate_segment_labels_timer("seg_propagate_segment_labels");

  LabelIndexMap propagated_labels = propagateSegmentLabels(segmentation, segment_map);
  seg_propagate_segment_labels_timer.Stop();

  int propagated_map_size = 0;
  for (auto item : propagated_labels)
    propagated_map_size += item.second.size();
  std::cout << "propagated map size: " << propagated_map_size << std::endl;

  timing::Timer seg_check_merge_candidates_timer("seg_check_merge_candidates");
  checkMergeCandidates(propagated_labels);
  seg_check_merge_candidates_timer.Stop();

  timing::Timer seg_update_global_segments_timer("seg_update_global_segments");
  updateGlobalSegments(propagated_labels);
  seg_update_global_segments_timer.Stop();
}

void SegmentedTsdfIntegrator::updateCameraModel(const Eigen::Matrix<FloatingPoint, 2, 1>& resolution, double focal_length)
{
  depth_cam_model_.setIntrinsicsFromFocalLength(resolution, focal_length, config_.min_ray_length_m, config_.max_ray_length_m);
}

void SegmentedTsdfIntegrator::getVisibleVoxels(const Transformation& T_G_C,
                                                         const Pointcloud& points_C,
                                                         ThreadSafeIndex* index_getter) {

  DCHECK(index_getter != nullptr);

  size_t point_idx;
  while (index_getter->getNextIndex(&point_idx)) {
    const Point& point_C = points_C[point_idx];

    bool is_clearing = false;
    if (!isPointValid(point_C)) {
      continue;
    }

    const Point origin = T_G_C.getPosition();
    const Point point_G = T_G_C * point_C;
    // Checks to see if another ray in this scan has already started 'close' to
    // this location. If it has then we skip ray casting this point. We measure
    // if a start location is 'close' to another points by inserting the point
    // into a set of voxels. This voxel set has a resolution
    // start_voxel_subsampling_factor times higher then the voxel size.
    AnyIndex global_voxel_idx = getGridIndexFromPoint(
        point_G, config_.start_voxel_subsampling_factor * voxel_size_inv_);
    if (!start_voxel_approx_set_.replaceHash(global_voxel_idx)) {
      continue;
    }

    constexpr bool cast_from_origin = false;
    RayCaster ray_caster(origin, point_G, is_clearing,
                         false,
                         config_.max_ray_length_m, voxel_size_inv_,
                         0.0f, cast_from_origin);

    int64_t consecutive_ray_collisions = 0;

    Block<SegmentedVoxel>::Ptr block = nullptr;
    BlockIndex block_idx;
    while (ray_caster.nextRayIndex(&global_voxel_idx)) {
      // Check if the current voxel has been seen by any ray cast this scan. If
      // it has increment the consecutive_ray_collisions counter, otherwise
      // reset it. If the counter reaches a threshold we stop casting as the ray
      // is deemed to be contributing too little new information.
      if (!voxel_observed_approx_set_.replaceHash(global_voxel_idx)) {
        ++consecutive_ray_collisions;
      } else {
        consecutive_ray_collisions = 0;
      }
      if (consecutive_ray_collisions > config_.max_consecutive_ray_collisions) {
        break;
      }

      SegmentedVoxel* voxel =
          allocateStorageAndGetVoxelPtr(global_voxel_idx, &block, &block_idx);

      if (voxel == nullptr)
        continue;

      //std::cout << "voxel segment: " << voxel->segment_id << " segment_map_[voxel->segment_id] size: " << segment_map_[voxel->segment_id].size() << std::endl;
      visible_voxels_[point_idx] = global_voxel_idx;
      segment_map_[voxel->segment_id].emplace_back(point_idx);
      segment_blocks_map_[voxel->segment_id].emplace(block_idx);
    }
  }
}

LabelIndexMap SegmentedTsdfIntegrator::propagateSegmentLabels(const Labels& segmentation,
                                                              const LabelIndexMap& segment_map) {

  LabelIndexMap propagated_labels;

  if (segment_map_.size() == 1 && segment_map_.find(0) != segment_map_.end()) {
    std::cout << "Map only has unsegmented voxels!" << std::endl;

    for (auto segment: segment_map) {
      Labels& prop_idxs  = propagated_labels[segment.first];
      const Labels& seg_idxs = segment_map.at(segment.first);
      prop_idxs.insert(prop_idxs.end(), seg_idxs.begin(), seg_idxs.end());
    }

    return  propagated_labels;
  }

  Label max_label = static_cast<Label>(segment_blocks_map_.size()-1);
  Label best_overlap_id;

  for (auto img_segmentation: segment_map) {

    // skip the unsegmented part and the noisy parts of the image
    if (img_segmentation.first == 0 || img_segmentation.second.size() < config_.min_segment_pixel_size)
      continue;

    float best_overlap = -std::numeric_limits<float>::max();

    for (auto global_segmentation: segment_map_) {

      float overlap = computeSegmentOverlap(global_segmentation.second, img_segmentation.second);

      if (overlap >= best_overlap) {
        best_overlap = overlap;
        best_overlap_id = global_segmentation.first;
      }

      if (overlap >= config_.min_segment_merge_overlap && global_segmentation.first != 0) {
        segment_merge_candidates_[img_segmentation.first].emplace_back(global_segmentation.first);
      }
    }

    // we could not find a match, skip the segment
    if (best_overlap < 0.01f)
      continue;

    std::cout << "max overlapping segment for segment " << img_segmentation.first << " is " << best_overlap_id <<
                 " with an overlap of " << best_overlap << std::endl;

    // if we have enough overlap, keep the global label, otherwise propagate the label of the depth img
    if (best_overlap >= config_.min_segment_overlap && best_overlap_id != 0) {

      Labels& prop_idxs  = propagated_labels[best_overlap_id];
      //const Labels& glob_idxs  = global_segmentation.second;
      const Labels& glob_idxs = img_segmentation.second;

      prop_idxs.insert(prop_idxs.end(), glob_idxs.begin(), glob_idxs.end());

    } else {
      Labels& prop_idxs  = propagated_labels[max_label++];
      const Labels& img_idxs = img_segmentation.second;

      prop_idxs.insert(prop_idxs.end(), img_idxs.begin(), img_idxs.end());
    }
  }

  return propagated_labels;
}

void SegmentedTsdfIntegrator::updateGlobalSegments(const LabelIndexMap& propagated_labels) {
  for (const auto& segment : propagated_labels) {
    for (const auto& point_idx: segment.second) {
      const VoxelIndex& glob_voxel_idx = visible_voxels_[point_idx];
      updateSegmentedVoxel(glob_voxel_idx, segment.first);
    }
  }
}

float SegmentedTsdfIntegrator::computeSegmentOverlap(Labels& segment1, Labels& segment2) {
  Labels intersection_indices;

  std::sort(segment1.begin(), segment1.end());
  std::sort(segment2.begin(), segment2.end());

  std::set_intersection(segment1.begin(), segment1.end(),
                        segment2.begin(), segment2.end(),
                        std::back_inserter(intersection_indices));

  return static_cast<float>(intersection_indices.size()) / static_cast<float>(segment1.size());
}

void SegmentedTsdfIntegrator::checkMergeCandidates(LabelIndexMap& propagated_labels) {

  LabelPairConfidenceMap candidate_pairs;

  // create all pairs of merge candidates
  for (auto candidate: segment_merge_candidates_) {
    const Labels& common_labels = candidate.second;

    if(common_labels.size() <= 1)
      continue;

    Label min_label = *std::min_element(common_labels.begin(), common_labels.end());

    for (Label l: common_labels) {

      if (min_label == l)
        continue;

      candidate_pairs.emplace(LabelPair(min_label, l), 0);
    }
  }

  // increase the confidence for all observed candidate pairs
  for (auto label_pair: candidate_pairs) {
    auto it = label_pair_confidences_.find(label_pair.first);

    if (it == label_pair_confidences_.end())
      label_pair_confidences_.emplace(label_pair.first, 1);
    else
      it->second++;
  }

  auto it = label_pair_confidences_.begin();

  while (it != label_pair_confidences_.end()) {

    // decrease the confidence for all unobserved candidate pairs
    if (candidate_pairs.count(it->first) == 0)
      it->second--;

    std::cout << "label confidence of (" << it->first.first << ", " << it->first.second << ") is " << it->second << std::endl;

    // merge the labels if our confidence is high enough and erase the entry afterwards
    if (it->second >= config_.min_merge_confidence) {
      std::cout << "merging the labels" << std::endl;
      mergeSegmentLabels(propagated_labels, it->first);
      it = label_pair_confidences_.erase(it);
    } else if (it->second == 0) {
      std::cout << "forgetting the label pair" << std::endl;
      it = label_pair_confidences_.erase(it);
    } else {
      it++;
    }
  }

  segment_merge_candidates_.clear();
}

void SegmentedTsdfIntegrator::applyLabelToVoxels(const BlockIndex& block_idx, Label old_segment, Label new_segment) {

  if (!segmentation_layer_->hasBlock(block_idx))
    return;

  Block<SegmentedVoxel>& block = segmentation_layer_->getBlockByIndex(block_idx);

  for (int i = 0; i < block.num_voxels(); i++) {

    SegmentedVoxel& seg_voxel = block.getVoxelByLinearIndex(i);

    if (seg_voxel.segment_id == old_segment) {

      // TODO: use mutex
      // Lookup the mutex that is responsible for this voxel and lock it
      //std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

      seg_voxel.segment_id = new_segment;
    }
  }
}

void SegmentedTsdfIntegrator::mergeSegmentLabels(LabelIndexMap& propagated_labels, const LabelPair& label_pair) {

  Label target_label = label_pair.first;
  Label old_label = label_pair.second;

  for (const BlockIndex& block_idx: segment_blocks_map_[old_label]) {
    applyLabelToVoxels(block_idx, old_label, target_label);
  }

  segment_blocks_map_.erase(old_label);

  Labels& target_idxs  = propagated_labels[target_label];
  const Labels& source_idxs = propagated_labels[old_label];
  target_idxs.insert(target_idxs.end(), source_idxs.begin(), source_idxs.end());
  propagated_labels.erase(old_label);
}

}  // namespace voxblox
