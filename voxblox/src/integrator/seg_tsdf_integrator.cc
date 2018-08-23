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
    : config_(config), tsdf_layer_(tsdf_layer), segmentation_layer_(segmentation_layer), max_label_(0), num_frames_(0) {
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

// Updates segmented voxel. Thread safe.
void SegmentedTsdfIntegrator::updateSegmentedVoxel(const GlobalIndex& global_voxel_idx,
                                                   const Label& segment) {

  SegmentedVoxel* seg_voxel = segmentation_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);

  if (seg_voxel == nullptr)
    return;

  if (seg_voxel->segment_id == 0 || seg_voxel->confidence == 0) {
    seg_voxel->segment_id = segment;
    seg_voxel->confidence = 0;
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
                                                  const LabelIndexMap& segment_map,
                                                  const std::map<uint, Color>& color_map) {

  timing::Timer seg_get_visible_voxels_timer("seg_get_visible_voxels");

  CHECK_EQ(points_C.size(), segmentation.size());

  visible_voxels_.clear();
  segment_map_.clear();
  updated_segments_.clear();

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

  // remember all the updated segments
  for (auto segment: propagated_labels) {
    updated_segments_.emplace_back(segment.first);
  }

  timing::Timer seg_update_global_segments_timer("seg_update_global_segments");
  updateGlobalSegments(propagated_labels);
  seg_update_global_segments_timer.Stop();

  if (config_.write_debug_data_) {
    pcl::PointCloud<pcl::PointXYZRGB> visible_points;
    pcl::PointCloud<pcl::PointXYZRGB> prop_points_merged;
    pcl::PointXYZRGB p;

    for (auto item : segment_map_) {

      Color color;

      auto it = color_map.find(item.first);
      if (it != color_map.end()) {
        color = it->second;
      } else {
        color.r = static_cast<uint8_t>(rand() % 256);
        color.g = static_cast<uint8_t>(rand() % 256);
        color.b = static_cast<uint8_t>(rand() % 256);
      }

      p.r = color.r;
      p.g = color.g;
      p.b = color.b;

      for (auto index: item.second) {

        const Point& point_C = points_C[index];
        const Point point_G = T_G_C * point_C;

        p.x = point_G(0);
        p.y = point_G(1);
        p.z = point_G(2);

        visible_points.points.push_back(p);
      }
    }

    visible_points.width = 1;
    visible_points.height = visible_points.points.size();

    if (!visible_points.points.empty())
      pcl::io::savePCDFile("/home/marius/pcds/visible_points_" + std::to_string(num_frames_) + ".pcd", visible_points);

    for (auto item : propagated_labels) {

      Color color;

      auto it = color_map.find(item.first);
      if (it != color_map.end()) {
        color = it->second;
      } else {
        color.r = static_cast<uint8_t>(rand() % 256);
        color.g = static_cast<uint8_t>(rand() % 256);
        color.b = static_cast<uint8_t>(rand() % 256);
      }

      p.r = color.r;
      p.g = color.g;
      p.b = color.b;

      for (auto index: item.second) {

        const Point& point_C = points_C[index];
        const Point point_G = T_G_C * point_C;

        p.x = point_G(0);
        p.y = point_G(1);
        p.z = point_G(2);

        prop_points_merged.points.push_back(p);
      }
    }

    prop_points_merged.width = 1;
    prop_points_merged.height = prop_points_merged.points.size();

    if (!prop_points_merged.points.empty())
      pcl::io::savePCDFile("/home/marius/pcds/prop_merged_" + std::to_string(num_frames_) + ".pcd", prop_points_merged);
  }

  num_frames_ += 1;
}

void SegmentedTsdfIntegrator::updateCameraModel(const Eigen::Matrix<FloatingPoint, 2, 1>& resolution, double focal_length)
{
  depth_cam_model_.setIntrinsicsFromFocalLength(resolution, focal_length, config_.min_ray_length_m, config_.max_ray_length_m);
}

void SegmentedTsdfIntegrator::getVisibleVoxels(const Transformation& T_G_C,
                                                         const Pointcloud& points_C,
                                                         ThreadSafeIndex* index_getter) {

  DCHECK(index_getter != nullptr);

  const Point& origin = T_G_C.getPosition();
  const FloatingPoint max_distance = config_.max_ray_length_m;

  for (size_t i = 0; i < points_C.size(); ++i) {
    Point surface_intersection = Point::Zero();

    const Point& point_C = points_C[i];

    if (!isPointValid(point_C)) {
      continue;
    }

    const Point point_G = T_G_C * point_C;

    Point direction = (point_G - origin).normalized();
    // Cast ray from the origin in sensor ray direction until
    // finding an intersection with a surface.
    bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(*tsdf_layer_, origin, direction,
                                                         max_distance, &surface_intersection);

    if (!success) {
      continue;
    }

    BlockIndex block_idx = segmentation_layer_->computeBlockIndexFromCoordinates(surface_intersection);
    Block<SegmentedVoxel>::Ptr block_ptr =
        segmentation_layer_->allocateBlockPtrByIndex(block_idx);

    GlobalIndex global_voxel_idx =
        getGridIndexFromPoint<GlobalIndex>(surface_intersection, voxel_size_inv_);

    SegmentedVoxel* voxel = segmentation_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);

    if (voxel == nullptr)
    {
      //std::cout << "[getVisibleVoxels] could not find voxel " << global_voxel_idx.transpose() << std::endl;
      continue;
    }

    visible_voxels_[i].emplace_back(global_voxel_idx);
    segment_map_[voxel->segment_id].emplace_back(i);
    segment_blocks_map_[voxel->segment_id].emplace(block_idx);

    // Now check the surrounding voxels along the bearing vector. If they have
    // never been observed, then fill in their value. Otherwise don't.
    Point close_voxel = surface_intersection;
    for (int voxel_offset = -config_.voxel_prop_radius;
         voxel_offset <= config_.voxel_prop_radius; voxel_offset++) {
      close_voxel =
          surface_intersection + direction * voxel_offset * voxel_size_;

      BlockIndex close_block_idx = segmentation_layer_->computeBlockIndexFromCoordinates(close_voxel);
      GlobalIndex close_global_voxel_idx =
          getGridIndexFromPoint<GlobalIndex>(close_voxel, voxel_size_inv_);

      SegmentedVoxel* close_voxel = segmentation_layer_->getVoxelPtrByGlobalIndex(close_global_voxel_idx);

      if (close_voxel == nullptr)
      {
        //std::cout << "[getVisibleVoxels] could not find voxel " << close_global_voxel_idx.transpose() << std::endl;
        continue;
      }

      visible_voxels_[i].emplace_back(close_global_voxel_idx);
      segment_blocks_map_[voxel->segment_id].emplace(close_block_idx);
    }
  }
}

LabelIndexMap SegmentedTsdfIntegrator::propagateSegmentLabels(const Labels& segmentation,
                                                              const LabelIndexMap& segment_map) {

  LabelIndexMap propagated_labels;
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

    std::cout << "max overlapping segment for segment " << img_segmentation.first << " (" << img_segmentation.second.size() << " points) is " << best_overlap_id <<
                 " (" << segment_map_[best_overlap_id].size() << " points) with an overlap of " << best_overlap << std::endl;

    // if we have enough overlap, keep the global label, otherwise propagate the label of the depth img
    if (best_overlap >= config_.min_segment_overlap && best_overlap_id != 0) {

      Labels& prop_idxs  = propagated_labels[best_overlap_id];
      //const Labels& glob_idxs  = global_segmentation.second;
      const Labels& glob_idxs = img_segmentation.second;

      prop_idxs.insert(prop_idxs.end(), glob_idxs.begin(), glob_idxs.end());

    } else {
      Labels& prop_idxs  = propagated_labels[max_label_++];
      const Labels& img_idxs = img_segmentation.second;

      prop_idxs.insert(prop_idxs.end(), img_idxs.begin(), img_idxs.end());
    }
  }

  return propagated_labels;
}

void SegmentedTsdfIntegrator::updateGlobalSegments(const LabelIndexMap& propagated_labels) {
  for (const auto& segment : propagated_labels) {
    for (const auto& point_idx: segment.second) {
      auto it = visible_voxels_.find(point_idx);
      if (it != visible_voxels_.end()) {
        for (const auto& voxel_idx: it->second) {
          updateSegmentedVoxel(voxel_idx, segment.first);
        }
      }
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

  return static_cast<float>(intersection_indices.size()) / static_cast<float>(segment2.size());
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

  block.set_updated(true);

  for (size_t i = 0; i < block.num_voxels(); i++) {

    SegmentedVoxel& seg_voxel = block.getVoxelByLinearIndex(i);

    if (seg_voxel.segment_id == old_segment) {
      seg_voxel.segment_id = new_segment;
    }
  }
}

void SegmentedTsdfIntegrator::mergeSegmentLabels(LabelIndexMap& propagated_labels, const LabelPair& label_pair) {

  Label target_label = label_pair.first;
  Label old_label = label_pair.second;

  for (const BlockIndex& block_idx: segment_blocks_map_[old_label]) {
    applyLabelToVoxels(block_idx, old_label, target_label);
    segment_blocks_map_[target_label].emplace(block_idx);
  }

  segment_blocks_map_.erase(old_label);

  Labels& target_idxs  = propagated_labels[target_label];
  const Labels& source_idxs = propagated_labels[old_label];
  target_idxs.insert(target_idxs.end(), source_idxs.begin(), source_idxs.end());
  propagated_labels.erase(old_label);
}

}  // namespace voxblox
