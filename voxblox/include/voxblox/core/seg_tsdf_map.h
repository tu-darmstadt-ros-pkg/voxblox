#ifndef VOXBLOX_CORE_SEGMENTED_TSDF_MAP_H_
#define VOXBLOX_CORE_SEGMENTED_TSDF_MAP_H_

#include <glog/logging.h>
#include <memory>
#include <utility>

#include "voxblox/core/common.h"
#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"

namespace voxblox {

class SegmentedTsdfMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::shared_ptr<SegmentedTsdfMap> Ptr;

  explicit SegmentedTsdfMap(FloatingPoint tsdf_voxel_size, size_t tsdf_voxels_per_side)
      : seg_tsdf_layer_(new Layer<SegmentedVoxel>(tsdf_voxel_size,
                                                  tsdf_voxels_per_side)) {
    block_size_ = tsdf_voxel_size * tsdf_voxels_per_side;
  }

  // Creates a new SegmentedTsdfMap based on a COPY of this layer.
  explicit SegmentedTsdfMap(const Layer<SegmentedVoxel>& layer)
      : SegmentedTsdfMap(aligned_shared<Layer<SegmentedVoxel>>(layer)) {}

  // Creates a new SegmentedTsdfMap that contains this layer.
  explicit SegmentedTsdfMap(Layer<SegmentedVoxel>::Ptr layer)
      : seg_tsdf_layer_(layer) {
    if (!layer) {
      /* NOTE(mereweth@jpl.nasa.gov) - throw std exception for Python to catch
       * This is idiomatic when wrapping C++ code for Python, especially with
       * pybind11
       */
      throw std::runtime_error(std::string("Null Layer<SegmentedVoxel>::Ptr") +
                               " in TsdfMap constructor");
    }

    CHECK(layer);
    block_size_ = layer->block_size();
  }

  virtual ~SegmentedTsdfMap() {}

  Layer<SegmentedVoxel>* getTsdfLayerPtr() { return seg_tsdf_layer_.get(); }
  const Layer<SegmentedVoxel>& getTsdfLayer() const { return *seg_tsdf_layer_; }

  FloatingPoint block_size() const { return block_size_; }
  FloatingPoint voxel_size() const { return seg_tsdf_layer_->voxel_size(); }

  /* NOTE(mereweth@jpl.nasa.gov)
   * EigenDRef is fully dynamic stride type alias for Numpy array slices
   * Use column-major matrices; column-by-column traversal is faster
   * Convenience alias borrowed from pybind11
   */
  using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  template <typename MatrixType>
  using EigenDRef = Eigen::Ref<MatrixType, 0, EigenDStride>;

  /* Extract all voxels on a slice plane that is parallel to one of the
   * axis-aligned planes.
   * free_plane_index specifies the free coordinate (zero-based; x, y, z order)
   * free_plane_val specifies the plane intercept coordinate along that axis
   */
  unsigned int coordPlaneSliceGetDistanceWeight(
      unsigned int free_plane_index, double free_plane_val,
      EigenDRef<Eigen::Matrix<double, 3, Eigen::Dynamic>>& positions,
      Eigen::Ref<Eigen::VectorXd> distances,
      Eigen::Ref<Eigen::VectorXd> weights,
      unsigned int max_points) const;

 protected:
  FloatingPoint block_size_;

  // The layers.
  Layer<SegmentedVoxel>::Ptr seg_tsdf_layer_;
};

}  // namespace voxblox

#endif  // VOXBLOX_CORE_SEGMENTED_TSDF_MAP_H_
