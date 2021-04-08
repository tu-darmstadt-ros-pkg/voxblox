#ifndef SEG_EDGE_FUSER_H_
#define SEG_EDGE_FUSER_H_

#include "voxblox_ros/SegDataFuser.h"

namespace voxblox {
template <class T>
class SegEdgeFuser : public SegDataFuser<T> {
 public:
  SegEdgeFuser(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf,
               std::shared_ptr<Layer<T>> label);

  void fuse(Transformation T_G_C, std::shared_ptr<labelMap> lmap,
            std::shared_ptr<rawDataPointer> data);
  void fuseEdgeMap(Transformation T_G_C, std::shared_ptr<labelMap> lmap,
                   std::shared_ptr<rawDataPointer> data);
  // std::shared_ptr<normalMap> calculateNormals(std::shared_ptr<vertexMap>
  // vertices); std::shared_ptr<normalMap>
  // calculateNormalsCrossproduct(std::shared_ptr<vertexMap> vertices);
  // std::shared_ptr<normalMap>
  // calculateNormalsCentralDifferences(std::shared_ptr<vertexMap> vertices);
  // std::shared_ptr<normalMap>
  // calculateNormalsVoxblox(std::shared_ptr<vertexMap> vertices, const
  // Layer<TsdfVoxel>& tsdf_layer);

  void set_integrator(std::shared_ptr<SegTsdfIntegrator<T>> integrator);

 protected:
  bool show_atan_image;
  bool show_pcl_image;
  bool show_pcl_debug;

  bool edge_debug_images;
  bool to_refine_edge_image;
  bool no_blur_refine;

  NormalCalculationMode normal_calculation;

  double thresh_concave;
  double thresh_distance;
  double max_dist_;
  int window_size;

  ros::Publisher pcl_normal_pub;
  ros::Publisher atan_img_pub;

  ros::Publisher concave_im_pub;
  ros::Publisher distance_im_pub;
  ros::Publisher merged_im_pub;
  ros::Publisher labeled_im_pub;
  const Layer<TsdfVoxel>& tsdf_layer;

  std::shared_ptr<SegTsdfIntegrator<T>> integrator;

  int downsample_datafuser_factor;

  double enforce_percentage;

  int bilateral_filter_size;
  double bilateral_filter_diff;

   bool aligned_rgbd;
};
}  // namespace voxblox

#endif