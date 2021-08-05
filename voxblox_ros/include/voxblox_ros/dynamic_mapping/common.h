#ifndef VOXBLOX_ROS_DYNAMIC_MAPPING_COMMON_H_
#define VOXBLOX_ROS_DYNAMIC_MAPPING_COMMON_H_

#include <pcl/point_types.h>


struct PointXYZRGBNormalCID {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB;
  int semantic_class;
  int id;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBNormalCID,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(float, rgb, rgb)
                                (int, semantic_class, semantic_class)
                                  (int, id, id))
typedef PointXYZRGBNormalCID InputPointType;

#endif  // VOXBLOX_ROS_DYNAMIC_MAPPING_COMMON_H_
