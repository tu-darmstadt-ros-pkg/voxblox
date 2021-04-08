#ifndef VOXBLOX_ROS_MRCNN_SEGMENTER_H_
#define VOXBLOX_ROS_MRCNN_SEGMENTER_H_

#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <std_srvs/Empty.h>
#include <visualization_msgs/MarkerArray.h>
#include <memory>
#include <string>

#include <pcl/features/integral_image_normal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/io/png_io.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/rgbd.hpp>
#include <opencv2/ximgproc.hpp>

#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/io/layer_io.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox_msgs/Mesh.h>

#include <voxblox_msgs/FilePath.h>
#include "voxblox_ros/mesh_vis.h"
#include "voxblox_ros/ptcloud_vis.h"
#include "voxblox_ros/transformer.h"
#include "voxblox_ros/image_operations.h"

namespace voxblox{

class MrcnnSegmenter {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    public:
    MrcnnSegmenter(const ros::NodeHandle& nh, float voxel_size);

    typedef boost::shared_ptr<cv_bridge::CvImage> CvImagePtr;
    typedef boost::shared_ptr<cv_bridge::CvImage const> CvImageConstPtr;

    typedef pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ> Octree;

    void transformMaskToMap(const cv::Mat& mask_img, const sensor_msgs::CameraInfoConstPtr& mask_camera_info_msg,
                                const cv::Mat& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_cam_info_msg,
                                const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, Pointcloud& cloud_out, LabelIndexMap& segment_map);

    //Color getSegmentColor(uint segment);
    Color getSegmentColor(uint segment);

    int getNormalsWindowSize() const { return normals_window_size_; }
    const std::map<uint, Color>& getColorMap() { return segment_colors_; }

    protected:

        std::map<uint, Color> segment_colors_;
        void initColorMap(int num_entries);
       
      
    void applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, const cv::Mat& segmentation_img, Pointcloud& cloud_out, LabelIndexMap& segment_map);

    ros::NodeHandle nh_private_;
    float voxel_size_;

    int edges_window_size_;
    int normals_window_size_;

    std::pair<ushort, int> getMostCommonLabel(const std::unordered_map<ushort, int>& x);

};



} //end namespace voxblos

#endif //VOXBLOX_ROS_MRCNN_SEGMENTER_H_