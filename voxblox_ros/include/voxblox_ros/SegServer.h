#ifndef SEGSERVER_H_
#define SEGSERVER_H_

// CHECK includes TODO
#include <ros/ros.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include "voxblox/utils/color_maps.h"
#include "voxblox_ros/tsdf_server.h"

#include <opencv2/core/mat.hpp>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"

#include "voxblox_ros/SegToolbox.h"

#include <cv_bridge/cv_bridge.h>
#include "voxblox_ros/SegToolbox.h"

#include "voxblox_ros/SegDataFuser.h"

#include "std_msgs/String.h"

namespace voxblox {

template <class T>
class SegServer : public TsdfServer {
 public:
  SegServer(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);

  virtual ~SegServer() {}

  virtual void updateMesh();
  virtual void publishPointclouds();

  bool receivePointCloud(sensor_msgs::PointCloud2::Ptr cloud_msg);

  void receiveInitialSegmentation(std::shared_ptr<labelMap> label_map,
                                  std::shared_ptr<rawDataPointer> data);

  std::shared_ptr<Layer<T>> getLabelLayer();

  void initDataFuser(std::shared_ptr<SegDataFuser<T>> fuser);

  virtual void newPoseCallback(const Transformation& T_G_C_);

  void initLookup(std::shared_ptr<LabelLookup> lookup);

  void recolorVoxbloxMeshMsgByLabel(const Layer<T>& label_layer,
                                    const std::shared_ptr<ColorMap>& color_map,
                                    voxblox_msgs::Mesh* mesh_msg);

  void recolorVoxbloxMeshByLabel(const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map);

  void outputMeshMsgAsPly(std:: string filepath);

  //void recolorAndSaveMesh(const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map), std::string filepath;


 protected:
  std::shared_ptr<Layer<T>> label_layer;
  std::shared_ptr<ColorMap> color_map_;

  ros::Publisher label_pointcloud_pub_;
  ros::Publisher label_mesh_pub_;
  std::shared_ptr<rawDataPointer> data;

  ros::Subscriber mesh_saver;

  sensor_msgs::PointCloud2::Ptr old_cloud;

  std::shared_ptr<SegDataFuser<T>> data_fuser;

  std::shared_ptr<LabelLookup> lookup;

  Transformation T_G_C;

  bool initedLookup;

  ColorLookup clookup;

  bool debug_color_mesh;

  bool recolor_real_mesh; //TODO set to sth.

  std::string filepath_saver;

  int debug_count;

  int msg_count;

  double delta_time;
  double delta_time2;
  ros::WallTime last_t;
  bool time_init;

   bool skip_tsdf;

   bool aligned_rgbd;

   bool fix_color;
};

}  // namespace voxblox
#endif