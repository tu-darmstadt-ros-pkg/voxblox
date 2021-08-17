#include "voxblox_ros/dynamic_mapping/dynamic_mapping_server.h"

#include <boost/filesystem.hpp>

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/ros_params.h"
#include "voxblox_ros/dynamic_mapping/ros_params_dynamic_mapping.h"

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace voxblox {

DynamicMappingServer::DynamicMappingServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private)
    : DynamicMappingServer(nh, nh_private, getTsdfMapConfigFromRosParam(nh_private),
                 getTsdfIntegratorConfigFromRosParam(nh_private),
                 getMeshIntegratorConfigFromRosParam(nh_private),
                 getDynamicMapperConfigFromRosParam(nh_private)) {}

DynamicMappingServer::DynamicMappingServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       const TsdfMap::Config& config,
                       const TsdfIntegratorBase::Config& integrator_config,
                       const MeshIntegratorConfig& mesh_config,
                       const DynamicMapper::Config& dynamic_mapperconfig)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(nh, nh_private),
      dynamic_mapper(dynamic_mapperconfig),
      offset_(0) {
  getServerConfigFromRosParam(nh_private);

  nh_private.param<std::vector<std::string>>(
      "semantic_classes", semantic_classes_, semantic_classes_);

  debug_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGBNormal> >(
          "debug_pointcloud", 1, true);

  nh_private_.param("pointcloud_queue_size", pointcloud_queue_size_,
                    pointcloud_queue_size_);
  std::string pointcloud_topic;
  nh_private_.param("pointcloud_topic", pointcloud_topic,
                    pointcloud_topic);

  pointcloud_sub_ = nh_.subscribe(pointcloud_topic, pointcloud_queue_size_,
                                  &DynamicMappingServer::DynamicMappingCallback, this);


  mesh_pub_ = nh_private_.advertise<voxblox_msgs::Mesh>("mesh", 1, true);
  multi_mesh_pub_ = nh_private_.advertise<voxblox_msgs::MultiMesh>("multi_mesh", 1, true);

  bbox_vis_pub_ = nh_private_.advertise<jsk_recognition_msgs::BoundingBoxArray>( "bbox_vis", 0 );
  label_vis_pub_ = nh_private_.advertise<visualization_msgs::MarkerArray>( "label_vis", 0 );

  // Advertise services.
  generate_mesh_srv_ = nh_private_.advertiseService(
      "save_background_mesh", &DynamicMappingServer::saveBackgroundMeshCallback, this);
  clear_map_srv_ = nh_private_.advertiseService(
      "clear_map", &DynamicMappingServer::clearMapCallback, this);


  // If set, use a timer to progressively integrate the mesh.
  double update_mesh_every_n_sec = 1.0;
  nh_private_.param("update_mesh_every_n_sec", update_mesh_every_n_sec,
                    update_mesh_every_n_sec);

  if (update_mesh_every_n_sec > 0.0) {
    update_mesh_timer_ =
        nh_private_.createTimer(ros::Duration(update_mesh_every_n_sec),
                                &DynamicMappingServer::updateMeshEvent, this);
  }

}

void DynamicMappingServer::getServerConfigFromRosParam(
    const ros::NodeHandle& nh_private) {
  // Before subscribing, determine minimum time between messages.
  // 0 by default.
  double min_time_between_msgs_sec = 0.0;
  nh_private.param("min_time_between_msgs_sec", min_time_between_msgs_sec,
                   min_time_between_msgs_sec);
  min_time_between_msgs_.fromSec(min_time_between_msgs_sec);

  nh_private.param("max_block_distance_from_body",
                   max_block_distance_from_body_,
                   max_block_distance_from_body_);
  nh_private.param("slice_level", slice_level_, slice_level_);
  nh_private.param("world_frame", world_frame_, world_frame_);
  nh_private.param("publish_pointclouds_on_update",
                   publish_pointclouds_on_update_,
                   publish_pointclouds_on_update_);
  nh_private.param("publish_slices", publish_slices_, publish_slices_);
  nh_private.param("publish_pointclouds", publish_pointclouds_,
                   publish_pointclouds_);

  nh_private.param("use_freespace_pointcloud", use_freespace_pointcloud_,
                   use_freespace_pointcloud_);
  nh_private.param("pointcloud_queue_size", pointcloud_queue_size_,
                   pointcloud_queue_size_);
  nh_private.param("enable_icp", enable_icp_, enable_icp_);
  nh_private.param("accumulate_icp_corrections", accumulate_icp_corrections_,
                   accumulate_icp_corrections_);

  nh_private.param("verbose", verbose_, verbose_);

  // Mesh settings.
  nh_private.param("meshing/mesh_filename", mesh_filename_, mesh_filename_);
  std::string color_mode("");
  nh_private.param("meshing/color_mode", color_mode, color_mode);
  color_mode_ = getColorModeFromString(color_mode);

}


void DynamicMappingServer::DynamicMappingCallback(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg) {

    pcl::PointCloud<InputPointType>::Ptr pointcloud_pcl(
                                      new pcl::PointCloud<InputPointType>());

    pcl::moveFromROSMsg(*pointcloud_msg, *pointcloud_pcl);

    if (!transformer_.lookupTransform(pointcloud_msg->header.frame_id,
                                world_frame_,
                                pointcloud_msg->header.stamp, &T_G_C_)){
            ROS_ERROR_STREAM("Error getting TF transform from frame "
                            << pointcloud_msg->header.frame_id << " to frame "
                                                      << world_frame_ << ".");
                  return;
    }

    dynamic_mapper.setInputCloud(pointcloud_pcl, T_G_C_);

    timing::Timer icp_timer("align");
    dynamic_mapper.align();
    icp_timer.Stop();

    timing::Timer integrate_timer("integrate");
    dynamic_mapper.integrate(T_G_C_);
    integrate_timer.Stop();

    timing::Timer block_remove_timer("remove_distant_blocks");
    dynamic_mapper.clearDistant(T_G_C_, max_block_distance_from_body_);
    block_remove_timer.Stop();

    timing::Timer generate_mesh_timer("mesh/update");
    dynamic_mapper.generateMesh();
    generate_mesh_timer.Stop();

    dynamic_mapper.updateObjectStates(T_G_C_);

    timing::Timer publish_mesh_timer("mesh/publish");
    updateMesh();
    publish_mesh_timer.Stop();

    visualizeBBoxes(pointcloud_msg->header.frame_id);

    if (verbose_) {
      ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    }
    //   ROS_INFO_STREAM(
    //       "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
    // }
}

void DynamicMappingServer::updateMesh() {
  if (verbose_) {
    // ROS_INFO("Updating mesh.");
  }



  voxblox_msgs::MultiMesh multi_mesh_msg;

  voxblox_msgs::Mesh bg_mesh_msg;
  bg_mesh_msg.object_id = 0;
  bg_mesh_msg.alpha = std::numeric_limits<uint8_t>::max();
  generateVoxbloxMeshMsg(dynamic_mapper.getBackgroundMeshLayer(), color_mode_, &bg_mesh_msg, false);
  multi_mesh_msg.meshes.push_back(bg_mesh_msg);

  // for (int i = 0; i < dynamic_mapper.getNumObjects(); i++){
  for (auto object : dynamic_mapper.getObjects()){
    voxblox_msgs::Mesh mesh_msg;
    mesh_msg.object_id = object.getID();
    mesh_msg.alpha = std::numeric_limits<uint8_t>::max();
    generateVoxbloxMeshMsg(object.getMeshLayer(), color_mode_, &mesh_msg, true);
    Eigen::Matrix4f transformation  = object.getTransformation();

    Eigen::Matrix4f result_transform =  T_G_C_.getTransformationMatrix() * transformation.inverse() ;

    Eigen::Quaternionf quat = Eigen::Quaternionf(result_transform.block<3,3>(0,0));

    pcl::PointXYZRGBNormal origMinPoint, origMaxPoint;
    pcl::getMinMax3D(*object.getMeshCloud(), origMinPoint, origMaxPoint);
    // std::cout<<"MESH TF CLOUD MID"<<object.getID()<<std::endl;
    // std::cout<<(origMaxPoint.x +  origMinPoint.x)/2.0<<" "
    //          <<(origMaxPoint.y +  origMinPoint.y)/2.0<<" "
    //          <<(origMaxPoint.z +  origMinPoint.z)/2.0<<" "<<std::endl;


    mesh_msg.transform.translation.x = result_transform(0,3);
    mesh_msg.transform.translation.y = result_transform(1,3);
    mesh_msg.transform.translation.z = result_transform(2,3);
    mesh_msg.transform.rotation.x = quat.x();
    mesh_msg.transform.rotation.y = quat.y();
    mesh_msg.transform.rotation.z = quat.z();
    mesh_msg.transform.rotation.w = quat.w();

    multi_mesh_msg.meshes.push_back(mesh_msg);
  }

  multi_mesh_msg.header.frame_id = world_frame_;
  multi_mesh_pub_.publish(multi_mesh_msg);

  if (cache_mesh_) {
    cached_mesh_msg_ = multi_mesh_msg;
  }




}

bool DynamicMappingServer::generateMesh() {
  timing::Timer generate_mesh_timer("mesh/generate");
  const bool clear_mesh = true;
  if (clear_mesh) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  } else {
    constexpr bool only_mesh_updated_blocks = true;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg, false);
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_.publish(mesh_msg);

  publish_mesh_timer.Stop();

  if (!mesh_filename_.empty()) {
    timing::Timer output_mesh_timer("mesh/output");
    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
    output_mesh_timer.Stop();
    if (success) {
      ROS_INFO("Output file as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output mesh as PLY: %s", mesh_filename_.c_str());
    }
  }

  ROS_INFO_STREAM("Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}


void DynamicMappingServer::visualizeBBoxes(std::string frame_id){

  jsk_recognition_msgs::BoundingBoxArray bboxes;
  visualization_msgs::MarkerArray label_marker;
  bboxes.header.frame_id = world_frame_;
  bboxes.header.stamp = ros::Time();

  for (auto object : dynamic_mapper.getObjects()){

    Eigen::Matrix4f transformation  = object.getTransformation();
    Eigen::Matrix4f result_transform =  T_G_C_.getTransformationMatrix() *
                                                  transformation.inverse();

    if (object.getMeshCloud()->points.empty()) continue;

    Eigen::Matrix<float, 7, 1> object_state = object.getState();

    float bbox_x = object_state(0);
    float bbox_y = object_state(1);
    float bbox_z = object_state(2);
    float bbox_l = object_state(4);
    float bbox_w = object_state(5);
    float bbox_h = object_state(6);
    float bbox_angle = object_state(3);

    // std::cout<<bbox_x<<" "<<bbox_y<<" "<<bbox_z<<std::endl;

    visualization_msgs::Marker marker;
    marker.header.frame_id = world_frame_;
    marker.header.stamp = ros::Time();
    marker.ns = "marker";
    marker.id = object.getID();
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = bbox_x;
    marker.pose.position.y = bbox_y;
    marker.pose.position.z = bbox_z + bbox_h/2.0 + 0.15;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.6;
    marker.scale.y = .15;
    marker.scale.z = .15;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.text = "Label: " + semantic_classes_[object.getSemanticClass() + 1]  +
                  "\nID: " +
                  std::to_string(object.getID());
    ros::Duration lifetime;
    marker.lifetime = lifetime.fromSec(2); // lifetime of 40ms : 25Hz
    label_marker.markers.push_back(marker);

    jsk_recognition_msgs::BoundingBox bbox;
    bbox.header.frame_id = world_frame_;
    bbox.header.stamp = ros::Time();

    bbox.pose.position.x = bbox_x;
    bbox.pose.position.y = bbox_y;
    bbox.pose.position.z = bbox_z;

    bbox.pose.orientation.x = 0;
    bbox.pose.orientation.y = 0;
    bbox.pose.orientation.z = std::sin(bbox_angle/2.0);
    bbox.pose.orientation.w = std::cos(bbox_angle/2.0);


    bbox.dimensions.x = bbox_l;
    bbox.dimensions.y = bbox_w;
    bbox.dimensions.z = bbox_h;

    bbox.value = 1.0;
    bbox.label = object.getSemanticClass();
    bboxes.boxes.push_back(bbox);
  }
  bbox_vis_pub_.publish( bboxes );
  label_vis_pub_.publish(label_marker);
}

bool DynamicMappingServer::saveBackgroundMeshCallback(
                                          std_srvs::Empty::Request& /*request*/,
                                          std_srvs::Empty::Response&
                                          /*response*/) {

  timing::Timer output_mesh_timer("mesh/output");
  if (!mesh_filename_.empty()) {


    const bool success = outputMeshLayerAsPly(mesh_filename_,
                                      *dynamic_mapper.getBackgroundMeshLayer());

    if (success) {
      ROS_INFO("Output background mesh as PLY: %s", mesh_filename_.c_str());
    } else {
      ROS_INFO("Failed to output background mesh as PLY: %s", mesh_filename_.c_str());
    }
  }
  bool success;
  boost::filesystem::create_directory("object_meshes");
  for (int i = 0; i < dynamic_mapper.getNumObjects(); i++){
    std::string mesh_filename = "object_meshes/object_" +
                        std::to_string(dynamic_mapper.getObjectID(i)) + ".ply";
    success = outputMeshLayerAsPly(mesh_filename,
                                      *dynamic_mapper.getObjectMeshLayer(i));
    if (success) {
      ROS_INFO("Output object mesh as PLY: %s", mesh_filename.c_str());
    } else {
      ROS_INFO("Failed to output object mesh as PLY: %s", mesh_filename.c_str());
    }

  }

  output_mesh_timer.Stop();
  return true;

}

bool DynamicMappingServer::saveObjectsCallback(std_srvs::Empty::Request& /*request*/,
                                     std_srvs::Empty::Response&
                                     /*response*/) {


  // std::map<ObjectID, ObjectVolume*>* object_volumes =
  //     map_->getObjectVolumesPtr();
  //
  // for (const auto& pair : *object_volumes) {
  //   if (!using_ground_truth_segmentation_ &&
  //       pair.second->getSemanticClass() == BackgroundClass &&
  //       pair.first != 2u) {
  //     continue;
  //   }
  //   CHECK_EQ(makePath("tpp_objects", 0777), 0);
  //
  //   std::string mesh_filename =
  //       "tpp_objects/tpp_object_" + std::to_string(pair.first) + ".ply";
  //
  //   bool success = voxblox::io::outputLayerAsPly(
  //       *pair.second->getTsdfLayerPtr(), mesh_filename,
  //       voxblox::io::PlyOutputTypes::kSdfIsosurface);
  //
  //   if (success) {
  //     LOG(INFO) << "Output object file as PLY: " << mesh_filename.c_str();
  //   } else {
  //     LOG(INFO) << "Failed to output mesh as PLY:" << mesh_filename.c_str();
  //   }
  // }
  //
  // return true;
}


bool DynamicMappingServer::clearMapCallback(std_srvs::Empty::Request& /*request*/,
                                  std_srvs::Empty::Response&
                                  /*response*/) {  // NOLINT
  clear();
  return true;
}

bool DynamicMappingServer::generateMeshCallback(std_srvs::Empty::Request& /*request*/,
                                      std_srvs::Empty::Response&
                                      /*response*/) {  // NOLINT
  return generateMesh();
}


void DynamicMappingServer::updateMeshEvent(const ros::TimerEvent& /*event*/) {
  // updateMesh();
}


void DynamicMappingServer::clear() {
  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
  mesh_layer_->clear();

}


}  // namespace voxblox
