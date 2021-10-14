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
    : DynamicMappingServer(nh, nh_private,
                 getDynamicMapperConfigFromRosParam(nh_private)) {}

DynamicMappingServer::DynamicMappingServer(const ros::NodeHandle& nh,
                       const ros::NodeHandle& nh_private,
                       const DynamicMapper::Config& dynamic_mapperconfig)
    : nh_(nh),
      nh_private_(nh_private),
      verbose_(true),
      world_frame_("world"),
      dynamic_mapper(dynamic_mapperconfig),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      cache_mesh_(false),
      pointcloud_queue_size_(1),
      transformer_(nh, nh_private)
      {
  getServerConfigFromRosParam(nh_private);

  nh_private.param<std::vector<std::string>>(
      "semantic_classes", semantic_classes_, semantic_classes_);

  nh_private.param<std::vector<int>>(
      "non_rigid_classes", non_rigid_classes_, non_rigid_classes_);
  dynamic_mapper.setNonRigidClasses(non_rigid_classes_);

  nh_private_.param("pointcloud_queue_size", pointcloud_queue_size_,
                    pointcloud_queue_size_);
  std::string pointcloud_topic;
  nh_private_.param("pointcloud_topic", pointcloud_topic,
                    pointcloud_topic);

  pointcloud_sub_ = nh_.subscribe(pointcloud_topic, pointcloud_queue_size_,
                                  &DynamicMappingServer::DynamicMappingCallback,
                                                                          this);
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  multi_mesh_pub_ = nh_private_.advertise<voxblox_msgs::MultiMesh>(
                                              "multi_mesh", 1, true);

  bbox_vis_pub_ = nh_private_.advertise<
                      jsk_recognition_msgs::BoundingBoxArray>( "bbox_vis", 0 );
  label_vis_pub_ = nh_private_.advertise<
                            visualization_msgs::MarkerArray>( "label_vis", 0 );

  // Advertise services.
  generate_mesh_srv_ = nh_private_.advertiseService(
      "save_meshes", &DynamicMappingServer::saveMeshCallback, this);

  generate_obj_trajectories_srv_ = nh_private_.advertiseService(
      "save_object_trajectories", &DynamicMappingServer::saveObjectTrajectories,
                                                                          this);
}

void DynamicMappingServer::getServerConfigFromRosParam(
    const ros::NodeHandle& nh_private) {

  nh_private.param("max_block_distance_from_body",
                   max_block_distance_from_body_,
                   max_block_distance_from_body_);
  nh_private.param("world_frame", world_frame_, world_frame_);

  nh_private.param("pointcloud_queue_size", pointcloud_queue_size_,
                   pointcloud_queue_size_);

  nh_private.param("verbose", verbose_, verbose_);

  // Mesh settings.
  std::string color_mode("");
  nh_private.param("meshing/color_mode", color_mode, color_mode);
  color_mode_ = getColorModeFromString(color_mode);

}

void DynamicMappingServer::DynamicMappingCallback(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg) {
      timing::Timer total_timer("total");
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

    timing::Timer dis_timer("distribution");
    dynamic_mapper.distributeInputCloud(pointcloud_pcl);
    dis_timer.Stop();

    timing::Timer icp_timer("align");
    dynamic_mapper.align(T_G_C_);
    icp_timer.Stop();

    timing::Timer integrate_timer("integrate_total");
    dynamic_mapper.integrate(T_G_C_);
    integrate_timer.Stop();

    timing::Timer block_remove_timer("remove_distant_blocks");
    dynamic_mapper.clearDistant(T_G_C_, max_block_distance_from_body_);
    block_remove_timer.Stop();

    timing::Timer generate_mesh_timer("mesh/update");
    dynamic_mapper.generateMesh();
    generate_mesh_timer.Stop();

    timing::Timer updatestate("object_stateupdate");
    dynamic_mapper.updateObjectStates(T_G_C_, pointcloud_msg);
    updatestate.Stop();

    timing::Timer publish_mesh_timer("mesh/publish");
    updateMesh();
    publish_mesh_timer.Stop();

    total_timer.Stop();

    visualizeBBoxes(pointcloud_msg);

    if (verbose_) {
      ROS_INFO_STREAM("Timings: " << std::endl << timing::Timing::Print());
    }
}

void DynamicMappingServer::updateMesh() {

  voxblox_msgs::MultiMesh multi_mesh_msg;

  voxblox_msgs::Mesh bg_mesh_msg;
  bg_mesh_msg.object_id = 0;

  generateVoxbloxMeshMsg(dynamic_mapper.getBackgroundMeshLayer(), color_mode_,
                                                          &bg_mesh_msg, false);
  multi_mesh_msg.meshes.push_back(bg_mesh_msg);

  for (auto object : dynamic_mapper.getObjects()){
    if (!object.isActive()) continue;
    voxblox_msgs::Mesh mesh_msg;
    mesh_msg.object_id = object.getID();
    mesh_msg.alpha = std::numeric_limits<uint8_t>::max();
    generateVoxbloxMeshMsg(object.getMeshLayer(), color_mode_, &mesh_msg, true);
    Eigen::Matrix4f transformation  = object.getTransformation();

    Eigen::Matrix4f result_transform =  transformation.inverse() ;

    Eigen::Quaternionf quat = Eigen::Quaternionf(
                          result_transform.block<3,3>(0,0));

    pcl::PointXYZRGBNormal origMinPoint, origMaxPoint;
    pcl::getMinMax3D(*object.getMeshCloud(), origMinPoint, origMaxPoint);

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

void DynamicMappingServer::visualizeBBoxes(
        const sensor_msgs::PointCloud2::Ptr& pointcloud_msg){

  jsk_recognition_msgs::BoundingBoxArray bboxes;
  visualization_msgs::MarkerArray label_marker;
  bboxes.header.frame_id = world_frame_;
  bboxes.header.stamp = pointcloud_msg->header.stamp;

  for (auto object : dynamic_mapper.getObjects()){

    Eigen::Matrix<float, 7, 1> object_state = object.getState();

    float bbox_x = object_state(0);
    float bbox_y = object_state(1);
    float bbox_z = object_state(2);
    float bbox_l = object_state(4);
    float bbox_w = object_state(5);
    float bbox_h = object_state(6);
    float bbox_angle = object_state(3);

    visualization_msgs::Marker marker;
    marker.header.frame_id = world_frame_;
    marker.header.stamp = ros::Time();
    marker.ns = "marker";
    marker.id = object.getID();
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = bbox_x;
    marker.pose.position.y = bbox_y;
    marker.pose.position.z = bbox_z + bbox_h/2.0 + 0.5;
    marker.pose.orientation.w = 1.0;

    marker.scale.z = .4;

    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    std::string marker_text = "";
    if (semantic_classes_[object.getSemanticClass() + 1] != "undefined"){
      marker_text += "Class: " +
                    semantic_classes_[object.getSemanticClass() + 1];
    }
    // marker_text += "\nID: " +  std::to_string(object.getID());
    marker.text = marker_text;

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
    bbox.label = object.getID();
    bboxes.boxes.push_back(bbox);
  }
  bbox_vis_pub_.publish( bboxes );
  label_vis_pub_.publish(label_marker);
}

bool DynamicMappingServer::saveMeshCallback(
                                          std_srvs::Empty::Request& /*request*/,
                                          std_srvs::Empty::Response&
                                          /*response*/) {

  timing::Timer output_mesh_timer("mesh/output");
  std::string bg_mesh_filename = "bg_mesh";

  bool success = outputMeshLayerAsPly(bg_mesh_filename,
                                    *dynamic_mapper.getBackgroundMeshLayer());

  if (success) {
    ROS_INFO("Output background mesh as PLY: %s", bg_mesh_filename.c_str());
  } else {
    ROS_INFO("Failed to output background mesh as PLY: %s",
                                                    bg_mesh_filename.c_str());
  }

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

bool DynamicMappingServer::saveObjectTrajectories(std_srvs::Empty::Request& /*request*/,
                                          std_srvs::Empty::Response&
                                          /*response*/) {

  boost::filesystem::create_directory("object_trajectories_improved");

  for (auto object : dynamic_mapper.getObjects()){

    int object_id = object.getID();
    boost::filesystem::create_directory("object_trajectories_improved/object_" +
                         std::to_string(object_id));


    std::string traj_filename = "object_trajectories_improved/object_" +
                         std::to_string(object_id) + "/stamped_traj_estimate.txt";
    std::ofstream stream(traj_filename.c_str());

    stream << "# timestamp tx ty tz qx qy qz qw" << std::endl;

    for (auto traj_pose : object.getTrajectory()) {
      double secs = traj_pose.header.stamp.toSec();
      stream << secs << " " <<
                traj_pose.pose.position.x << " " <<
                traj_pose.pose.position.y << " " <<
                traj_pose.pose.position.z << " " <<
                traj_pose.pose.orientation.x << " " <<
                traj_pose.pose.orientation.y << " " <<
                traj_pose.pose.orientation.z << " " <<
                traj_pose.pose.orientation.w << std::endl;
    }
    ROS_INFO("Output object trajectory as txt: %s", traj_filename.c_str());

  }

  return true;

}

}  // namespace voxblox
