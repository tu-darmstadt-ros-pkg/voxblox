#include "voxblox_rviz_plugin/voxblox_mesh_display.h"

#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>

#include <tf/transform_listener.h>

#include <rviz/frame_manager.h>
#include <rviz/visualization_manager.h>

namespace voxblox_rviz_plugin {

VoxbloxMeshDisplay::VoxbloxMeshDisplay() {}

void VoxbloxMeshDisplay::onInitialize() {
  MFDClass::onInitialize();
  visual_.reset(
      new VoxbloxMeshVisual(context_->getSceneManager(), scene_node_));
}

VoxbloxMeshDisplay::~VoxbloxMeshDisplay() {}

void VoxbloxMeshDisplay::reset() {
  MFDClass::reset();
  visual_.reset(
      new VoxbloxMeshVisual(context_->getSceneManager(), scene_node_));
}

void VoxbloxMeshDisplay::processMessage(
    const voxblox_msgs::Mesh::ConstPtr& msg) {
  new_msg_ = msg;
}

void VoxbloxMeshDisplay::update(float wall_dt, float ros_dt) {
  Display::update(wall_dt, ros_dt);
  if (visual_ == nullptr) return;
  if (new_msg_ == nullptr) return;

  Ogre::Quaternion orientation;
  Ogre::Vector3 position;
  if (!context_->getFrameManager()->getTransform(
          new_msg_->header.frame_id, new_msg_->header.stamp, position, orientation)) {
    ROS_WARN("Error transforming from frame '%s' to frame '%s'",
              new_msg_->header.frame_id.c_str(), qPrintable(fixed_frame_));
    return;
  }

  visual_->setMessage(new_msg_);
  // Now set or update the contents of the chosen visual.
  visual_->setFramePosition(position);
  visual_->setFrameOrientation(orientation);
  new_msg_ = nullptr;
}

}  // namespace voxblox_rviz_plugin

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(voxblox_rviz_plugin::VoxbloxMeshDisplay, rviz::Display)
