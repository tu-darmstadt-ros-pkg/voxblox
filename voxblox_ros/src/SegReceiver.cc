#include "voxblox_ros/SegReceiver.h"

namespace voxblox {

SegReceiver::SegReceiver(const ros::NodeHandle& nh,
                         const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private), transformer_(nh, nh_private) {
  // instantiate subscribers and publisher here in subclass
  initedLookup = false;
}

/*void SegReceiver::setServerPointer(std::shared_ptr<SegServer<T>> server_ptr){
    server = server_ptr;
}*/

// callback
/*
 * synchronized callback from subscribers
 *
 * call create pointcloud
 *
 * give pointcloud to server
 *
 * do initial segmentation
 *
 * give initial segmentation to server
 *
 */

void SegReceiver::initLookup(std::shared_ptr<LabelLookup> lookup_) {
  lookup = lookup_;
  initedLookup = true;
}

}  // namespace voxblox