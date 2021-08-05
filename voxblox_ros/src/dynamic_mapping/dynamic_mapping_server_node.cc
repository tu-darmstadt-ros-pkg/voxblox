#include "voxblox_ros/dynamic_mapping/dynamic_mapping_server.h"

#include <gflags/gflags.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "voxblox");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  voxblox::DynamicMappingServer node(nh, nh_private);

  ros::spin();
  return 0;
}
