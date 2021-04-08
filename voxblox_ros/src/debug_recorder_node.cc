#include "voxblox_ros/debug_recorder.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "debug_cnn_recorder");


    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    voxblox::DebugRecorder node(nh, nh_private, "/camera/rgb/image_color", "/camera/depth/image", "/camera/rgb/camera_info", "/camera/depth/camera_info");

    ros::spin();
    return 0;
}