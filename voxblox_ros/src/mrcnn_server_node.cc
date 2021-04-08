#include "voxblox_ros/mrcnn_server.h"
#include "voxblox_ros/mrcnn_receiver.h"
#include "voxblox/integrator/tsdf_seq_integrator.h"


int main(int argc, char** argv){
    ros::init(argc, argv, "voxblox_mrcnn");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    voxblox::MrcnnServer node(nh, nh_private);
    
    ros::spin();
    return 0;
}