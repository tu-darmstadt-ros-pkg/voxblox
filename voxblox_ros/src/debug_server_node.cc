#include "voxblox_ros/tsdf_seq_server.h"
#include "voxblox_ros/debug_receiver.h"
#include "voxblox/integrator/tsdf_seq_integrator.h"
#include "voxblox/core/voxel.h"

int main(int argc, char** argv){

   ros::init(argc, argv, "voxblox_tsdf_segmentation_server");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    voxblox::TsdfSeqServer<voxblox::LabelVoxel> node(nh, nh_private);
    /*voxblox::MrcnnReceiver rec(nh, nh_private,
        "/front_rgbd_cam/color/image_rect_color", 
        "/front_rgbd_cam/depth/image_raw",
        "/front_rgbd_cam/color/camera_info",
        "/front_rgbd_cam/depth/camera_info");*/
    std::shared_ptr<voxblox::DebugReceiver> receiver;

    //TODO set scale (can't be done automatic)
    //receiver->scale = 1.25;

    receiver.reset(new voxblox::DebugReceiver(nh, nh_private,
        //"/front_rgbd_cam/color/image_rect_color", 
        //"/front_rgbd_cam/depth/image_raw",
        //"/front_rgbd_cam/color/camera_info",
        //"/front_rgbd_cam/depth/camera_info"));
        "/debug/color/image", "/debug/depth/image", "/debug/color/info", "/debug/depth/info", "/debug/cnn/result"));
    
    std::shared_ptr<voxblox::TsdfSeqIntegrator<voxblox::LabelVoxel>> integrator;

    std::string mode;
    nh.param<std::string>("integration_mode", mode, "DEFAULT");
    integrator.reset(new voxblox::TsdfSeqIntegrator<voxblox::LabelVoxel>(nh, node.getTsdfMapPtr()->getTsdfLayer(), node.getLabelLayer().get(), mode));

    node.initialize(receiver, integrator);

    ros::spin();
    return 0; 
}