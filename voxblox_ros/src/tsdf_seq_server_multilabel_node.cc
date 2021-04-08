#include "voxblox_ros/tsdf_seq_server.h"
#include "voxblox_ros/mrcnn_receiver.h"
#include "voxblox/integrator/tsdf_seq_integrator.h"
#include "voxblox/core/voxel.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "voxblox_tsdf_segmentation_server");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    voxblox::TsdfSeqServer<voxblox::MultiLabelVoxel> node(nh, nh_private);
    /*voxblox::MrcnnReceiver rec(nh, nh_private,
        "/front_rgbd_cam/color/image_rect_color", 
        "/front_rgbd_cam/depth/image_raw",
        "/front_rgbd_cam/color/camera_info",
        "/front_rgbd_cam/depth/camera_info");*/
    std::shared_ptr<voxblox::MrcnnReceiver> receiver;
    receiver.reset(new voxblox::MrcnnReceiver(nh, nh_private,
        //"/front_rgbd_cam/color/image_rect_color", 
        //"/front_rgbd_cam/depth/image_raw",
        //"/front_rgbd_cam/color/camera_info",
        //"/front_rgbd_cam/depth/camera_info"));
        "/color/image", "/depth/image", "/color/info", "/depth/info"));
    
    std::shared_ptr<voxblox::TsdfSeqIntegrator<voxblox::MultiLabelVoxel>> integrator;
    std::string mode;
    nh.param<std::string>("integration_mode", mode, "DEFAULT");
    integrator.reset(new voxblox::TsdfSeqIntegrator<voxblox::MultiLabelVoxel>(nh, node.getTsdfMapPtr()->getTsdfLayer(), node.getLabelLayer().get(), mode));

    node.initialize(receiver, integrator);

    ros::spin();
    return 0;
}