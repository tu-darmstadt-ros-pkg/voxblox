#include "voxblox/core/voxel.h"
#include "voxblox_ros/SegDebugReceiver.h"
#include "voxblox_ros/SegEdgeFuser.h"
#include "voxblox_ros/SegServer.h"

#include "voxblox/integrator/SegTsdfIntegrator.h"

// TODO more options here + own template functions

int main(int argc, char** argv) {
  ros::init(argc, argv, "segmentation debug server");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  // TODO use nodehandle to adapt this
  std::string voxel_type;
  nh.param<std::string>("seg_voxel_type", voxel_type, "single");

  if (voxel_type.compare("single") == 0) {
    std::shared_ptr<voxblox::SegServer<voxblox::LabelVoxel>> node(
        new voxblox::SegServer<voxblox::LabelVoxel>(nh, nh_private));

    std::shared_ptr<voxblox::SegDebugReceiver<voxblox::LabelVoxel>> receiver;
    receiver.reset(new voxblox::SegDebugReceiver<voxblox::LabelVoxel>(
        nh, nh_private, "/debug/color/image", "/debug/depth/image",
        "/debug/color/info", "/debug/depth/info", "/debug/cnn/result"));
    receiver->setServerPointer(node);

    const voxblox::Layer<voxblox::TsdfVoxel>& tsdflayer =
        (node->getTsdfMapPtr())->getTsdfLayer();
    std::shared_ptr<voxblox::Layer<voxblox::LabelVoxel>> labellayer =
        node->getLabelLayer();

    std::shared_ptr<voxblox::SegEdgeFuser<voxblox::LabelVoxel>> fuser(
        new voxblox::SegEdgeFuser<voxblox::LabelVoxel>(nh, tsdflayer,
                                                       labellayer));
    node->initDataFuser(fuser);

    std::shared_ptr<voxblox::SegTsdfIntegrator<voxblox::LabelVoxel>> integrator(
        new voxblox::SegTsdfIntegrator<voxblox::LabelVoxel>(nh, tsdflayer,
                                                            labellayer));
    fuser->set_integrator(integrator);

    std::shared_ptr<voxblox::LabelLookup> lookup(new voxblox::LabelLookup());
    node->initLookup(lookup);
    receiver->initLookup(lookup);

    ros::spin();

  } else if (voxel_type.compare("multi") == 0) {
    std::shared_ptr<voxblox::SegServer<voxblox::MultiLabelVoxel>> node(
        new voxblox::SegServer<voxblox::MultiLabelVoxel>(nh, nh_private));

    std::shared_ptr<voxblox::SegDebugReceiver<voxblox::MultiLabelVoxel>>
        receiver;
    receiver.reset(new voxblox::SegDebugReceiver<voxblox::MultiLabelVoxel>(
        nh, nh_private, "/debug/color/image", "/debug/depth/image",
        "/debug/color/info", "/debug/depth/info", "/debug/cnn/result"));
    receiver->setServerPointer(node);

    const voxblox::Layer<voxblox::TsdfVoxel>& tsdflayer =
        (node->getTsdfMapPtr())->getTsdfLayer();
    std::shared_ptr<voxblox::Layer<voxblox::MultiLabelVoxel>> labellayer =
        node->getLabelLayer();

    std::shared_ptr<voxblox::SegEdgeFuser<voxblox::MultiLabelVoxel>> fuser(
        new voxblox::SegEdgeFuser<voxblox::MultiLabelVoxel>(nh, tsdflayer,
                                                            labellayer));
    node->initDataFuser(fuser);

    std::shared_ptr<voxblox::SegTsdfIntegrator<voxblox::MultiLabelVoxel>>
        integrator(new voxblox::SegTsdfIntegrator<voxblox::MultiLabelVoxel>(
            nh, tsdflayer, labellayer));
    fuser->set_integrator(integrator);

    std::shared_ptr<voxblox::LabelLookup> lookup(new voxblox::LabelLookup());
    node->initLookup(lookup);
    receiver->initLookup(lookup);

    ros::spin();

  } else {
    std::cout << "unknown seg voxel type" << std::endl;
  }

  return 0;
}