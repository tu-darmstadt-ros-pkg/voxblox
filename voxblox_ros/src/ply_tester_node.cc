#include <voxblox_ros/ply_reader.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "segmentation debug server");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  voxblox::PlyReader reader(nh);

  std::vector<voxblox::mvertex> m1 = reader.read_mesh("/home/frederik/hector/src/voxblox/voxblox_ros/mesh_results/mesh1.ply");

  std::vector<voxblox::mvertex> m2 = reader.read_mesh("/home/frederik/hector/src/voxblox/voxblox_ros/mesh_results/mesh2.ply");

  std::shared_ptr<voxblox::MeshLayer> layer2 = reader.mvertex_to_layer(m2);

  reader.compare_to_layer(m1, layer2);
}