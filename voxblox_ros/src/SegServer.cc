#include <voxblox_ros/SegServer.h>
#include "std_msgs/String.h"
#include <voxblox/io/mesh_ply.h>

//#include <ofstream>

namespace voxblox {

template <class T>
SegServer<T>::SegServer(const ros::NodeHandle& nh,
                        const ros::NodeHandle& nh_private)
    : TsdfServer(nh, nh_private) {
  label_layer.reset(new Layer<T>(tsdf_map_->getTsdfLayer().voxel_size(),
                                 tsdf_map_->getTsdfLayer().voxels_per_side()));

  //todo
  //if(recolor_real_mesh || debug_color_mesh){
  //  cache_mesh_ = false;
  //}else{
    cache_mesh_ = true;
  //}
  color_map_.reset(new RainbowColorMap());
  color_map_->setMinValue(0.0f);
  color_map_->setMaxValue(50.0f);

  // Publishers for output. //like intensity
  label_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "labeled_pointcloud", 1, true);
  label_mesh_pub_ =
      nh_private_.advertise<voxblox_msgs::Mesh>("label_mesh", 1, true);

  initedLookup = false;

  debug_count = 0;

  msg_count = 0;

  delta_time = 0;
  delta_time2 = 0;
  time_init = false;

   nh.param<bool>("skip_tsdf", skip_tsdf, false);

  //mesh_saver = nh.subscribe("/debug/mesh_saver", 1, boost::bind(&SegServer<T>::outputMeshMsgAsPly, this, _1));

  //to color the mesh just with one color
  nh_.param<bool>("debug_color_mesh", debug_color_mesh, false);

  nh_.param<bool>("aligned_rgbd", aligned_rgbd, true);

  nh_.param<bool>("fix_color", fix_color, true);
  //DEBUG adding values to lookup, do from file later, TODO
  /*
  'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush'
    */

   recolor_real_mesh = false; //TODO set to sth.

  clookup.add("BG", 0, 0, 0);
  clookup.add("person", 0, 0, 64);
  clookup.add("bicycle", 0, 0, 128);
  clookup.add("car", 0, 0, 192);
  clookup.add("motorcycle", 0, 64, 0);
  clookup.add("airplane", 0, 64, 0);
  clookup.add("fire hydrant", 0, 64, 64);
  clookup.add("stop sign", 0, 64, 128);
  clookup.add("parking meter", 0, 64, 192);
  clookup.add("bench", 0, 128, 0);
  clookup.add("bird", 0, 128, 64);
  clookup.add("cat", 0, 128, 128);
  clookup.add("dog", 0, 128, 192);
  clookup.add("horse", 0, 192, 0);
  clookup.add("sheep", 0, 192, 64);
  clookup.add("cow", 0, 192, 128);
  clookup.add("elephant", 0, 192, 192);
  clookup.add("bear", 64, 0, 0);
  clookup.add("zebra", 64, 0, 64);
  clookup.add("giraffe", 64, 0, 128);
  clookup.add("backpack", 64, 0, 192);
  clookup.add("umbrella", 64, 64, 0);
  clookup.add("handbag", 64, 64, 64);
  clookup.add("tie", 64, 64, 128);
  clookup.add("suitcase", 64, 64, 192);
  clookup.add("frisbee", 64, 128, 0);
  clookup.add("skis", 64, 128, 64);
  clookup.add("snowboard", 64, 128, 128);
  clookup.add("sports ball", 64, 128, 192);
  clookup.add("kite", 64, 192, 0);
  clookup.add("baseball bat", 64, 192, 64);
  clookup.add("baseball glove", 64, 192, 128);
  clookup.add("skateboard", 64, 192, 192);
  clookup.add("surfboard", 128, 0, 0);
  clookup.add("tennis racket", 128, 0, 64);
  clookup.add("bottle", 128, 0, 128);
  clookup.add("wine glass", 128, 0, 192);
  clookup.add("cup", 128, 64, 0);
  clookup.add("fork", 128, 64, 64);
  clookup.add("knife",128, 64, 128);
  clookup.add("spoon", 128, 64, 192);
  clookup.add("bowl", 128, 128, 0);
  clookup.add("banana", 128, 128, 64);
  clookup.add("apple", 128, 128, 128);
  clookup.add("sandwich", 128, 128, 192);
  clookup.add("orange", 128, 192, 0);
  clookup.add("broccoli", 128, 192, 64);
  clookup.add("carrot", 128, 192, 128);
  clookup.add("hot dog", 128, 192, 192);
  clookup.add("pizza", 192, 0, 0);
  clookup.add("donut", 192, 0, 64);
  clookup.add("cake", 192, 0, 128);
  clookup.add("chair", 192, 0, 192);
  clookup.add("couch", 192, 64, 0);
  clookup.add("potted plant", 192, 64, 64);
  clookup.add("bed", 192, 64, 128);
  clookup.add("dining table", 192, 64, 192);
  clookup.add("toilet", 192, 128, 0);
  clookup.add("tv", 192, 128, 64);
  clookup.add("laptop", 192, 128, 128);
  clookup.add("mouse", 192, 128, 192);
  clookup.add("remote", 192, 192, 0);
  clookup.add("keyboard", 192, 192, 64);
  clookup.add("cell phone", 192, 192, 128);
  clookup.add("microwave", 192, 192, 192);
  clookup.add("oven", 255, 0, 0);
  clookup.add("toaster", 255, 0, 64);
  clookup.add("sink", 255, 0, 128);
  clookup.add("refrigerator", 255, 0, 192);
  clookup.add("book", 255, 64, 0);
  clookup.add("clock", 255, 64, 64);
  clookup.add("vase", 255, 64, 128);
  clookup.add("scissors", 255, 64, 192);
  clookup.add("teddy bear", 255, 128, 0);
  clookup.add("hair drier", 255, 128, 64);
  clookup.add("toothbrush", 255, 128, 128);

  // old_cloud = NULL;
}

template <class T>
void SegServer<T>::initDataFuser(std::shared_ptr<SegDataFuser<T>> fuser) {
  data_fuser = fuser;
}

template <class T>
std::shared_ptr<Layer<T>> SegServer<T>::getLabelLayer() {
  return label_layer;
}

template <class T>
void SegServer<T>::initLookup(std::shared_ptr<LabelLookup> lookup_) {
  lookup = lookup_;
  initedLookup = true;
}

template <class T>
bool SegServer<T>::receivePointCloud(sensor_msgs::PointCloud2::Ptr cloud_msg) {
  // cloud_msg->header = depth_img->header;
  // TODO, check if header is valid, else pass valid one

  // TODO other way???
  // Do this to avoid endless circle? (clouds not being removed from vxoblox
  // queue???)
  Transformation T_G_C_last;
  std::string world_frame_id = "odom";
  if (!transformer_.lookupTransform(cloud_msg->header.frame_id, world_frame_id,
                                    cloud_msg->header.stamp, &T_G_C_last)) {
    ROS_ERROR("Failed to get transform to world frame");
    return false;
  }

  std::cout << "inserting pointcloud " << std::endl;
  std::cout << cloud_msg->header.stamp << std::endl;

  if(!skip_tsdf){
    insertPointcloud(cloud_msg);

    
  }else{
    newPoseCallback(T_G_C_last);
    std::cout << "updated T_G_C" << std::endl;
  }

  old_cloud = cloud_msg;

  return true;
}

template <class T>
void SegServer<T>::receiveInitialSegmentation(
    std::shared_ptr<labelMap> label_map,
    std::shared_ptr<rawDataPointer> data_ptr) {
  data = data_ptr;

  //update time
  last_t = ros::WallTime::now();
  time_init = true;


  //DEBUG recoloring of tsdf
  if(!aligned_rgbd && fix_color){


  }

  //get time from ros like voxblox
  ros::WallTime begin = ros::WallTime::now();

  data_fuser->fuse(T_G_C, label_map, data);

  ros::WallTime end = ros::WallTime::now();
  delta_time += (end-begin).toSec();

  double dt = (end-begin).toSec();
  if(dt > delta_time2){
    delta_time2 = dt;
  }

  lookup->debugPrint();
  msg_count++;

  std::cout << msg_count << ". segmentation received by server." << std::endl;
  std::cout << "average integration time: " << delta_time/msg_count << " seconds" << std::endl;
  std::cout << "highest integration time: " << delta_time2 << " seconds" << std::endl;
}

template <class T>
void SegServer<T>::updateMesh() {
  debug_count++;
  std::cout << "update: " << debug_count << std::endl;
  TsdfServer::updateMesh();
  
  //TODO rethink how to save the mesh in the best way
  if(!time_init){
    time_init = true;
    last_t = ros::WallTime::now();
  }

  ros::WallTime current_t = ros::WallTime::now();

  //if 1 minute has passed without update
  std::cout << (current_t - last_t).toSec() << std::endl;
  //if(debug_count == 2500 || (current_t - last_t).toSec() > 180.0){
  if((current_t - last_t).toSec() > 180.0){
    ROS_INFO("finished TSDF creation, returning mesh");
    
    //ROS_INFO("saving mesh");
    //std::string t = "/home/frederik/mesh.ply";
    //outputMeshMsgAsPly(t);
    debug_count = 0;

    bool old = debug_color_mesh;
    //trying to recolor it before the save???
    //TODO delete later
    debug_color_mesh = false;
    recolorVoxbloxMeshByLabel(*label_layer, color_map_);
    Mesh connected;
    mesh_layer_->getMesh(&connected);
    outputMeshAsPly("/home/frederik/mesh1.ply", connected);

    debug_color_mesh = true;
    recolorVoxbloxMeshByLabel(*label_layer, color_map_);
    Mesh connected2;
    mesh_layer_->getMesh(&connected2);
    outputMeshAsPly("/home/frederik/mesh2.ply", connected2);

    debug_color_mesh = old;

    std::cout << "finished saving the mesh" << std::endl;

    last_t = ros::WallTime::now();
  }

  // Now recolor the mesh...
  // std::cout << "updating mesh with color" <<std::endl;
  timing::Timer publish_mesh_timer("label_mesh/publish");
  
  //TODO debug, trying recoloring the real mesh
  //
  //if(recolor_real_mesh || debug_color_mesh){
  //  recolorVoxbloxMeshByLabel(*label_layer, color_map_);
  //}else{
    recolorVoxbloxMeshMsgByLabel(*label_layer, color_map_, &cached_mesh_msg_);
  //}
  label_mesh_pub_.publish(cached_mesh_msg_);
  publish_mesh_timer.Stop();



}

// from intensity vis, fixed version by marius
template <class T>
void SegServer<T>::recolorVoxbloxMeshMsgByLabel(
    const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map,
    voxblox_msgs::Mesh* mesh_msg) {
  CHECK_NOTNULL(mesh_msg);
  CHECK(color_map);
  // std::cout << label_layer.getNumberOfAllocatedBlocks() << std::endl;
  // std::cout << tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() <<
  // std::endl; std::cout << "recoloring mesh" << std::endl;
  // Go over all the blocks in the mesh.
  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg->mesh_blocks) {
    // Look up verticies in the thermal layer.
    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {
      // only needed if color information was originally missing
      // mesh_block.r.resize(mesh_block.x.size());
      // mesh_block.g.resize(mesh_block.x.size());
      // mesh_block.b.resize(mesh_block.x.size());
      // std::cout << "iterating over voxel" << std::endl;

      // from segmentation server (variant from intensity doesn't work); ask
      // /rewrite TODO
      constexpr float point_conv_factor =
          2.0f / std::numeric_limits<uint16_t>::max();
      const float mesh_x =
          (static_cast<float>(mesh_block.x[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[0])) *
          mesh_msg->block_edge_length;
      const float mesh_y =
          (static_cast<float>(mesh_block.y[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[1])) *
          mesh_msg->block_edge_length;
      const float mesh_z =
          (static_cast<float>(mesh_block.z[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[2])) *
          mesh_msg->block_edge_length;

      /*const float mesh_x = static_cast<float>(mesh_block.x[vert_idx]);
      const float mesh_y = static_cast<float>(mesh_block.y[vert_idx]);
      const float mesh_z = static_cast<float>(mesh_block.z[vert_idx]);*/

      // std::cout << "server" << mesh_x << std::endl;
      const T* voxel =
          label_layer.getVoxelPtrByCoordinates(Point(mesh_x, mesh_y, mesh_z));
      /*if(voxel != nullptr){
          std::cout << voxel->weight << std::endl;
      }*/
      float epsilon = 0.0001;
      


        Color new_color;
        std::vector<int> color;

        if(debug_color_mesh){
          
          if (voxel != nullptr && voxel->weight >= (0.0 - epsilon)) {
            std::string label = lookup->get(voxel->label_id);

            color = clookup.get(label);            

            if(color.size() == 0){
              ROS_ERROR("Unknown label for coloring");
              color.push_back(0);
              color.push_back(0);
              color.push_back(0);
            }
          }
          else{
            color.push_back(0);
            color.push_back(0);
            color.push_back(0);

          }
          new_color.r = color[0];
          new_color.g = color[1];
          new_color.b = color[2];
          //new_color.alpha = 255;

        }else{

          if (voxel != nullptr && voxel->weight >= (0.0 - epsilon)) {
          // std::cout << "recolor" << std::endl;
          float label = 0.0f + voxel->label_id;
          // std::cout << label << std::endl;
          new_color = color_map->colorLookup(label);

          // get colors from mesh
          Color mesh_col;
          mesh_col.r = mesh_block.r[vert_idx];
          mesh_col.g = mesh_block.g[vert_idx];
          mesh_col.b = mesh_block.b[vert_idx];

          // convert to hue
          std::vector<uint> mesh_col_hue = rgbToHue(mesh_col);
          std::vector<uint> label_col_hue = rgbToHue(new_color);

          // merge
          label_col_hue[2] = mesh_col_hue[2];

          // reconvert
          new_color = hueToRgb(label_col_hue);
          }
          else{
            continue;
          }
        }
          mesh_block.r[vert_idx] = new_color.r;
          mesh_block.g[vert_idx] = new_color.g;
          mesh_block.b[vert_idx] = new_color.b;
          //mesh_block.a[vert_idx] = 255;
        
      
    }
  }

  
}

// from intensity server, TODO
template <class T>
void SegServer<T>::publishPointclouds() {
  // Create a pointcloud with color from labels.
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;  // TODO

  /*createLabelPointcloudFromLabelLayer(*labely_layer_, &pointcloud);

  pointcloud.header.frame_id = world_frame_id;
  label_pointcloud_pub_.publish(pointcloud);*/

  TsdfServer::publishPointclouds();
}

template <class T>
void SegServer<T>::newPoseCallback(const Transformation& T_G_C_) {
  T_G_C = T_G_C_;
}

//Recoding the ply creation from voxblox to get a mesh from an mesh msg
template <class T>
void SegServer<T>::outputMeshMsgAsPly(std::string filepath_saver){
  voxblox_msgs::Mesh mesh_msg = cached_mesh_msg_;
  //std::string filepath = msg->data.c_str();
  std::ofstream stream(filepath_saver);

  if(!stream){
    ROS_ERROR("couldn't open file");
    return;
  }

  size_t num_points = 0;

  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg.mesh_blocks) {
    // Look up verticies in the thermal layer.
    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {
      num_points++;
    }
  }

  stream << "ply" << std::endl;
  stream << "format ascii 1.0" << std::endl;
  stream << "element vertex " << num_points << std::endl;
  stream << "property float x" << std::endl;
  stream << "property float y" << std::endl;
  stream << "property float z" << std::endl;

  //TODO leave normals out and fix script from before
  stream << "property float normal_x" << std::endl;
  stream << "property float normal_y" << std::endl;
  stream << "property float normal_z" << std::endl;

  stream << "property uchar red" << std::endl;
  stream << "property uchar green" << std::endl;
  stream << "property uchar blue" << std::endl;
  stream << "property uchar alpha" << std::endl;

  //TODO maybe save traingles?
  stream << "element face " << 0 << std::endl;
  stream << "property list uchar int vertex_index" << std::endl;

  stream << "end_header" << std::endl;

  //resuing this from recolor
  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg.mesh_blocks) {
    // Look up verticies in the thermal layer.
    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {
      // only needed if color information was originally missing
      // mesh_block.r.resize(mesh_block.x.size());
      // mesh_block.g.resize(mesh_block.x.size());
      // mesh_block.b.resize(mesh_block.x.size());
      // std::cout << "iterating over voxel" << std::endl;

      // from segmentation server (variant from intensity doesn't work); ask
      // /rewrite TODO
      /*constexpr float point_conv_factor =
          2.0f / std::numeric_limits<uint16_t>::max();
      const float mesh_x =
          (static_cast<float>(mesh_block.x[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[0])) *
          mesh_msg->block_edge_length;
      const float mesh_y =
          (static_cast<float>(mesh_block.y[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[1])) *
          mesh_msg->block_edge_length;
      const float mesh_z =
          (static_cast<float>(mesh_block.z[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[2])) *
          mesh_msg->block_edge_length;*/

          //maybe needing above?

      stream << mesh_block.x[vert_idx] << " " << mesh_block.y[vert_idx] << " " << mesh_block.z[vert_idx];

      stream << " " << 0.6 << " " << 0.2 << " " << 0.2;
      stream << " " << mesh_block.r[vert_idx] << " " << mesh_block.g[vert_idx] << " " << mesh_block.b[vert_idx] << " " << 255;
      stream << std::endl;
    }
  }

/*for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg->mesh_blocks) {
    // Look up verticies in the thermal layer.
    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {
      stream << "3"
    }
}*/
  

//triangles?
  //don't know where this info is, maybe not known??
}

//oriented by mesh_layer, get Mesh
template <class T>
void SegServer<T>::recolorVoxbloxMeshByLabel(const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map){
  BlockIndexList meshes;
  mesh_layer_->getAllAllocatedMeshes(&meshes);

  if(meshes.empty()){
    ROS_ERROR("couldn't recolor mesh");
    return;
  }

  for(const BlockIndex& block_idx : meshes){
    Mesh::Ptr m = mesh_layer_->getMeshPtrByIndex(block_idx);

    if(!m->vertices.empty()){

      for(size_t i = 0; i < m->vertices.size(); i++){
        Point p = m->vertices[i];

        const T* voxel =
          label_layer.getVoxelPtrByCoordinates(p);

        float epsilon = 0.0001;
      


        Color new_color;
        std::vector<int> color;

        if(debug_color_mesh){
          
          if (voxel != nullptr && voxel->weight >= (0.0 - epsilon)) {
            std::string label = lookup->get(voxel->label_id);

            color = clookup.get(label);            

            if(color.size() == 0){
              ROS_ERROR("Unknown label for coloring");
              color.push_back(0);
              color.push_back(0);
              color.push_back(0);
            }
          }
          else{
            color.push_back(0);
            color.push_back(0);
            color.push_back(0);

          }
          new_color.r = color[0];
          new_color.g = color[1];
          new_color.b = color[2];
          //new_color.alpha = 255;

        }else{

          if (voxel != nullptr && voxel->weight >= (0.0 - epsilon)) {
          // std::cout << "recolor" << std::endl;
          float label = 0.0f + voxel->label_id;
          // std::cout << label << std::endl;
          new_color = color_map->colorLookup(label);

          // get colors from mesh
          Color mesh_col;
          mesh_col.r = m->colors[i].r;
          mesh_col.g = m->colors[i].g;
          mesh_col.b = m->colors[i].b;

          // convert to hue
          std::vector<uint> mesh_col_hue = rgbToHue(mesh_col);
          std::vector<uint> label_col_hue = rgbToHue(new_color);

          // merge
          label_col_hue[2] = mesh_col_hue[2];

          // reconvert
          new_color = hueToRgb(label_col_hue);
          }
        }
        //ROS_INFO("writing color");
        m->colors[i].r = new_color.r;
        m->colors[i].g = new_color.g;
        m->colors[i].b = new_color.b;
        m->colors[i].a = 255;

      }

    }
  }


}

/*template <class T>
void SegServer<T>::recolorAndSaveMesh(const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map, filepath){

}*/


template class SegServer<LabelVoxel>;
template class SegServer<MultiLabelVoxel>;

}  // namespace voxblox