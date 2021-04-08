#include "voxblox_ros/tsdf_seq_server.h"
#include <boost/bind.hpp>
#include <math.h>

namespace voxblox {


template<class T>
TsdfSeqServer<T>::TsdfSeqServer(const ros::NodeHandle& nh,
                             const ros::NodeHandle& nh_private)
    : TsdfServer(nh, nh_private) {
  label_layer.reset(
      new Layer<T>(tsdf_map_->getTsdfLayer().voxel_size(),
                            tsdf_map_->getTsdfLayer().voxels_per_side()));

  // rec.initialize(this);
  debug_pub =
      nh_.advertise<sensor_msgs::PointCloud2>("/tsdf_seq/debug_cloud", 2);

  cache_mesh_ = true;
  color_map_.reset(new RainbowColorMap());
  color_map_->setMinValue(0.0f);
  color_map_->setMaxValue(25.0f);

  // Publishers for output. //like intensity
  label_pointcloud_pub_ =
      nh_private_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>(
          "labeled_pointcloud", 1, true);
  label_mesh_pub_ =
      nh_private_.advertise<voxblox_msgs::Mesh>("label_mesh", 1, true);

  edge_image_pub = nh_private_.advertise<sensor_msgs::Image>("/debug/edge/image", 1, true);
  nh_.param<bool>("use_tsdf_segmentation", use_tsdf_segmentation, false);
  nh_.param<bool>("raw_image_segmentation", raw_image_segmentation, false);

  looked_up_tf = false;
}

template<class T>
void TsdfSeqServer<T>::initialize(std::shared_ptr<TsdfSeqReceiver> rec,
                               std::shared_ptr<TsdfSeqIntegrator<T>> integ) {
  receiver = rec;
  integrator = integ;
  //if(use_tsdf_segmentation){
  //  receiver->initialize_goemtric(boost::bind(&TsdfSeqServer<T>::receiver_geometric_callback, this, _1, _2, _3, _4, _5, _6, _7, _8));
  //  return;
  //}

  if (rec->projects_weight){
    receiver->initialize_weight3(boost::bind(&TsdfSeqServer<T>::receiver_weight_callback3, this, _1, _2, _3, _4, _5, _6, _7));
    ROS_INFO("using image projection");
    return;
  }

  if (rec->provides_weight_image) {
    /*receiver->initialize_weight(
        boost::bind(&TsdfSeqServer::receiver_weight_callback, this, _1, _2, _3,
                    _4, _5, _6, _7));*/
      receiver->initialize_weight2(boost::bind(&TsdfSeqServer<T>::receiver_weight_callback2, this, _1, _2, _3, _4, _5, _6, _7, _8));
  } else {
    receiver->initialize(
        boost::bind(&TsdfSeqServer<T>::receiver_callback, this, _1, _2, _3, _4));
  }

  if(rec->early_depth){
    receiver->init_early_depth_callback(
        boost::bind(&TsdfSeqServer<T>::depth_callback, this, _1, _2));
    
  }
}

template<class T>//TODO also early callback for color image/it's transform
void TsdfSeqServer<T>::depth_callback(const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_info){
  

  ROS_INFO("received early depth image");

  if (!transformer_.lookupTransform(depth_img->header.frame_id, world_frame_id,
                                      depth_img->header.stamp, &T_G_C_last)) {
      ROS_ERROR("Failed to get transform for segmented image");
      return;
  }

  //TODO skip the rest cycle in this case


  cv::Mat depth_img_mat;
  
  depth_img_mat = convertImagePtr(depth_img)->image;

  sensor_msgs::CameraInfoPtr depth_cam_info_ptr =
    boost::make_shared<sensor_msgs::CameraInfo>(*depth_info);

  // done similar to segmentation server
  // now convert image to coordinates/pointcloud and vector of labels? then add
  // them to map like intensity integrator?
  pcl::PointCloud<pcl::PointXYZRGB> cloud(
      static_cast<uint>(depth_img_mat.cols),
      static_cast<uint>(depth_img_mat.rows));

  convertToCloudMonoColor<uint16_t>(depth_img_mat, depth_cam_info_ptr,
                           cloud);

  sensor_msgs::PointCloud2::Ptr cloud_msg =
      boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header = depth_img->header;

  debug_pub.publish(cloud_msg);

  // give cloud to tsdf_server so the normal values are generated
  std::cout << "inserting pointcloud " << std::endl;
  insertPointcloud(cloud_msg);



  looked_up_tf = true;
}

template<class T>
void TsdfSeqServer<T>::receiver_callback(
    const sensor_msgs::ImageConstPtr& seg_img,
    const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info) {
  ROS_INFO("server callback");
}

template<class T>
void TsdfSeqServer<T>::receiver_weight_callback2(
    const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::ImageConstPtr& color_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info,
    const std::vector<std::vector<LabelVoxel>> label_map, 
    const std::vector<cv::Vec3f> coord_map, 
    const std::unordered_map<std::string, int> label_string_mapping,
    const bool aligned
){
  ROS_INFO("server callback");

  // set transform T_G_C ?? TODO
  if(!looked_up_tf){
    // from intensity server
    if (!transformer_.lookupTransform(depth_img->header.frame_id, world_frame_id,
                                      depth_img->header.stamp, &T_G_C_last)) {
      ROS_ERROR("Failed to get transform for segmented image");
      return;
    }
  
    cv::Mat depth_img_mat;
    cv::Mat color_img_mat;
  
    depth_img_mat = convertImagePtr(depth_img)->image;
    
    color_img_mat  = cv_bridge::toCvShare(color_img, sensor_msgs::image_encodings::TYPE_8UC3)->image;

    sensor_msgs::CameraInfoPtr depth_cam_info_ptr =
      boost::make_shared<sensor_msgs::CameraInfo>(*depth_info);

    // done similar to segmentation server
    // now convert image to coordinates/pointcloud and vector of labels? then add
    // them to map like intensity integrator?
    pcl::PointCloud<pcl::PointXYZRGB> cloud(
        static_cast<uint>(depth_img_mat.cols),
        static_cast<uint>(depth_img_mat.rows));


    std::cout << "pre crit2" << std::endl;
    std::cout << aligned << std::endl;
    if(color_img->height == depth_img->height && color_img->width == depth_img->width && aligned){
      convertToCloud<uint16_t>(depth_img_mat, color_img_mat, depth_cam_info_ptr,
                               cloud);
    }
    else{
      convertToCloudMonoColor<uint16_t>(depth_img_mat, depth_cam_info_ptr,
                               cloud);
    }

    sensor_msgs::PointCloud2::Ptr cloud_msg =
        boost::make_shared<sensor_msgs::PointCloud2>();
    pcl::toROSMsg(cloud, *cloud_msg);
    cloud_msg->header = depth_img->header;

    debug_pub.publish(cloud_msg);

    // give cloud to tsdf_server so the normal values are generated
    std::cout << "inserting pointcloud " << std::endl;
    insertPointcloud(cloud_msg);
  }else{
    ROS_INFO("depth image was processed before, skipping it now");
  }
  looked_up_tf = false;

  //now directly integrate the vectors
  std::cout << "before integrate" << std::endl;
  pcl::PointCloud<pcl::PointXYZRGB> cloud2 = integrator->directlyIntegrate(label_map, coord_map, T_G_C_last);

  //convert debug cloud
  sensor_msgs::PointCloud2 cl;
  pcl::toROSMsg(cloud2,cl);
  cl.header.stamp = ros::Time::now();
  cl.header.seq = depth_img->header.seq;
  cl.header.frame_id = "world";

  label_pointcloud_pub_.publish(cl);
}


template<class T>
void TsdfSeqServer<T>::receiver_weight_callback3(const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::ImageConstPtr& color_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info,
    const std::vector<std::vector<LabelVoxel>> label_map, 
    const std::unordered_map<std::string, int> label_string_mapping,
    const bool aligned
){

  //if(!looked_up_tf){ //TODO fix
    // from intensity server
    if (!transformer_.lookupTransform(color_info->header.frame_id, world_frame_id,
                                      color_info->header.stamp, &T_G_C_color_last)) {
      ROS_ERROR("Failed to get transform for segmented image");
      return;
    }

    uint32_t height = color_img->height;
    uint32_t width = color_img->width;
  
    cv::Mat depth_img_mat;
    cv::Mat color_img_mat;
  
    depth_img_mat = convertImagePtr(depth_img)->image;
    color_img_mat = color_img_mat  = cv_bridge::toCvShare(color_img, sensor_msgs::image_encodings::TYPE_8UC3)->image;

    sensor_msgs::CameraInfoPtr depth_cam_info_ptr =
      boost::make_shared<sensor_msgs::CameraInfo>(*depth_info);

    // done similar to segmentation server
    // now convert image to coordinates/pointcloud and vector of labels? then add
    // them to map like intensity integrator?
    pcl::PointCloud<pcl::PointXYZRGB> cloud(
        static_cast<uint>(depth_img_mat.cols),
        static_cast<uint>(depth_img_mat.rows));

    std::cout << "pre crit" << std::endl;
    std::cout << aligned << std::endl;
    if(color_img->height == depth_img->height && color_img->width == depth_img->width && aligned){
      convertToCloud<uint16_t>(depth_img_mat, color_img_mat, depth_cam_info_ptr,
                               cloud);
    }
    else{
      convertToCloudMonoColor<uint16_t>(depth_img_mat, depth_cam_info_ptr,
                               cloud);
    }

    sensor_msgs::PointCloud2::Ptr cloud_msg =
        boost::make_shared<sensor_msgs::PointCloud2>();
    pcl::toROSMsg(cloud, *cloud_msg);
    cloud_msg->header = depth_img->header;

    debug_pub.publish(cloud_msg);

    // give cloud to tsdf_server so the normal values are generated
    std::cout << "inserting pointcloud " << std::endl;
    insertPointcloud(cloud_msg);
  //}else{
    //ROS_INFO("depth image was processed before, skipping it now");
  //}
  looked_up_tf = false;

  double focal_length = color_info->K[0];
  double px = color_info->K[2];
  double py = color_info->K[5];

  //now directly integrate the vectors
  std::cout << "before integrate" << std::endl;

  //ROS_INFO("before decision");
  if(use_tsdf_segmentation){
  //  //sensor_msgs::Image debug = 
    integrator->integrateByGeometricSegmentation(color_img, color_info, label_map, T_G_C_color_last);
    
  }else if(raw_image_segmentation){
    integrator->integrateByGeometricSegmentationAligned(depth_img_mat, depth_info, label_map, T_G_C_color_last);
  }
    //debug, aligned
    //ROS_INFO("using aligned segmentation");
    
    
    //edge_image_pub.publish(debug);
  else{
    integrator->integrateByProjection(label_map, width, height, focal_length, px, py, T_G_C_color_last);
  }

  //convert debug cloud
  //sensor_msgs::PointCloud2 cl;
  //pcl::toROSMsg(cloud2,cl);
  //cl.header.stamp = ros::Time::now();
  //cl.header.seq = depth_img->header.seq;
  //cl.header.frame_id = "world";

  //label_pointcloud_pub_.publish(cl);

}

template<class T>
void TsdfSeqServer<T>::receiver_weight_callback(
    const sensor_msgs::ImageConstPtr& seg_img,
    const sensor_msgs::ImageConstPtr& depth_img,
    const sensor_msgs::CameraInfoConstPtr& color_info,
    const sensor_msgs::CameraInfoConstPtr& depth_info,
    const cv::Mat weight_image,
    const std::unordered_map<int, std::string> label_lookup_new,
    const std::unordered_map<int, std::vector<cv::Point>> label_map_new) {
  ROS_INFO("server callback");

  label_lookup = label_lookup_new;
  label_map = label_map_new;

  // now process and insert

  // set transform T_G_C ?? TODO
  Transformation T_G_C;  // from intensity server
  if (!transformer_.lookupTransform(depth_img->header.frame_id, world_frame_id,
                                    depth_img->header.stamp, &T_G_C)) {
    ROS_ERROR("Failed to get transform for segmented image");
    return;
    
  }

  cv::Mat depth_img_mat;
  cv::Mat seg_img_mat;

  // TODO check if they are the same size
  depth_img_mat = convertImagePtr(depth_img)->image;
  seg_img_mat = convertImagePtr(seg_img)->image;

  sensor_msgs::CameraInfoPtr depth_cam_info_ptr =
      boost::make_shared<sensor_msgs::CameraInfo>(*depth_info);

  // done similar to segmentation server
  // now convert image to coordinates/pointcloud and vector of labels? then add
  // them to map like intensity integrator?
  pcl::PointCloud<pcl::PointXYZRGB> cloud(
      static_cast<uint>(depth_img_mat.cols),
      static_cast<uint>(depth_img_mat.rows));

  convertToCloud<uint16_t>(depth_img_mat, seg_img_mat, depth_cam_info_ptr,
                           cloud);

  sensor_msgs::PointCloud2::Ptr cloud_msg =
      boost::make_shared<sensor_msgs::PointCloud2>();
  pcl::toROSMsg(cloud, *cloud_msg);
  cloud_msg->header = depth_img->header;

  debug_pub.publish(cloud_msg);

  // give cloud to tsdf_server so the normal values are generated
  insertPointcloud(cloud_msg);

  // also do stuff from tsdf integ? (also done there); try to imitate it but
  // make own label layer? now calling integrator to project segmentation on to
  // the surface of the tsdf
  calculateVectors(
      depth_img, T_G_C,
      color_info->K[0]);  // think about where to reuse LabelMap etc.
}

// like intensity server (ripped from there)
template<class T>
void TsdfSeqServer<T>::calculateVectors(
    const sensor_msgs::ImageConstPtr& depth_img, Transformation T_G_C,
    float focal_length_px) {
  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(depth_img);

  CHECK(cv_ptr);

  const size_t num_pixels =
      cv_ptr->image.rows * cv_ptr->image.cols;  // maybe subsample factor

  float half_row = cv_ptr->image.rows / 2.0;
  float half_col = cv_ptr->image.cols / 2.0;

  Pointcloud bearing_vectors;
  bearing_vectors.reserve(num_pixels + 1);
  std::vector<int> labels;
  labels.reserve(num_pixels + 1);

  size_t k = 0;
  size_t m = 0;
  //TODO maybe do this with a cloud(colored) instead
  for (int i = 0; i < cv_ptr->image.rows; i++) {
    const float* image_row = cv_ptr->image.ptr<float>(i);
    for (int j = 0; j < cv_ptr->image.cols; j++) {
      // subsample?
      bearing_vectors.push_back(
          T_G_C.getRotation().toImplementation() *
          Point(j - half_col, i - half_row, focal_length_px).normalized());
      labels.push_back((int)image_row[j]);
      //std::cout << "label " <<(int) image_row[i] << std::endl;
      k++;
    }
    m++;
  }

  // call integrator; also think about labelmap, weights etc. TODO
  integrator->addVectors(T_G_C.getPosition(), bearing_vectors, labels);
}

// like intensity
template<class T>
void TsdfSeqServer<T>::updateMesh() {
  TsdfServer::updateMesh();
  // Now recolor the mesh...
  //std::cout << "updating mesh with color" <<std::endl;
  timing::Timer publish_mesh_timer("label_mesh/publish");
  recolorVoxbloxMeshMsgByLabel(*label_layer, color_map_, &cached_mesh_msg_);
  label_mesh_pub_.publish(cached_mesh_msg_);
  publish_mesh_timer.Stop();
}

inline std::vector<uint> rgbToHue(Color color){
  //calc hue
  float r = color.r / 255.0;
  float g = color.g / 255.0;
  float b = color.b / 255.0;

  float max_c = std::max(r, std::max(g, b));
  float min_c = std::min(r, std::min(g, b));

  float h = 0;
  uint h_final;
  float diff = max_c - min_c;

  if(max_c == min_c)
    h = 0;
  else if(max_c == r)
    h = 60 * (0 + (g - b) / diff);
  else if(max_c == g)
    h = 60 * (2 + (b - r) / diff);
  else if(max_c == b)
    h = 60 * (4 + (r - g) / diff);

  if(h > 360.0)
    h_final = 360;
  else if(h < 0.0)
    h_final = 0;
  else 
    h_final = uint(std::round(h));

  float v = std::round(max_c * 255);
  uint v_final = 0;

  if(v > 255.0)
    v_final = 255;
  else if(v < 0.0)
    v_final = 0;
  else
    v_final = uint(v);

  uint s_final = 0;

  float s = 0.0;
  if(max_c == 0.0)
    s = 0;
  else 
    s = std::round((diff / max_c) * 255);

  if(s > 255.0)
    s_final = 255;
  else if(s < 0.0)
    s_final = 0;
  else
    s_final = uint(s);



  std::vector<uint> vec;
  vec.push_back(h_final);
  vec.push_back(s_final);
  vec.push_back(v_final);

  return vec;

}

inline Color hueToRgb(std::vector<uint> hue){
  
  int h = std::round(hue[0] / 60.0);
  float f = (hue[0] / 60.0 - h);

  float v = hue[2] / 255.0;
  float s = hue[1] / 255.0;

  float p = v * (1 - s);
  float q = v * (1 - s * f);
  float t = v * (1 - s * (1 - f));

  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  if(h == 0 || h == 6){
    r = v;
    g = t;
    b = p;
  }else if(h == 1){
    r = q;
    g = v;
    b = p;
  }else if(h == 2){
    r = p;
    g = v;
    b = t;
  }else if(h == 3){
    r = p;
    g = q;
    b = v;
  }else if(h == 4){
    r = t;
    g = p;
    b = v;
  }else if(h == 5){
    r = v;
    g = p;
    b = q;
  }

  r = std::round(r * 255);
  g = std::round(g * 255);
  b = std::round(b * 255);
  Color res;

  if(r > 255)
    res.r = 255;
  else if(r < 0)
    res.r = 0;
  else
    res.r = r;

  if(g > 255)
    res.g = 255;
  else if(r < 0)
    res.g = 0;
  else
    res.g = g;

  if(b > 255)
    res.b = 255;
  else if(b < 0)
    res.b = 0;
  else
    res.b = b;

  return res;

}

//from intensity vis
template<class T>
void TsdfSeqServer<T>::recolorVoxbloxMeshMsgByLabel(
    const Layer<T>& label_layer,
    const std::shared_ptr<ColorMap>& color_map, voxblox_msgs::Mesh* mesh_msg) {

  CHECK_NOTNULL(mesh_msg);
  CHECK(color_map);
  //std::cout << label_layer.getNumberOfAllocatedBlocks() << std::endl;
  //std::cout << tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() << std::endl;
  //std::cout << "recoloring mesh" << std::endl;
  // Go over all the blocks in the mesh.
  for (voxblox_msgs::MeshBlock& mesh_block : mesh_msg->mesh_blocks) {
    // Look up verticies in the thermal layer.
    for (size_t vert_idx = 0u; vert_idx < mesh_block.x.size(); ++vert_idx) {
      // only needed if color information was originally missing
      //mesh_block.r.resize(mesh_block.x.size());
      //mesh_block.g.resize(mesh_block.x.size());
      //mesh_block.b.resize(mesh_block.x.size());
      //std::cout << "iterating over voxel" << std::endl;


      //from segmentation server (variant from intensity doesn't work); ask /rewrite TODO
      constexpr float point_conv_factor = 2.0f / std::numeric_limits<uint16_t>::max();
      const float mesh_x =
          (static_cast<float>(mesh_block.x[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[0])) * mesh_msg->block_edge_length;
      const float mesh_y =
          (static_cast<float>(mesh_block.y[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[1])) * mesh_msg->block_edge_length;
      const float mesh_z =
          (static_cast<float>(mesh_block.z[vert_idx]) * point_conv_factor +
           static_cast<float>(mesh_block.index[2])) * mesh_msg->block_edge_length;

      /*const float mesh_x = static_cast<float>(mesh_block.x[vert_idx]);
      const float mesh_y = static_cast<float>(mesh_block.y[vert_idx]);
      const float mesh_z = static_cast<float>(mesh_block.z[vert_idx]);*/


      //std::cout << "server" << mesh_x << std::endl;
      const T* voxel = label_layer.getVoxelPtrByCoordinates(
          Point(mesh_x, mesh_y, mesh_z));
     /*if(voxel != nullptr){
         std::cout << voxel->weight << std::endl;
     }*/
      float epsilon = 0.0001;
      if (voxel != nullptr && voxel->weight >= (0.0 - epsilon)) {
          //std::cout << "recolor" << std::endl;
        float label = 0.0f + voxel->label_id;
        //std::cout << label << std::endl;
        Color new_color = color_map->colorLookup(label);

        //get colors from mesh
        Color mesh_col;
        mesh_col.r = mesh_block.r[vert_idx];
        mesh_col.g = mesh_block.g[vert_idx];
        mesh_col.b = mesh_block.b[vert_idx];

        //convert to hue
        std::vector<uint> mesh_col_hue = rgbToHue(mesh_col);
        std::vector<uint> label_col_hue = rgbToHue(new_color);

        //merge
        label_col_hue[2] = mesh_col_hue[2];

        //reconvert
        new_color = hueToRgb(label_col_hue);

        mesh_block.r[vert_idx] = new_color.r;
        mesh_block.g[vert_idx] = new_color.g;
        mesh_block.b[vert_idx] = new_color.b;
      }
    }
  }
}



//from intensity server
template<class T>
void TsdfSeqServer<T>::publishPointclouds() {
  // Create a pointcloud with color from labels.
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;//TODO

  /*createLabelPointcloudFromLabelLayer(*labely_layer_, &pointcloud);

  pointcloud.header.frame_id = world_frame_id;
  label_pointcloud_pub_.publish(pointcloud);*/
  
  TsdfServer::publishPointclouds();

}

//intensity server stuff
/*void TsdfSeqServer::createLabelPointcloudFromLabelLayer(
    const Layer<LabelVoxel>& layer,
    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<LabelVoxel>(
      layer, &visualizeLabelVoxels, pointcloud);
}*/

/*void createColorPointcloudFromLayer(
    const Layer<LabelVoxel>& layer,
    const ShouldVisualizeVoxelIntensityFunctionType<VoxelType>& vis_function,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  // Cache layer settings.
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  // Temp variables.
  double intensity = 0.0;
  // Iterate over all blocks.
  for (const BlockIndex& index : blocks) {
    // Iterate over all voxels in said blocks.
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord,
                       &intensity)) {
        pcl::PointXYZI point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.intensity = intensity;
        pointcloud->push_back(point);
      }
    }
  }
}*/
template<class T>
cv_bridge::CvImageConstPtr TsdfSeqServer<T>::convertImagePtr(
    const sensor_msgs::ImageConstPtr& depth_img) {
  cv_bridge::CvImageConstPtr depth_img_ptr;
  // from image_operations.h
  // convert the unit to mm if needed
  if (depth_img->encoding == "32FC1") {
    std::cout << "converting to mm" << std::endl;
    cv_bridge::CvImagePtr depth_img_mm = cv_bridge::toCvCopy(
        depth_img, sensor_msgs::image_encodings::TYPE_32FC1);
    depth_img_mm->image.convertTo(depth_img_mm->image, CV_16U, 1000.0);
    depth_img_ptr = depth_img_mm;
  } else {
    depth_img_ptr = cv_bridge::toCvShare(
        depth_img, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  return depth_img_ptr;
}


//like work from Marius
//warning images must be aligned!!!
template <class T> template <class C>
void TsdfSeqServer<T>::convertToCloud(
    const cv::Mat& depth_img, const cv::Mat& rgb_img,
    const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
    pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  //std::cout << rgb_img.size() << std::endl;
  //std::cout << depth_img.size() << std::endl;

  // Use correct principal point from calibration
  float center_x = static_cast<float>(depth_cam_info->K[2]);
  float center_y = static_cast<float>(depth_cam_info->K[5]);

  float unit_scaling = 0.001f;

  // unit_scaling = 1.0f;

  if (std::is_same<C, float>::value) unit_scaling = 1.0f;

  float f_x = static_cast<float>(depth_cam_info->K[0]);
  float f_y = static_cast<float>(depth_cam_info->K[4]);

  for (int row = 0; row < depth_img.rows; row++) {
    for (int col = 0; col < depth_img.cols; col++) {
      const cv::Vec3b& rgb = rgb_img.at<cv::Vec3b>(row, col);
      C depth = depth_img.at<C>(row, col);
      float scaled_depth = unit_scaling * float(depth);

      pcl::PointXYZRGB& p = cloud.at(col, row);
      // pcl::PointXYZRGB p;

      // Check for invalid measurements
      // if (!isDepthValid(depth)) { //do depth valid check TODO
      if (!(depth != 0)) {
        p.x = p.y = p.z = NAN;
      } else {
        // Fill in XYZ
        p.x = (col - center_x) * scaled_depth / f_x;
        p.y = (row - center_y) * scaled_depth / f_y;
        p.z = scaled_depth;
      }

      // Fill in color //lookupColor TODO

      //use only 3rd hue channel
      Color gray;
      gray.r = 100;
      gray.g = 100;
      gray.b = 100;

      Color i_color;
      i_color.r = rgb[0];
      i_color.g = rgb[1];
      i_color.b = rgb[2];

      std::vector<uint> mesh_col_hue = rgbToHue(i_color);
      std::vector<uint> label_col_hue = rgbToHue(gray);

      //merge
      label_col_hue[2] = mesh_col_hue[2];

      //reconvert
      Color new_color = hueToRgb(label_col_hue);

      p.r = new_color.r;
      p.g = new_color.g;
      p.b = new_color.b;

      //cloud.push_back(p);
    }
  }
}


template <class T> template <class C>
void TsdfSeqServer<T>::convertToCloudMonoColor(
    const cv::Mat& depth_img, 
    const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
    pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  
  // Use correct principal point from calibration
  float center_x = static_cast<float>(depth_cam_info->K[2]);
  float center_y = static_cast<float>(depth_cam_info->K[5]);

  float unit_scaling = 0.001f;

  // unit_scaling = 1.0f;

  if (std::is_same<C, float>::value) unit_scaling = 1.0f;

  float f_x = static_cast<float>(depth_cam_info->K[0]);
  float f_y = static_cast<float>(depth_cam_info->K[4]);

  for (int row = 0; row < depth_img.rows; row++) {
    for (int col = 0; col < depth_img.cols; col++) {
      // const cv::Vec3b& rgb = rgb_img.at<cv::Vec3b>(row, col);
      C depth = depth_img.at<C>(row, col);
      float scaled_depth = unit_scaling * float(depth);

      pcl::PointXYZRGB& p = cloud.at(col, row);
      // pcl::PointXYZRGB p;

      // Check for invalid measurements
      // if (!isDepthValid(depth)) { //do depth valid check TODO
      if (!(depth != 0)) {
        p.x = p.y = p.z = NAN;
      } else {
        // Fill in XYZ
        p.x = (col - center_x) * scaled_depth / f_x;
        p.y = (row - center_y) * scaled_depth / f_y;
        p.z = scaled_depth;
      }

      // Fill in color //lookupColor TODO
      p.r = 100;  // rgb[0];
      p.g = 100;  // rgb[1];
      p.b = 100;  // rgb[2];

      // cloud.push_back(p);
    }
  }
}

template<class T>
std::shared_ptr<Layer<T>> TsdfSeqServer<T>::getLabelLayer() {
  return label_layer;
}

/*void TsdfSeqServer::newPoseCallback(const Transformation& T_G_C){
  T_G_C_cur_ = T_G_C;
}*/


template class TsdfSeqServer<LabelVoxel>;
template class TsdfSeqServer<MultiLabelVoxel>;

}  // namespace voxblox