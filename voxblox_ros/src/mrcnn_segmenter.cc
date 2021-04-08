#include "voxblox_ros/mrcnn_segmenter.h"

//like segmenter
namespace voxblox {

    MrcnnSegmenter::MrcnnSegmenter(const ros::NodeHandle& nh_private, float voxel_size) : 
    nh_private_(nh_private), voxel_size_(voxel_size) {

        initColorMap(255);
        normals_window_size_ = 2;
        edges_window_size_ = 3;
    }

//from segmenter
Color MrcnnSegmenter::getSegmentColor(uint segment) {

  //make sure the first segment is always black
  if (segment == 0)
    return Color::Black();

  auto it = segment_colors_.find(segment);

  if (it != segment_colors_.end()) {
    return it->second;
  } else {

    // seed is not set on purpose
    uint8_t r = static_cast<uint8_t>(rand() % 256);
    uint8_t g = static_cast<uint8_t>(rand() % 256);
    uint8_t b = static_cast<uint8_t>(rand() % 256);

    Color c(r, g, b);
    segment_colors_.emplace(segment, c);

    return c;
  }
}


    //from segmenter
void MrcnnSegmenter::initColorMap(int num_entries) {
    segment_colors_.emplace(0, Color::Black());

    for (int i = 1; i < num_entries; i++) {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        uint8_t c = static_cast<uint8_t>(i);

        for (int j = 0; j < 8; j++){
            r |= ((c >> 0) & 1) << (7-j);
            g |= ((c >> 1) & 1) << (7-j);
            b |= ((c >> 2) & 1) << (7-j);
            c = c >> 3;
        }

        Color color(r, g, b);
        segment_colors_.emplace(i, color);
    }
}

//from segmenter, adjusted
void MrcnnSegmenter::transformMaskToMap(const cv::Mat& mask_img, const sensor_msgs::CameraInfoConstPtr& mask_camera_info_msg,
                                const cv::Mat& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_cam_info_msg,
                                const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, Pointcloud& cloud_out, LabelIndexMap& segment_map){

    if(cloud_in->points.empty()){
        return;
    }

    image_geometry::PinholeCameraModel depth_camera_model_;
    depth_camera_model_.fromCameraInfo(depth_cam_info_msg);

    segment_map.clear();

    //edge points?

    applyVoxelGridFilter(cloud_in, mask_img, cloud_out, segment_map);

    int seg_map_size =0;
    std::cout << "segment_map size" << std::endl;
    for(auto item:segment_map)
    seg_map_size += item.second.size();
    std::cout << seg_map_size << std::endl;
}


//from segmenter
std::pair<ushort, int> MrcnnSegmenter::getMostCommonLabel(const std::unordered_map<ushort, int>& x) {
  using pairtype=std::pair<ushort, int>;
  return *std::max_element(x.begin(), x.end(), [] (const pairtype& p1, const pairtype& p2) {
        return p1.second < p2.second;
  });
}

//from segmenter, adjusted
void MrcnnSegmenter::applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in, const cv::Mat& segmentation_img, Pointcloud& cloud_out, LabelIndexMap& segment_map) {
  Octree octree(static_cast<double>(voxel_size_));
  octree.setInputCloud(cloud_in);
  octree.addPointsFromInputCloud();

  int width = static_cast<int>(cloud_in->width);
  int height = static_cast<int>(cloud_in->height);


  cloud_out.reserve(octree.getLeafCount());

  Point point;
  int num_points;
  std::unordered_map<ushort, int> label_counts;

  int index = 0;
  for (auto leaf_it = octree.leaf_begin(); leaf_it != octree.leaf_end(); ++leaf_it) {
    auto& leaf_container = leaf_it.getLeafContainer();
    const auto& indices = leaf_container.getPointIndicesVector();

    point = Point::Zero();
    num_points = 0;
    label_counts.clear();

    //std::cout << indices.size() << std::endl;
    for (int index: indices) {

      const pcl::PointXYZ& point_pcl = cloud_in->points[static_cast<size_t>(index)];

      int row = index / width;
      int col = index % width;

      if (row < normals_window_size_ || col < normals_window_size_ ||
          row >= height - normals_window_size_ || col >= width - normals_window_size_) {
        break;
      }

      ushort label = segmentation_img.at<ushort>(row-normals_window_size_, col-normals_window_size_);
      // keep track of the labels and centroid of the points in this voxel
      if (label_counts.count(label) == 0) {
        label_counts[label] = 1;
      } else {
        label_counts[label]++;
      }

      point.x() += point_pcl.x;
      point.y() += point_pcl.y;
      point.z() += point_pcl.z;

      num_points++;
    }

    if (num_points > 0) {
      // add the centroid of the points in this cell
      cloud_out.emplace_back(point/num_points);

      // select the most common label as label for this cell
      ushort label = getMostCommonLabel(label_counts).first;
      segment_map[label].emplace_back(index);
      index++;
    }
  }
}

} //namespace voxblox