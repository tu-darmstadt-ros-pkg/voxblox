#ifndef VOXBLOX_ROS_DYNAMIC_OBJECT_H_
#define VOXBLOX_ROS_DYNAMIC_OBJECT_H_

#include <vector>
#include <list>

#include <pcl/common/common.h>

#include <geometry_msgs/PoseStamped.h>

#include "voxblox/core/tsdf_map.h"
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/dynamic_mapping/pcl_icp.h"
#include "voxblox_ros/dynamic_mapping/common.h"

namespace voxblox
{

  class DynamicObject {

   public:

    DynamicObject() {};

    DynamicObject(const TsdfMap::Config& tsdf_config,
                  TsdfIntegratorBase::Config& integrator_config,
                  const MeshIntegratorConfig& mesh_config,
                  const std::shared_ptr<PCL_ICP>& icp,
                  std::string method,
                  const int id, const int semantic_class, Color mesh_color,
                  const int max_num_resets_before_inactive);
    virtual ~DynamicObject() = default;

    void convertPointclouds(std::shared_ptr<ColorMap> color_map_);

    void integrate(const Transformation& T_G_C, const bool is_background);

    void generateMesh();

    void updateState(const Transformation& T_G_C,
                        const sensor_msgs::PointCloud2::Ptr& pointcloud_msg);

    void updateMeshMinMax();

    bool isPointInside(const InputPointType point);

    int align(const Transformation& T_G_C);

    void updatePosition(const Transformation& T_G_C);

    void reset();

    void initNextStep();

    void setSemanticClass(const int semantic_class);

    std::shared_ptr<TsdfMap> getTsdfMap() const { return map_; }
    std::shared_ptr<MeshLayer> getMeshLayer() const { return mesh_layer_; }

    void setFrameOccurence(bool occured) {occured_in_current_frame_ = occured; }
    void increaseOccurenceCounter() {time_since_last_occurence_++; }

    int getTimeSinceLastOccurence() const { return time_since_last_occurence_; }
    Eigen::Matrix4f getTransformation() const { return T_O_accumulated_; }
    Eigen::Matrix<float, 7, 1> getState() const { return state_; }
    std::vector<geometry_msgs::PoseStamped> getTrajectory()
                                      const { return stamped_trajectory_; }
    void setVoxelSize(float min_size, float max_size);
    void setInactive() {active_ = false; }
    void setDelete() {delete_ = true; }
    bool getDelete() const { return delete_; }
    bool isActive() const { return active_; }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getTransformedCloud()
                    const { return cloud_transformed_; }
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getMeshCloud()
                    const { return mesh_cloud_; }

    int getID() const { return id_; }
    int getSemanticClass() const { return semantic_class_; }



    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_current_;
    Eigen::Matrix4f T_O_accumulated_;
    Eigen::Matrix4f T_O_last_;

  private:

    int id_;
    int semantic_class_;
    int semantic_class_confidence_;
    bool semantic_class_set_;
    bool occured_in_current_frame_;
    int time_since_last_occurence_;
    int frames_not_aligned_;
    Color mesh_color_;

    std::shared_ptr<PCL_ICP> icp_;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_last_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_current_transformed_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_last_transformed_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_transformed_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_accumulated_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud_;


    Eigen::Matrix<float, 7, 1> state_;

    std::vector<Point> trajectory_;
    std::vector<geometry_msgs::PoseStamped> stamped_trajectory_;
    std::shared_ptr<ColorMap> color_map_;

    // Maps and integrators.
    std::shared_ptr<TsdfMap> map_;
    std::shared_ptr<TsdfIntegratorBase> integrator_;
    std::shared_ptr<MeshLayer> mesh_layer_;
    std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;


    pcl::PointXYZRGBNormal p_mesh_min_, p_mesh_max_;

    TsdfMap::Config tsdf_config_;
    TsdfIntegratorBase::Config integrator_config_;
    MeshIntegratorConfig mesh_config_;
    std::string integrator_method_;

    int reset_counter_;
    bool active_;
    bool delete_;
    bool not_aligned_;
    bool static_;
    bool voxel_size_set_;

    float xy_in_object_padding_;
    int max_num_resets_before_inactive_;

  };

}  // namespace voxblox

#endif  // VOXBLOX_ROS_DYNAMIC_OBJECT_H_
