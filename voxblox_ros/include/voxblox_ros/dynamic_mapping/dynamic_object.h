#ifndef VOXBLOX_ROS_DYNAMIC_OBJECT_H_
#define VOXBLOX_ROS_DYNAMIC_OBJECT_H_

#include <vector>
#include <list>

#include <pcl/common/common.h> //TODO: clean includes

#include "voxblox/core/tsdf_map.h"
#include <voxblox_ros/dynamic_mapping/pcl_icp.h>
#include "voxblox_ros/dynamic_mapping/common.h"


#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>
#include <voxblox/utils/color_maps.h>

#include "voxblox_ros/conversions.h"
// #include "voxblox_ros/ros_params.h"

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
                  bool dynamic_voxel_size,
                  const int id, const int semantic_class, Color mesh_color);
    virtual ~DynamicObject() = default;

    void convertPointclouds(std::shared_ptr<ColorMap> color_map_);

    void integrate(const Transformation& T_G_C, const bool is_background);

    void generateMesh();

    void updateState(const Transformation& T_G_C);

    void align();

    void updatePosition(const Transformation& T_G_C);

    void reset();

    void setSemanticClass(const int semantic_class); //TODO everything const which can be const

    std::shared_ptr<TsdfMap> getTsdfMap() const { return map_; }
    std::shared_ptr<MeshLayer> getMeshLayer() const { return mesh_layer_; }

    void setFrameOccurence(bool occured) {occured_in_current_frame_ = occured; }
    void increaseOccurenceCounter() {time_since_last_occurence_++; }

    int getTimeSinceLastOccurence() const { return time_since_last_occurence_; }

    // Point getTranslation() const { return trajectory_.back() - trajectory_.front(); }
    Eigen::Matrix4f getTransformation() const { return T_O_accumulated_; }

    Eigen::Matrix<float, 7, 1> getState() const { return state_; }




    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getTransformedCloud() const { return cloud_transformed_; }
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr getMeshCloud() const { return mesh_cloud_; }

    int getID() const { return id_; }
    int getSemanticClass() const { return semantic_class_; }



    /**
     * Integrates background and objects
    */
    //void integrate(const Transformation& T_G_C, const bool is_freespace_pointcloud);

    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_start_;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_current_;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_last_;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_transformed_;

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud_;

    Eigen::Matrix4f T_O_accumulated_;

    Eigen::Matrix4f T_O_last_; //TODO Propper naming and public private reorder

  private:

    int id_;
    int semantic_class_;
    int semantic_class_confidence_;
    bool semantic_class_set_;

    bool occured_in_current_frame_;
    int time_since_last_occurence_;

    Eigen::Matrix<float, 7, 1> state_;

    Color mesh_color_;

    std::vector<Point> trajectory_;

    std::shared_ptr<ColorMap> color_map_;

    // Maps and integrators.
    std::shared_ptr<TsdfMap> map_;
    std::shared_ptr<TsdfIntegratorBase> integrator_;
    std::shared_ptr<MeshLayer> mesh_layer_;
    std::shared_ptr<MeshIntegrator<TsdfVoxel>> mesh_integrator_;

    std::shared_ptr<PCL_ICP> icp_;

  };


}  // namespace voxblox

#endif  // VOXBLOX_ROS_DYNAMIC_OBJECT_H_
