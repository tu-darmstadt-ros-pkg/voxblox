#ifndef VOXBLOX_TSDF_SEQ_INTEGRATOR_H_
#define VOXBLOX_TSDF_SEQ_INTEGRATOR_H_

#include <ros/ros.h>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"
#include "voxblox/utils/timing.h"
#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/utils/distance_utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cv_bridge/cv_bridge.h>


#include <pcl_ros/point_cloud.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <sensor_msgs/CameraInfo.h>

namespace voxblox {

    //struct to track the vertices belonging to the color pixels
            struct point_map_entry {
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                bool valid_point = false;
                int segment_id = -1;
                int pcl_index = 0;
                bool edge = false;
            };


    enum IntegrationMode {DEFAULT, CONFIDENCE, CONFIDENCE_WEIGHT};
    enum NormalCalculationMode {CROSSPRODUCT, CENTRALDIFFERENCES, CENTRALDIFFERENCES_NORMALIZED, VOXBLOX};

    template<class T>
    class TsdfSeqIntegrator{
        public:
            TsdfSeqIntegrator(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf_layer, Layer<T>* label_layer, std::string integ_mode);
            //: tsdf_layer_(tsdf_layer), label_layer_(label_layer)

            //like intensity
            void addVectors(const Point& origin, const Pointcloud& bearing_vectors, std::vector<int>& labels);

            void integrateByProjection(std::vector<std::vector<LabelVoxel>> label_map, 
                                              uint32_t width, 
                                              uint32_t height,
                                              double focal_length,
                                              double px,
                                              double py,
                                              Transformation T_G_C);


            //stays this time
            pcl::PointCloud<pcl::PointXYZRGB> directlyIntegrate(std::vector<std::vector<LabelVoxel>> label_map, 
                                              std::vector<cv::Vec3f> coord_map,
                                              Transformation T_G_C);

            void integrateVectors(const Point& origin, const Pointcloud& bearing_vectors, std::vector<std::vector<LabelVoxel>> label_map);

            void integrate(Point g, std::vector<LabelVoxel> v, pcl::PointCloud<pcl::PointXYZRGB> cloud);

            void integrateVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

            void integrateConfidenceVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud);
    
            void integrateConfidenceWeightVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud);

            void integrateByGeometricSegmentation(const sensor_msgs::ImageConstPtr& color_img, 
                                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                                          std::vector<std::vector<LabelVoxel>> label_map, 
                                          Transformation T_G_C);

            std::pair<std::vector<point_map_entry>, std::vector<point_map_entry>> createVertexMap(double px, double py, double focal_length, Transformation T_G_C, uint32_t width, uint32_t height);

            std::vector<point_map_entry> calculateNormalsCentralDifferences(std::vector<point_map_entry> vertex_map, uint32_t width, uint32_t height, double px, double py, double focal_lengthx, double focal_lengthy);

            cv::Mat createEdgeImage(std::vector<point_map_entry> vertex_map, std::vector<point_map_entry> normal_map, uint32_t width, uint32_t height);

            std::pair<std::vector<std::unordered_map<int, double>>, cv::Mat> createSegmentation(cv::Mat edge_im, std::vector<std::vector<LabelVoxel>> label_map, uint32_t width, uint32_t height);
    
            std::vector<LabelVoxel> getWinningLabels(std::vector<std::unordered_map<int, double>> segment_labels);

            std::vector<point_map_entry> estimateNormalsCrossProduct(std::vector<point_map_entry> vertex_map, uint32_t width, uint32_t height);

            std::vector<point_map_entry> getVoxbloxNormals(std::vector<point_map_entry> vertex_map);

         
            void integrateByGeometricSegmentationAligned(cv::Mat depth_img,
                                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                                          std::vector<std::vector<LabelVoxel>> label_map,
                                          Transformation T_G_C);
    


        protected:
            IntegrationMode mode;
            NormalCalculationMode normal_calculation;
            const Layer<TsdfVoxel>& tsdf_layer_;
            Layer<T>* label_layer_;
            FloatingPoint max_distance_;//ray

            ros::NodeHandle nh_;

            ros::Publisher edge_im_c_pub;
            ros::Publisher edge_im_d_pub;
            ros::Publisher segmented_pub;
            ros::Publisher fake_depth_image_pub;

            bool init;
            bool show_pcl_debug;

            double tresh_concave;
            double tresh_distance;

            double max_dist;
    
    };
}

#endif