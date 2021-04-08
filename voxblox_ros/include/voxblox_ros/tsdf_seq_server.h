#ifndef VOXBLOX_ROS_TSDF_SEQ_SERVER_H_
#define VOXBLOX_ROS_TSDF_SEQ_SERVER_H_

#include <ros/ros.h>

#include "voxblox_ros/tsdf_seq_receiver.h"
#include "voxblox/integrator/tsdf_seq_integrator.h"
#include "voxblox_ros/tsdf_server.h"
#include "voxblox/utils/color_maps.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/mat.hpp>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"



#include <cv_bridge/cv_bridge.h>



namespace voxblox {

    template<class T>
    class TsdfSeqServer : public TsdfServer {

        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

            TsdfSeqServer(const ros::NodeHandle& nh, 
                          const ros::NodeHandle& nh_private);
            virtual ~TsdfSeqServer() {}

            virtual void updateMesh();
            virtual void publishPointclouds();

            void receiver_callback(const sensor_msgs::ImageConstPtr& seg_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info);
            
            
            void initialize(std::shared_ptr<TsdfSeqReceiver> rec,
                            std::shared_ptr<TsdfSeqIntegrator<T>> integ);

            void receiver_weight_callback(const sensor_msgs::ImageConstPtr& seg_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info, 
                                          const cv::Mat weight_image, std::unordered_map<int, std::string> label_lookup, std::unordered_map<int, std::vector<cv::Point>> label_map);

            void receiver_weight_callback2(
                const sensor_msgs::ImageConstPtr& depth_img,
                const sensor_msgs::ImageConstPtr& color_img,
                const sensor_msgs::CameraInfoConstPtr& color_info,
                const sensor_msgs::CameraInfoConstPtr& depth_info,
                const std::vector<std::vector<LabelVoxel>> label_map, 
                const std::vector<cv::Vec3f> coord_map, 
                const std::unordered_map<std::string, int> label_string_mapping,
                const bool aligned
                );

            void receiver_weight_callback3(
                const sensor_msgs::ImageConstPtr& depth_img,
                const sensor_msgs::ImageConstPtr& color_img,
                const sensor_msgs::CameraInfoConstPtr& color_info,
                const sensor_msgs::CameraInfoConstPtr& depth_info,
                const std::vector<std::vector<LabelVoxel>> label_map, 
                const std::unordered_map<std::string, int> label_string_mapping,
                const bool aligned
                );


                //virtual void newPoseCallback(const Transformation& T_G_C);

            void depth_callback(const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& depth_info);

            std::shared_ptr<Layer<T>> getLabelLayer();
        protected:

           

            cv_bridge::CvImageConstPtr convertImagePtr(const sensor_msgs::ImageConstPtr& depth_img);
            void calculateVectors(const sensor_msgs::ImageConstPtr& depth_img, Transformation T_G_C, float focal_length_px);

            
            void recolorVoxbloxMeshMsgByLabel(const Layer<T>& label_layer, const std::shared_ptr<ColorMap>& color_map, voxblox_msgs::Mesh* mesh_msg);
            void createLabelPointcloudFromLabelLayer(
                const Layer<LabelVoxel>& layer,
                pcl::PointCloud<pcl::PointXYZRGB>* pointcloud);

            std::string world_frame_id = "world";

            
            std::shared_ptr<TsdfSeqIntegrator<T>> integrator;
            std::shared_ptr<TsdfSeqReceiver> receiver;
            std::unordered_map<int, std::vector<cv::Point>> label_map;
            std::unordered_map<int, std::string> label_lookup;
            std::shared_ptr<ColorMap> color_map_;


            Transformation T_G_C_depth_last;
            Transformation T_G_C_color_last;
            Transformation T_G_C_last;  

            template<typename C>
            void convertToCloud(const cv::Mat& depth_msg,
                            const cv::Mat& rgb_msg,
                            const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
                            pcl::PointCloud<pcl::PointXYZRGB>& cloud);

            template<typename C>
            void convertToCloudMonoColor(const cv::Mat& depth_msg,
                            const sensor_msgs::CameraInfoConstPtr& depth_cam_info,
                            pcl::PointCloud<pcl::PointXYZRGB>& cloud);

        
            std::shared_ptr<Layer<T>> label_layer;

            bool looked_up_tf;
            bool use_tsdf_segmentation;
            bool raw_image_segmentation;


            ros::Publisher debug_pub;
            ros::Publisher label_pointcloud_pub_;
            ros::Publisher label_mesh_pub_;
            ros::Publisher edge_image_pub;
            //Transformation T_G_C_cur_;

    };

}//namepsace
#endif