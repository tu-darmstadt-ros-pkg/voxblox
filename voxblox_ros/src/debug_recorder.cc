#include "voxblox_ros/debug_recorder.h"


namespace voxblox {


    DebugRecorder::DebugRecorder(ros::NodeHandle& nh, 
                                 ros::NodeHandle& nh_private,
                                 std::string color_image_topic,
                                 std::string depth_image_topic,
                                 std::string color_info_topic,
                                 std::string depth_info_topic) 
        :                 color_img_sub(nh, color_image_topic, 1),
                          color_info_sub(nh, color_info_topic, 1),
                          depth_img_sub(nh, depth_image_topic, 1),
                          depth_info_sub(nh, depth_info_topic, 1),
                          sync(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>(10), color_img_sub, depth_img_sub, color_info_sub, depth_info_sub),
                          nh_(nh), nh_private_(nh_private)  {
       
        published_to_cnn = false;
        first_cb = true;

            debug_color_image_pub = nh.advertise<sensor_msgs::Image>("/debug/color/image", 1);
            debug_depth_image_pub = nh.advertise<sensor_msgs::Image>("/debug/depth/image", 1);
            debug_color_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/debug/color/info", 1);
            debug_depth_info_pub = nh.advertise<sensor_msgs::CameraInfo>("/debug/depth/info", 1);
            debug_cnn_result_pub = nh.advertise<mask_rcnn_ros::Result>("/debug/cnn/result", 1);

            cnn_image_pub = nh.advertise<sensor_msgs::Image>("/cnn/image", 1);
            cnn_result_sub = nh.subscribe("/mask_rcnn/result", 1, &DebugRecorder::cnn_callback, this);
       

        
    
        sync.setAgePenalty(0.5);
        sync.setMaxIntervalDuration(ros::Duration(0.1));
        sync.registerCallback(boost::bind(&DebugRecorder::callback, this, _1, _2, _3, _4));
    }


    void DebugRecorder::callback(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info){
        
        if(first_cb){
            

            first_cb = false;
        }


        //TODO make a way to escape from this, if cnn fails 
        if(published_to_cnn)
            return;

        ROS_INFO("callback recorder");



        //save image and camera info
        color_img_cam = *color_img;
        depth_img_cam = *depth_img;
        color_info_cam = *color_info;
        depth_info_cam = *depth_info;


        //if color image hasnt right scale, rescale it || using marius versions to avoid mask rccn error || upsample later? discuss this
        if(color_img_cam.height > 1024 || color_img_cam.width > 1024){
            //cv::Mat color_img_downsampled = downSampleColorImg(color_img_msg, downsampling_factor);
            double downsample_factorx = std::max(1.0, double(color_img_cam.width / 1024.0));
            double downsample_factory = std::max(1.0, double(color_img_cam.height / 1024.0));
            double downsample_factor = std::max(downsample_factorx, downsample_factory);

            //ripped from downsample
            cv::Mat img_downsampled;
            cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(color_img, sensor_msgs::image_encodings::TYPE_8UC3);
            cv::resize(img->image, img_downsampled, cv::Size(0, 0),
            1.0/downsample_factor, 1.0/downsample_factor,
            cv::INTER_AREA);
            
            //small changes to make it work in this context
            cv_bridge::CvImage i = *img;
            i.image = img_downsampled;
            i.header = color_img_cam.header;
            i.encoding = color_img_cam.encoding;
            sensor_msgs::ImagePtr color_img_new = i.toImageMsg();


            //ripped from downsampleCameraInfo but modified double support
            //sensor_msgs::CameraInfoPtr color_cam_info = downsampleCameraInfo(color_info, downsample_factor);
            sensor_msgs::CameraInfo cam_info_downsampled = *color_info;
            cam_info_downsampled.height = static_cast<uint>(img_downsampled.rows);
            cam_info_downsampled.width = static_cast<uint>(img_downsampled.cols);

            cam_info_downsampled.K[0] /= downsample_factor;  // fx
            cam_info_downsampled.K[2] /= downsample_factor;  // cx
            cam_info_downsampled.K[4] /= downsample_factor;  // fy
            cam_info_downsampled.K[5] /= downsample_factor;  // cy

            cam_info_downsampled.P[0] /= downsample_factor;  // fx
            cam_info_downsampled.P[2] /= downsample_factor;  // cx
            cam_info_downsampled.P[3] /= downsample_factor;  // T
            cam_info_downsampled.P[5] /= downsample_factor;  // fy
            cam_info_downsampled.P[6] /= downsample_factor;  // cy


            color_info_cam = cam_info_downsampled;
            color_img_cam = *color_img_new;

            std::cout << "Downsample factor" << std::endl;
            std::cout << downsample_factor << std::endl;
        }

        

        cnn_image_pub.publish(color_img_cam);
        published_to_cnn = true;
        
    }

    void  DebugRecorder::cnn_callback(const mask_rcnn_ros::Result res){
        

        mask_rcnn_ros::Result result;
        result.header = res.header;
        result.header.stamp = color_img_cam.header.stamp;
        result.boxes = res.boxes;
        result.class_ids = res.class_ids;
        result.class_names = res.class_names;
        result.scores = res.scores;
        result.masks = res.masks;

        debug_color_image_pub.publish(color_img_cam);
        debug_depth_image_pub.publish(depth_img_cam);
        debug_color_info_pub.publish(color_info_cam);
        debug_depth_info_pub.publish(depth_info_cam);
        debug_cnn_result_pub.publish(result);

        published_to_cnn = false;

    }
}