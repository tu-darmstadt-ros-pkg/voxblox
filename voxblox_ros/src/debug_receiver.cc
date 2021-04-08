#include "voxblox_ros/debug_receiver.h"



namespace voxblox {

    DebugReceiver::DebugReceiver(const ros::NodeHandle& nh, 
                                 const ros::NodeHandle& nh_private,
                                 std::string color_image_topic,
                                 std::string depth_image_topic,
                                 std::string color_info_topic,
                                 std::string depth_info_topic,
                                 std::string cnn_result_topic) 
        : TsdfSeqReceiver(nh, nh_private), 
                          color_img_sub(nh_, color_image_topic, 1),
                          color_info_sub(nh_, color_info_topic, 1),
                          depth_img_sub(nh_, depth_image_topic, 1),
                          depth_info_sub(nh_, depth_info_topic, 1),
                          cnn_result_sub(nh_, cnn_result_topic, 1),
                          sync(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo, mask_rcnn_ros::Result>(10), color_img_sub, depth_img_sub, color_info_sub, depth_info_sub, cnn_result_sub) {


        
        sync.setAgePenalty(0.5);
        sync.setMaxIntervalDuration(ros::Duration(0.1));

        //sync.registerCallback(boost::bind(&DebugReceiver::callback, this, _1, _2, _3, _4, _5));
        sync.registerCallback(boost::bind(&DebugReceiver::callback2, this, _1, _2, _3, _4, _5));
    
        id_counter = 1;

        //flags
        //debug
        projects_weight = true; //TODO nh param
        provides_weight_image = true;
        aligned = false;
        nh.param<bool>("aligned_rgbd", aligned, false);


        //TODO read from param server
        scale = 1.0;

        //scale = 1.25;
    }

    void DebugReceiver::initialize_weight2(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                 const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> func){
        //TsdfSeqReceiver::initialize(func);
        server_weight_callback2 = func;
        got_callback = true;
        //cnn_image_pub = nh_.advertise<sensor_msgs::Image>("/cnn/image", 1);
        //cnn_result_sub = nh_.subscribe("/mask_rcnn/result", 1, &DebugReceiver::cnn_callback, this);

    }

    void DebugReceiver::initialize_weight3(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                 const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> func){
        //TsdfSeqReceiver::initialize(func);
        server_weight_callback3 = func;
        got_callback = true;

    }


    //TODO maybe improve
    void DebugReceiver::callback(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info, const mask_rcnn_ros::ResultConstPtr& cnn_result){
        if(!got_callback)
            return;

        
        
        
        
        color_img_cam = *color_img;
        depth_img_cam = *depth_img;
        color_info_cam = *color_info;
        depth_info_cam = *depth_info;
        cnn_res = *cnn_result;



        label_map.clear();
        coord_map.clear();

        //on empty res
        if(!(cnn_res.masks.size() > 0)){
            return;
        }

        //convert masks to images
        std::vector<cv_bridge::CvImagePtr> image_vec;

        for(int i = 0; i < cnn_res.masks.size(); i++){
            cv_bridge::CvImagePtr ptr;
            
            try{
                ptr = cv_bridge::toCvCopy(cnn_res.masks[i], cnn_res.masks[i].encoding);
            }
            catch(cv_bridge::Exception& e){
                ROS_ERROR("Cv_bridge Exception: %s", e.what());
            }
            image_vec.push_back(ptr);
        }


        int cols = image_vec[0]->image.cols;
        int rows = image_vec[0]->image.rows;
        int type = image_vec[0]->image.type();

        //convert depth image
        cv_bridge::CvImagePtr depth = cv_bridge::toCvCopy(depth_img_cam, depth_img_cam.encoding);
        //std::cout << depth_img_cam.encoding << std::endl;


        //get 3d points from depth image (like in segmenter)
        image_geometry::PinholeCameraModel depth_cam_model;
        depth_cam_model.fromCameraInfo(depth_info_cam);


        //lookup label for every string (if ids don't work)
        //also push back new vectors for each label
        //std::cout << "t1" << std::endl;
        std::vector<int> ids;
        for(int i = 0; i < cnn_res.class_names.size(); i++){
            //std::cout << "t2" << std::endl;
            std::unordered_map<std::string,int>::iterator it = label_string_mapping.find(cnn_res.class_names[i]);
            if(!(label_string_mapping.count(cnn_res.class_names[i]))){
                //std::cout << "t3" << std::endl;
                label_string_mapping[cnn_res.class_names[i]] = id_counter;
                ids.push_back(id_counter);
                id_counter++; 
            }else{
                //std::cout << "t4" << std::endl;
                ids.push_back(label_string_mapping[cnn_res.class_names[i]]);
            }
            //std::cout << "t5" << std::endl;
            
        }
        //std::cout << "masks: " << cnn_res.masks.size() << std::endl;


        //setup cam params //the coord calculation is similiar to convertCloud from segmentation server
        float center_x = static_cast<float>(depth_info_cam.K[2]);
        float center_y = static_cast<float>(depth_info_cam.K[5]);
        float f_x = static_cast<float>(depth_info_cam.K[0]);
        float f_y = static_cast<float>(depth_info_cam.K[4]);

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        //iterate over row and col
        for(int x = 0; x < cols; x++){
            for(int y = 0; y < rows; y++){

                //calculate point per hand because problems with prev
                //cv::Vec3b coord = points3d.at<cv::Vec3b>(y, x);
                cv::Vec3f coord;
                float d = depth->image.at<float>(y,x);

                if(!(d > 0.001)){
                    continue;
                }

                //TODO check for invalid points
                coord[0] = (x - center_x) * d / f_x;
                coord[1] = (y - center_y) * d / f_y;
                coord[2] = d;

                pcl::PointXYZRGB p;
                p.x = coord[0];
                p.y = coord[1];
                p.z = coord[2];
                p.r = 0;
                p.g = 0;
                p.b = 0;
                cloud.push_back(p);
                coord_map.push_back(coord);
                //std::cout << coord << std::endl;

                //iterate over masks and add values
                std::vector<LabelVoxel> vec;
                
                for(int i = 0; i < image_vec.size(); i++){
                    int val = (int) image_vec[i]->image.at<uchar>(y, x);
                    //std::cout <<"receiver_label" << val << res.class_names[i] << res.class_ids[i] << std::endl;
                    if(val != 0){
                        //std::cout << ids[i] << std::endl;
                        LabelVoxel voxel;
                        voxel.label_id = ids[i];
                        voxel.weight = cnn_res.scores[i];
                        vec.push_back(voxel);
                    }
                }
                //push back this pixel
                label_map.push_back(vec);
                
                
            }

        }

        sensor_msgs::ImagePtr depth_img_msg = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr color_info_msg = boost::make_shared<sensor_msgs::CameraInfo >(color_info_cam);
        sensor_msgs::CameraInfoPtr depth_info_msg = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        sensor_msgs::ImagePtr color_img_msg = boost::make_shared<sensor_msgs::Image >(color_img_cam);

        std::cout << "callback to server" << std::endl;
        server_weight_callback2(depth_img_msg, color_img_msg, color_info_msg, depth_info_msg, label_map, coord_map, label_string_mapping, aligned);
        
    }


    void DebugReceiver::callback2(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info, const mask_rcnn_ros::ResultConstPtr& cnn_result){
        label_map.clear();

        

        ROS_INFO("1");
        color_img_cam = *color_img;
        depth_img_cam = *depth_img;
        color_info_cam = *color_info;
        depth_info_cam = *depth_info;
        cnn_res = *cnn_result;

        if(scale != 1.0){
            //DEBUG rescale images

            double downsample_factor = 1.0/scale;

            mask_rcnn_ros::Result res_new;
            res_new.class_ids = cnn_result->class_ids;
            res_new.class_names = cnn_result->class_names;
            res_new.scores = cnn_result->scores;
            
            ROS_INFO("2");
            if((cnn_result->masks).size() == 0){
                return;
                //nothing to do without masks
            }

            for(int i = 0; i < (cnn_result->masks).size(); i++){
            //ripped from downsample
                cv::Mat img_downsampled;
                
                sensor_msgs::ImageConstPtr ptr(new sensor_msgs::Image(cnn_result->masks[i]));
                cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(ptr, sensor_msgs::image_encodings::TYPE_8UC1);
                cv::resize(img->image, img_downsampled, cv::Size(0, 0),
                1.0/downsample_factor, 1.0/downsample_factor,
                cv::INTER_AREA);
            
                //small changes to make it work in this context
                cv_bridge::CvImage im = *img;
                im.image = img_downsampled;
                im.header = cnn_result->masks[i].header;
                im.encoding = cnn_result->masks[i].encoding;
                sensor_msgs::ImagePtr color_img_new = im.toImageMsg();

                res_new.masks.push_back(*color_img_new);
                ROS_INFO("3");
            }
            

            //ripped from downsampleCameraInfo but modified double support
            //sensor_msgs::CameraInfoPtr color_cam_info = downsampleCameraInfo(color_info, downsample_factor);
            sensor_msgs::CameraInfo cam_info_downsampled = *color_info;
            cam_info_downsampled.height = static_cast<uint>(res_new.masks[0].height);
            cam_info_downsampled.width = static_cast<uint>(res_new.masks[0].width);

            cam_info_downsampled.K[0] /= downsample_factor;  // fx
            cam_info_downsampled.K[2] /= downsample_factor;  // cx
            cam_info_downsampled.K[4] /= downsample_factor;  // fy
            cam_info_downsampled.K[5] /= downsample_factor;  // cy

            cam_info_downsampled.P[0] /= downsample_factor;  // fx
            cam_info_downsampled.P[2] /= downsample_factor;  // cx
            cam_info_downsampled.P[3] /= downsample_factor;  // T
            cam_info_downsampled.P[5] /= downsample_factor;  // fy
            cam_info_downsampled.P[6] /= downsample_factor;  // cy
            ROS_INFO("4");

            color_info_cam = cam_info_downsampled;
            ROS_INFO("5");
            cnn_res = res_new;

            ROS_INFO("6");

        }

        

        
        
        

        if(!(cnn_res.masks.size() > 0)){
            return;
        }

        std::vector<cv_bridge::CvImagePtr> image_vec;

        for(int i = 0; i < cnn_res.masks.size(); i++){
            cv_bridge::CvImagePtr ptr;
            
            try{
                ptr = cv_bridge::toCvCopy(cnn_res.masks[i], cnn_res.masks[i].encoding);
            }
            catch(cv_bridge::Exception& e){
                ROS_ERROR("Cv_bridge Exception: %s", e.what());
            }
            image_vec.push_back(ptr);
        }

        int cols = image_vec[0]->image.cols;
        int rows = image_vec[0]->image.rows;
        int type = image_vec[0]->image.type();

       
        std::vector<int> ids;
        for(int i = 0; i < cnn_res.class_names.size(); i++){
            //std::cout << "t2" << std::endl;
            std::unordered_map<std::string,int>::iterator it = label_string_mapping.find(cnn_res.class_names[i]);
            if(!(label_string_mapping.count(cnn_res.class_names[i]))){
                //std::cout << "t3" << std::endl;
                label_string_mapping[cnn_res.class_names[i]] = id_counter;
                ids.push_back(id_counter);
                id_counter++; 
            }else{
                //std::cout << "t4" << std::endl;
                ids.push_back(label_string_mapping[cnn_res.class_names[i]]);
            }
            //std::cout << "t5" << std::endl;
            
        }

        //pushing back coordinates, TODO only push back valid values
        //iterate over row and col
        for(int x = 0; x < cols; x++){
            for(int y = 0; y < rows; y++){

                //image_coord_map.push_back(coord);
                //std::cout << coord << std::endl;

                //iterate over masks and add values
                std::vector<LabelVoxel> vec;
                
                for(int i = 0; i < image_vec.size(); i++){
                    int val = (int) image_vec[i]->image.at<uchar>(y, x);
                    //std::cout <<"receiver_label" << val << res.class_names[i] << res.class_ids[i] << std::endl;
                    if(val != 0){
                        //std::cout << ids[i] << std::endl;
                        LabelVoxel voxel;
                        voxel.label_id = ids[i];
                        voxel.weight = cnn_res.scores[i];
                        vec.push_back(voxel);
                    }
                }
                //push back this pixel
                label_map.push_back(vec);
                
            }
        }


        sensor_msgs::ImagePtr depth_img_msg = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr color_info_msg = boost::make_shared<sensor_msgs::CameraInfo >(color_info_cam);
        sensor_msgs::CameraInfoPtr depth_info_msg = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        sensor_msgs::ImagePtr color_img_msg = boost::make_shared<sensor_msgs::Image >(color_img_cam);


        std::cout << "callback to server" << std::endl;
        server_weight_callback3(depth_img_msg, color_img_msg, color_info_msg, depth_info_msg, label_map, label_string_mapping, aligned);


    }
}