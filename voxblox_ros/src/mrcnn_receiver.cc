#include "voxblox_ros/mrcnn_receiver.h"

namespace voxblox {


    MrcnnReceiver::MrcnnReceiver(const ros::NodeHandle& nh, 
                                 const ros::NodeHandle& nh_private,
                                 std::string color_image_topic,
                                 std::string depth_image_topic,
                                 std::string color_info_topic,
                                 std::string depth_info_topic) 
        : TsdfSeqReceiver(nh, nh_private), 
                          color_img_sub(nh_, color_image_topic, 1),
                          color_info_sub(nh_, color_info_topic, 1),
                          depth_img_sub(nh_, depth_image_topic, 1),
                          depth_info_sub(nh_, depth_info_topic, 1),
                          sync(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>(10), color_img_sub, depth_img_sub, color_info_sub, depth_info_sub) {
       
        published_to_cnn = false;
        got_callback = false;
        sync.setAgePenalty(0.5);
        sync.setMaxIntervalDuration(ros::Duration(0.1));
        sync.registerCallback(boost::bind(&MrcnnReceiver::callback, this, _1, _2, _3, _4));
        seg_img_pub = nh_.advertise<sensor_msgs::Image>("/segmented_image", 2);
        debug_seg_cloud_pub = nh_.advertise<sensor_msgs::PointCloud2>("/segmented/debugcloud", 2);

        id_counter = 1;

        //flags
        //DEBUG, reorder this
        projects_weight = true;
        provides_weight_image = true;
        early_depth = true;
        aligned = false;
        nh.param<bool>("aligned_rgbd", aligned, false);
    }

    void MrcnnReceiver::callback(const sensor_msgs::ImageConstPtr& color_img, const sensor_msgs::ImageConstPtr& depth_img, const sensor_msgs::CameraInfoConstPtr& color_info, const sensor_msgs::CameraInfoConstPtr& depth_info){
        
        if(!got_callback)
            return;

        //TODO make a way to escape from this, if cnn fails
        if(published_to_cnn)
            return;

        ROS_INFO("callback receiver");

        //save image and camera info
        color_img_cam = *color_img;
        depth_img_cam = *depth_img;
        color_info_cam = *color_info;
        depth_info_cam = *depth_info;

        cnn_image_pub.publish(color_img_cam);
        published_to_cnn = true;


        //let voxblox generate mesh before sending data to cnn
        sensor_msgs::ImagePtr depth_img_msg = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr depth_info_msg = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        early_depth_callback(depth_img_msg, depth_info_msg);
        
    }

    void MrcnnReceiver::initialize_weight(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&,
                                                                const cv::Mat, const std::unordered_map<int, std::string>, const std::unordered_map<int, std::vector<cv::Point>>)> func){
        //TsdfSeqReceiver::initialize(func);
        server_weight_callback = func;
        got_callback = true;
        cnn_image_pub = nh_.advertise<sensor_msgs::Image>("/cnn/image", 1);
        cnn_result_sub = nh_.subscribe("/mask_rcnn/result", 1, &MrcnnReceiver::cnn_callback, this);

    }

    void MrcnnReceiver::initialize_weight2(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                 const std::vector<std::vector<LabelVoxel>>, const std::vector<cv::Vec3f>, const std::unordered_map<std::string, int>, const bool)> func){
        //TsdfSeqReceiver::initialize(func);
        server_weight_callback2 = func;
        got_callback = true;
        cnn_image_pub = nh_.advertise<sensor_msgs::Image>("/cnn/image", 1);
        cnn_result_sub = nh_.subscribe("/mask_rcnn/result", 1, &MrcnnReceiver::cnn_callback, this);

    }

    void MrcnnReceiver::initialize_weight3(boost::function<void (const sensor_msgs::ImageConstPtr&, const sensor_msgs::ImageConstPtr&, const sensor_msgs::CameraInfoConstPtr&, const sensor_msgs::CameraInfoConstPtr&, 
                                                                 const std::vector<std::vector<LabelVoxel>>, const std::unordered_map<std::string, int>, const bool)> func){
        //TsdfSeqReceiver::initialize(func);
        server_weight_callback3 = func;
        got_callback = true;
        cnn_image_pub = nh_.advertise<sensor_msgs::Image>("/cnn/image", 1);
        cnn_result_sub = nh_.subscribe("/mask_rcnn/result", 1, &MrcnnReceiver::cnn_callback2, this);

    }



    //TODO rework this 
    /*void MrcnnReceiver::cnn_callback(const mask_rcnn_ros::Result res){
        std::cout << "Segments: ";
        std::cout << res.class_ids.size() << std::endl;

        if(res.masks.size() > 0){
        std::cout << "starting to work on masks" << std::endl;
        cv_bridge::CvImagePtr initial;
                try{
                    initial = cv_bridge::toCvCopy(res.masks[0], res.masks[0].encoding);
                }
                catch(cv_bridge::Exception& e){
                    ROS_ERROR("Cv_bridge Exception: %s", e.what());
                }
        
        int rows = initial->image.rows;
        int cols = initial->image.cols;
        int type = initial->image.type();

        //https://www.geeksforgeeks.org/opencv-c-plus-plus-program-to-create-a-single-colored-blank-image/
        cv::Mat res_image(rows, cols, CV_16UC1, cv::Scalar(0));
        cv::Mat weight_image(rows, cols, CV_32F, cv::Scalar(0));
        std::cout << "created res image " << std::endl;

        //for(auto &mask : res.masks){
        for(int i = 0; i < res.masks.size(); i++){

            //we take the label ids from mrcnn, maybe change later
            if (label_lookup.find(res.class_ids[i])==label_lookup.end()){
                label_lookup[res.class_ids[i]] = res.class_names[i];
            }
            
            //maybe change this
            int color = res.class_ids[i];
            float weight = res.scores[i];

            cv_bridge::CvImagePtr ptr;
            try{
                ptr = cv_bridge::toCvCopy(res.masks[i], res.masks[i].encoding);
            }
            catch(cv_bridge::Exception& e){
                ROS_ERROR("Cv_bridge Exception: %s", e.what());
            }

            //adding label from mrcnn here TODO maybe change
            color = 1;
            cv::Mat color_img(rows, cols, CV_16UC1, cv::Scalar(color));
            cv::Mat weights(rows, cols, CV_32F, cv::Scalar(weight));

            cv::add(res_image, color_img, res_image, ptr->image);
            cv::add(weight_image, weights, weight_image, ptr->image);

            std::cout << "completed mask" << std::endl;

            //iterate over mask to get indexes; TODO see if this can be improved ; Check if vectors are initiated
            for(int x; x < ptr->image.cols; x++){
                for(int y; y < ptr->image.rows; y++){
                    cv::Point p(x,y);
                    
                    cv::Scalar s = ptr->image.at<uchar>(p);
                    if(s.val[0] != 0){
                        label_map[color].emplace_back(p); //inspired by example
                    }
                }
            }
        }
        //TODO check if this is correct
        //http://wiki.ros.org/image_transport/Tutorials/PublishingImages
        
        std::string encoding = res.masks[0].encoding;
        //encoding = "bgr8";
        encoding = "16UC1";
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), encoding, res_image).toImageMsg();

        msg->header.frame_id = color_img_cam.header.frame_id;
        msg->header.stamp = color_img_cam.header.stamp;

        sensor_msgs::ImagePtr depth_img = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr color_info = boost::make_shared<sensor_msgs::CameraInfo >(color_info_cam);
        sensor_msgs::CameraInfoPtr depth_info = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        seg_img_pub.publish(*msg);
        
        server_weight_callback(msg, depth_img,color_info, depth_info, weight_image, label_lookup, label_map);

    }
        //TODO this is slow
        published_to_cnn = false;

        
        
    }*/

    //can this be enhanced? TODO
    void MrcnnReceiver::cnn_callback(const mask_rcnn_ros::Result res){
        label_map.clear();
        coord_map.clear();

        if(!(res.masks.size() > 0)){
            published_to_cnn = false;
            return;
        }


        std::vector<cv_bridge::CvImagePtr> image_vec;

        for(int i = 0; i < res.masks.size(); i++){
            cv_bridge::CvImagePtr ptr;
            
            try{
                ptr = cv_bridge::toCvCopy(res.masks[i], res.masks[i].encoding);
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

        //from segmentation server, scale image up to mm
        /*if (depth_img_cam.encoding == "32FC1") {
            cv_bridge::CvImagePtr depth_img_mm = cv_bridge::toCvCopy(depth_img_cam, sensor_msgs::image_encodings::TYPE_32FC1);
            depth_img_mm->image.convertTo(depth_img_mm->image, CV_16U, 1000.0);
            depth = depth_img_mm;
        } else {
          depth = cv_bridge::toCvCopy(depth_img_cam, sensor_msgs::image_encodings::TYPE_16UC1);
        }*/

        //get 3d points from depth image (like in segmenter)
        image_geometry::PinholeCameraModel depth_cam_model;
        depth_cam_model.fromCameraInfo(depth_info_cam);

        /*cv::Mat points3d;
        cv::Matx33f intrinsic_matrix(depth_cam_model.fullIntrinsicMatrix());
        std::cout << intrinsic_matrix << std::endl;
        cv::rgbd::depthTo3d(depth->image, intrinsic_matrix, points3d);*/


        //lookup label for every string (if ids don't work)
        //also push back new vectors for each label
        //std::cout << "t1" << std::endl;
        std::vector<int> ids;
        for(int i = 0; i < res.class_names.size(); i++){
            //std::cout << "t2" << std::endl;
            std::unordered_map<std::string,int>::iterator it = label_string_mapping.find(res.class_names[i]);
            if(!(label_string_mapping.count(res.class_names[i]))){
                //std::cout << "t3" << std::endl;
                label_string_mapping[res.class_names[i]] = id_counter;
                ids.push_back(id_counter);
                id_counter++; 
            }else{
                //std::cout << "t4" << std::endl;
                ids.push_back(label_string_mapping[res.class_names[i]]);
            }
            //std::cout << "t5" << std::endl;
            
        }
        //std::cout << "masks: " << res.masks.size() << std::endl;

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
                        voxel.weight = res.scores[i];
                        vec.push_back(voxel);
                    }
                }
                //push back this pixel
                label_map.push_back(vec);
                
            }

            
        }

        sensor_msgs::PointCloud2 cl;
            pcl::toROSMsg(cloud,cl);
            cl.header = depth_img_cam.header;/*.stamp = ros::Time::now();
            cl.header.seq = depth_img_cam.header.seq;
            cl.header.frame_id = depth_img_cam.header.frame_id;*/

            debug_seg_cloud_pub.publish(cl);


        sensor_msgs::ImagePtr depth_img = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr color_info = boost::make_shared<sensor_msgs::CameraInfo >(color_info_cam);
        sensor_msgs::CameraInfoPtr depth_info = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        sensor_msgs::ImagePtr color_img_msg = boost::make_shared<sensor_msgs::Image >(color_img_cam);

        std::cout << "callback to server" << std::endl;
        server_weight_callback2(depth_img, color_img_msg, color_info, depth_info, label_map, coord_map, label_string_mapping, aligned);

        published_to_cnn = false;


    }



//to project the image just the image coordinates are needed
    void MrcnnReceiver::cnn_callback2(const mask_rcnn_ros::Result res){
        label_map.clear();
    

        if(!(res.masks.size() > 0)){
            published_to_cnn = false;
            return;
        }

        std::vector<cv_bridge::CvImagePtr> image_vec;

        for(int i = 0; i < res.masks.size(); i++){
            cv_bridge::CvImagePtr ptr;
            
            try{
                ptr = cv_bridge::toCvCopy(res.masks[i], res.masks[i].encoding);
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
        for(int i = 0; i < res.class_names.size(); i++){
            //std::cout << "t2" << std::endl;
            std::unordered_map<std::string,int>::iterator it = label_string_mapping.find(res.class_names[i]);
            if(!(label_string_mapping.count(res.class_names[i]))){
                //std::cout << "t3" << std::endl;
                label_string_mapping[res.class_names[i]] = id_counter;
                ids.push_back(id_counter);
                id_counter++; 
            }else{
                //std::cout << "t4" << std::endl;
                ids.push_back(label_string_mapping[res.class_names[i]]);
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
                        voxel.weight = res.scores[i];
                        vec.push_back(voxel);
                    }
                }
                //push back this pixel
                label_map.push_back(vec);
                
            }
        }


        sensor_msgs::ImagePtr depth_img = boost::make_shared<sensor_msgs::Image >(depth_img_cam);
        sensor_msgs::CameraInfoPtr color_info = boost::make_shared<sensor_msgs::CameraInfo >(color_info_cam);
        sensor_msgs::CameraInfoPtr depth_info = boost::make_shared<sensor_msgs::CameraInfo >(depth_info_cam);
        sensor_msgs::ImagePtr color_img_msg = boost::make_shared<sensor_msgs::Image >(color_img_cam);

        std::cout << "callback to server" << std::endl;
        server_weight_callback3(depth_img, color_img_msg, color_info, depth_info, label_map, label_string_mapping, aligned);

        published_to_cnn = false;


    }



}