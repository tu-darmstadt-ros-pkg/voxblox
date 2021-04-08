#include "voxblox/integrator/tsdf_seq_integrator.h"

#include <pcl/visualization/cloud_viewer.h>

#include <thread>
#include <chrono>


#include "voxblox/interpolator/interpolator.h"

//#include<"pcl/point_cloud.h">

//#include <pcl/common/common_headers.h>

namespace voxblox {
    
    template<class T>
    TsdfSeqIntegrator<T>::TsdfSeqIntegrator(ros::NodeHandle nh, const Layer<TsdfVoxel>& tsdf_layer, Layer<T>* label_layer, std::string integ_mode)
    : tsdf_layer_(tsdf_layer), label_layer_(label_layer), max_distance_(30.0), nh_(nh){
        
        if(integ_mode.compare("DEFAULT") == 0){
            mode = DEFAULT;
            ROS_INFO("DEFAULT integration mode");
        }else if(integ_mode.compare("CONFIDENCE") == 0){
            mode = CONFIDENCE;
            ROS_INFO("Integration based on confidence");
        }else if(integ_mode.compare("CONFIDENCE_WEIGHT") == 0){
            mode = CONFIDENCE_WEIGHT;
            ROS_INFO("Integration based on confidence and weight");
        }
        else{
            mode = DEFAULT;
            ROS_INFO("Wrong parameter for integration_mode, using default");
        }
        
        edge_im_c_pub = nh.advertise<sensor_msgs::Image>("/debug/edge_im_concave", 1, true);
        edge_im_d_pub = nh.advertise<sensor_msgs::Image>("/debug/edge_im_distance", 1, true);
        segmented_pub = nh.advertise<sensor_msgs::Image>("/debug/segmented_geometric", 1, true);
        fake_depth_image_pub = nh.advertise<sensor_msgs::Image>("/fake_depth_image", 1, true);

        nh.param<double>("treshold_concave", tresh_concave, 0.94);
        nh.param<double>("treshold_distance", tresh_distance, 0.94);
        nh.param<bool>("pcl_debug", show_pcl_debug, false);
        nh.param<double>("max_distance", max_dist, 20.0);

        std::string normal_mode;
        nh.param<std::string>("normal_calculation_mode", normal_mode, "CENTRALDIFFERENCES");

        if(normal_mode.compare("CENTRALDIFFERENCES") == 0){
            normal_calculation = CENTRALDIFFERENCES;
        }else if(normal_mode.compare("CROSSPRODUCT") == 0){
            normal_calculation = CROSSPRODUCT;
        }else if(normal_mode.compare("CENTRALDIFFERENCES_NORMALIZED") == 0){
            normal_calculation = CENTRALDIFFERENCES_NORMALIZED;
        }else if(normal_mode.compare("VOXBLOX") == 0){
            normal_calculation = VOXBLOX;
        }else{
            ROS_INFO("Wrong parameter for normal_calculation_mode, using central differences, using default");
        }


        

    }

    //TODO redo
    //like intensity integrator
    template<class T>
    void TsdfSeqIntegrator<T>::addVectors(const Point& origin, const Pointcloud& bearing_vectors, std::vector<int>& labels){

    }


    //like intensity integrator
    template<class T>
    void TsdfSeqIntegrator<T>::integrateByProjection(std::vector<std::vector<LabelVoxel>> label_map, 
                                              uint32_t width, 
                                              uint32_t height,
                                              double focal_length,
                                              double px,
                                              double py,
                                              Transformation T_G_C
    ){
        ROS_INFO("integrating with projection");

        const size_t num_pixels = width * height; //TODO add subsampling here
      

        //float half_row = height / 2.0;
        //float half_col = width / 2.0;
        Pointcloud bearing_vectors;
        bearing_vectors.reserve(num_pixels + 1);


        size_t k = 0;
        size_t m = 0;

        //TODO maybe do this with a cloud(colored) instead
        for (int j = 0; j < width; j++) {
            
            for (int i = 0; i < height; i++) {
                // subsample?
                bearing_vectors.push_back(
                  T_G_C.getRotation().toImplementation() *
                  Point(j - px, i - py, focal_length).normalized());
                
                //std::cout << "label " <<(int) image_row[i] << std::endl;
                k++;
            }      
            m++;
        }

        integrateVectors(T_G_C.getPosition(), bearing_vectors, label_map);
    }
        

    //also like intensity integrator
    template<class T>
    void TsdfSeqIntegrator<T>::integrateVectors(const Point& origin, const Pointcloud& bearing_vectors, std::vector<std::vector<LabelVoxel>> label_map){

        const FloatingPoint voxel_size = tsdf_layer_.voxel_size();

        FloatingPoint max_distance = (float) max_dist;


        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        for(size_t i = 0; i < bearing_vectors.size(); ++i){
            Point intersect = Point::Zero();


            //if no label exists
            if(label_map[i].size() == 0){
                continue;
            }

            bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(tsdf_layer_, origin, bearing_vectors[i], max_distance, &intersect);

            if(!success){
                continue;
            }

            integrate(intersect, label_map[i], cloud);

            //TODO maybe look at neighbours
        }

        //do sth. with the pcl

    }

    template<class T>
    void TsdfSeqIntegrator<T>::integrate(Point g, std::vector<LabelVoxel> v, pcl::PointCloud<pcl::PointXYZRGB> cloud){

        switch(mode){
                case DEFAULT:
                    integrateVoxel(g, v, cloud);
                    break;
                case CONFIDENCE:
                    integrateConfidenceVoxel(g, v, cloud);
                    break;
                case CONFIDENCE_WEIGHT:
                    integrateConfidenceWeightVoxel(g, v, cloud);
            }

    }


    //deprecated
    template<>
    void TsdfSeqIntegrator<LabelVoxel>::addVectors(const Point& origin, const Pointcloud& bearing_vectors, std::vector<int>& labels){

        timing::Timer label_timer("label/integrate");

        CHECK_EQ(bearing_vectors.size(), labels.size()) << "label and vector sizes don't match";

        const FloatingPoint voxel_size = tsdf_layer_.voxel_size();

        for(size_t i = 0; i < bearing_vectors.size(); ++i) {
            Point surface_intersection = Point::Zero();

            //now casting ray from origin along the vector to find surface intersection

            FloatingPoint max_distance = (float) max_dist;
            bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(tsdf_layer_, origin, bearing_vectors[i], max_distance, &surface_intersection);

            if(!success) {
                continue;
            }
            //std::cout << "got surface voxel" << std::endl;

            //lookup voxel and mark  (does this create always a new voxel??? TODO check)
            Block<LabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(surface_intersection);

            LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(surface_intersection);
            if(!labels[i] != 0){
              voxel.label_id = labels[i];
              voxel.weight = 1.0f;
            }
            //std::cout << "updated voxel" << std::endl;
            //std::cout << "integ " << surface_intersection[0] << std::endl;
            


            //check surrounding voxel???
        }


        //std::cout << label_layer_->getNumberOfAllocatedBlocks() << std::endl;
    }


    template<class T>
    pcl::PointCloud<pcl::PointXYZRGB> TsdfSeqIntegrator<T>::directlyIntegrate(std::vector<std::vector<LabelVoxel>> label_map, 
                                              std::vector<cv::Vec3f> coord_map,
                                              Transformation T_G_C){

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        
        //now just directly iterate over coordinates, multiply them by T_G_C, add voxel
        for(int i = 0; i < coord_map.size(); i++){
            
            //with no label, do nothing
            if(label_map[i].size() == 0){
                continue;
            }
            //get point in world g
            Point c(coord_map[i][0], coord_map[i][1], coord_map[i][2]);
            Point g = T_G_C * c;
            //std::cout << T_G_C << std::endl;

            //std::cout << "before directly integrate voxel" << std::endl;
            //integrate all information for this voxel; TODO make enhanced behaviour here
            switch(mode){
                case DEFAULT:
                    integrateVoxel(g, label_map[i], cloud);
                    break;
                case CONFIDENCE:
                    integrateConfidenceVoxel(g, label_map[i], cloud);
                    break;
                case CONFIDENCE_WEIGHT:
                    integrateConfidenceWeightVoxel(g, label_map[i], cloud);
            }
            
        }
        return cloud;
    }


    //case single label//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class T>
    void TsdfSeqIntegrator<T>::integrateVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){

    }

    template <>
    void TsdfSeqIntegrator<LabelVoxel>::integrateVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){
         
        //std::cout << "in integrate voxel" << std::endl;
        Block<LabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);

        //std::cout << "integrator " << g[0] << std::endl;
        LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);

        //just add the last one
        for(int i = 0; i < segments.size(); i++){
            //std::cout << "reading from voxel" << std::endl;
            voxel.label_id = segments[i].label_id;
            voxel.weight = segments[i].weight;
        }

        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);
        
    }

    

    template<class T>
    void TsdfSeqIntegrator<T>::integrateConfidenceVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){

    }

    template <>
    void TsdfSeqIntegrator<LabelVoxel>::integrateConfidenceVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){
        
        Block<LabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);
        LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);


        float epsilon = 0.0001;
        if(segments.size() == 0)
            return;

        //check if voxel was updated before
        if(voxel.label_id == 0){

            //if more than one label
            //maybe just order by weight TODO
            if(segments.size() > 1){
                voxel.label_id = segments[segments.size() - 1].label_id;
                voxel.weight = 0;
            }else{
                voxel.label_id = segments[0].label_id;
                voxel.weight = 1.0;
            }
        }else{

            
            if(segments.size() > 1){
                float normed = 1.0 / (float) segments.size();
                //if label of voxel is in list, don't change anything
                bool found = false;
                for(int i = 0; i < segments.size(); i++){
                
                    if(voxel.label_id == segments[i].label_id){
                        found = true;
                    }
                }
                if(!found){
                    //if low, set new voxel
                    if(voxel.weight - epsilon <= 1.0){
                        voxel.weight = 0.0;
                        voxel.label_id = segments[segments.size() -1].label_id;
                    }else{
                        voxel.weight -= 1.0;
                    }
                }
                
            }else{
                if(voxel.label_id == segments[0].label_id){
                    voxel.weight += 1.0;
                }
                else{
                    if(voxel.weight - epsilon <= 1.0){
                        voxel.label_id = segments[0].label_id;
                        voxel.weight = 0.0;
                    }else{
                        voxel.weight -= 1.0;
                    }
                }
            }
        }


        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);
    }

    template<class T>
    void TsdfSeqIntegrator<T>::integrateConfidenceWeightVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){

    }
    //if a voxel has multiple labels, multiply highest with internal prob ( weight/sum_weight) * weight
    template<>
    void TsdfSeqIntegrator<LabelVoxel>::integrateConfidenceWeightVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){

        Block<LabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);
        LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);

        if(segments.size() == 0)
            return;

        //check if voxel was updated before
        if(voxel.label_id == 0){

            //if more than one label
            //maybe just order by weight TODO
            if(segments.size() > 1){
                int highest = segments[0].label_id;
                int second_highest = segments[0].label_id;
                float highest_weight = segments[0].weight;
                float second_highest_weight = segments[0].weight;
                float sum_weight = 0.0;
            
                //get highest entry, and subtract second highest
                for(int i = 0; i < segments.size(); i++){
                    float weight = segments[i].weight;
                    if(weight > highest_weight){
                        second_highest_weight = highest_weight;
                        highest_weight = weight;
                        second_highest = highest;
                        highest = segments[i].label_id;
                    }else if(weight > second_highest){
                        second_highest_weight = weight;
                        second_highest = segments[i].label_id;
                    }
                    sum_weight += sum_weight;
                }
                voxel.label_id = highest;
                voxel.weight = highest_weight /sum_weight * highest_weight;
            }else{
                voxel.label_id = segments[0].label_id;
                voxel.weight = segments[0].weight;
            }
        }else{

            
            if(segments.size() > 1){
                int highest = segments[0].label_id;
                int second_highest = segments[0].label_id;
                float highest_weight = segments[0].weight;
                float second_highest_weight = segments[0].weight;
            
                //get highest entry, and subtract second highest
                float sum_weight = 0.0;
            
                //get highest entry, and subtract second highest
                for(int i = 0; i < segments.size(); i++){
                    float weight = segments[i].weight;
                    if(weight > highest_weight){
                        second_highest_weight = highest_weight;
                        highest_weight = weight;
                        second_highest = highest;
                        highest = segments[i].label_id;
                    }else if(weight > second_highest){
                        second_highest_weight = weight;
                        second_highest = segments[i].label_id;
                    }
                    sum_weight += weight;
                }
                float weight_normed = highest_weight/sum_weight * highest_weight;
                //check if label are the same
                if(voxel.label_id == highest){
                    voxel.weight += weight_normed;
                }
                else{
                    if(voxel.weight < weight_normed){
                        voxel.label_id = segments[0].label_id;
                        voxel.weight = weight_normed - voxel.weight;
                    }else{
                        voxel.weight -= weight_normed;
                    }
                }
                
            }else{
                if(voxel.label_id == segments[0].label_id){
                    voxel.weight += segments[0].weight;
                }
                else{
                    if(voxel.weight < segments[0].weight){
                        voxel.label_id = segments[0].label_id;
                        voxel.weight = segments[0].weight - voxel.weight;
                    }else{
                        voxel.weight -= segments[0].weight;
                    }
                }
            }
        }


        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);



    }


    //case multilabel///////////////////////////////////////////////////////////////

    //not that interesting
    template <>
    void TsdfSeqIntegrator<MultiLabelVoxel>::integrateVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){
        Block<MultiLabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);
        MultiLabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);

        //just add the last one
        for(int i = 0; i < segments.size(); i++){
            //std::cout << "reading from voxel" << std::endl;
            voxel.label_id = segments[i].label_id;
            voxel.weight = segments[i].weight;
            voxel.weights[segments[i].label_id] = 1.0;
        }

        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);
        
    }

    template <>
    void TsdfSeqIntegrator<MultiLabelVoxel>::integrateConfidenceVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){
        
        Block<MultiLabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);
        MultiLabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);


        float epsilon = 0.0001;
        if(segments.size() == 0)
            return;

        //check for each key if it exists and then add the weight
        for(int i = 0; i < segments.size(); i++){
            //std::cout << "reading from voxel" << std::endl;
            int label = segments[i].label_id;
            if(voxel.weights.find(label) != voxel.weights.end()){
                voxel.weights[label] += 1.0;
            }
            else{
                voxel.weights[label] = 1.0;
            }
        }

        //now iterate over all known keys and normalize
        int succ_label = 0;
        float succ_weight = 0.0;
        float acc_weight = 0.0;
        for(auto const& entry : voxel.weights){
            if(succ_weight <= entry.second){
                succ_label = entry.first;
                succ_weight = entry.second;
            }

            acc_weight += entry.second;

        }
        voxel.label_id = succ_label;
        voxel.weight = succ_weight / acc_weight;

        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);
    }



    template<>
    void TsdfSeqIntegrator<MultiLabelVoxel>::integrateConfidenceWeightVoxel(Point g, std::vector<LabelVoxel> segments, pcl::PointCloud<pcl::PointXYZRGB>& cloud){


        Block<MultiLabelVoxel>::Ptr block_ptr = label_layer_->allocateBlockPtrByCoordinates(g);
        MultiLabelVoxel& voxel = block_ptr->getVoxelByCoordinates(g);


        float epsilon = 0.0001;
        if(segments.size() == 0)
            return;

        //check for each key if it exists and then add the weight
        for(int i = 0; i < segments.size(); i++){
            //std::cout << "reading from voxel" << std::endl;
            int label = segments[i].label_id;
            if(voxel.weights.find(label) != voxel.weights.end()){
                voxel.weights[label] += segments[i].weight;
            }
            else{
                voxel.weights[label] = segments[i].weight;
            }
        }

        //now iterate over all known keys and normalize
        int succ_label = 0;
        float succ_weight = 0.0;
        float acc_weight = 0.0;
        for(auto const& entry : voxel.weights){
            if(succ_weight <= entry.second){
                succ_label = entry.first;
                succ_weight = entry.second;
            }

            acc_weight += entry.second;

        }
        voxel.label_id = succ_label;
        voxel.weight = succ_weight / acc_weight;


        pcl::PointXYZRGB p;
        p.x = g(0);
        p.y = g(1);
        p.z = g(2);
        if(voxel.label_id > 25){
            p.r = 255;
        }else{
            p.r = voxel.label_id * 10;
        }
        p.g = 0;
        p.b = 0;

        cloud.push_back(p);



    }

    //////////////////////////////////////////////////////////////////////
    inline std::vector<std::pair<int, int>> getNeighbours(int x, int y, int width, int height){
        //ROS_INFO("getting neighbours")
        std::vector<std::pair<int, int>> vec;

        for(int dx = -1; dx < 2; dx++){
            if(x + dx < 0){
                continue;
            }else if(x + dx >= width){
                continue;
            }


            for(int dy = -1; dy < 2; dy++){
                if(y + dy < 0){
                    continue;
                }else if(y + dy >= height){
                    continue;
                }

                //skip the point itself
                if(dx == 0 && dy == 0){
                    continue;
                }

                vec.push_back(std::make_pair(x + dx, y + dy));
            }
        }
        return vec;
    }

    inline void show_debug_normals(std::vector<point_map_entry> vertex_map, std::vector<point_map_entry> normal_map){
        //convert vertices and normals to point clouds (color by z???)
        pcl::PointCloud<pcl::PointXYZ>::Ptr vertex_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normal_cloud(new pcl::PointCloud<pcl::Normal>);

        ROS_INFO("in pcl debug");

        for(auto const& vertex : vertex_map){
            //ROS_INFO("iterating");
            pcl::PointXYZ p(vertex.x, vertex.y, vertex.z);
            //p.x = vertex.x;
            //p.y = vertex.y;
            //p.z = vertex.z;
            //ROS_INFO("adding point");
            vertex_cloud->push_back(p);
        }

        for(auto const& normal : normal_map){
            pcl::Normal n(normal.x, normal.y, normal.z);
            //n.n_x = normal.x;
            //n.n_y = normal.y;
            //n.n_z = normal.z;
            normal_cloud->push_back(n);
        }

        ROS_INFO("before creating viewer");
        //add them to viewer
        //http://pointclouds.org/documentation/tutorials/pcl_visualizer.php
        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
        viewer->initCameraParameters();
        ROS_INFO("before color handler");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(vertex_cloud, 0, 255, 0);
        ROS_INFO("before adding cloud");
        viewer->addPointCloud<pcl::PointXYZ> (vertex_cloud, single_color, "vertex cloud");
        ROS_INFO("before adding normals");
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (vertex_cloud, normal_cloud, 10, 0.05, "normals");

        
        //show viewer
        ROS_INFO("before spinning");
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }



    template<class T>
    std::pair<std::vector<point_map_entry>, std::vector<point_map_entry>> TsdfSeqIntegrator<T>::createVertexMap(double px, double py, double focal_length, Transformation T_G_C, uint32_t width, uint32_t height){
        //create matrix with image size (vertex map like tateno)
        std::vector<point_map_entry> vertex_map;
        std::vector<point_map_entry> real_vertex_map;

        const FloatingPoint voxel_size = tsdf_layer_.voxel_size();
        FloatingPoint max_distance = (float) max_dist; //TODO adjust to nh param

        //for each pixel shoot a vector to look up tsdf data

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
            
            
                // subsample?
                Point bearing_vector = T_G_C.getRotation().toImplementation() * Point(j - px, i - py, focal_length).normalized();
                

                Point intersect = Point::Zero();
                
                bool success = getSurfaceDistanceAlongRay<TsdfVoxel>(tsdf_layer_, T_G_C.getPosition(), bearing_vector, max_distance, &intersect);

                if(!success){
                    point_map_entry vertex;//TODO emplace?
                    real_vertex_map.push_back(vertex);
                    vertex_map.push_back(vertex);
                    continue;
                }
                else{
                    point_map_entry vertex;
                    vertex.x = intersect[0];
                    vertex.y = intersect[1];
                    vertex.z = intersect[2];
                    vertex.valid_point = true;

                    real_vertex_map.push_back(vertex);

                    Point camera_frame_point = T_G_C.inverseTransform(intersect);

                    point_map_entry cvertex;
                    cvertex.x = camera_frame_point[0];
                    cvertex.y = camera_frame_point[1];
                    cvertex.z = camera_frame_point[2];
                    cvertex.valid_point = true;
                    //std::cout << cvertex.x << " " << cvertex.y << " " << cvertex.z << std::endl;

                    vertex_map.push_back(cvertex);
                }
            }      
        }

        return std::make_pair(vertex_map, real_vertex_map);
    }
    template<class T>
    std::vector<point_map_entry> TsdfSeqIntegrator<T>::getVoxbloxNormals(std::vector<point_map_entry> vertex_map){
        //in voxblox case, work with real coordinates?
        ROS_INFO("creating interpolator");
        Interpolator<TsdfVoxel> interpolator(&tsdf_layer_);
        ROS_INFO("created interpolator");

        std::vector<point_map_entry> normal_map;


        //iterate over vertex map
        for(auto const& vertex : vertex_map){
            Point p(vertex.x, vertex.y, vertex.z);
            Point gradient;
            bool success = interpolator.getGradient(p.cast<FloatingPoint>(),
                                         &gradient, false);

            if(success){
                point_map_entry normal;
                normal.valid_point = true;
                normal.x = gradient[0];
                normal.y = gradient[1];
                normal.z = gradient[2];
                normal_map.push_back(normal);
            }else{
                point_map_entry normal;
                normal_map.push_back(normal);
            }
        }

        return normal_map;
    }

    template<class T>
    std::vector<point_map_entry> TsdfSeqIntegrator<T>::calculateNormalsCentralDifferences(std::vector<point_map_entry> vertex_map, uint32_t width, uint32_t height, double px, double py, double focal_lengthx, double focal_lengthy){
        //for border values mirror values
        //TODO set max_dist else
        bool actual_diff = false;
        if(normal_calculation ==CENTRALDIFFERENCES_NORMALIZED){
            actual_diff = true;
        }

        std::vector<point_map_entry> normal_map;

        
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                int xl = x - 1;
                int yu = y - 1;
                int xr = x + 1;
                int yd = y + 1;

                //if point is not valid, don't calculate a normal for it
                if(!vertex_map[x + y * width].valid_point){
                    point_map_entry normal;
                    normal_map.push_back(normal);
                    continue;
                }

                //replicate on border case
                if(xl < 0)
                    xl = 0;
                if(yu < 0)
                    yu = 0;
                if(xr >= width)
                    xr = width - 1;
                if(yd >= height)
                    yd = height - 1;

                //calculate x difference
                //if point is not valid, treat it as max range
                double xleft = max_dist;
                double x1;
                if(vertex_map[xl + y * width].valid_point){
                    xleft = vertex_map[xl + y * width].z;
                    x1 = vertex_map[xl + y * width].x;
                }else{
                    //calculate missing x
                    x1 = (xl - px) * max_dist / focal_lengthx; //maybe just calculate the distance at max range?
                }
                double xright = max_dist;
                double x2;
                if(vertex_map[xr + y * width].valid_point){
                    xright = vertex_map[xr + y * width].z;
                    x2 = vertex_map[xr + y * width].x;
                    
                }
                else{
                    //calculate missing x
                    x2 = (xr - px) * max_dist / focal_lengthx;
                }

                double xdiff = xleft - xright;
                
                //normalize by actual difference
                if(actual_diff)
                    xdiff /= std::abs(x2-x1); //normalize the normal with the actual difference and not the difference in the image
                else
                    xdiff/= (2.0/1000.0); //scaling by 1000 to get mm?
                

                //calculate y difference
                double yup = max_dist;
                double y1;
                if(vertex_map[x + yu * width].valid_point){
                    yup = vertex_map[x + yu * width].z;
                    y1 = vertex_map[x + yu * width].y;
                }else{
                    //calculate missing y
                    y1 = (yu - py) * max_dist / focal_lengthy;
                }
                double ydown = max_dist; 
                double y2;
                if(vertex_map[x + yd * width].valid_point){
                    ydown = vertex_map[x + yd * width].z;
                    y2 = vertex_map[x + yd * width].y;
                }else{
                    //calculate missing x
                    y2 = (yd - py) * max_dist / focal_lengthy;
                }

                double ydiff = yup - ydown;
                if(actual_diff)
                    ydiff /= std::abs(y2 - y1);
                else
                    ydiff /= (2.0/1000.0); //scaling by 1000 to get mm?

                //calulate magnitude
                double magnitude = std::sqrt(xdiff * xdiff + ydiff * ydiff + 1.0);

                //create normal
                point_map_entry normal;
                normal.valid_point = true;
                normal.x = -xdiff / magnitude; 
                normal.y = -ydiff / magnitude;
                normal.z = -1.0 / magnitude; //-1 to point at the camera? //for all directions -1?
                normal_map.push_back(normal);

            }
        }

        return normal_map;
    }

    template<class T>
    cv::Mat TsdfSeqIntegrator<T>::createEdgeImage(std::vector<point_map_entry> vertex_map, std::vector<point_map_entry> normal_map, uint32_t width, uint32_t height){
        //now do the calculations from https://arxiv.org/pdf/1804.09194.pdf
        //also add them to an image

        //https://answers.ros.org/question/195979/creating-sensor_msgsimage-from-scratch/
        //creating empty debug images
        sensor_msgs::Image image_concave;
        sensor_msgs::Image image_distance;
        
        image_concave.height = height;
        image_concave.width = width;
        image_concave.encoding = "8UC1";
        image_concave.is_bigendian = false;
        image_concave.step = width;

        image_distance.height = height;
        image_distance.width = width;
        image_distance.encoding = "8UC1";
        image_distance.is_bigendian = false;
        image_distance.step = width;

        cv::Mat edge_im(cv::Size(width, height), CV_8UC1, cv::Scalar(255));

        //debug errors
        int tresh = 0;
        int neigh = 0;
        int invalid_p = 0;
        int invalid_n = 0;
        int no_tresh = 0;

        ROS_INFO("calculate edges");
        //iterate over image
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
            
                //now calculate for all neighbours
                std::vector<std::pair<int, int>> neighbours = getNeighbours(x, y, width, height);

                double phi_d = std::numeric_limits<double>::min();
                double phi_c = std::numeric_limits<double>::min();

                //if point is not valid or has no normal, set it to edge
                if(!vertex_map[x + y * width].valid_point || !normal_map[x + y * width].valid_point){
                    vertex_map[x + y * width].edge = true;

                        //coloring debug image
                        uint8_t color = 100;
                        image_concave.data.push_back(color);
                        image_distance.data.push_back(color);

                        edge_im.at<uint8_t>(y, x) = 0;//set to edge

                        if(!vertex_map[x + y * width].valid_point)//debug counter
                            invalid_p++;

                        if(!normal_map[x + y * width].valid_point)
                        invalid_n++;
                        continue;
                }

                bool worked = false;
                bool invalid = false;
                for(int i = 0; i < neighbours.size(); i++){
                    int xi = neighbours[i].first;
                    int yi = neighbours[i].second;
                    //lookup if point is valid
                    if(!vertex_map[xi + yi * width].valid_point){//} || !normal_map[xi][yi].valid_point){
                        //neighbour is not valid, so this must be an edge point
                        //TODO notate this IMPORTANT
                        
                        //treat as if not worked
                        vertex_map[x + y * width].edge = true;
                        uint8_t color = 50;
                        image_concave.data.push_back(color);
                        image_distance.data.push_back(color);

                        edge_im.at<uint8_t>(y, x) = 0;//set to edge


                        invalid = true;
                        neigh++;
                        break;
                    }

                    //if neighbour has no normal, skip him
                    if(!normal_map[xi + yi * width].valid_point){
                        continue;
                    }

                    //written out dot product
                    double delta = (vertex_map[xi + yi * width].x - vertex_map[x + y * width].x) * normal_map[x + y * width].x +
                                   (vertex_map[xi + yi * width].y - vertex_map[x + y * width].y) * normal_map[x + y * width].y +
                                   (vertex_map[xi + yi * width].z - vertex_map[x + y * width].z) * normal_map[x + y * width].z;
                    phi_d = std::max(abs(delta), phi_d);

                    if(delta < 0){//error in tateno? not inverting?
                        phi_c = std::max(0.0, phi_c);
                    }else{
                        double diff = 1 -
                                      (normal_map[xi + yi * width].x * normal_map[x + y * width].x + 
                                       normal_map[xi + yi * width].y * normal_map[x + y * width].y + 
                                       normal_map[xi + yi * width].z * normal_map[x + y * width].z);
                        phi_c = std::max(diff, phi_c);
                    }
                    //to check that we worked on this point
                    worked = true;
                    
                }
                if(invalid){
                    continue;
                }

                //if we didn't work on this point, treat it as edge anyway (becuase no real neighbour exists)
                if(!worked){
                    vertex_map[x + y * width].edge = true;
                    uint8_t color = 50;
                    image_concave.data.push_back(color);
                    image_distance.data.push_back(color);

                    edge_im.at<uint8_t>(y, x) = 0;//set to edge

                    neigh++;
                    continue;
                }


                //compare to treshold
                if(phi_d > tresh_distance){
                    
                    vertex_map[x + y * width].edge = true;
                    uint8_t color = 0;
                    image_distance.data.push_back(color);

                    edge_im.at<uint8_t>(y, x) = 0;//set to edge

                    if(phi_c > tresh_concave){
                        image_concave.data.push_back(color);
                    }else{
                        color = 255;
                        image_concave.data.push_back(color);
                    }

                    tresh++;

                }
                else if(phi_c > tresh_concave){
                    vertex_map[x + y * width].edge = true;
                    uint8_t color = 0;
                    image_concave.data.push_back(color);

                    //distance must be white
                    color = 255;
                    image_distance.data.push_back(color);

                    edge_im.at<uint8_t>(y, x) = 0;//set to edge

                    tresh++;
                }
                else{
                    //no edge
                    uint8_t color = 255;
                    image_distance.data.push_back(color);
                    image_concave.data.push_back(color);

                    no_tresh++;
                }
            }


        }

        //publish image
        //edge_im_pub.publish(image)
        image_distance.header.stamp = ros::Time::now();
        image_concave.header.stamp = ros::Time::now();
        ROS_INFO("finished, returning image");
        /*std::cout << image_distance.data.size() << std::endl;
        std::cout << image_concave.data.size() << std::endl;
        
        std::cout << width * height << std::endl;
        std::cout << "tresh " <<  tresh << std::endl;
        std::cout << "no tresh" << no_tresh << std::endl;
        std::cout << "neighbour errors" << neigh << std::endl;
        std::cout << "invalid point" << invalid_p << std::endl;
        std::cout << "invalid_normal" << invalid_n << std::endl;*/

        //return image;

        edge_im_c_pub.publish(image_concave);
        edge_im_d_pub.publish(image_distance);

        return edge_im;
    }

    template<class T>
    std::pair<std::vector<std::unordered_map<int, double>>, cv::Mat> TsdfSeqIntegrator<T>::createSegmentation(cv::Mat edge_im, std::vector<std::vector<LabelVoxel>> label_map, uint32_t width, uint32_t height){
        //segment with cv connected components, thanks to marius
        cv::Mat connected;
        int num_connected = cv::connectedComponents(edge_im, connected, 4, CV_16U);
        ROS_INFO("connected comp");
        //keep only labels without edge
        cv::Mat segmented(cv::Size(width, height), CV_8UC1, cv::Scalar(0));
        /*cv::add(connected, segmented, segmented, edge_im);

        */
        //now iterate over connected image
        for(int x = 0; x < width; x++){
            for(int y = 0; y < height; y++){

                //check if edge
                if(edge_im.at<uint8_t>(y, x) > 0){
                    
                    ushort pixel_label = connected.at<ushort>(y, x);

                    //add to image
                    segmented.at<uint8_t>(y, x) = pixel_label;

                }

            }
        }

        cv_bridge::CvImage segmented_img;
        ROS_INFO("cv bridge without header");
        //segmented_img.header = image_distance.header;
        segmented_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
        segmented_img.image = segmented;

        //segmented_pub.publish(segmented_img.toImageMsg());
        ROS_INFO("edge image debug pub");


        //create vector for calculation
        std::vector<std::unordered_map<int, double>> segment_labels;

        //skip background
        for(int i = 0; i < num_connected; i++ ){
            std::unordered_map<int, double> labels;
            segment_labels.push_back(labels); 
        }


        //iterate over label_map
        int i = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                //TODO maybe also use weight here
                i = y + x * height;

                //check if point is valid
                ushort seg_id = connected.at<ushort>(y, x);
                uint8_t edge_id = edge_im.at<uint8_t>(y, x);

                if(seg_id == 0 || edge_id == 0){
                    
                    continue;
                }

                //now iterate over possible labels
                for(int j = 0; j < label_map[i].size(); j++){
                    LabelVoxel v = label_map[i][j];

                    //check if label already exists
                    std::unordered_map<int, double>::iterator it = segment_labels[seg_id].find(v.label_id);
                    if(it == segment_labels[seg_id].end()){
                        segment_labels[seg_id][v.label_id] = 0.0;
                    }

                    //TODO maybe use modes here
                    segment_labels[seg_id][v.label_id] += 1.0;
                }

            }
        }
        ROS_INFO("matched labels onto segmented");

        return std::make_pair(segment_labels, connected);
    }
    

    template<class T>
    std::vector<LabelVoxel> TsdfSeqIntegrator<T>::getWinningLabels(std::vector<std::unordered_map<int, double>> segment_labels){
        //now get the winning label
        std::vector<LabelVoxel> winning_labels;
        for(int i = 0; i < segment_labels.size(); i++){
            std::unordered_map<int, double> labels = segment_labels[i];

            LabelVoxel v;

            if(segment_labels[i].size() == 0){
                v.label_id = 0;
                v.weight = 0.0;
                winning_labels.push_back(v); //no label found for this segment -> background
                continue;
            }
            int winning_label = 0;
            double weight = 0.0;
            double sum_weight = 0.0;

            for( auto l : labels){
                if(l.second > weight){
                    weight = l.second;
                    winning_label = l.first;
                    sum_weight += l.second;
                }
            }
            
            v.label_id = winning_label;
            v.weight = weight / sum_weight; //normalize
            winning_labels.push_back(v);
        }

        return winning_labels;
    }


    //changed from marius version
    template<class T>
    std::vector<point_map_entry> TsdfSeqIntegrator<T>::estimateNormalsCrossProduct(std::vector<point_map_entry> vertex_map, uint32_t width, uint32_t height) {


        std::vector<point_map_entry> normal_map;

        
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                int xl = x - 1;
                int yu = y - 1;
                int xr = x + 1;
                int yd = y + 1;

                //if point is not valid, don't calculate a normal for it
                if(!vertex_map[x + y * width].valid_point){
                    point_map_entry normal;
                    normal_map.push_back(normal);
                    continue;
                }

                //replicate on border case
                if(xl < 0)
                    xl = 0;
                if(yu < 0)
                    yu = 0;
                if(xr >= width)
                    xr = width - 1;
                if(yd >= height)
                    yd = height - 1;


            //scaling by 1000, because depth_img mat was scaled by 1000
            double z;
            //nw
            z = max_dist;
            if(vertex_map[xl + yu * width].valid_point)
                z = vertex_map[xl + yu * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d nw(xl, yu, z);

            //n
            z = max_dist;
            if(vertex_map[x + yu * width].valid_point)
                z = vertex_map[x + yu * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d n(x, yu, z);

            //ne
            z = max_dist;
            if(vertex_map[xr + yu * width].valid_point)
                z = vertex_map[xr + yu * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d ne(xr, yu, z);    

            //w
            z = max_dist;
            if(vertex_map[xl + y * width].valid_point)
                z = vertex_map[xl + y * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d w(xl, y, z);    

            //e
            z = max_dist;
            if(vertex_map[xr + y * width].valid_point)
                z = vertex_map[xr + y * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d e(xr, y, z);   

            //sw
            z = max_dist;
            if(vertex_map[xl + yd * width].valid_point)
                z = vertex_map[xl + yd * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d sw(xl, yd, z);    

            //s
            z = max_dist;
            if(vertex_map[x + yd * width].valid_point)
                z = vertex_map[x + yd * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d s(x, yd, z); 

            //se
            z = max_dist;
            if(vertex_map[xr + yd * width].valid_point)
                z = vertex_map[xr + yd * width].z;//  * 1000; //scaling by 1000 to match the original depth image? (does it even matter?);
            cv::Vec3d se(xr, yd, z);

            cv::Vec3d n1 = (sw-n).cross(se-n);
            cv::Vec3d n2 = (nw-e).cross(sw-e);
            cv::Vec3d n3 = (ne-s).cross(nw-s);
            cv::Vec3d n4 = (se-w).cross(ne-w);

            cv::Vec3d normal_vec = cv::normalize(0.25 * (n1 + n2 + n3 + n4));

            //create normal
            point_map_entry normal;
            normal.valid_point = true;
            normal.x = normal_vec[0]; 
            normal.y = normal_vec[1];
            normal.z = normal_vec[2];
            normal_map.push_back(normal);


            }
        }

        return normal_map;
}
    

    //Projecting rgb image to get fake depth image
    template<class T>
    void TsdfSeqIntegrator<T>::integrateByGeometricSegmentation(const sensor_msgs::ImageConstPtr& color_img, 
                                          const sensor_msgs::CameraInfoConstPtr& color_info, 
                                          std::vector<std::vector<LabelVoxel>> label_map,
                                          Transformation T_G_C
    ){
        //project color image onto tsdf to get hits for fake depth image

        //like intensity integrator
        ROS_INFO("creating vectors to get depth from tsdf");

        uint32_t height = color_info->height;
        uint32_t width = color_info->width;
        double px = color_info->K[2];
        double py = color_info->K[5];
        double focal_lengthx = color_info->K[0];
        double focal_lengthy = color_info->K[4];

        const size_t num_pixels = width * height; //TODO add subsampling here

        size_t k = 0;
        size_t m = 0;


        std::pair<std::vector<point_map_entry>, std::vector<point_map_entry>> vertex_maps = createVertexMap(px, py, focal_lengthx, T_G_C, width, height);

        std::vector<point_map_entry> vertex_map = vertex_maps.first;
        std::vector<point_map_entry> real_vertex_map = vertex_maps.second;
       
       
        //debug convert vertex map to depth image and publish it
        sensor_msgs::Image fake_depth;
        fake_depth.height = height;
        fake_depth.width = width;
        fake_depth.encoding = "8UC1";
        fake_depth.is_bigendian = false;
        fake_depth.step = width;
        
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                
                uint8_t color = 0;
                if(vertex_map[x + y * width].valid_point){
                    //std::cout << vertex_map[x + y * width].z * 100 << std::endl;
                    if(std::round(vertex_map[x + y * width].z * 100) < 0){
                        color = 0;
                    }else if(std::round(vertex_map[x + y * width].z * 100) > 255)
                        color = 255;
                    else
                        color = (uint8_t) std::round(vertex_map[x + y * width].z * 100);
                    
                }
                fake_depth.data.push_back(color);
            }
        }
        fake_depth_image_pub.publish(fake_depth);

        
        

        ROS_INFO("created vertex map");

        
        ROS_INFO("calculating normals");
        std::vector<point_map_entry> normal_map;


        switch(normal_calculation){
            case CROSSPRODUCT://marius approach
                normal_map = estimateNormalsCrossProduct(vertex_map, width, height);
                break;
            case CENTRALDIFFERENCES:
            case CENTRALDIFFERENCES_NORMALIZED:
                normal_map = calculateNormalsCentralDifferences(vertex_map, width, height, px, py, focal_lengthx, focal_lengthy);
                break;
            case VOXBLOX:
                
                //set vertex_map to real_vertex_map, as we work in the world frame now
                vertex_map = real_vertex_map;
                normal_map = getVoxbloxNormals(vertex_map);
                
                break;

        }
        //debug option
        if(show_pcl_debug)
            show_debug_normals(vertex_map, normal_map);

        ROS_INFO("creating image");

        cv::Mat edge_im = createEdgeImage(vertex_map, normal_map, width, height);

        ROS_INFO("create segmentation");

        std::pair<std::vector<std::unordered_map<int, double>>, cv::Mat> pair = createSegmentation(edge_im, label_map, width, height);
        std::vector<std::unordered_map<int, double>> segment_labels = pair.first;
        cv::Mat connected = pair.second;


        ROS_INFO("created segment labels");
        
        std::vector<LabelVoxel> winning_labels = getWinningLabels(segment_labels);


        
        ROS_INFO("got best label per region");

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        //now integrate
        //iterate over image points
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){

                //check if no edge
                if(!edge_im.at<uint8_t>(y, x) > 0){
                    continue;
                }
                //check if it has a label (other than 0)
                short segment_id = connected.at<short>(y, x);
                if(winning_labels[segment_id].label_id == 0){
                    continue;
                }

                //get point, use real_vertex map here (to correctly reproject)
                Point p;
                p[0] = real_vertex_map[x + y * width].x;
                p[1] = real_vertex_map[x + y * width].y;
                p[2] = real_vertex_map[x + y * width].z;


                std::vector<LabelVoxel> label;
                label.push_back(winning_labels[segment_id]);

                switch(mode){
                case DEFAULT:
                    integrateVoxel(p, label, cloud);
                    break;
                case CONFIDENCE:
                    integrateConfidenceVoxel(p, label, cloud);
                    break;
                case CONFIDENCE_WEIGHT:
                    integrateConfidenceWeightVoxel(p, label, cloud);
                }
            }
        }
        ROS_INFO("integrated");

        //maybe publish cloud?

        
    }

    //trying this if image and depth are aligned
    //Projecting rgb image to get fake depth image
    template<class T>
    void TsdfSeqIntegrator<T>::integrateByGeometricSegmentationAligned(cv::Mat depth_img,
                                          const sensor_msgs::CameraInfoConstPtr& depth_info, 
                                          std::vector<std::vector<LabelVoxel>> label_map,
                                          Transformation T_G_C
    ){
        //project color image onto tsdf to get hits for fake depth image

        //like intensity integrator
        ROS_INFO("aligned case");
       

        uint32_t height = depth_info->height;
        uint32_t width = depth_info->width;
        double px = depth_info->K[2];
        double py = depth_info->K[5];
        double focal_lengthx = depth_info->K[0];
        double focal_lengthy = depth_info->K[4];

        const size_t num_pixels = width * height; //TODO add subsampling here

        size_t k = 0;
        size_t m = 0;


        //std::pair<std::vector<point_map_entry>, std::vector<point_map_entry>> vertex_maps = createVertexMapFromDepth(px, py, focal_length, T_G_C, width, height);

        //std::vector<point_map_entry> vertex_map = vertex_maps.first;
        //std::vector<point_map_entry> real_vertex_map = vertex_maps.second;
       
       
        //debug convert vertex map to depth image and publish it
        /*sensor_msgs::Image fake_depth;
        fake_depth.height = height;
        fake_depth.width = width;
        fake_depth.encoding = "8UC1";
        fake_depth.is_bigendian = false;
        fake_depth.step = width;
        
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                
                uint8_t color = 0;
                if(vertex_map[x + y * width].valid_point){
                    //std::cout << vertex_map[x + y * width].z * 100 << std::endl;
                    if(std::round(vertex_map[x + y * width].z * 100) < 0){
                        color = 0;
                    }else if(std::round(vertex_map[x + y * width].z * 100) > 255)
                        color = 255;
                    else
                        color = (uint8_t) std::round(vertex_map[x + y * width].z * 100);
                    
                }
                fake_depth.data.push_back(color);
            }
        }
        fake_depth_image_pub.publish(fake_depth);*/

        //creating vertex map from depth image
        //resuing from marius
        float center_x = static_cast<float>(depth_info->K[2]);
        float center_y = static_cast<float>(depth_info->K[5]);
        float f_x = static_cast<float>(depth_info->K[0]);
        float f_y = static_cast<float>(depth_info->K[4]);

        std::vector<point_map_entry> vertex_map;
        std::vector<point_map_entry> real_vertex_map;
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                //TODO check if this skips max range
                if(depth_img.at<uint16_t>(y, x) == 0){
                    point_map_entry vertex;
                    vertex_map.push_back(vertex);
                    real_vertex_map.push_back(vertex);
                    continue;
                }
                //resuing from marius
                double scaled_depth = 0.001f * static_cast<double>(depth_img.at<uint16_t>(y, x));

                point_map_entry vertex;
                vertex.valid_point = true;

                //resuing from marius
                vertex.x = (x - center_x) * scaled_depth / f_x;;
                vertex.y = (y - center_y) * scaled_depth / f_y;
                vertex.z = scaled_depth;
                vertex_map.push_back(vertex);

                Point p(vertex.x, vertex.y, vertex.z);

                //now convert to real world point
                Point world_point = T_G_C.transform(p);

                point_map_entry wvertex;
                wvertex.x = world_point[0];
                wvertex.y = world_point[1];
                wvertex.z = world_point[2];
                wvertex.valid_point = true;
                    

                real_vertex_map.push_back(wvertex);
            }
        }
        
        
        

        ROS_INFO("created vertex map");

        
        ROS_INFO("calculating normals");
        std::vector<point_map_entry> normal_map;

        switch(normal_calculation){
            case CROSSPRODUCT://marius approach
                normal_map = estimateNormalsCrossProduct(vertex_map, width, height);
                break;
            case CENTRALDIFFERENCES:
            case CENTRALDIFFERENCES_NORMALIZED:
                normal_map = calculateNormalsCentralDifferences(vertex_map, width, height, px, py, focal_lengthx, focal_lengthy);
                break;
            case VOXBLOX:
                ROS_INFO("ERROR Voxblox normals not possible for raw input image, using central differences");
                normal_map = calculateNormalsCentralDifferences(vertex_map, width, height, px, py, focal_lengthx, focal_lengthy);
                break;

        }
        

 
        //debug option
        if(show_pcl_debug)
            show_debug_normals(vertex_map, normal_map);


        ROS_INFO("creating image");

        cv::Mat edge_im = createEdgeImage(vertex_map, normal_map, width, height);

        ROS_INFO("create segmentation");

        std::pair<std::vector<std::unordered_map<int, double>>, cv::Mat> pair = createSegmentation(edge_im, label_map, width, height);
        std::vector<std::unordered_map<int, double>> segment_labels = pair.first;
        cv::Mat connected = pair.second;


        ROS_INFO("created segment labels");
        
        std::vector<LabelVoxel> winning_labels = getWinningLabels(segment_labels);


        
        ROS_INFO("got best label per region");

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        //now integrate
        //iterate over image points
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){

                //check if no edge
                if(!edge_im.at<uint8_t>(y, x) > 0){
                    continue;
                }
                //check if it has a label (other than 0)
                short segment_id = connected.at<short>(y, x);
                if(winning_labels[segment_id].label_id == 0){
                    continue;
                }

                //get point, use real_vertex map here (to correctly reproject)
                Point p;
                p[0] = real_vertex_map[x + y * width].x;
                p[1] = real_vertex_map[x + y * width].y;
                p[2] = real_vertex_map[x + y * width].z;


                std::vector<LabelVoxel> label;
                label.push_back(winning_labels[segment_id]);

                switch(mode){
                case DEFAULT:
                    integrateVoxel(p, label, cloud);
                    break;
                case CONFIDENCE:
                    integrateConfidenceVoxel(p, label, cloud);
                    break;
                case CONFIDENCE_WEIGHT:
                    integrateConfidenceWeightVoxel(p, label, cloud);
                }
            }
        }
        ROS_INFO("integrated");

        //maybe publish cloud?

        
    }

    


    template class TsdfSeqIntegrator<LabelVoxel>;
    template class TsdfSeqIntegrator<MultiLabelVoxel>;
}