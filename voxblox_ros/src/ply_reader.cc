#include <voxblox_ros/ply_reader.h>
#include <fstream>
#include <sstream>
#include <string>

namespace voxblox{

    PlyReader::PlyReader(const ros::NodeHandle& nh) : nh_(nh){
        //mesh_pub = nh.advertise<voxblox_msgs::Mesh>("mesh", 1, true);
        nh.param<float>("block_size", block_size, 0.02 * 16); //voxel_size * voxels_per_side

    }


    std::vector<mvertex> PlyReader::read_mesh(std::string filepath){
        std::ifstream in;
        in.open(filepath);

        std::vector<mvertex> vertices_vector;
      

        if(!in){
            ROS_ERROR("couldn't open mesh file");
            return vertices_vector;
        }

        bool header = true;
        bool vert = false;
        bool tria = false;
        int vertices = 0;
        int count = 0;
        std::string x;
        while(!in.eof()){
            in >> x;
            if(header){
                //maybe math stuff
                std::cout << x << std::endl;
                if(x.compare("element") == 0){
                    in >> x;
                    if(x.compare("vertex") == 0){
                        //http://www.cplusplus.com/reference/string/stoi/
                        //std::string::size_type sz;   // alias of size_t
                        in >> x;
                        vertices = std::stoi(x);
                        ROS_INFO("got vertice count");
                        std::cout << vertices << std::endl; 
                    }
                }

                if(x.compare("end_header") == 0){
                    header = false;
                    vert = true;
                    ROS_INFO("Header ended, beginning with vertices");
                }
            }else
            if(vert){
                if(count < vertices){
                    count++;
                
                
                //matching in default order
                //http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html
                mvertex v;

                //match element for element
                v.x = stof(x);
                
                in >> x;
                v.y = stof(x);
                
                in >> x;
                v.z = stof(x);
                
                in >> x;
                v.nx = stof(x);
                
                in >> x;
                v.ny = stof(x);

                in >> x;
                v.nz = stof(x);
                
                in >> x;
                v.r = stoi(x);
                
                in >> x;
                v.g = stoi(x);
                
                in >> x;
                v.b = stoi(x);
                
                in >> x;
                v.alpha = stoi(x);

                vertices_vector.push_back(v);
                
                }
                else{
                    vert = false;
                    tria = true;
                }

            }
            if(tria){

            }
        }

        in.close();
        std::cout << vertices_vector.size() << std::endl;

        return vertices_vector;
    }

    void PlyReader::compare(std::vector<mvertex> m1, std::vector<mvertex> m2){
        std::cout << m1.size() << " vertices and " << m2.size() << " vertices" << std::endl; 
        //iterate over vector 1

        int correct = 0;
        int wrong = 0;
        int labeled_not_labeled = 0;
        int not_labeled_labeled = 0;
        int unlabeled = 0;


        for(int i = 0; i < m1.size(); i++){
            if(i % 1000 == 0)
                std::cout << "step: " << i << std::endl;
            

            int smallest = 0;
            float distance = std::numeric_limits<float>::max();
            for(int j = 0; j < m2.size(); j++){
                float d = m1[i].getDistance(m2[j].x, m2[j].y, m2[j].z);
                if(d < distance){
                    smallest = j;
                    distance = d;
                }

                //maybe erase element?

            }

            //now compare
            mvertex v1 = m1[i];
            mvertex v2 = m2[smallest];
            if(v1.r == 0 && v1.g == 0 && v1.b == 0){
                if(v2.r == 0 && v2.g == 0 && v2.b == 0){
                    unlabeled++;
                }else{
                    not_labeled_labeled++;
                }
            }else{
                if(v1.r == v2.r && v1.g == v2.g && v1.b == v2.b){
                    correct++;
                }else{
                    if(v2.r == 0 && v2.g == 0 && v2.b == 0){
                        labeled_not_labeled++;
                    }
                    else{
                        wrong++;
                    }
                }
            }

        }

        std::cout << "correct: " << correct << std::endl;
        std::cout << "wrong: " << wrong << std::endl;
        std::cout << "unlabeled: " << unlabeled << std::endl;
        std::cout << "labeled_not_labeled: " << labeled_not_labeled << std::endl;
        std::cout << "not_labeled_labeled: " << not_labeled_labeled << std::endl; 
        std::cout << "vertices were compared to mesh 1, there may be duplicates: " << std::endl;
    }

    std::shared_ptr<MeshLayer> PlyReader::mvertex_to_layer(std::vector<mvertex> m){
        std::shared_ptr<MeshLayer> layer;
        layer.reset(new MeshLayer(block_size));

        //std::shared_ptr<MeshLayer> layer(block_size);

        //pushing_back points based on the blocks
        for(int i = 0; i < m.size(); i++){
            mvertex v = m[i];
            Point p(v.x, v.y, v.z);
            Mesh::Ptr mesh = layer->allocateMeshPtrByCoordinates(p);

            //resize by one (can we just add???)
            //Colors color;
            //mesh->resize(mesh->size() + 1);
            //Pointcloud pi;
            Point n(v.nx, v.ny, v.nz);
            Color c(v.r, v.g, v.b, v.alpha);
            mesh->vertices.push_back(p);
            mesh->normals.push_back(n);
            mesh->colors.push_back(c);
        }

        ROS_INFO("pushed back points");

        return layer;
    }

    void PlyReader::compare_to_layer(std::vector<mvertex> m1, std::shared_ptr<MeshLayer> layer2){
        std::cout << m1.size() << " vertices in m1, m2 unknown." << std::endl; 
        //iterate over vector 1

        int correct = 0;
        int wrong = 0;
        int labeled_not_labeled = 0;
        int not_labeled_labeled = 0;
        int unlabeled = 0;
        int no_corresp = 0;


        for(int i = 0; i < m1.size(); i++){
            if(i % 1000 == 0)
                std::cout << "step: " << i << std::endl;
            

            

            //get second mesh by block
            Point p1(m1[i].x, m1[i].y, m1[i].z);
            Mesh::Ptr m2 = layer2->getMeshPtrByCoordinates(p1);

            //no Ptr found
            if(!m2){
                no_corresp++;
                continue;
            }

            //No iterate over Mesh
            Pointcloud pc2 = m2->vertices;

            int smallest = 0;
            float distance = std::numeric_limits<float>::max();
            for(int j = 0; j < pc2.size(); j++){
                float d = m1[i].getDistance(pc2[j][0], pc2[j][1], pc2[j][2]);
                if(d < distance){
                    smallest = j;
                    distance = d;
                }

                //maybe erase element?

            }

            //now compare
            mvertex v1 = m1[i];
            Color c2 = m2->colors[smallest];
            //Point p2 = pc2->vertices[smallest];

            
            if(v1.r == 0 && v1.g == 0 && v1.b == 0){
                if(c2.r == 0 && c2.g == 0 && c2.b == 0){
                    unlabeled++;
                }else{
                    not_labeled_labeled++;
                }
            }else{
                if(v1.r == c2.r && v1.g == c2.g && v1.b == c2.b){
                    correct++;
                }else{
                    if(c2.r == 0 && c2.g == 0 && c2.b == 0){
                        labeled_not_labeled++;
                    }
                    else{
                        //std::cout << v1.r << " " << v1.g << " " << v1.b << ", " << (int) c2.r << " " << (int) c2.g << " " << (int) c2.b << std::endl;
                        wrong++;
                    }
                }
            }

        }

        std::cout << "correct: " << correct << std::endl;
        std::cout << "wrong: " << wrong << std::endl;
        std::cout << "unlabeled: " << unlabeled << std::endl;
        std::cout << "labeled_not_labeled: " << labeled_not_labeled << std::endl;
        std::cout << "not_labeled_labeled: " << not_labeled_labeled << std::endl; 
        std::cout << "vertices were compared to mesh 1, there may be duplicates: " << std::endl;
        std::cout << "no correspondence: " << no_corresp << std::endl;
    }
}