#ifndef VOXBLOX_PLY_READER_H_
#define VOXBLOX_PLY_READER_H_

#include <ros/ros.h>

#include "voxblox/core/layer.h"
#include "voxblox/core/voxel.h"

#include <voxblox/mesh/mesh_integrator.h>

#include <voxblox_msgs/Mesh.h>

namespace voxblox{

    struct mvertex{
        float x, y, z, nx, ny, nz;
        int r, g, b, alpha;

        

        bool isNear(float ix, float iy, float iz){
            if(std::abs(x - ix) < 10e-4 && std::abs(y - iy) < 10e-4 && std::abs(z -iz) < 10e-4){
                return true;
            }
            return false;
        }

        float getDistance(float ix, float iy, float iz){
            return (x - ix) * (x - ix) + (y - iy) * (y - iy) + (z - iz) * (z - iz);
        }

    };

    

    class PlyReader {
        public:
            PlyReader(const ros::NodeHandle& nh);

            std::vector<mvertex> read_mesh(std::string filepath);

            void compare(std::vector<mvertex> m1, std::vector<mvertex> m2);

            void compare_to_layer(std::vector<mvertex> m1, std::shared_ptr<MeshLayer> layer2);

            std::shared_ptr<MeshLayer> mvertex_to_layer(std::vector<mvertex> m);

        private:
            ros::NodeHandle nh_;
            
            std::string file1;
            std:: string file2;

            ros::Publisher mesh_pub_;

            float block_size;

    };
}

#endif