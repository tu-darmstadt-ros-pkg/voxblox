#ifndef VERTEX_MAP_H_
#define VERTEX_MAP_H_

#include <ros/ros.h>
#include <voxblox/core/common.h>

#include "voxblox/integrator/integrator_utils.h"
#include "voxblox/utils/distance_utils.h"

namespace voxblox {
// struct to track the vertices belonging to the color pixels
struct point_entry {
  Point p;
  bool valid_point = false;
  int segment_id = -1;
  int pcl_index = 0;
  bool edge = false;
};

struct vertexMap {
  std::vector<point_entry> vertices;
  int width;
  int height;

  void init(int width, int height) {
    if (vertices.size() == 0) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          point_entry p;
          vertices.push_back(p);
        }
      }
    } else {
      std::cout << "already inited vertex map" << std::endl;
    }
  }

  point_entry* get(int x, int y) { return &vertices[x + y * width]; }

  void set(Point p, int x, int y, bool valid) {
    vertices[x + y * width].p = p;
    vertices[x + y * width].valid_point = valid;
  }

  vertexMap(int w, int h) {
    width = w;
    height = h;
    init(width, height);
  }
};

typedef vertexMap normalMap;
}  // namespace voxblox
#endif