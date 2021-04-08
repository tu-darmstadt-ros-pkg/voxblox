#ifndef LABEL_MAP_H_
#define LABEL_MAP_H_

#include <ros/ros.h>

namespace voxblox {
struct labelMap {
  //std::vector<std::vector<std::pair<int, double>>> labels;
  std::vector<std::unordered_map<int, double>> labels;
  int width;
  int height;

  void init(int width, int height) {
    if (labels.size() == 0) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          //std::vector<std::pair<int, double>> l;
          std::unordered_map<int, double> l;
          labels.push_back(l);
        }
      }
      std::cout << labels.size() << std::endl;
    } else {
      std::cout << "already inited label map" << std::endl;
    }
  }

  std::unordered_map<int, double> get(int x, int y) {
    return labels[x + y * width];
  }

  void add(int x, int y, int label, double weight) {
    //std::pair<int, double> p = std::make_pair(label, weight);
    // maybe label map is now inverted (as before iterated over y first??)
    //labels[x + y * width].push_back(p);

    //check if element exists
    if((labels[x + y * width]).find(label) != (labels[x + y * width]).end()){ 
      labels[x + y * width][label] += weight;
    }else{
      labels[x + y * width][label] = weight;
    }
  }

  std::vector<std::pair<int, double>> get_labels(int x, int y){
    std::vector<std::pair<int, double>> labels_i;
    if(labels[x + y * width].size() == 0){
      //std::cout << "size 0" << std::endl;
    }

    for(auto const& entry : labels[x + y * width]){
      std::pair<int, double> p = std::make_pair(entry.first, entry.second);
      labels_i.push_back(p);
    }

    //std::cout << "got labels" << std::endl;

    return labels_i;
  }

  
  labelMap(int w, int h) {
    width = w;
    height = h;
    init(width, height);
  }
};
}  // namespace voxblox

#endif