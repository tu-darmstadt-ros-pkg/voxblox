#include "voxblox/integrator/SegTsdfIntegrator.h"

namespace voxblox {

template <class T>
SegTsdfIntegrator<T>::SegTsdfIntegrator(ros::NodeHandle nh,
                                        const Layer<TsdfVoxel>& tsdf,
                                        std::shared_ptr<Layer<T>> label)
    : tsdf_layer(tsdf), label_layer(label), nh_(nh) {
  // std::string normal_mode;
  // nh.param<std::string>("normal_calculation_mode", normal_mode,
  // "CENTRALDIFFERENCES");

  /*if(normal_mode.compare("CENTRALDIFFERENCES") == 0){
      normal_calculation = CENTRALDIFFERENCES;
  }else if(normal_mode.compare("CROSSPRODUCT") == 0){
      normal_calculation = CROSSPRODUCT;
  }else if(normal_mode.compare("CENTRALDIFFERENCES_NORMALIZED") == 0){
      normal_calculation = CENTRALDIFFERENCES_NORMALIZED;
  }else if(normal_mode.compare("VOXBLOX") == 0){
      normal_calculation = VOXBLOX;
  }else{
      ROS_INFO("Wrong parameter for normal_calculation_mode, using
  default(central differences"); normal_calculation = CENTRALDIFFERENCES;
  }*/

  std::string integration_mode;
  nh.param<std::string>("integration_mode", integration_mode, "HIGHEST");
  if (integration_mode.compare("CONFIDENCE") == 0) {
    mode = CONFIDENCE;
    std::cout << "integratin with confidence" << std::endl;
  } else if (integration_mode.compare("CONFIDENCE_WEIGHT") == 0) {
    mode = CONFIDENCE_WEIGHT;
    std::cout << "integrating with confidence weight" << std::endl;
  } else if (integration_mode.compare("HIGHEST") == 0) {
    mode = DEFAULT;
    std::cout << "integrating with highest" << std::endl;
  } else if (integration_mode.compare("DEFAULT") == 0) {
    mode = DEFAULT;
    std::cout << "integrating with highest" << std::endl;
  } else if (integration_mode.compare("CONFIDENCE_HIGHEST") == 0) {
    std::cout << "integrating with confidence highest" << std::endl;
    mode = CONFIDENCE_HIGHEST;
    
  } else {
    ROS_INFO("Wrong parameter for integration_mode, using default");
    mode = DEFAULT;
  }
}

template <class T>
void SegTsdfIntegrator<T>::direct_integration(
    std::shared_ptr<vertexMap> vertices, std::shared_ptr<labelMap> lmap) {
  if (vertices->height != lmap->height || vertices->width != lmap->width) {
    ROS_ERROR("labelmap and vertices aren't aligned, no integration possible");
    return;
  }

  for (int y = 0; y < vertices->height; y++) {
    for (int x = 0; x < vertices->width; x++) {
      // get Point
      Point p = (vertices->get(x, y))->p;
      std::vector<std::pair<int, double>> labels = lmap->get_labels(x, y);
      
      
      if (labels.size() == 0) {
        continue;
      }
      
      

      switch (mode) {
        
        case CONFIDENCE:
          integrateConfidence(p, labels);
          break;
        case CONFIDENCE_WEIGHT:
          integrateConfidenceWeight(p, labels);
          break;

        case CONFIDENCE_HIGHEST:
          integrateConfidenceHighest(p, labels);
          break;      
        case DEFAULT:
          integrateHighest(p, labels);
          break;
      }
    }
  }
}

template <class T>
void SegTsdfIntegrator<T>::integrateHighest(
    Point p, std::vector<std::pair<int, double>> labels) {}

template <class T>
void SegTsdfIntegrator<T>::integrateConfidenceHighest(
    Point p, std::vector<std::pair<int, double>> labels) {}

template <class T>
void SegTsdfIntegrator<T>::integrateConfidence(
    Point p, std::vector<std::pair<int, double>> labels) {}

template <class T>
void SegTsdfIntegrator<T>::integrateConfidenceWeight(
    Point p, std::vector<std::pair<int, double>> labels) {}
//////////////////////////////////LabelVoxel

// getting block pointers like voxblox/marius work

template <>
void SegTsdfIntegrator<LabelVoxel>::integrateHighest(
    Point p, std::vector<std::pair<int, double>> labels) {
  Block<LabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);

  int label = labels[labels.size() - 1].first;

  // TODO think about what to do with the background (substract confidence or
  // not???)
  voxel.label_id = 0;
  voxel.weight = 0.0;
  for (int i = 0; i < labels.size(); i++) {
    if (labels[i].second > voxel.weight) {
      voxel.weight = labels[i].second;
      voxel.label_id = labels[i].first;
    }
  }
}

// only increasing the confidence based on one voxel
template <>
void SegTsdfIntegrator<LabelVoxel>::integrateConfidenceHighest(
    Point p, std::vector<std::pair<int, double>> labels) {
  Block<LabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);

  int label_highest = 0;
  double weight_highest = 0.0;
  // get highest
  for (int i = 0; i < labels.size(); i++) {
    if (labels[i].second > weight_highest) {
      weight_highest = labels[i].second;
      label_highest = labels[i].first;
    }
  }

  if (voxel.label_id == label_highest) {
    voxel.weight = std::min(voxel.weight + 1, 10.0);
  } else {
    --voxel.weight;
    if (voxel.weight <= 0) {
      voxel.weight = 0;
      voxel.label_id = label_highest;
    }
  }

  
}

template <>
void SegTsdfIntegrator<LabelVoxel>::integrateConfidence(
    Point p, std::vector<std::pair<int, double>> labels) {
  Block<LabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);



  if (labels.size() == 1) {
    if (voxel.label_id == labels[0].first) {
      voxel.weight = std::min(10.0, voxel.weight + 1);
    } else {
      voxel.weight--;
      if (voxel.weight <= 0) {
        voxel.weight = 0;
        voxel.label_id = labels[0].first;
      }
    }
    return;
  }

  int label_highest = 0;
  double weight_highest = 0.0;
  bool found = false;
  // get highest
  for (int i = 0; i < labels.size(); i++) {

    if(labels[i].first == 0){
      std::cout << "zero label received warning" << std::endl;
    }
    //std::cout << labels[i].first << std::endl;
    if (labels[i].second > weight_highest) {
      weight_highest = labels[i].second;
      label_highest = labels[i].first;
    }

    // check if current voxel is in update, dont make changes
    if (labels[i].first == voxel.label_id) {
      found = true;
    }
  }

  if (found) {
    return;
  }

  if (!found) {
    voxel.weight--;
    if (voxel.weight <= 0) {
      voxel.weight = 0;
      voxel.label_id = label_highest;
    }
  }

  //std::cout << voxel.label_id << std::endl;
}

template <>
void SegTsdfIntegrator<LabelVoxel>::integrateConfidenceWeight(
    Point p, std::vector<std::pair<int, double>> labels) {
  Block<LabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  LabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);

  // TODO think about what to do with the background (substract confidence or
  // not???)
  int label_highest = 0;
  double weight_highest = 0.0;

  int label_second = 0;
  double weight_second = 0.0;

  for (int i = 0; i < labels.size(); i++) {
    if (labels[i].second > weight_highest) {
      weight_second = weight_highest;
      label_second = label_highest;
      weight_highest = labels[i].second;
      label_highest = labels[i].first;
    } else if (labels[i].second > weight_second) {
      weight_second = labels[i].second;
      label_second = labels[i].first;
    }
  }

  double diff = weight_highest - weight_second;

  if (label_highest == voxel.label_id) {
    voxel.weight = std::min(voxel.weight + diff, 10.0);
  } else if (label_second == voxel.label_id) {
    if (voxel.weight < diff) {
      voxel.label_id = label_highest;
      voxel.weight = diff - voxel.weight;
    } else {
      voxel.weight -= diff;
    }

  } else {
    if (voxel.weight < weight_highest) {
      voxel.weight = weight_highest - voxel.weight;
      voxel.label_id = label_highest;
    } else {
      voxel.weight -= weight_highest;
    }
  }
}

////////////////////////////////////Multilabel

// reducing al labels once we reach the maximum of 10
template <>
void SegTsdfIntegrator<MultiLabelVoxel>::integrateConfidence(
    Point p, std::vector<std::pair<int, double>> labels) {
  Block<MultiLabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  MultiLabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);

  bool overstepped = false;
  for (int i = 0; i < labels.size(); i++) {
    if (voxel.weights.find(labels[i].first) != voxel.weights.end()) {
      voxel.weights[labels[i].first] += 1.0;

      if (voxel.weights[labels[i].first] >= 10.0) {
        overstepped = true;
      }

    } else {
      voxel.weights[labels[i].first] = 1.0;
    }
  }

  // now iterate over all known keys and normalize, also substract 1 if
  // overstepped
  int succ_label = 0;
  float succ_weight = 0.0;
  float acc_weight = 0.0;

  std::vector<int> to_erase;
  for (auto it = voxel.weights.begin(); it != voxel.weights.end(); it++) {
    acc_weight += it->second;
    if (succ_weight <= it->second) {
      succ_label = it->first;
      succ_weight = it->second;
      if (overstepped) {
        it->second = it->second - 1.0;

        if (it->second < 0.0) {
          to_erase.push_back(it->first);
        }
      }
    }
  }

  // remove
  for (int i = 0; i < to_erase.size(); i++) {
    auto it = voxel.weights.find(to_erase[i]);
    voxel.weights.erase(it);
  }

  voxel.label_id = succ_label;
  if(acc_weight > 0)
    voxel.weight = succ_weight / acc_weight;
}

template <>
void SegTsdfIntegrator<MultiLabelVoxel>::integrateConfidenceWeight(
    Point p, std::vector<std::pair<int, double>> labels) {


  //std::cout << "case multilabel" << std::endl;
  Block<MultiLabelVoxel>::Ptr block_ptr =
      label_layer->allocateBlockPtrByCoordinates(p);
  MultiLabelVoxel& voxel = block_ptr->getVoxelByCoordinates(p);

  bool overstepped = false;
  for (int i = 0; i < labels.size(); i++) {

    if (voxel.weights.find(labels[i].first) != voxel.weights.end()) {
      voxel.weights[labels[i].first] += labels[i].second;
      if (voxel.weights[labels[i].first] >= 10.0) {
        overstepped = true;
      }

    } else {
      voxel.weights[labels[i].first] = labels[i].second;
    }
  }

  // now iterate over all known keys and normalize, also substract 1 if
  // overstepped
  int succ_label = 0;
  float succ_weight = 0.0;
  float acc_weight = 0.0;

  std::vector<int> to_erase;
  for (auto it = voxel.weights.begin(); it != voxel.weights.end(); it++) {
    acc_weight += it->second;
    if (succ_weight <= it->second) {
      succ_label = it->first;
      succ_weight = it->second;
      if (overstepped) {
        it->second = it->second - 1.0;

        if (it->second < 0.0) {
          to_erase.push_back(it->first);
        }
      }
    }
  }

  // remove
  for (int i = 0; i < to_erase.size(); i++) {
    auto it = voxel.weights.find(to_erase[i]);
    voxel.weights.erase(it);
  }

  voxel.label_id = succ_label;
  if(acc_weight > 0)
    voxel.weight = succ_weight / acc_weight;
}

template class SegTsdfIntegrator<LabelVoxel>;
template class SegTsdfIntegrator<MultiLabelVoxel>;
}  // namespace voxblox