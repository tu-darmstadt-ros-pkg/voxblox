cmake_minimum_required(VERSION 2.8.3)
project(voxblox_ros)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-std=c++11 -Wall -Wextra)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED)

find_package(mask_rcnn_ros REQUIRED)
include_directories(${mask_rcnn_ros_INCLUDE_DIRS})

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}
  src/esdf_server.cc
  src/interactive_slider.cc
  src/simulation_server.cc
  src/intensity_server.cc
  src/transformer.cc
  src/tsdf_server.cc
  src/segmentation_server.cc
  src/segmenter.cc
  src/mrcnn_server.cc
  src/mrcnn_segmenter.cc

  src/tsdf_seq_server.cc
  src/tsdf_seq_receiver.cc
  src/mrcnn_receiver.cc

  src/debug_recorder.cc
  src/debug_receiver.cc

  src/SegReceiver.cc
  src/SegMrcnnReceiver.cc
  src/SegDebugReceiver.cc
  src/SegServer.cc
  src/SegDataFuser.cc
  src/SegEdgeFuser.cc

  src/ply_reader.cc
  

)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

############
# BINARIES #
############

cs_add_executable(voxblox_eval
  src/voxblox_eval.cc
)
target_link_libraries(voxblox_eval ${PROJECT_NAME})

cs_add_executable(tsdf_server
  src/tsdf_server_node.cc
)
target_link_libraries(tsdf_server ${PROJECT_NAME})

cs_add_executable(esdf_server
  src/esdf_server_node.cc
)
target_link_libraries(esdf_server ${PROJECT_NAME})

cs_add_executable(intensity_server
  src/intensity_server_node.cc
)
target_link_libraries(intensity_server ${PROJECT_NAME})

cs_add_executable(simulation_eval
  src/simulation_eval.cc
)
target_link_libraries(simulation_eval ${PROJECT_NAME})

cs_add_executable(visualize_tsdf
  src/visualize_tsdf.cc
)
target_link_libraries(visualize_tsdf ${PROJECT_NAME})

cs_add_executable(segmentation_server
  src/segmentation_server_node.cc
)
target_link_libraries(segmentation_server ${PROJECT_NAME} ${OpenCV_LIBRARIES})

cs_add_executable(mrcnn_server
  src/mrcnn_server_node.cc
)
target_link_libraries(mrcnn_server ${PROJECT_NAME} ${OpenCV_LIBRARIES})

cs_add_executable(tsdf_seq_server
  src/tsdf_seq_server_node.cc
)
target_link_libraries(tsdf_seq_server ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(tsdf_seq_server_multilabel
  src/tsdf_seq_server_multilabel_node.cc
)
target_link_libraries(tsdf_seq_server_multilabel ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(mrcnn_result_recorder
  src/debug_recorder_node.cc
)
target_link_libraries(mrcnn_result_recorder ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(tsdf_seq_server_presegmented
  src/debug_server_node.cc
)
target_link_libraries(tsdf_seq_server_presegmented ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(tsdf_seq_server_presegmented_multilabel
  src/debug_server_multilabel_node.cc
)
target_link_libraries(tsdf_seq_server_presegmented_multilabel ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(new_seg_debug_server
  src/SegDebugServer_node.cc
)
target_link_libraries(new_seg_debug_server ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(new_seg_debug_edge_server
  src/SegDebugEdgeServer_node.cc
)
target_link_libraries(new_seg_debug_edge_server ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

cs_add_executable(ply_tester
  src/ply_tester_node.cc
)
target_link_libraries(ply_tester ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${mask_rcnn_ros_LIBRARIES})

#TODO add mrcnn node

##########
# EXPORT #
##########
cs_install()
cs_export()
