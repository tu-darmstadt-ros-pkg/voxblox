# A mesh message that support multiple instances of meshes.

# header of the MultiMesh message. This header is also used for the 'meshes' field when read by the rviz plugin.
std_msgs/Header header

# The message containing the voxblox meshes.
# NOTE: if 'meshes[i].mesh_blocks' is empty the according visual will be cleared.
voxblox_msgs/Mesh[] meshes
