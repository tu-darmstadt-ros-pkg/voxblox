# vertex positions rounded to nearest BLOCK_EDGE_LENGTH/65535
# no alpha information

std_msgs/Header header

uint32 object_id

# sets the opacity of the entire mesh.
uint8 alpha

geometry_msgs/Transform transform

float32 block_edge_length

voxblox_msgs/MeshBlock[] mesh_blocks
