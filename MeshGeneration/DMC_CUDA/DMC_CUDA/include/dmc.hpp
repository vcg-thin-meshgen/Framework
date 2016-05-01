#ifndef DMC_H
#define DMC_H

#include <stdint.h>
#include <vector>

#include "utils.hpp"
#include "cuda_includes.h"

namespace dmc
{
    typedef uint8_t voxel_pt_index_type;
    typedef uint8_t voxel_edge_index_type;
    typedef uint8_t voxel_face_index_type;
    
    typedef uint8_t voxel_config_type;
    typedef unsigned voxel_index1D_type;
    typedef uint8_t iso_vertex_m_type;
    typedef unsigned vertex_index_type;
    // typedef unsigned char flag_type;
	typedef unsigned flag_type;
	typedef uint8_t check_dir_type;
    
    const unsigned INVALID_UINT8 = 0xff;
    const unsigned INVALID_UINT32 = 0xffffffff;
    
    const unsigned VOXEL_NUM_PTS = 8;
    const unsigned VOXEL_NUM_EDGES = 12;
    const unsigned VOXEL_NUM_FACES = 6;
	const unsigned NUM_CONFIGS = 256;
    
    // For inactive voxel index_1D in d_full_voxel_index_map, this value is stored.
    const voxel_index1D_type INVALID_INDEX_1D = INVALID_UINT32;
    const voxel_config_type MAX_VOXEL_CONFIG_MASK = INVALID_UINT8;
    // Used in config_edge_lut1[2]
    const iso_vertex_m_type NO_VERTEX = INVALID_UINT8;
    
	typedef utils::Array3D<float> scalar_grid_type;

	void setup_device_luts();

	void setup_device_luts(iso_vertex_m_type** d_config_edge_lut1, iso_vertex_m_type** d_config_edge_lut2,
		uint8_t** d_num_vertex_lut1, uint8_t** d_num_vertex_lut2, 
		voxel_config_type** d_config_2B_3B_lut, voxel_face_index_type** d_config_2B_3B_ambiguous_face, 
		voxel_face_index_type** d_opposite_face_lut, check_dir_type** d_face_to_check_dir_lut, 
		uint8_t** d_edge_belonged_voxel_lut, voxel_edge_index_type** d_circular_edge_lut, voxel_edge_index_type** d_voxel_local_edges);

	void cleanup_device_luts();

	void cleanup_device_luts(iso_vertex_m_type* d_config_edge_lut1, iso_vertex_m_type* d_config_edge_lut2,
		uint8_t* d_num_vertex_lut1, uint8_t* d_num_vertex_lut2, 
		voxel_config_type* d_config_2B_3B_lut, voxel_face_index_type* d_config_2B_3B_ambiguous_face, 
		voxel_face_index_type* d_opposite_face_lut, check_dir_type* d_face_to_check_dir_lut, 
		uint8_t* d_edge_belonged_voxel_lut, voxel_edge_index_type* d_circular_edge_lut, voxel_edge_index_type* d_voxel_local_edges);

	void run_dmc(std::vector<float3>& vertices, std::vector<uint3>& triangles, const scalar_grid_type& h_scalar_grid, 
				 const float3& xyz_min, const float3& xyz_max, float iso_value, unsigned num_smooth = 0);

}; // namespace dmc
#endif