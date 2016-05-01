
#include <cassert>
#include <cmath>
#include <algorithm>
#include <memory>								// std::unique_ptr
#include <iostream>

#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\reduce.h>
#include <thrust\count.h>

#include "cuda_texture_types.h"					// texture
#include "texture_fetch_functions.h"			// tex1Dfetch

#include "helper_math.hpp"

#include "dmc.hpp"

template <typename T>
void print_d_arr(const T* d_in, unsigned size, const std::string& prefix)
{
	T* h_in = new T[size];
	checkCudaErrors(cudaMemcpy(h_in, d_in, sizeof(T) * size, cudaMemcpyDeviceToHost));
	std::cout << prefix << std::endl;
	for (unsigned i = 0; i < size; ++i)
	{
		std::cout << "[" << i << "] " << h_in[i] << std::endl;
	}
	delete[] h_in;
}

namespace dmc
{
    
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif
    
    using namespace utils;

    // For each voxel config and an edge index, return the associated iso vertex in DMC.
    // This is LUT 1. Voxel with 3B config with its adjacent voxel being 2B config CANNOT use this LUT.
    const iso_vertex_m_type config_edge_lut1[NUM_CONFIGS][VOXEL_NUM_EDGES] =
    {
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 0, 0, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 1, 0, 0, 1, 1, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0, 1, 1, 1, 1, 0xff, 0xff, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0xff },
        { 1, 1, 0, 0, 0, 1, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0, 0, 0, 0, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 1, 0, 1, 1, 1, 1, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 1, 1, 0xff, 0, 1, 0xff, 0xff, 1, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0, 1, 1, 0, 0, 0xff, 0xff, 1, 0xff, 1, 0xff, 1 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0, 0, 1, 1, 0xff, 1, 0xff, 1 },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0, 0xff, 0 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0, 0xff },
        { 0, 1, 1, 0, 0xff, 0, 1, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0, 0xff, 1, 1, 0, 1, 0xff, 1, 1, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0 },
        { 0, 0, 0xff, 0xff, 0, 1, 0, 0xff, 1, 1, 0, 0 },
        { 0xff, 1, 0xff, 1, 0xff, 0, 1, 0xff, 0, 0, 1, 1 },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0xff, 0xff, 0xff, 0xff, 0, 0, 1, 1 },
        { 1, 0xff, 1, 0xff, 1, 0, 0xff, 0xff, 0, 0, 1, 1 },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0, 0 },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 1, 0, 0, 1, 1 },
        { 1, 0xff, 1, 0xff, 0xff, 0xff, 1, 0, 1, 1, 0, 0 },
        { 0, 1, 1, 0, 0, 2, 1, 3, 2, 2, 3, 3 },
        { 0xff, 0, 0, 0xff, 0xff, 1, 0, 2, 1, 1, 2, 2 },
        { 0xff, 1, 0xff, 1, 1, 0xff, 0xff, 0, 1, 1, 0, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0, 0, 1, 1 },
        { 0, 0xff, 0xff, 0, 0, 2, 0xff, 1, 2, 2, 1, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 1, 1, 1, 0, 0xff, 0xff, 1, 0xff, 1, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0, 0, 0, 0xff, 0, 0xff },
        { 1, 0xff, 1, 0xff, 0, 1, 1, 0, 0, 0xff, 0, 0xff },
        { 1, 0, 0, 1, 0xff, 0xff, 0, 1, 1, 0xff, 1, 0xff },
        { 0xff, 1, 1, 0xff, 0, 0xff, 1, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff },
        { 1, 1, 0xff, 0xff, 0, 1, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 1, 1, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 1, 1, 1, 1, 1, 0, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 1, 0xff, 0xff, 1, 1 },
        { 1, 2, 2, 1, 1, 0xff, 2, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0 },
        { 0, 0, 1, 1, 0, 0xff, 0, 1, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0, 0, 1, 1, 1, 0, 0xff, 0xff, 1, 1 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0, 0xff },
        { 1, 1, 0, 0, 0xff, 0xff, 1, 0, 1, 0xff, 1, 0xff },
        { 0, 0xff, 0, 0xff, 0, 1, 1, 0, 1, 0xff, 1, 0xff },
        { 0xff, 0xff, 1, 1, 0xff, 0, 0, 1, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 1, 1, 0, 0, 1, 0xff, 0xff, 1, 0xff, 1, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0, 0 },
        { 0, 0xff, 0xff, 0, 1, 0, 0xff, 0, 1, 0, 0, 1 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0 },
        { 0xff, 1, 0xff, 1, 0, 0xff, 0xff, 1, 0, 1, 1, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 1, 0, 0, 1, 1, 0 },
        { 0, 0, 1, 1, 3, 0, 2, 1, 3, 2, 2, 3 },
        { 1, 0xff, 1, 0xff, 0xff, 0xff, 0, 1, 1, 0, 0, 1 },
        { 0xff, 0xff, 0, 0, 2, 0xff, 1, 0, 2, 1, 1, 2 },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0 },
        { 1, 0xff, 1, 0xff, 0, 1, 0xff, 0xff, 0, 1, 1, 0 },
        { 0, 1, 1, 0, 0xff, 0xff, 0xff, 0xff, 0, 1, 1, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 1, 0, 0, 1 },
        { 0xff, 1, 0xff, 1, 0xff, 1, 0, 0xff, 1, 0, 0, 1 },
        { 0, 0, 0xff, 0xff, 1, 0, 2, 0xff, 1, 2, 2, 1 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0 },
        { 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 0xff, 1, 0, 0, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 1, 1, 0xff, 1, 1, 0, 1, 0xff, 0, 0, 0xff },
        { 2, 2, 1, 1, 0xff, 2, 0, 1, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 1, 0, 0, 1, 1, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 1, 0xff, 0xff, 1, 1, 0xff },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 1, 0xff, 1, 0, 0, 1, 1, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 1, 1, 0, 0, 1, 0xff, 0xff, 0, 0xff, 1, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0, 0xff, 1, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 1, 0, 0, 1, 0xff, 1, 0, 0xff, 0xff, 1, 0xff, 1 },
        { 0xff, 1, 1, 0xff, 0, 0, 1, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 1, 1, 0xff, 0xff, 1, 0, 1, 1, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 1, 0, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0, 0, 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 1, 0, 0, 1, 1, 2, 0, 0xff, 2, 2, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0 },
        { 1, 0xff, 0xff, 1, 0, 1, 1, 1, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 1, 0xff, 0, 0, 1, 0xff, 0xff, 1 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 2, 2, 0, 0, 1, 2, 0xff, 0, 1, 0xff, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 1, 0, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 1, 1, 0, 0xff, 0xff, 1, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff }, 
        { 0, 0, 1, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff }, 
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0xff, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 1, 0, 0, 1, 1, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }, 
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }
    };

    // Number of iso vertices for DMC for each voxel config, this is LUT_1.
    // Voxel with 3B config with its adjacent voxel being 2B config CANNOT use this LUT.
    const uint8_t num_vertex_lut1[NUM_CONFIGS] =
    {
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1,
        1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1,
        1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 3, 2,
        1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1,
        1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1,
        1, 2, 1, 2, 2, 4, 2, 3, 1, 2, 2, 2, 2, 3, 2, 2,
        1, 1, 1, 1, 2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1,
        1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1,
        1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 3, 2, 1, 1, 2, 1,
        1, 2, 1, 2, 1, 3, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0
    };
    
    const iso_vertex_m_type config_edge_lut2[NUM_CONFIGS][VOXEL_NUM_EDGES] =
    {
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 1, 0, 0, 1, 1, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 1, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 1, 0xff, 0xff, 1 },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 1, 1, 0, 0xff, 0xff, 1, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 1, 0, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 1, 0, 0, 1, 1, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 1, 0xff, 0, 0, 1, 0xff, 0xff, 1 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0, 1, 1, 1, 1, 0xff, 0xff, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff, 0xff },
        { 1, 1, 0, 0, 0, 1, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0, 0, 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 1, 0, 0, 1, 1, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 1, 0, 1, 1, 1, 1, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 1, 1, 0xff, 0, 0, 1, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 1, 1, 0xff, 0, 1, 0xff, 0xff, 1, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0, 0xff, 1, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0, 1, 1, 0, 0, 0xff, 0xff, 1, 0xff, 1, 0xff, 1 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0, 0, 1, 1, 0xff, 1, 0xff, 1 },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0, 0xff, 0 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0xff },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 1, 0xff, 0xff, 1, 1, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 1, 0, 0, 1, 1, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0, 1, 1, 0, 0xff, 0, 1, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0, 0xff, 1, 1, 0, 1, 0xff, 1, 1, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 0xff, 1, 0, 0, 1 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0 },
        { 0, 0, 0xff, 0xff, 0, 1, 0, 0xff, 1, 1, 0, 0 },
        { 0xff, 1, 0xff, 1, 0xff, 0, 1, 0xff, 0, 0, 1, 1 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 1, 0, 0, 1 },
        { 0, 0, 1, 1, 0xff, 0xff, 0xff, 0xff, 0, 0, 1, 1 },
        { 1, 0xff, 1, 0xff, 1, 0, 0xff, 0xff, 0, 0, 1, 1 },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0, 0 },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 1, 0, 0, 1, 1 },
        { 1, 0xff, 1, 0xff, 0xff, 0xff, 1, 0, 1, 1, 0, 0 },
        { 0, 1, 1, 0, 0, 2, 1, 3, 2, 2, 3, 3 },
        { 0xff, 0, 0, 0xff, 0xff, 1, 0, 2, 1, 1, 2, 2 },
        { 0xff, 1, 0xff, 1, 1, 0xff, 0xff, 0, 1, 1, 0, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0, 0, 1, 1 },
        { 0, 0xff, 0xff, 0, 0, 2, 0xff, 1, 2, 2, 1, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 1, 1, 1, 0, 0xff, 0xff, 1, 0xff, 1, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 1, 1, 0xff, 0, 0, 1, 0, 0xff, 0, 0xff },
        { 1, 0xff, 1, 0xff, 0, 1, 1, 0, 0, 0xff, 0, 0xff },
        { 1, 0, 0, 1, 0xff, 0xff, 0, 1, 1, 0xff, 1, 0xff },
        { 0xff, 1, 1, 0xff, 0, 0xff, 1, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff },
        { 1, 1, 0xff, 0xff, 0, 1, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 1, 1, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 1, 1, 1, 1, 1, 0, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 1, 0xff, 0xff, 1, 1 },
        { 1, 2, 2, 1, 1, 0xff, 2, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0xff, 0, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 1, 0xff, 0xff, 1, 1 },
        { 0, 0, 1, 1, 0, 0xff, 0, 1, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 1, 0xff, 0xff, 1, 1 },
        { 0xff, 0xff, 0, 0, 1, 1, 1, 0, 0xff, 0xff, 1, 1 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 1, 1, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff },
        { 1, 1, 0xff, 0xff, 0, 1, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff },
        { 0xff, 1, 1, 0xff, 0, 0xff, 1, 0, 0, 0xff, 0, 0xff },
        { 1, 1, 0, 0, 0xff, 0xff, 1, 0, 1, 0xff, 1, 0xff },
        { 0, 0xff, 0, 0xff, 0, 1, 1, 0, 1, 0xff, 1, 0xff },
        { 0xff, 0xff, 1, 1, 0xff, 0, 0, 1, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0, 1, 1, 0, 0, 1, 0xff, 0xff, 1, 0xff, 1, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0, 0 },
        { 0, 0xff, 0xff, 0, 1, 0, 0xff, 0, 1, 0, 0, 1 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 1, 0, 0, 1, 1 },
        { 0xff, 1, 0xff, 1, 0, 0xff, 0xff, 1, 0, 1, 1, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 1, 0, 0, 1, 1, 0 },
        { 0, 0, 1, 1, 3, 0, 2, 1, 3, 2, 2, 3 },
        { 1, 0xff, 1, 0xff, 0xff, 0xff, 0, 1, 1, 0, 0, 1 },
        { 0xff, 0xff, 0, 0, 2, 0xff, 1, 0, 2, 1, 1, 2 },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0, 0 },
        { 1, 0xff, 1, 0xff, 0, 1, 0xff, 0xff, 0, 1, 1, 0 },
        { 0, 1, 1, 0, 0xff, 0xff, 0xff, 0xff, 0, 1, 1, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 0xff, 1, 0, 0, 1 },
        { 0xff, 1, 0xff, 1, 0xff, 1, 0, 0xff, 1, 0, 0, 1 },
        { 0, 0, 0xff, 0xff, 1, 0, 2, 0xff, 1, 2, 2, 1 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 1, 0xff, 0, 1, 1, 0 },
        { 0xff, 0xff, 0xff, 0xff, 1, 0xff, 0, 0xff, 1, 0, 0, 1 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff },
        { 0xff, 1, 1, 0xff, 1, 1, 0, 1, 0xff, 0, 0, 0xff },
        { 2, 2, 1, 1, 0xff, 2, 0, 1, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 1, 0, 0, 1, 1, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 1, 0xff, 0xff, 1, 1, 0xff },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0xff, 1, 1, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 1, 0xff, 0xff, 1, 1, 0xff, 0, 0, 0xff, 0, 0xff, 0 },
        { 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 1, 0xff, 1, 0, 0, 1, 1, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 1, 1, 0, 0, 1, 0xff, 0xff, 0, 0xff, 1, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0, 0xff, 1, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0 },
        { 1, 0, 0, 1, 0xff, 1, 0, 0xff, 0xff, 1, 0xff, 1 },
        { 0xff, 1, 1, 0xff, 0, 0, 1, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff },
        { 1, 1, 0xff, 0xff, 1, 0, 1, 1, 0, 0, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 1, 0, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0, 0xff, 0xff },
        { 0, 0, 1, 1, 0xff, 0xff, 0xff, 1, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 1, 0xff, 0, 1, 1, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff },
        { 1, 0, 0, 1, 1, 2, 0, 0xff, 2, 2, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 1, 0, 0xff, 1, 1, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 1, 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0 },
        { 1, 0xff, 0xff, 1, 0, 1, 1, 1, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0xff, 0, 1, 0xff, 0, 0, 1, 0xff, 0xff, 1 },
        { 0xff, 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 2, 2, 0, 0, 1, 2, 0xff, 0, 1, 0xff, 0xff, 1 },
        { 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 1, 1, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0xff, 0, 0xff, 1, 0, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 1, 1, 0, 0xff, 0xff, 1, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0, 0, 0xff, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 1 },
        { 0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0, 0, 0xff, 0xff, 1, 0, 0xff, 0xff, 1, 0xff, 0xff, 1 },
        { 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0 },
        { 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0, 0, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 1, 1, 0xff, 0, 0xff, 1, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 1, 0, 0, 1, 1, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0, 0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff },
        { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff }
    };
    
    const uint8_t num_vertex_lut2[NUM_CONFIGS] =
    {
        0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
        1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,
        1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1,
        1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1,
        1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 2, 2, 3, 2,
        1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 1,
        1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1,
        1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1,
        2, 2, 2, 2, 2, 4, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2,
        1, 1, 1, 1, 2, 3, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1,
        1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1,
        1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 3, 2, 1, 1, 2, 1,
        1, 2, 1, 2, 1, 3, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1,
        1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0 
    };
    
    const unsigned NUM_AMBIGUOUS_CONFIGS = 36;
    const voxel_config_type config_2B_3B_lut[NUM_AMBIGUOUS_CONFIGS] =
    {
        0xa0, 0x21, 0x42, 0x84, 0x05, 0x81, 0x48, 0x0a, 0x50, 0x12, 0x18, 0x24, // 2B
        0xc1, 0xc2, 0x83, 0x45, 0x86, 0x49, 0x8a, 0x51, 0x92, 0x43, 0x54, 0x15, // 3B
        0x16, 0x1c, 0x61, 0xa2, 0xa8, 0x29, 0x2a, 0x2c, 0x68, 0x34, 0x38, 0x94
    };
    
    const voxel_face_index_type config_2B_3B_ambiguous_face[NUM_AMBIGUOUS_CONFIGS] =
    {
        5,    1,    2,    3,    0,    4,    3,    0,    5,    1,    4,    2,    // 2B
        4,    2,    4,    0,    3,    3,    0,    5,    1,    2,    5,    0,    // 3B
        1,    4,    1,    5,    5,    1,    0,    2,    3,    2,    4,    3
    };
    
    const voxel_face_index_type opposite_face_lut[VOXEL_NUM_FACES] = {5, 3, 4, 1, 2, 0};
    
    const check_dir_type POS_X_DIR = 0;
	const check_dir_type NEG_X_DIR = 1;
	const check_dir_type POS_Y_DIR = 2;
	const check_dir_type NEG_Y_DIR = 3;
	const check_dir_type POS_Z_DIR = 4;
	const check_dir_type NEG_Z_DIR = 5;

	const check_dir_type face_to_check_dir_lut[VOXEL_NUM_FACES] =
	{ NEG_Z_DIR, NEG_Y_DIR, POS_X_DIR, POS_Y_DIR, NEG_X_DIR, POS_Z_DIR };

    const uint8_t LOCAL_EDGE_ENTRY = 0xff;
    const uint8_t edge_belonged_voxel_lut[VOXEL_NUM_EDGES] =
    {
        ( 0x00 | 0x40 | 0x20 ) | 10,    // 0
        ( 0x00 | 0x00 | 0x20 ) | 9,     // 1
        ( 0x00 | 0x00 | 0x20 ) | 10,    // 2
        ( 0x80 | 0x00 | 0x20 ) | 9,     // 3
        ( 0x80 | 0x40 | 0x00 ) | 6,     // 4
        ( 0x00 | 0x40 | 0x00 ) | 6,     // 5
        LOCAL_EDGE_ENTRY,               // 6
        ( 0x80 | 0x00 | 0x00 ) | 6,     // 7
        ( 0x00 | 0x40 | 0x00 ) | 10,    // 8
        LOCAL_EDGE_ENTRY,               // 9
        LOCAL_EDGE_ENTRY,               // 10
        ( 0x80 | 0x00 | 0x00 ) | 9,     // 11
    };
        
    // Same edge shared by four voxels. Default in CCW order when looking align the positive
    // direction of the axis.
    voxel_edge_index_type circular_edge_lut[3][4] =
    {
        {6, 7, 4, 5},
        {9, 1, 3, 11},
        {10, 8, 0, 2}
    };

	const uint8_t VOXEL_NUM_LOCAL_EDGES = 3;
	voxel_edge_index_type voxel_local_edges[VOXEL_NUM_LOCAL_EDGES] = {6, 9, 10};
    
	// LUT on device memory
	texture<iso_vertex_m_type, cudaTextureType1D, cudaReadModeElementType> config_edge_lut1_tex;
	texture<iso_vertex_m_type, cudaTextureType1D, cudaReadModeElementType> config_edge_lut2_tex;
	
	texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> num_vertex_lut1_tex;
	texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> num_vertex_lut2_tex;

	texture<voxel_config_type, cudaTextureType1D, cudaReadModeElementType> config_2B_3B_lut_tex;
	texture<voxel_face_index_type, cudaTextureType1D, cudaReadModeElementType> config_2B_3B_ambiguous_face_tex;
	texture<voxel_face_index_type, cudaTextureType1D, cudaReadModeElementType> opposite_face_lut_tex;
	texture<check_dir_type, cudaTextureType1D, cudaReadModeElementType> face_to_check_dir_lut_tex;

	texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> edge_belonged_voxel_lut_tex;

	texture<voxel_edge_index_type, cudaTextureType1D, cudaReadModeElementType> circular_edge_lut_tex;

	texture<voxel_edge_index_type, cudaTextureType1D, cudaReadModeElementType> voxel_local_edges_tex;
	
	// A singleton class to hold all the device pointers needed by static LUTs. Saves trouble
	// for maintaining these pointers on the client side.
	class LutPtrsCollection
	{
	private:
		static std::unique_ptr<LutPtrsCollection> m_instance;

	public:
		static LutPtrsCollection* instance()
		{
			if (!m_instance)
			{
				m_instance = std::unique_ptr<LutPtrsCollection>(new LutPtrsCollection);
			}
			return m_instance.get();
		}

		iso_vertex_m_type* d_config_edge_lut1;
		iso_vertex_m_type* d_config_edge_lut2;
		uint8_t* d_num_vertex_lut1;
		uint8_t* d_num_vertex_lut2;
		voxel_config_type* d_config_2B_3B_lut;
		voxel_face_index_type* d_config_2B_3B_ambiguous_face;
		voxel_face_index_type* d_opposite_face_lut;
		check_dir_type* d_face_to_check_dir_lut;
		uint8_t* d_edge_belonged_voxel_lut;
		voxel_edge_index_type* d_circular_edge_lut;
		voxel_edge_index_type* d_voxel_local_edges;
	};

	std::unique_ptr<LutPtrsCollection> LutPtrsCollection::m_instance = nullptr;

	void setup_device_luts()
	{
		LutPtrsCollection* luts = LutPtrsCollection::instance();
		setup_device_luts(&(luts->d_config_edge_lut1), &(luts->d_config_edge_lut2), &(luts->d_num_vertex_lut1), &(luts->d_num_vertex_lut2), 
			&(luts->d_config_2B_3B_lut), &(luts->d_config_2B_3B_ambiguous_face), &(luts->d_opposite_face_lut), &(luts->d_face_to_check_dir_lut), 
			&(luts->d_edge_belonged_voxel_lut), &(luts->d_circular_edge_lut), &(luts->d_voxel_local_edges));
	}

	void setup_device_luts(iso_vertex_m_type** d_config_edge_lut1, iso_vertex_m_type** d_config_edge_lut2,
		uint8_t** d_num_vertex_lut1, uint8_t** d_num_vertex_lut2, 
		voxel_config_type** d_config_2B_3B_lut, voxel_face_index_type** d_config_2B_3B_ambiguous_face, 
		voxel_face_index_type** d_opposite_face_lut, check_dir_type** d_face_to_check_dir_lut, 
		uint8_t** d_edge_belonged_voxel_lut, voxel_edge_index_type** d_circular_edge_lut, voxel_edge_index_type** d_voxel_local_edges)
	{
		const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		// setup for d_config_edge_lut1 2D array
		checkCudaErrors(cudaMalloc(d_config_edge_lut1, sizeof(voxel_config_type) * NUM_CONFIGS * VOXEL_NUM_EDGES));
		checkCudaErrors(cudaMemcpy(*d_config_edge_lut1, (voxel_config_type*)(*config_edge_lut1), 
									sizeof(voxel_config_type) * NUM_CONFIGS * VOXEL_NUM_EDGES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, config_edge_lut1_tex, *d_config_edge_lut1, channel_desc));
		// setup for d_config_edge_lut2 2D array
		checkCudaErrors(cudaMalloc(d_config_edge_lut2, sizeof(voxel_config_type) * NUM_CONFIGS * VOXEL_NUM_EDGES));
		checkCudaErrors(cudaMemcpy(*d_config_edge_lut2, (voxel_config_type*)(*config_edge_lut2), 
									sizeof(voxel_config_type) * NUM_CONFIGS * VOXEL_NUM_EDGES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, config_edge_lut2_tex, *d_config_edge_lut2, channel_desc));
		// setup for d_num_vertex_lut1
		checkCudaErrors(cudaMalloc(d_num_vertex_lut1, sizeof(uint8_t) * NUM_CONFIGS));
		checkCudaErrors(cudaMemcpy(*d_num_vertex_lut1, num_vertex_lut1, sizeof(uint8_t) * NUM_CONFIGS, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, num_vertex_lut1_tex, *d_num_vertex_lut1, channel_desc));
		// setup for d_num_vertex_lut2
		checkCudaErrors(cudaMalloc(d_num_vertex_lut2, sizeof(uint8_t) * NUM_CONFIGS));
		checkCudaErrors(cudaMemcpy(*d_num_vertex_lut2, num_vertex_lut2, sizeof(uint8_t) * NUM_CONFIGS, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, num_vertex_lut2_tex, *d_num_vertex_lut2, channel_desc));
		// setup for d_config_2B_3B_lut
		checkCudaErrors(cudaMalloc(d_config_2B_3B_lut, sizeof(voxel_config_type) * NUM_AMBIGUOUS_CONFIGS));
		checkCudaErrors(cudaMemcpy(*d_config_2B_3B_lut, config_2B_3B_lut, sizeof(voxel_config_type) * NUM_AMBIGUOUS_CONFIGS, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, config_2B_3B_lut_tex, *d_config_2B_3B_lut, channel_desc));
		// setup for d_config_2B_3B_ambiguous_face
		checkCudaErrors(cudaMalloc(d_config_2B_3B_ambiguous_face, sizeof(voxel_face_index_type) * NUM_AMBIGUOUS_CONFIGS));
		checkCudaErrors(cudaMemcpy(*d_config_2B_3B_ambiguous_face, config_2B_3B_ambiguous_face, 
									sizeof(voxel_face_index_type) * NUM_AMBIGUOUS_CONFIGS, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, config_2B_3B_ambiguous_face_tex, *d_config_2B_3B_ambiguous_face, channel_desc));
		// setup for d_opposite_face_lut
		checkCudaErrors(cudaMalloc(d_opposite_face_lut, sizeof(voxel_face_index_type) * VOXEL_NUM_FACES));
		checkCudaErrors(cudaMemcpy(*d_opposite_face_lut, opposite_face_lut, sizeof(voxel_face_index_type) * VOXEL_NUM_FACES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, opposite_face_lut_tex, *d_opposite_face_lut, channel_desc));
		// setup for d_face_to_check_dir_lut
		checkCudaErrors(cudaMalloc(d_face_to_check_dir_lut, sizeof(check_dir_type) * VOXEL_NUM_FACES));
		checkCudaErrors(cudaMemcpy(*d_face_to_check_dir_lut, face_to_check_dir_lut, sizeof(check_dir_type) * VOXEL_NUM_FACES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, face_to_check_dir_lut_tex, *d_face_to_check_dir_lut, channel_desc));

		// setup for d_edge_belonged_voxel_lut
		checkCudaErrors(cudaMalloc(d_edge_belonged_voxel_lut, sizeof(uint8_t) * VOXEL_NUM_EDGES));
		checkCudaErrors(cudaMemcpy(*d_edge_belonged_voxel_lut, edge_belonged_voxel_lut, sizeof(uint8_t) * VOXEL_NUM_EDGES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, edge_belonged_voxel_lut_tex, *d_edge_belonged_voxel_lut, channel_desc));
		// setup for d_circular_edge_lut
		checkCudaErrors(cudaMalloc(d_circular_edge_lut, sizeof(voxel_edge_index_type) * 12));
		checkCudaErrors(cudaMemcpy(*d_circular_edge_lut, (voxel_edge_index_type*)(*circular_edge_lut), 
									sizeof(voxel_edge_index_type) * 12, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, circular_edge_lut_tex, *d_circular_edge_lut, channel_desc));

		// setup for d_voxel_local_edges
		checkCudaErrors(cudaMalloc(d_voxel_local_edges, sizeof(voxel_edge_index_type) * VOXEL_NUM_LOCAL_EDGES));
		checkCudaErrors(cudaMemcpy(*d_voxel_local_edges, voxel_local_edges, 
									sizeof(voxel_edge_index_type) * VOXEL_NUM_LOCAL_EDGES, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaBindTexture(0, voxel_local_edges_tex, *d_voxel_local_edges, channel_desc));
	}

	void cleanup_device_luts()
	{
		LutPtrsCollection* luts = LutPtrsCollection::instance();
		cleanup_device_luts(luts->d_config_edge_lut1, luts->d_config_edge_lut2, luts->d_num_vertex_lut1, luts->d_num_vertex_lut2, 
			luts->d_config_2B_3B_lut, luts->d_config_2B_3B_ambiguous_face, luts->d_opposite_face_lut, luts->d_face_to_check_dir_lut, 
			luts->d_edge_belonged_voxel_lut, luts->d_circular_edge_lut, luts->d_voxel_local_edges);
	}

	void cleanup_device_luts(iso_vertex_m_type* d_config_edge_lut1, iso_vertex_m_type* d_config_edge_lut2,
		uint8_t* d_num_vertex_lut1, uint8_t* d_num_vertex_lut2, 
		voxel_config_type* d_config_2B_3B_lut, voxel_face_index_type* d_config_2B_3B_ambiguous_face, 
		voxel_face_index_type* d_opposite_face_lut, check_dir_type* d_face_to_check_dir_lut, 
		uint8_t* d_edge_belonged_voxel_lut, voxel_edge_index_type* d_circular_edge_lut, voxel_edge_index_type* d_voxel_local_edges)
	{
		checkCudaErrors(cudaFree(d_config_edge_lut1));
		checkCudaErrors(cudaFree(d_config_edge_lut2));

		checkCudaErrors(cudaFree(d_num_vertex_lut1));
		checkCudaErrors(cudaFree(d_num_vertex_lut2));

		checkCudaErrors(cudaFree(d_config_2B_3B_lut));
		checkCudaErrors(cudaFree(d_config_2B_3B_ambiguous_face));
		checkCudaErrors(cudaFree(d_opposite_face_lut));
		checkCudaErrors(cudaFree(d_face_to_check_dir_lut));

		checkCudaErrors(cudaFree(d_edge_belonged_voxel_lut));

		checkCudaErrors(cudaFree(d_circular_edge_lut));

		checkCudaErrors(cudaFree(d_voxel_local_edges));
	}

    // Stores the minimum required information for each voxel
    class _VoxelInfo
    {
        typedef uint8_t info_type;
        
        static const uint8_t EDGE_6_SHIFT = 0;
        static const uint8_t EDGE_9_SHIFT = 1;
        static const uint8_t EDGE_10_SHIFT = 2;
        static const uint8_t USE_LUT2_SHIFT = 7;
    public:

        __host__ __device__ _VoxelInfo() = default;
        __host__ __device__ _VoxelInfo(voxel_index1D_type index) : m_index1D(index), m_info(0) { }
        
        __host__ __device__ void encode_edge_is_bipolar(voxel_edge_index_type edge, bool is_bipolar)
        {
            uint8_t shift = get_edge_shift(edge);
            info_write_bit(shift, is_bipolar);
        }
        
        __host__ __device__ void encode_edge_bipolar_info(voxel_edge_index_type edge, bool is_bipolar, bool use_ccw)
        {
            uint8_t shift = get_edge_shift(edge);
            info_write_bit(shift, is_bipolar);
            
            if (is_bipolar)
            {
                shift += 3;
                info_write_bit(shift, use_ccw);
            }
        }
        
        __host__ __device__ bool is_edge_bipolar(voxel_edge_index_type edge) const
        {
			//if (edge != 6 && edge != 9 && edge != 10) {
			//	printf("is_edge_bipolar: edge: %d\n", edge);
			//}
            uint8_t shift = get_edge_shift(edge);
            return (bool)info_read_bit(shift);
        }
        
        // An edge that is 'CCW' means the polarization direction of the edge aligns with the positive axis.
        // [precondition] 'edge' must be bipolar
        __host__ __device__ bool is_edge_ccw(voxel_edge_index_type edge) const
        {
            assert(is_edge_bipolar(edge));
            
            uint8_t shift = get_edge_shift(edge) + 3;
            
			return (bool)info_read_bit(shift);
			// return true;
        }
        
        __host__ __device__ inline void encode_use_lut2(bool use_lut2) { info_write_bit(USE_LUT2_SHIFT, use_lut2); }
        
        __host__ __device__ inline bool use_lut2() const { return (bool)info_read_bit(USE_LUT2_SHIFT); }
        
        __host__ __device__ voxel_index1D_type index1D() const { return m_index1D; }
        
        __host__ __device__ voxel_config_type config() const { return m_config; }
        
        __host__ __device__ void set_config(voxel_config_type c) { m_config = c; }
        
        __host__ __device__ inline vertex_index_type vertex_begin() const { return m_vertex_begin; }
        
        __host__ __device__ void set_vertex_begin(vertex_index_type begin) { m_vertex_begin = begin; }
        
        __host__ __device__ uint8_t num_vertices() const { return m_num_vertices; }
        
        __host__ __device__ void set_num_vertices(uint8_t num) { m_num_vertices = num; }
        
		__host__ __device__ uint8_t info() const { return m_info; }

        __host__ __device__ uint8_t num_edge_vertices() const
        {
            uint8_t num = 0;
            num += info_read_bit(EDGE_6_SHIFT);
            num += info_read_bit(EDGE_9_SHIFT);
            num += info_read_bit(EDGE_10_SHIFT);
            return num;
        }
        
        __host__ __device__ inline uint8_t num_iso_vertices() const
        {
            // return use_lut2() ? num_vertex_lut2[config] : num_vertex_lut1[config];
            return m_num_vertices - num_edge_vertices();
        }
        
        __host__ __device__ inline vertex_index_type iso_vertex_begin() const { return vertex_begin(); }
        
        __host__ __device__ inline vertex_index_type iso_vertex_index(iso_vertex_m_type iso_vertex_m) const
        {
            assert(iso_vertex_m < num_iso_vertices());
            return iso_vertex_begin() + iso_vertex_m;
        }
        
        __host__ __device__ inline vertex_index_type edge_vertex_begin() const { return m_vertex_begin + num_iso_vertices(); }
        
        __host__ __device__ vertex_index_type edge_vertex_index(voxel_edge_index_type edge) const
        {
            assert(is_edge_bipolar(edge));
            
            uint8_t offset = 0;
            for (uint8_t i = 0; i < get_edge_shift(edge); ++i)
            {
                offset += info_read_bit(i);
            }
            return edge_vertex_begin() + offset;
        }
        
        __device__ iso_vertex_m_type iso_vertex_m_by_edge(voxel_edge_index_type edge) const
        {
            if (use_lut2())
            {
				// return config_edge_lut2[m_config][edge];
				return tex1Dfetch(config_edge_lut2_tex, m_config * VOXEL_NUM_EDGES + edge);
            }
            else
            {
                // return config_edge_lut1[m_config][edge];
				return tex1Dfetch(config_edge_lut1_tex, m_config * VOXEL_NUM_EDGES + edge);
            }
        }
        
    private:
        __host__ __device__ uint8_t get_edge_shift(voxel_edge_index_type edge) const
        {
            uint8_t shift;
            switch (edge)
            {
                case 6:
                    shift = EDGE_6_SHIFT;
                    break;
                case 9:
                    shift = EDGE_9_SHIFT;
                    break;
                case 10:
                    shift = EDGE_10_SHIFT;
                    break;
                default:
                    assert(false);
                    break;
            }
            return shift;
        }
        
        __host__ __device__ void info_write_bit(uint8_t shift, bool flag)
        {
            info_type mask;
            if (flag)
            {
                mask = 0x01 << shift;
                m_info |= mask;
            }
            else
            {
                mask = ~(0x01 << shift);
                m_info &= mask;
            }
        }
        
        __host__ __device__ inline info_type info_read_bit(uint8_t shift) const
        {
            info_type shifted_info = m_info >> shift;
            return shifted_info & 0x01;
        }

        friend std::ostream& operator<<(std::ostream& os, const _VoxelInfo vx_info);
        
        // Its index_1D
        voxel_index1D_type m_index1D = INVALID_INDEX_1D;
        // The beginning index of the vertices (both DMC iso_vertex and iso-surface edge
        // intersection point).
        vertex_index_type m_vertex_begin = INVALID_UINT32;
        // The voxel config mask, each bit corresponds to one unique vertex corner point.
        // LSB (bit 0) represents corner pt 0, MSB (bit 7) represents corner pt 7
        voxel_config_type m_config = 0x00;
        // Compact bit vector:
        // bit 7: should use LUT2?
        // bit 5: is edge 10 using ccw?
        // bit 4: is edge 9 using ccw?
        // bit 3: is edge 6 using ccw?
        // bit 2: is edge 10 bipolar?
        // bit 1: is edge 9 bipolar?
        // bit 0: is edge 6 bipolar?
        // other bits: not used
        info_type m_info = 0x00;
        // Since this class will be enforeced aligned, we can use another 8 bit to store the
        // number of vertices, although we can fully retrieve this information under the help
        // of both 'info' and other 'LUT's. 8_bit is quite enough because each voxel will have
        // a maximum of 4 + 3 = 7 vertices. (4 for DMC iso-vertices, 3 for bipolar edge pts)
        uint8_t m_num_vertices = 0;
		uint8_t m_pad = 0xff;
    };
    
	std::ostream& operator<<(std::ostream& os, const _VoxelInfo vx_info)
	{
		os << "index1D: " << vx_info.index1D()
			<< " config: " << std::hex <<(unsigned) vx_info.config() << std::dec
			<< " num_vertices: " << (unsigned)vx_info.num_vertices()
			<< " vertex_begin: " << vx_info.vertex_begin()
			<< " info: " << std::hex << (unsigned)vx_info.m_info << std::dec;
		return os;
	}
    // Calculate a voxel's config mask.
    __device__ voxel_config_type voxel_config_mask(const float* d_voxel_vals, float iso_value)
    {
        voxel_config_type mask = 0;
        for (uint8_t i = 0; i < 8; ++i)
        {
            mask |= (d_voxel_vals[i] < iso_value) << i;
        }
        
        return mask;
    }
    
	__device__ bool is_out_of_grid_bound(const uint3& index3D, const uint3& grid_size)
	{
		return (index3D.x >= grid_size.x) || (index3D.y >= grid_size.y) || (index3D.z >= grid_size.z);
	}

    // Scan through and flag out the active voxels according to its voxel config.
    __global__ void 
	flag_active_voxels_kern(flag_type* d_voxel_flags, const float* d_scalar_grid, const uint3 num_voxels_dim, const float iso_value)
    {
		uint3 index3D;
		index3D.x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
		index3D.y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
		index3D.z = __mul24(blockDim.z, blockIdx.z) + threadIdx.z;
		if (is_out_of_grid_bound(index3D, num_voxels_dim)) return;

		float voxel_vals[8] = 
		{
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y,		index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y,		index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y + 1,	index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y + 1,	index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y,		index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y,		index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y + 1,	index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y + 1,	index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
		};

		voxel_config_type voxel_config = voxel_config_mask(voxel_vals, iso_value);
		unsigned index1D = index3D_to_1D(index3D, num_voxels_dim);
		d_voxel_flags[index1D] = (voxel_config && voxel_config < MAX_VOXEL_CONFIG_MASK);
		// printf("i: %d, j: %d, k: %d, index1D: %d, config: %x, flag: %d\n", 
		//		index3D.x, index3D.y, index3D.z, index1D, voxel_config, d_voxel_flags[index1D]);
    }
    
	void launch_flag_active_voxels(flag_type* d_voxel_flags, const float* d_scalar_grid, const uint3 num_voxels_dim, const float iso_value,
			const dim3 blocks_dim3, const dim3 threads_dim3)
	{
		flag_active_voxels_kern<<<blocks_dim3, threads_dim3>>>(d_voxel_flags, d_scalar_grid, num_voxels_dim, iso_value);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	size_t launch_thrust_count(const unsigned* d_arr, size_t size)
	{
		return thrust::reduce(thrust::device, d_arr, d_arr + size);
	}

	void launch_thrust_scan(unsigned* d_scan, const unsigned* d_data, size_t size)
	{
		thrust::exclusive_scan(thrust::device, d_data, d_data + size, d_scan);
	}

	template <typename OutT, typename InT, typename UnaryOp>
	void launch_thrust_transform_scan(OutT* d_scan, const InT* d_data, size_t size, const UnaryOp& op)
	{
		auto begin = thrust::make_transform_iterator(d_data, op);
		auto end = thrust::make_transform_iterator(d_data + size, op);
		thrust::exclusive_scan(thrust::device, begin, end, d_scan);
	}

    // Calculate the position of the isosurface's vertex
    __device__ float3 lerp_float3(const float3& p0, const float3& p1, const float v0, const float v1, const float iso_value)
    {
        float interp = (iso_value - v0) / (v1 - v0);
        float one_minus_interp = 1.0f - interp;
        
        float3 iso_vertex = p0 * one_minus_interp + p1 * interp;
        
        return iso_vertex;
    }
    
    // Check if an edge is bipolar given its two endpoints' value
    __device__ bool is_edge_bipolar(float val0, float val1, float iso_value)
    {
        if (val0 == val1) return false;
        else if (val0 > val1) return is_edge_bipolar(val1, val0, iso_value);
        return !((val0 < iso_value && val1 < iso_value) || (val0 > iso_value && val1 > iso_value));
    }
    
    // Return an edge index given its two point indices
    __device__ voxel_edge_index_type pt_pair_edge_lut(voxel_pt_index_type p0, voxel_pt_index_type p1)
    {
        // assert(p0 != p1);
        if (p0 > p1)
            return pt_pair_edge_lut(p1, p0);
        
        if      (p0 == 0 && p1 == 1) return 0;
        else if (p0 == 1 && p1 == 2) return 1;
        else if (p0 == 2 && p1 == 3) return 2;
        else if (p0 == 0 && p1 == 3) return 3;
        else if (p0 == 0 && p1 == 4) return 4;
        else if (p0 == 1 && p1 == 5) return 5;
        else if (p0 == 2 && p1 == 6) return 6;
        else if (p0 == 3 && p1 == 7) return 7;
        else if (p0 == 4 && p1 == 5) return 8;
        else if (p0 == 5 && p1 == 6) return 9;
        else if (p0 == 6 && p1 == 7) return 10;
        else if (p0 == 4 && p1 == 7) return 11;

        // assert(false);
    }
    
    // Compact to get the active voxels, for each compacted voxel, store its index_1D.
    // [invariant] for 0 <= i < d_compact_voxel_info.size(),
    //                  d_full_voxel_index_map[d_compact_voxel_info[i].index1D] == i
    __global__ void 
	compact_voxel_flags_kern(_VoxelInfo* d_compact_voxel_info, voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim,
                             const flag_type* d_flags, const unsigned* d_flags_scan, const unsigned flags_size)
    {
		// index = (gridDim.x * blockDim.x) * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x
		// unsigned index1D = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		// index1D = __mul24(index1D, blockDim.x) + threadIdx.x;

		// if (index1D >= flags_size) return;
		uint3 index3D;
		index3D.x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
		index3D.y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
		index3D.z = __mul24(blockDim.z, blockIdx.z) + threadIdx.z;
		if (is_out_of_grid_bound(index3D, num_voxels_dim)) return;

		unsigned index1D = index3D_to_1D(index3D, num_voxels_dim);
		unsigned compact_index = d_flags_scan[index1D];

		// printf("index1D: %d, scan: %d, flag: %d\n", index1D, compact_index, d_flags[index1D]);
		if (d_flags[index1D])
		{
			d_full_voxel_index_map[index1D] = compact_index; // d_flags_scan[index1D];
			d_compact_voxel_info[compact_index] = _VoxelInfo(index1D);
		}
    }
    
	void launch_compact_voxel_flags(_VoxelInfo* d_compact_voxel_info, voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim,
                             const flag_type* d_flags, const unsigned* d_flags_scan, const unsigned flags_size,
							 const dim3 blocks_dim3, const dim3 threads_dim3)
	{
		compact_voxel_flags_kern<<<blocks_dim3, threads_dim3>>>(d_compact_voxel_info,
				d_full_voxel_index_map, num_voxels_dim, d_flags, d_flags_scan, flags_size);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

    // Initialize the voxel info. During this stage we only store the voxel config and
    // the edges this voxel manages (edge 6, 9, 10) are bipolar. The possible situation
    // where voxels with 2B config and 3B config are adjacent are not resolved at this stage.
    __global__ void 
	init_voxels_info_kern(_VoxelInfo* d_compact_voxel_info, const unsigned compact_size,
						const float* d_scalar_grid, const uint3 num_voxels_dim, const float iso_value)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		uint3 index3D = index1D_to_3D(vx_info.index1D(), num_voxels_dim);

		float voxel_vals[8] = 
		{
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y,		index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y,		index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y + 1,	index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y + 1,	index3D.z,		num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y,		index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y,		index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x + 1,	index3D.y + 1,	index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
			d_scalar_grid[index3D_to_1D(index3D.x,		index3D.y + 1,	index3D.z + 1,	num_voxels_dim.x + 1, num_voxels_dim.y + 1)],
		};

		vx_info.set_config(voxel_config_mask(voxel_vals, iso_value));
		
		auto encode_voxel_edge_info = [=, &vx_info](voxel_pt_index_type p0, voxel_pt_index_type p1)
        {
			voxel_edge_index_type edge_index = pt_pair_edge_lut(p0, p1);
            bool is_bipolar = is_edge_bipolar(voxel_vals[p0], voxel_vals[p1], iso_value);
            if (is_bipolar)
            {
                bool use_ccw = voxel_vals[p0] <= iso_value;
                vx_info.encode_edge_bipolar_info(edge_index, is_bipolar, use_ccw);
            }
            else
            {
                vx_info.encode_edge_is_bipolar(edge_index, is_bipolar);
            }
        };

		encode_voxel_edge_info(2, 6);   // edge 6
        encode_voxel_edge_info(5, 6);   // edge 9
        encode_voxel_edge_info(7, 6);   // edge 10

		d_compact_voxel_info[compact_index] = vx_info;
		// printf("compact index: %d, index1D: %d, config: %x, info: %x\n", 
		//		compact_index, vx_info.index1D(), (unsigned)vx_info.config(), (unsigned)vx_info.info());
    }
    
	void launch_init_voxels_info(_VoxelInfo* d_compact_voxel_info, const unsigned compact_size,
						const float* d_scalar_grid, const uint3 num_voxels_dim, const float iso_value,
						const dim3 blocks_dim3, const dim3 threads_dim3)
	{
		init_voxels_info_kern<<<blocks_dim3, threads_dim3>>>(d_compact_voxel_info, compact_size, d_scalar_grid, num_voxels_dim, iso_value);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}
    // Check if the given voxel config belongs to 2B or 3B ambiguous config category.
    __device__ bool is_ambiguous_config(voxel_config_type config, uint8_t& index)
    {
		for (unsigned i = 0; i < NUM_AMBIGUOUS_CONFIGS; ++i)
		{
			// if (config_2B_3B_lut[i] == config)
			// if (d_config_2B_3B_lut[i] == config)
			if (tex1Dfetch(config_2B_3B_lut_tex, i) == config)
			{
				index = i;
				return true;
			}
		}
		return false;
    }
    
    // Check if after we advance the 'index3D' according to 'dir', the new result will
    // exceed the boundary or not. Have to use this function because we are using unsigned
    // int instead of int.
    __device__ bool will_exceed_boundary(uint3 index3D, uint3 dims, const check_dir_type dir)
    {
        switch (dir) 
		{
		case POS_X_DIR: // CHECK_DIR::PX:
			return index3D.x + 1 >= dims.x;
                
		case NEG_X_DIR: // CHECK_DIR::NX:
            return index3D.x == 0;
            
		case POS_Y_DIR: // CHECK_DIR::PY:
			return index3D.y + 1 >= dims.y;
                
		case NEG_Y_DIR: // CHECK_DIR::NY:
			return index3D.y == 0;
                
		case POS_Z_DIR: // CHECK_DIR::PZ:
			return index3D.z + 1 >= dims.z;
                
		case NEG_Z_DIR: // CHECK_DIR::NZ:
			return index3D.z == 0;
                
		default:
			return false;
        }
    }
    
    // Execute the 'dir' on 'index3D' to get the new result. It is the user's responsibility
    // to make sure that the result won't exceed the boundary.
    __device__ uint3 get_index3D_by_dir(uint3 index3D, const check_dir_type dir)
    {
        switch (dir) 
		{
		case POS_X_DIR: // CHECK_DIR::PX:
			return make_uint3(index3D.x + 1, index3D.y, index3D.z);
		case NEG_X_DIR: // CHECK_DIR::NX:
			return make_uint3(index3D.x - 1, index3D.y, index3D.z);
		case POS_Y_DIR: // CHECK_DIR::PY:
			return make_uint3(index3D.x, index3D.y + 1, index3D.z);
		case NEG_Y_DIR: // CHECK_DIR::NY:
			return make_uint3(index3D.x, index3D.y - 1, index3D.z);
		case POS_Z_DIR: // CHECK_DIR::PZ:
			return make_uint3(index3D.x, index3D.y, index3D.z + 1);
		case NEG_Z_DIR: // CHECK_DIR::NZ:
			return make_uint3(index3D.x, index3D.y, index3D.z - 1);
        }
    }
    
    // Check if the active voxel indicated by 'cur_compact_index' has an adjacent voxel which has
    // an ambiguous config that will result in non-manifold situation.
    // [precondition] d_compact_voxel_info[cur_compact_index].config == config_2B_3B_lut[cur_config_index]
    __device__ bool is_adjacent_ambiguous_config(voxel_index1D_type& adjacent_compact_index,
												voxel_index1D_type cur_index1D, uint8_t cur_config_index,
												const _VoxelInfo* d_compact_voxel_info, const voxel_index1D_type* d_full_voxel_index_map,
												const uint3& num_voxels_dim)
    {
        // assert(compact_voxel_info[cur_compact_index].config() == config_2B_3B_lut[cur_config_index]);
        // Get the 3D coordinate of the current active voxel
		uint3 cur_index3D = index1D_to_3D(cur_index1D, num_voxels_dim);
        // uint3 cur_index3D = index1D_to_3D(compact_voxel_info[cur_compact_index].index1D(), num_voxels_dim);

        // Get the checking direction, or offset, according to 'cur_ambiguous_face'
        // voxel_face_index_type cur_ambiguous_face = config_2B_3B_ambiguous_face[cur_config_index];
		voxel_face_index_type cur_ambiguous_face = tex1Dfetch(config_2B_3B_ambiguous_face_tex, cur_config_index);
        // CHECK_DIR dir = face_to_check_dir_lut[cur_ambiguous_face];
		check_dir_type dir = tex1Dfetch(face_to_check_dir_lut_tex, cur_ambiguous_face);
        
        if (will_exceed_boundary(cur_index3D, num_voxels_dim, dir))
        {
            return false;
        }
        
        // Compute the index of the voxel to be checked in 'd_compact_voxel_info'
        uint3 index3D_to_check = get_index3D_by_dir(cur_index3D, dir);
        voxel_index1D_type index1D_to_check;
        index3D_to_1D(index3D_to_check, num_voxels_dim, index1D_to_check);
        
        voxel_index1D_type adjc_compact_index_to_check = d_full_voxel_index_map[index1D_to_check];
        // assert(adjc_compact_index_to_check != INVALID_INDEX_1D);
        
        uint8_t adj_config_index;
        if (is_ambiguous_config(d_compact_voxel_info[adjc_compact_index_to_check].config(), adj_config_index))
        {
            // voxel_face_index_type adj_ambiguous_face = config_2B_3B_ambiguous_face[adj_config_index];
            // assert(opposite_face_lut[cur_ambiguous_face] == adj_ambiguous_face);
            adjacent_compact_index = adjc_compact_index_to_check;
            return true;
        }
        
        return false;
    }

    // Correct some of the voxels when it and its adjacent voxel are having ambiguous configs that will
    // result in non-manifold. Returns the actual number of vertices, including both iso-vertex and
    // intersection vertex between voxel bipolar edge and iso-surface.
    __global__ void correct_voxels_info_kern(_VoxelInfo* d_compact_voxel_info, unsigned compact_size,
											const voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);

		uint8_t ambiguous_config_index = INVALID_UINT8;
            
		// if ((vx_info.use_lut2()) || (!is_ambiguous_config(vx_info.config(), ambiguous_config_index)))
		if (!is_ambiguous_config(vx_info.config(), ambiguous_config_index))
        {
            return;
        }

		voxel_index1D_type adjacent_compact_index;
		if (is_adjacent_ambiguous_config(adjacent_compact_index, vx_info.index1D(), ambiguous_config_index,
										d_compact_voxel_info, d_full_voxel_index_map, num_voxels_dim))
        {
			printf("compact_index %d uses lut2!\n", compact_index);
            d_compact_voxel_info[compact_index].encode_use_lut2(true);
            // d_compact_voxel_info[adjacent_compact_index].encode_use_lut2(true);
        }
    }

	void launch_correct_voxels_info(_VoxelInfo* d_compact_voxel_info, unsigned num_compact_voxels,
									const voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim,
									const dim3 blocks_dim3, const dim3 threads_dim3)
	{
		correct_voxels_info_kern<<<blocks_dim3, threads_dim3>>>(d_compact_voxel_info, num_compact_voxels, d_full_voxel_index_map, num_voxels_dim);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	__global__ void calc_num_vertices_per_voxel_kern(_VoxelInfo* d_compact_voxel_info, unsigned compact_size)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		uint8_t num_voxel_vertices = 0;
		// num iso-vertices
		if (vx_info.use_lut2())
        {
            // num_voxel_vertices += num_vertex_lut2[vx_info.config()];
			num_voxel_vertices += tex1Dfetch(num_vertex_lut2_tex, vx_info.config());
        }
        else
        {
            // num_voxel_vertices += num_vertex_lut1[vx_info.config()];
			num_voxel_vertices += tex1Dfetch(num_vertex_lut1_tex, vx_info.config());
        }
		// num edge iso-surface intersection vertices
        num_voxel_vertices += vx_info.num_edge_vertices();

		d_compact_voxel_info[compact_index].set_num_vertices(num_voxel_vertices);
    }

	void launch_calc_num_vertices_per_voxel(_VoxelInfo* d_compact_voxel_info, unsigned num_compact_voxels,
											const dim3 blocks_dim3, const dim3 threads_dim3)
	{
		calc_num_vertices_per_voxel_kern<<<blocks_dim3, threads_dim3>>>(d_compact_voxel_info, num_compact_voxels);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

	__global__ void set_vertices_begin_kern(_VoxelInfo* d_compact_voxel_info, const vertex_index_type* d_vertices_begin_scan, unsigned compact_size)
	{
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		d_compact_voxel_info[compact_index].set_vertex_begin(d_vertices_begin_scan[compact_index]);
	}
	
	void launch_set_vertices_begin(_VoxelInfo* d_compact_voxel_info, const vertex_index_type* d_vertices_begin_scan, unsigned compact_size,
								dim3 blocks_dim3, dim3 threads_dim3)
	{
		set_vertices_begin_kern<<<blocks_dim3, threads_dim3>>>(d_compact_voxel_info, d_vertices_begin_scan, compact_size);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
	}

    __device__ void 
	decode_edge_belong_voxel_entry(uint8_t entry, int8_t& x_offset, int8_t& y_offset, int8_t& z_offset, uint8_t& belonged_edge_index)
    {
        if (entry == LOCAL_EDGE_ENTRY) return;
        // extract the edge
        belonged_edge_index = 0x0f & entry;
        
        auto get_offset = [](uint8_t first_bit)
        {
            switch (first_bit) {
                case 0x00:
                    return (int8_t)0;
                case 0x80:
                    return (int8_t)-1;
                default:
                    assert(false);
					return (int8_t)0xff;
            }
        };
        
        uint8_t first_bit = entry & 0x80;
        x_offset = get_offset(first_bit);
        
        entry <<= 1;
        first_bit = entry & 0x80;
        y_offset = get_offset(first_bit);
        
        entry <<= 1;
        first_bit = entry & 0x80;
        z_offset = get_offset(first_bit);
    }

    // Sample the intersection vertices positions between voxel bipolar edges and iso-surface.
    // Each voxel is only responsible for its local edges, namely 6, 9 and 10.
    __global__ void sample_edge_intersection_vertices_kern(float3* d_vertices, const _VoxelInfo* d_compact_voxel_info, const unsigned compact_size,
                                           const float* d_scalar_grid, const uint3 num_voxels_dim,
                                           const float3 xyz_min, const float3 xyz_range, const float iso_value)
    {
        unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		uint3 index3D = index1D_to_3D(d_compact_voxel_info[compact_index].index1D(), num_voxels_dim);
		vertex_index_type vx_edge_vertex_index = d_compact_voxel_info[compact_index].edge_vertex_begin();
		
		float x1 = ijk_to_xyz(index3D.x + 1, num_voxels_dim.x, xyz_range.x, xyz_min.x);
		float y1 = ijk_to_xyz(index3D.y + 1, num_voxels_dim.y, xyz_range.y, xyz_min.y);
        float z1 = ijk_to_xyz(index3D.z + 1, num_voxels_dim.z, xyz_range.z, xyz_min.z);
		float val6 = d_scalar_grid[index3D_to_1D(index3D.x + 1, index3D.y + 1, index3D.z + 1, num_voxels_dim.x + 1, num_voxels_dim.y + 1)]; 

		float xyz_changed = ijk_to_xyz(index3D.z, num_voxels_dim.z, xyz_range.z, xyz_min.z);
		if (d_compact_voxel_info[compact_index].is_edge_bipolar(6))
		{
			// edge 6, pt 2 & 6
			d_vertices[vx_edge_vertex_index] = lerp_float3(make_float3(x1, y1, xyz_changed), make_float3(x1, y1, z1),
					d_scalar_grid[index3D_to_1D(index3D.x + 1, index3D.y + 1, index3D.z, num_voxels_dim.x + 1, num_voxels_dim.y + 1)], val6, iso_value);

			++vx_edge_vertex_index;
		}

		xyz_changed = ijk_to_xyz(index3D.y, num_voxels_dim.y, xyz_range.y, xyz_min.y);
		if (d_compact_voxel_info[compact_index].is_edge_bipolar(9))
		{
			// edge 9, pt 5 & 6
			d_vertices[vx_edge_vertex_index] = lerp_float3(make_float3(x1, xyz_changed, z1), make_float3(x1, y1, z1),
					d_scalar_grid[index3D_to_1D(index3D.x + 1, index3D.y, index3D.z + 1, num_voxels_dim.x + 1, num_voxels_dim.y + 1)], val6, iso_value);

			++vx_edge_vertex_index;
		}
		
		xyz_changed = ijk_to_xyz(index3D.x, num_voxels_dim.x, xyz_range.x, xyz_min.x);
		if (d_compact_voxel_info[compact_index].is_edge_bipolar(10))
		{
			// edge 10, pt 6 & 7
			d_vertices[vx_edge_vertex_index] = lerp_float3(make_float3(x1, y1, z1), make_float3(xyz_changed, y1, z1),
					val6, d_scalar_grid[index3D_to_1D(index3D.x, index3D.y + 1, index3D.z + 1, num_voxels_dim.x + 1, num_voxels_dim.y + 1)], iso_value);

			++vx_edge_vertex_index;
		}
    }
    
    // Calculate the iso vertices positions in each voxel.
    __global__ void calc_iso_vertices_kern(float3* d_vertices, const _VoxelInfo* d_compact_voxel_info, const unsigned compact_size,
                           const voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		extern __shared__ _VoxelInfo sh_vx_info[];
		sh_vx_info[threadIdx.x] = d_compact_voxel_info[compact_index];
		// _VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		
		uint3 index3D = index1D_to_3D(sh_vx_info[threadIdx.x].index1D(), num_voxels_dim);
		uint8_t iso_vertex_num_incident[4] = {0, 0, 0, 0};
		
		for (voxel_edge_index_type edge = 0; edge < VOXEL_NUM_EDGES; ++edge)
		{
			iso_vertex_m_type iso_vertex_m = sh_vx_info[threadIdx.x].iso_vertex_m_by_edge(edge);

            if (iso_vertex_m == NO_VERTEX)
            {
                continue;
            }
			
			// uint8_t entry = edge_belonged_voxel_lut[edge];
			uint8_t entry = tex1Dfetch(edge_belonged_voxel_lut_tex, edge);
			voxel_edge_index_type belonged_edge = 0xff;
			voxel_index1D_type belonged_index1D = INVALID_INDEX_1D;
			
			if (entry == LOCAL_EDGE_ENTRY)
			{
				// edge belongs to current voxel
				belonged_index1D = sh_vx_info[threadIdx.x].index1D();
				belonged_edge = edge;
			}
			else
			{
				int8_t x_offset = 0xff, y_offset = 0xff, z_offset = 0xff;
				decode_edge_belong_voxel_entry(entry, x_offset, y_offset, z_offset, belonged_edge);
				bool exceed_boundary  = (x_offset < 0 && index3D.x == 0) ||
										(y_offset < 0 && index3D.y == 0) ||
										(z_offset < 0 && index3D.z == 0);
                if (exceed_boundary)
                {
                    continue;
                }
                    
                belonged_index1D = index3D_to_1D(index3D.x + x_offset, index3D.y + y_offset, index3D.z + z_offset,
												num_voxels_dim.x, num_voxels_dim.y);
			}

			// Get the 'belonged_voxel' which manages 'belonged_edge'
			vertex_index_type edge_intersect_vertex_index = d_compact_voxel_info[d_full_voxel_index_map[belonged_index1D]].edge_vertex_index(belonged_edge);
			vertex_index_type iso_vertex_index = sh_vx_info[threadIdx.x].iso_vertex_index(iso_vertex_m);
			if (iso_vertex_num_incident[iso_vertex_m] == 0)
			{
				// If this is the first time we see 'iso_vertex_m', we just assign it
				d_vertices[iso_vertex_index] = d_vertices[edge_intersect_vertex_index];
			}
			else
			{
				// Otherwise we increase it
				d_vertices[iso_vertex_index] += d_vertices[edge_intersect_vertex_index];
			}
			++iso_vertex_num_incident[iso_vertex_m];
		}
		// For each iso-vertex managed by 'vx_info', calculate its new position by averaging its
		// associated edges intersection vertex positions.
		for (iso_vertex_m_type iso_vertex_m = 0; iso_vertex_m < sh_vx_info[threadIdx.x].num_iso_vertices(); ++iso_vertex_m)
		{
			vertex_index_type iso_vertex_index = sh_vx_info[threadIdx.x].iso_vertex_index(iso_vertex_m);
			if (iso_vertex_num_incident[iso_vertex_m])
			{
				d_vertices[iso_vertex_index] /= (float)(iso_vertex_num_incident[iso_vertex_m]);
			}
		}
    }

    class CircularEdgeRange
    {
    public:
        class CircularEdgeIterator
        {
        public:
            typedef CircularEdgeIterator iterator_type;
            
            __device__ CircularEdgeIterator(voxel_edge_index_type edge, bool ccw)
            : m_lut_index(get_lut_index_by_edge(edge)), m_cur_state(0), m_ccw(ccw) { }
            // For end iterator
            __device__ CircularEdgeIterator(voxel_edge_index_type edge)
            : m_lut_index(get_lut_index_by_edge(edge)), m_cur_state(4), m_ccw(true) { }
            
            // We are using CircularEdgeIterator itself, it does not represent any data underlying it. However,
            // for range object to work in c++11, we have to define dereference opreator*(). Therefore we let
            // it to dereference to itself.
            __device__ const CircularEdgeIterator& operator*() const { return *this; }
            
            // We've been lazy here and only compares 'm_lut_index' and 'm_cur_state'.
            // It's not absolutely safe, but we don't expect the client should use this class at all!
            __device__ bool operator==(const iterator_type& other) const
            {
                return (m_lut_index == other.m_lut_index) && (m_cur_state == other.m_cur_state);
            }
            
            __device__ bool operator!=(const iterator_type& other) const { return !(this->operator==(other)); }
            
            __device__ iterator_type& operator++()
            {
                if (m_cur_state < 4) ++m_cur_state;
                return (*this);
            }
            
            // Retrieve the information of the adjacent voxel that shares the edge, along with
            // the edge index in that voxel, in circular order
            __device__ void retrieve(uint3& circular_index3D, voxel_edge_index_type& circular_edge, const uint3& src_index3D) const
            {
				// ccw order: 0, 1, 2, 3
				// cw order: 0, 3, 2, 1
				// cw[i] = (3 - ccw[i] + 1) % 4 = (4 - ccw[i]) % 4
                if (m_ccw)
                {
                    // circular_edge = circular_edge_lut[m_lut_index][ccw_order[m_cur_state]];
					circular_edge = tex1Dfetch(circular_edge_lut_tex, m_lut_index + m_cur_state);
                }
                else
                {
                    // circular_edge = circular_edge_lut[m_lut_index][cw_order[m_cur_state]];
					circular_edge = tex1Dfetch(circular_edge_lut_tex, m_lut_index + ((4 - m_cur_state) % 4));
                }
                // reverse calculate the adjacent voxel that shares the edge
                // uint8_t entry = edge_belonged_voxel_lut[circular_edge];
				uint8_t entry = tex1Dfetch(edge_belonged_voxel_lut_tex, circular_edge);
                if (entry == LOCAL_EDGE_ENTRY)
                {
                    circular_index3D = src_index3D;
                }
                else
                {
                    int8_t x_offset, y_offset, z_offset;
                    voxel_edge_index_type src_edge;
                    decode_edge_belong_voxel_entry(entry, x_offset, y_offset, z_offset, src_edge);
                    assert(get_lut_index_by_edge(src_edge) == m_lut_index);
                
                    x_offset = -x_offset; y_offset = -y_offset; z_offset = -z_offset;
                    circular_index3D = src_index3D;
                    circular_index3D.x += x_offset;
                    circular_index3D.y += y_offset;
                    circular_index3D.z += z_offset;
                }
            }
            
        private:
            __device__ uint8_t get_lut_index_by_edge(voxel_edge_index_type edge) const
            {
                if (edge == 6) return 0;
                else if (edge == 9) return 4;  // 1;
                else if (edge == 10) return 8; // 2;
                assert(false);
            }
            
            uint8_t m_lut_index;
            uint8_t m_cur_state;
            bool m_ccw;
        };
        
        __device__ CircularEdgeRange(voxel_edge_index_type edge, bool ccw = true)
        : m_edge(edge), m_ccw(ccw) { }
        
        __device__ CircularEdgeIterator begin() const { return {m_edge, m_ccw}; }
        __device__ CircularEdgeIterator end() const { return {m_edge}; }
        
    private:
        uint8_t m_edge;
        bool m_ccw;
    };
    
    // Check, when we want to retrieve all the four voxels sharing the same 'edge', if any of these voxels
    // will actually exceed the boundary. Notice that all the circular edges are carefully designed so that
    // the adjacent voxels will only increase their position along the positive axis direction.
    __device__ bool circular_edge_exceed_boundary(voxel_edge_index_type edge, const uint3& index3D, const uint3& num_voxels_dim)
    {
        switch (edge) {
            case 6:
                return (index3D.x + 1 >= num_voxels_dim.x) || (index3D.y + 1 >= num_voxels_dim.y);
            case 9:
                return (index3D.x + 1 >= num_voxels_dim.x) || (index3D.z + 1 >= num_voxels_dim.z);
            case 10:
                return (index3D.y + 1 >= num_voxels_dim.y) || (index3D.z + 1 >= num_voxels_dim.z);
            default:
                assert(false);
        }
    }
    
	__device__ void project_vertices_by_shared_edge(float2* projected_vertex_pos, voxel_edge_index_type edge,
												const vertex_index_type* iso_vertex_indices, const float3* compact_vertices)
    {
        if (edge == 6)
        {
			for (uint8_t i = 0; i < 4; ++i)
			{
				projected_vertex_pos[i] = xy(compact_vertices[iso_vertex_indices[i]]);
			}
        }
        else if (edge == 9)
        {
			for (uint8_t i = 0; i < 4; ++i)
			{
				projected_vertex_pos[i] = xz(compact_vertices[iso_vertex_indices[i]]);
			}
        }
        else if (edge == 10)
        {
			for (uint8_t i = 0; i < 4; ++i)
			{
				projected_vertex_pos[i] = yz(compact_vertices[iso_vertex_indices[i]]);
			}
        }
        else
        {
            assert(false);
        }
    }

    inline __device__ int8_t calc_cross_z_sign(const float2& p_left, const float2& p_mid, const float2& p_right)
    {
        float dx1 = p_right.x - p_mid.x, dy1 = p_right.y - p_mid.y;
        float dx2 = p_left.x - p_mid.x, dy2 = p_left.y - p_mid.y;
        float cross_z = dx1 * dy2 - dx2 * dy1;
        return cross_z >= 0 ? 1 : -1;
    }
    
    __device__ void calc_quadrilateral_signs(const float2* pts, uint8_t& pos_info, uint8_t& neg_info)
    {
        pos_info = 0x00; neg_info = 0x00;
        auto encode_sign_info = [&](uint8_t& info, uint8_t index)
        {
			// info: 
			//		bit 3-0, count of pos/neg signs
			//		bit 7-4, index
            info &= 0x0f; info += 1;
            index = (index & 0x0f) << 4;
            info |= index;
        };
        
        auto calc_sign = [&](uint8_t index)
        {
            int8_t sign = calc_cross_z_sign(pts[(index + 4 - 1) % 4], pts[index], pts[(index + 1) % 4]);
            if (sign == 1)
            {
                encode_sign_info(pos_info, index);
            }
            else
            {
                encode_sign_info(neg_info, index);
            }
        };
        
        for (uint8_t i = 0; i < 4; ++i)
        {
            calc_sign(i);
        }
    }
    
	// The only case for this is when (pos_info & 0x0f) == (pos_info & 0x0f) == 2
    __device__ bool is_quadrilateral_complex(uint8_t pos_info, uint8_t neg_info)
    {
        return (pos_info & 0x0f) == (neg_info & 0x0f);
    }
    
    // is_quadrilateral_convex function acts a bit weird. It tests if the four points
    // in 'pts' form a convex quadrilateral. If they does, then 'split_index' will not
    // be changed. Otherwise if they form a concave quadrilateral, 'split_index' stores
    // the index of the point (in range [0, 3]) that causes the concavity.
    __device__ bool is_quadrilateral_convex(uint8_t pos_info, uint8_t neg_info, uint8_t& unique_index)
    {
        if (((pos_info & 0x0f) == 0) || ((neg_info & 0x0f) == 0))
        {
            return true;
        }
        else if ((pos_info & 0x0f) < (neg_info & 0x0f))
        {
            unique_index = (pos_info & 0xf0) >> 4;
        }
        else if ((neg_info & 0x0f) < (pos_info & 0x0f))
        {
            unique_index = (neg_info & 0xf0) >> 4;
        }
        else
        {
            assert(false);
        }
        
        return false;
    }
    
    __device__ bool is_quadrilateral_convex(const float2* pts, uint8_t& unique_index)
    {
        uint8_t pos_info = 0x00, neg_info = 0x00;
        calc_quadrilateral_signs(pts, pos_info, neg_info);
        
        return is_quadrilateral_convex(pos_info, neg_info, unique_index);
    }
    
	__device__ float calc_radian(const float2& p_left, const float2& p_mid, const float2& p_right)
    {
        float2 v_ml = p_left - p_mid;
        normalize(v_ml);

		float2 v_mr = p_right - p_mid;
        normalize(v_mr);
        
        float theta = acosf(v_ml.x * v_mr.x + v_ml.y * v_mr.y);
        return theta;
    }

    __device__ void find_quadrilateral_split(const float2* pts, uint8_t pos_info, uint8_t neg_info,
                                  uint8_t& split0, uint8_t& split1)
    {
        uint8_t split_index;
        
        if (is_quadrilateral_convex(pos_info, neg_info, split_index))
        {
            // If it is convex, then we split the quadrilateral with the diagonal that connects the
            // point that forms the largest angle.
			float radians[4] = 
			{
				calc_radian(pts[3], pts[0], pts[1]), 
				calc_radian(pts[0], pts[1], pts[2]),
				calc_radian(pts[1], pts[2], pts[3]),
				calc_radian(pts[2], pts[3], pts[0])
			};

			uint8_t max_radian_index = 0;
			for (uint8_t i = 1; i < 4; ++i)
			{
				if (radians[i] > radians[max_radian_index]) max_radian_index = i;
			}
			split_index = max_radian_index;
            // split_index = (uint8_t)argmax(radian0, radian1, radian2, radian3);
        }
        split0 = split_index;
        split1 = (split0 + 2) % 4; // pts.size();
    }
    
    __device__ void find_quadrilateral_split(const float2* pts, uint8_t& split0, uint8_t& split1)
    {
        uint8_t pos_info = 0x00, neg_info = 0x00;
        calc_quadrilateral_signs(pts, pos_info, neg_info);
        find_quadrilateral_split(pts, pos_info, neg_info, split0, split1);
    }
    
    __device__ void 
	get_circular_vertices_by_edge(vertex_index_type* iso_vertex_indices, const voxel_edge_index_type edge, 
								const uint3& index3D, const _VoxelInfo& vx_info, const _VoxelInfo* d_compact_voxel_info, 
								const voxel_index1D_type* d_full_voxel_index_map, const uint3& num_voxels_dim)
    {
		uint8_t iter = 0;
        for (auto circular_edge_iter : CircularEdgeRange(edge, vx_info.is_edge_ccw(edge)))
        {
            uint3 circular_index3D;
            voxel_edge_index_type circular_edge;
            circular_edge_iter.retrieve(circular_index3D, circular_edge, index3D);
            
            voxel_index1D_type circular_index1D = index3D_to_1D(circular_index3D, num_voxels_dim);
            
            assert(d_full_voxel_index_map[circular_index1D] != INVALID_INDEX_1D);
            const _VoxelInfo& circular_vx_info = d_compact_voxel_info[d_full_voxel_index_map[circular_index1D]];
            
            iso_vertex_m_type circular_iso_vertex_m = circular_vx_info.iso_vertex_m_by_edge(circular_edge);
            assert(circular_iso_vertex_m != NO_VERTEX);
            
            vertex_index_type circular_iso_vertex_index = circular_vx_info.iso_vertex_index(circular_iso_vertex_m);
            iso_vertex_indices[iter] = circular_iso_vertex_index;
			++iter;
        }
    }
    
    template <typename Vec>
    __device__ bool is_inside_triangle(const Vec& p0, const Vec& p1, const Vec& p2, const Vec& pt,
										float& alpha, float& beta, float& gamma)
    {
        Vec v0(p1 - p0), v1(p2 - p0), v2(pt - p0);
        float d00 = dot(v0, v0);
        float d10 = dot(v1, v0);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);
        
        float denom_inv = d00 * d11 - d10 * d10;
        denom_inv = 1.0f / denom_inv;
        beta = (d11 * d20 - d10 * d21) * denom_inv;
        gamma = (d00 * d21 - d10 * d20) * denom_inv;
        alpha = 1.0f - beta - gamma;
        
        return (-1e-4 < beta) && (-1e-4 < gamma) && (beta + gamma < 1.0 + 1e-4);
    }
    
    __global__ void 
	smooth_edge_vertices(float3* d_vertices, const _VoxelInfo* d_compact_voxel_info, const unsigned compact_size,
						const voxel_index1D_type* d_full_voxel_index_map, const float3 xyz_min, const float3 xyz_range, const uint3 num_voxels_dim)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		extern __shared__ float2 sh_projected_vertex_pos[];
		float2* projected_vertex_pos = sh_projected_vertex_pos + threadIdx.x * 4;
		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		uint3 index3D = index1D_to_3D(vx_info.index1D(), num_voxels_dim);
		
		for (uint8_t edge_iter = 0; edge_iter < VOXEL_NUM_LOCAL_EDGES; ++edge_iter)
        {
			// voxel_edge_index_type edge = voxel_local_edges[edge_iter];
			voxel_edge_index_type edge = tex1Dfetch(voxel_local_edges_tex, edge_iter);
            if ((!vx_info.is_edge_bipolar(edge)) ||
                circular_edge_exceed_boundary(edge, index3D, num_voxels_dim))
            {
                continue;
            }

			vertex_index_type iso_vertex_indices[4] = {INVALID_UINT8, INVALID_UINT8, INVALID_UINT8, INVALID_UINT8};
			get_circular_vertices_by_edge(iso_vertex_indices, edge, index3D, vx_info, 
									d_compact_voxel_info, d_full_voxel_index_map, num_voxels_dim);
			project_vertices_by_shared_edge(projected_vertex_pos, edge, iso_vertex_indices, d_vertices);
			uint8_t pos_info = 0x00, neg_info = 0x00;
            calc_quadrilateral_signs(projected_vertex_pos, pos_info, neg_info);
            if (is_quadrilateral_complex(pos_info, neg_info))
            {
                continue;
            }

			uint8_t split0 = INVALID_UINT8, split1 = INVALID_UINT8;
            find_quadrilateral_split(projected_vertex_pos, pos_info, neg_info, split0, split1);

			float x1 = ijk_to_xyz(index3D.x + 1, num_voxels_dim.x, xyz_range.x, xyz_min.x);
			float y1 = ijk_to_xyz(index3D.y + 1, num_voxels_dim.y, xyz_range.y, xyz_min.y);
            float z1 = ijk_to_xyz(index3D.z + 1, num_voxels_dim.z, xyz_range.z, xyz_min.z);

            float2 origin;
            if (edge == 6) origin = make_float2(x1, y1);
            else if (edge == 9) origin = make_float2(x1, z1);
            else origin = make_float2(y1, z1);

			float alpha, beta, gamma;
			if (is_inside_triangle(projected_vertex_pos[split0], projected_vertex_pos[(split0 + 1) % 4],
									projected_vertex_pos[split1], origin, alpha, beta, gamma))
            {
                float3& edge_vertex = d_vertices[vx_info.edge_vertex_index(edge)];
                    
                edge_vertex  = alpha * d_vertices[iso_vertex_indices[split0]];
                edge_vertex += beta  * d_vertices[iso_vertex_indices[(split0 + 1) % 4]];
                edge_vertex += gamma * d_vertices[iso_vertex_indices[split1]];
            }
            else if (is_inside_triangle(projected_vertex_pos[split1], projected_vertex_pos[(split1 + 1) % 4],
                                        projected_vertex_pos[split0], origin, alpha, beta, gamma))
            {
                float3& edge_vertex = d_vertices[vx_info.edge_vertex_index(edge)];
                    
                edge_vertex  = alpha * d_vertices[iso_vertex_indices[split1]];
                edge_vertex += beta  * d_vertices[iso_vertex_indices[(split1 + 1) % 4]];
                edge_vertex += gamma * d_vertices[iso_vertex_indices[split0]];
            }
		}
    }
	
	__global__ void 
	calc_num_triangles_per_voxel_kern(unsigned* d_num_triangles, const _VoxelInfo* d_compact_voxel_info, 
									const unsigned compact_size, const uint3 num_voxels_dim)
	{
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		uint3 index3D = index1D_to_3D(vx_info.index1D(), num_voxels_dim);
		
		uint8_t vx_num_triangles = 0;
		for (uint8_t edge_iter = 0; edge_iter < VOXEL_NUM_LOCAL_EDGES; ++edge_iter)
        {
			// voxel_edge_index_type edge = voxel_local_edges[edge_iter];
			voxel_edge_index_type edge = tex1Dfetch(voxel_local_edges_tex, edge_iter);
            if ((!vx_info.is_edge_bipolar(edge)) ||
                circular_edge_exceed_boundary(edge, index3D, num_voxels_dim))
            {
                continue;
            }
            vx_num_triangles += 2;
        }
		d_num_triangles[compact_index] = (unsigned)vx_num_triangles;
	}

    // Genreate the actual triangles information of the mesh.
    __global__ void 
	generate_triangles_kern(uint3* d_triangles, const unsigned* d_triangles_scan,
						const _VoxelInfo* d_compact_voxel_info, const unsigned compact_size, 
						const voxel_index1D_type* d_full_voxel_index_map, const uint3 num_voxels_dim)
    {
		unsigned compact_index = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
		compact_index = __mul24(compact_index, blockDim.x) + threadIdx.x;

		if (compact_index >= compact_size) return;

		_VoxelInfo vx_info(d_compact_voxel_info[compact_index]);
		uint3 index3D = index1D_to_3D(vx_info.index1D(), num_voxels_dim);
		
		unsigned vx_triangle_index = d_triangles_scan[compact_index];
		for (uint8_t edge_iter = 0; edge_iter < VOXEL_NUM_LOCAL_EDGES; ++edge_iter)
        {
			// voxel_edge_index_type edge = voxel_local_edges[edge_iter];
			voxel_edge_index_type edge = tex1Dfetch(voxel_local_edges_tex, edge_iter);
            if ((!vx_info.is_edge_bipolar(edge)) ||
                circular_edge_exceed_boundary(edge, index3D, num_voxels_dim))
            {
                continue;
            }

			vertex_index_type iso_vertex_indices[4] = {INVALID_UINT8, INVALID_UINT8, INVALID_UINT8, INVALID_UINT8};
			get_circular_vertices_by_edge(iso_vertex_indices, edge, index3D, vx_info,
                                d_compact_voxel_info, d_full_voxel_index_map, num_voxels_dim);
			
			uint3 triangle = make_uint3(iso_vertex_indices[0], iso_vertex_indices[1], iso_vertex_indices[2]);
			d_triangles[vx_triangle_index] = triangle;
			++vx_triangle_index;

			triangle = make_uint3(iso_vertex_indices[2], iso_vertex_indices[3], iso_vertex_indices[0]);
			d_triangles[vx_triangle_index] = triangle;
			++vx_triangle_index;
        }
    }
    
	inline void get_num_voxels_dim_from_scalar_grid(uint3& num_voxels_dim,
                                                    const scalar_grid_type& h_scalar_grid)
    {
        num_voxels_dim.x = h_scalar_grid.dim_x() - 1;
        num_voxels_dim.y = h_scalar_grid.dim_y() - 1;
        num_voxels_dim.z = h_scalar_grid.dim_z() - 1;
    }

	class _VoxelInfoToNumVerticesUniOp
	{
	public:
		typedef _VoxelInfo argument_type;
		typedef unsigned result_type;

		__device__ result_type operator()(const argument_type& vx_info) const
		{
			return (unsigned)(vx_info.num_vertices());
		}
	};

    void run_dmc(std::vector<float3>& vertices, std::vector<uint3>& triangles, const scalar_grid_type& h_scalar_grid, 
				 const float3& xyz_min, const float3& xyz_max, float iso_value, unsigned num_smooth)
    {

        uint3 num_voxels_dim;
        get_num_voxels_dim_from_scalar_grid(num_voxels_dim, h_scalar_grid);
        const size_t num_total_voxels = num_voxels_dim.x * num_voxels_dim.y * num_voxels_dim.z;

        float* d_scalar_grid;
		checkCudaErrors(cudaMalloc(&d_scalar_grid, sizeof(float) * h_scalar_grid.size()));
		checkCudaErrors(cudaMemcpy(d_scalar_grid, h_scalar_grid.data(), sizeof(float) * h_scalar_grid.size(), cudaMemcpyHostToDevice));

		flag_type* d_voxel_flags;
		checkCudaErrors(cudaMalloc(&d_voxel_flags, sizeof(flag_type) * num_total_voxels));
		checkCudaErrors(cudaMemset(d_voxel_flags, 0, sizeof(unsigned) * num_total_voxels));

		dim3 threads_dim3(16, 16, 1);
		dim3 blocks_dim3((num_voxels_dim.x + threads_dim3.x - 1) / threads_dim3.x, 
						(num_voxels_dim.y + threads_dim3.y - 1) / threads_dim3.y,
						(num_voxels_dim.z + threads_dim3.z - 1) / threads_dim3.z);

		launch_flag_active_voxels(d_voxel_flags, d_scalar_grid, num_voxels_dim, iso_value, blocks_dim3, threads_dim3);
        // print_d_arr(d_voxel_flags, num_total_voxels, "voxel flag: ");
		size_t num_compact_voxels = launch_thrust_count(d_voxel_flags, num_total_voxels);
		
		unsigned* d_voxel_flags_scan;
		checkCudaErrors(cudaMalloc(&d_voxel_flags_scan, sizeof(unsigned) * num_total_voxels));
		checkCudaErrors(cudaMemset(d_voxel_flags_scan, 0, sizeof(unsigned) * num_total_voxels));

		launch_thrust_scan(d_voxel_flags_scan, d_voxel_flags, num_total_voxels);
		// print_d_arr(d_voxel_flags_scan, num_total_voxels, "flags scan: ");
		
		// thrust::device_vector<_VoxelInfo> d_compact_voxel_info_vec(num_compact_voxels);
		_VoxelInfo* d_compact_voxel_info; // = thrust::raw_pointer_cast(d_compact_voxel_info_vec.data());
		checkCudaErrors(cudaMalloc(&d_compact_voxel_info, sizeof(_VoxelInfo) * num_compact_voxels));
		checkCudaErrors(cudaMemset(d_compact_voxel_info, 0xff, sizeof(_VoxelInfo) * num_compact_voxels));
		voxel_index1D_type* d_full_voxel_index_map;
		checkCudaErrors(cudaMalloc(&d_full_voxel_index_map, sizeof(voxel_index1D_type) * num_total_voxels));
		checkCudaErrors(cudaMemset(d_full_voxel_index_map, 0xff, sizeof(voxel_index1D_type) * num_total_voxels));

		launch_compact_voxel_flags(d_compact_voxel_info, d_full_voxel_index_map, num_voxels_dim, d_voxel_flags, d_voxel_flags_scan, 
									num_total_voxels, blocks_dim3, threads_dim3);
        // print_d_arr(d_full_voxel_index_map, num_total_voxels, "full voxel map: ");

		threads_dim3 = dim3(128, 1, 1);
		blocks_dim3 = dim3((num_compact_voxels + 127) / 128, 1, 1);

		while (blocks_dim3.x > 32768)
		{
			blocks_dim3.x /= 2;
			blocks_dim3.y *= 2;
		}
		
		checkCudaErrors(cudaFree(d_voxel_flags));
		checkCudaErrors(cudaFree(d_voxel_flags_scan));

		launch_init_voxels_info(d_compact_voxel_info, num_compact_voxels, d_scalar_grid, 
							num_voxels_dim, iso_value, blocks_dim3, threads_dim3);
		launch_correct_voxels_info(d_compact_voxel_info, num_compact_voxels, 
							d_full_voxel_index_map, num_voxels_dim, blocks_dim3, threads_dim3);
		
		launch_calc_num_vertices_per_voxel(d_compact_voxel_info, num_compact_voxels, blocks_dim3, threads_dim3);
		// print_d_arr(d_compact_voxel_info, num_compact_voxels, "vx_info: ");
		
		unsigned num_vertices = 
			thrust::transform_reduce(thrust::device, d_compact_voxel_info, d_compact_voxel_info + num_compact_voxels, _VoxelInfoToNumVerticesUniOp(), 0, thrust::plus<unsigned>());
        
		unsigned* d_vertices_begin_scan;
		checkCudaErrors(cudaMalloc(&d_vertices_begin_scan, sizeof(unsigned) * num_compact_voxels));
		checkCudaErrors(cudaMemset(d_vertices_begin_scan, 0x00, sizeof(unsigned) * num_compact_voxels));
		launch_thrust_transform_scan(d_vertices_begin_scan, d_compact_voxel_info, num_compact_voxels, _VoxelInfoToNumVerticesUniOp());
		
		launch_set_vertices_begin(d_compact_voxel_info, d_vertices_begin_scan, num_compact_voxels, blocks_dim3, threads_dim3);
		// print_d_arr(d_vertices_begin_scan, num_compact_voxels, "vertices begin: ");
		checkCudaErrors(cudaFree(d_vertices_begin_scan));
		// print_d_arr(d_compact_voxel_info, num_compact_voxels, "vx_info: ");
		
		float3* d_vertices;
		checkCudaErrors(cudaMalloc(&d_vertices, sizeof(float3) * num_vertices));

		float3 xyz_range = xyz_max - xyz_min;
		sample_edge_intersection_vertices_kern<<<blocks_dim3, threads_dim3>>>(d_vertices, d_compact_voxel_info, 
							num_compact_voxels, d_scalar_grid, num_voxels_dim, xyz_min, xyz_range, iso_value);
		
		checkCudaErrors(cudaFree(d_scalar_grid));
		calc_iso_vertices_kern<<<blocks_dim3, threads_dim3, threads_dim3.x * sizeof(_VoxelInfo)>>>(d_vertices, 
						d_compact_voxel_info, num_compact_voxels, d_full_voxel_index_map, num_voxels_dim);
		
		for (unsigned smooth_iter = 0; smooth_iter < num_smooth; ++ smooth_iter)
		{
			smooth_edge_vertices<<<blocks_dim3, threads_dim3, threads_dim3.x * sizeof(float2) * 4>>>(d_vertices, 
					d_compact_voxel_info, num_compact_voxels, d_full_voxel_index_map, xyz_min, xyz_range, num_voxels_dim);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			
			calc_iso_vertices_kern<<<blocks_dim3, threads_dim3, threads_dim3.x * sizeof(_VoxelInfo)>>>(d_vertices, 
					d_compact_voxel_info, num_compact_voxels, d_full_voxel_index_map, num_voxels_dim);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			std::cout << "done for smooth iteration: " << smooth_iter << std::endl;
		}
		// print_d_arr(d_vertices, num_vertices, "all vertices: ");

		unsigned* d_num_triangles;
		checkCudaErrors(cudaMalloc(&d_num_triangles, sizeof(unsigned) * num_compact_voxels));
		checkCudaErrors(cudaMemset(d_num_triangles, 0, sizeof(unsigned) * num_compact_voxels));

		calc_num_triangles_per_voxel_kern<<<blocks_dim3, threads_dim3>>>(d_num_triangles, d_compact_voxel_info, num_compact_voxels, num_voxels_dim);

		size_t num_triangles = launch_thrust_count(d_num_triangles, num_compact_voxels);
		unsigned* d_triangles_scan;
		checkCudaErrors(cudaMalloc(&d_triangles_scan, sizeof(unsigned) * num_compact_voxels));
		checkCudaErrors(cudaMemset(d_triangles_scan, 0, sizeof(unsigned) * num_compact_voxels));

		launch_thrust_scan(d_triangles_scan, d_num_triangles, num_compact_voxels);
		checkCudaErrors(cudaFree(d_num_triangles));

		uint3* d_triangles;
		checkCudaErrors(cudaMalloc(&d_triangles, sizeof(uint3) * num_triangles));
		checkCudaErrors(cudaMemset(d_triangles, 0xff, sizeof(uint3) * num_triangles));

		generate_triangles_kern<<<blocks_dim3, threads_dim3>>>(d_triangles, d_triangles_scan, d_compact_voxel_info, num_compact_voxels, d_full_voxel_index_map, num_voxels_dim);
		// print_d_arr(d_triangles, num_triangles, "all triangles: ");
		
		checkCudaErrors(cudaFree(d_compact_voxel_info));
		checkCudaErrors(cudaFree(d_full_voxel_index_map));
		checkCudaErrors(cudaFree(d_triangles_scan));

		vertices.clear(); triangles.clear();

		vertices.resize(num_vertices);
		checkCudaErrors(cudaMemcpy(vertices.data(), d_vertices, sizeof(float3) * num_vertices, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_vertices));

		triangles.resize(num_triangles);
		checkCudaErrors(cudaMemcpy(triangles.data(), d_triangles, sizeof(uint3) * num_triangles, cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_triangles));

		std::cout << "Dual Marching Cubes done!" << std::endl;
    }
}; // namespace dmc

/*
class Isosurface
    {
    public:
        virtual ~Isosurface() = default;
        
        virtual float value(float x, float y, float z) const = 0;
    };
    
class SphereSurface : public Isosurface
{
public:
    float value(float x, float y, float z) const override
    {
        return sqrtf(x * x + y * y + z * z);
    }
};
    
class GyroidSurface : public Isosurface
{
public:
    float value(float x, float y, float z) const override
    {
        return 2.0 * (cosf(x) * sinf(y) + cosf(y) * sinf(z) + cosf(z) * sinf(x));
    }
};

void dump_obj(const char* filename, const std::vector<float3>& compact_vertices, const std::vector<uint3>& compact_triangles)
{
	std::ofstream of(filename);
	for (const auto& v : compact_vertices)
		of << "v " << v.x << " " << v.y << " " << v.z << std::endl;
	for (const auto& t: compact_triangles)
		of << "f " << t.x + 1 << " " << t.y + 1 << " " << t.z + 1 << std::endl;

}

void test_dmc()
{
    using namespace utils;
    using namespace dmc;
        

	iso_vertex_m_type* d_config_edge_lut1, * d_config_edge_lut2;
	uint8_t* d_num_vertex_lut1, * d_num_vertex_lut2;
	voxel_config_type* d_config_2B_3B_lut;
	voxel_face_index_type* d_config_2B_3B_ambiguous_face;
	voxel_face_index_type* d_opposite_face_lut;
	check_dir_type* d_face_to_check_dir_lut;
	uint8_t* d_edge_belonged_voxel_lut;
	voxel_edge_index_type* d_circular_edge_lut;
	voxel_edge_index_type* d_voxel_local_edges;
	setup_device_luts(&d_config_edge_lut1, &d_config_edge_lut2, &d_num_vertex_lut1, &d_num_vertex_lut2, 
		&d_config_2B_3B_lut, &d_config_2B_3B_ambiguous_face, &d_opposite_face_lut, &d_face_to_check_dir_lut, 
		&d_edge_belonged_voxel_lut, &d_circular_edge_lut, &d_voxel_local_edges);


    SphereSurface surface;
    // GyroidSurface surface;
    float3 xyz_min = make_float3(-5, -5, -5);
    float3 xyz_max = make_float3(5, 5, 5);
    float3 xyz_range = xyz_max - xyz_min;
    float iso_value = 4.1f;
        
    unsigned resolution = 20;
    Array3D<float> scalar_grid(resolution + 1, resolution + 1, resolution + 1);

    for (unsigned k = 0; k < scalar_grid.dim_z(); ++k)
    {
        float z = ijk_to_xyz(k, resolution, xyz_range.z, xyz_min.z);
        for (unsigned j = 0; j < scalar_grid.dim_y(); ++j)
        {
            float y = ijk_to_xyz(j, resolution, xyz_range.y, xyz_min.y);
            for (unsigned i = 0; i < scalar_grid.dim_x(); ++i)
            {
                float x = ijk_to_xyz(i, resolution, xyz_range.x, xyz_min.x);
                scalar_grid(i, j, k) = surface.value(x, y, z);
            }
        }
    }

    std::vector<float3> compact_vertices;
    std::vector<uint3> compact_triangles;
    dmc::run_dmc(compact_vertices, compact_triangles, scalar_grid, xyz_min, xyz_max, iso_value, 15);
    
	dump_obj("sphere.obj", compact_vertices, compact_triangles);
	
	for (const auto& vertex : compact_vertices)
    {
        std::cout << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
    }
    for (const auto& tri : compact_triangles)
    {
        std::cout << "f " << tri.x+1 << " " << tri.y+1 << " " << tri.z+1 << std::endl;
    }
}

int main()
{
	test_dmc();
	return 0;
}
*/