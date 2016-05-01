#ifndef GPU_UTILS_CUDA_INCLUDES_H
#define GPU_UTILS_CUDA_INCLUDES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <iomanip>
#include <cassert>

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

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

// __device__ function can call another __host__ __device__ function
// __host__ __deivce__ function CANNOT call another __host__ or __device__ function
#endif