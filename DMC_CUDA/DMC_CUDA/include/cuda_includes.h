#ifndef GPU_UTILS_CUDA_INCLUDES_H
#define GPU_UTILS_CUDA_INCLUDES_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <iostream>
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

#endif