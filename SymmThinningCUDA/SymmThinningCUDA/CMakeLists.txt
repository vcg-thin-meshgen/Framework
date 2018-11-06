cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(skeletonization LANGUAGES CXX CUDA)

include_directories(include)
include_directories(libs) # for h5_io.h. HDF5 10.2 DOESN't work.


set(CONDA_ROOT "$ENV{HOME}/miniconda3" CACHE STRING "Location of the conda installation" FORCE)

add_subdirectory(src)

add_executable(main main.cu)
target_link_libraries(main skeletonize)
#target_include_directories(main PRIVATE ${CONDA_ROOT}/include/)
#target_link_libraries(main -L${CONDA_ROOT}/lib)
set(HDF5_ROOT "/usr/local/HDF_Group/HDF5/1.11.4")
include_directories(main PRIVATE "${HDF5_ROOT}/include/")
target_link_libraries(main -L${HDF5_ROOT}/lib -lhdf5)



set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_compile_features(main PUBLIC cxx_std_11)
