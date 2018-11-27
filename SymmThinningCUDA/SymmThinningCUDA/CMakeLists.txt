cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(skeletonization LANGUAGES CXX CUDA)

# For compiling against a libhdf5 compiled from source from:
# https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git
# commit e6545faa5699aae4d5dca23ed40e3080fa9dc72d
set(HDF5_ROOT "/usr/local/HDF_Group/HDF5/1.11.4")

# For compiling against an hdf5 installed by miniconda
#set(CONDA_ROOT "$ENV{HOME}/miniconda3" CACHE STRING "Location of the conda installation" FORCE)
#target_include_directories(main PRIVATE ${CONDA_ROOT}/include/)
#target_link_libraries(main -L${CONDA_ROOT}/lib)

include_directories(io_shared PRIVATE "${HDF5_ROOT}/include/")
include_directories(include)
include_directories(libs) # for h5_io.h. HDF5 10.2 DOESN't work.


add_subdirectory(src)
add_subdirectory(libs)

add_executable(main main.cu)
target_link_libraries(main skeletonize)

target_link_libraries(main io_shared) # WARNING! CMake is failing to detect changes to this dependency!! make clean!

target_link_libraries(main "-L${HDF5_ROOT}/lib -lhdf5 -lhdf5_cpp -lhdf5_hl")

# Failed attempt to build everything statically.
#find_library(HDF5_LIB libhdf5.a)
#find_library(HDF5_CPP_LIB libhdf5_cpp.a)
#find_library(HDF5_CPP_HL libhdf5_hl.a)
#target_link_libraries(main "-L${HDF5_ROOT}/lib ${HDF5_LIB} ${HDF5_CPP_LIB} ${HDF5_CPP_HL}")
#target_link_libraries(main "-static")


target_compile_features(main PUBLIC cxx_std_11)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
