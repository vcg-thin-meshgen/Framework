add_library(io_shared STATIC io_shared.cpp)
target_link_libraries(io_shared "-L${HDF5_ROOT}/lib -lhdf5 -lhdf5_cpp -lhdf5_hl")
target_link_libraries(io_shared -lboost_system -lboost_filesystem)
