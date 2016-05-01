#include <vector>
#include <string>
#include "dmc.hpp"

int main()
{
	// minimal and maximal (x, y, z) coordinate
	float3 xyz_min, xyz_max;
	// volumetric dataset size in 3D
	uint3 size3D;
	// grid points each with a scalar value
	utils::Array3D<float> scalar_grid(size3D.x, size3D.y, size3D.z);
	// isovalue
	float iso_value;
	float max_iter;
	// output mesh vertices
	std::vector<float3> vertices;
	// output mesh triangles
    std::vector<uint3> triangles;
	// initialize the DMC algorithm
	dmc::setup_device_luts();
	// run the DMC algorithm
	dmc::run_dmc(vertices, triangles, scalar_grid, xyz_min, xyz_max, iso_value, max_iter);
	// shutdown the DMC algorithm
	dmc::cleanup_device_luts();

	return 0;
}