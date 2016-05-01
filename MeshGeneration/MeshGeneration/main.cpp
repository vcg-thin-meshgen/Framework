#include "meshgen.hpp"

int main()
{
	// mesh vertices
	std::vector<float3> vertices;
	// mesh triangles
	std::vector<uint3> triangles;
	// volumetric size of the raw segmented dataset 
	uint3 segSize3D;
	// load the segmented dataset
	utils::Array3D<float> segData(segSize3D); 
	// the minimum and the maximum of (x, y, z) coordinate
	float3 xyzMin, xyzMax;
	float isoVal; 
	// the volumetric size of the re-sampling grid
	uint3 meshSize3D;
	// MPU error bound
	float maxError;

	meshgen::generateMesh(vertices, triangles, segData, xyzMin, xyzMax, isoVal, meshSize3D, maxError);

	return 0;
}