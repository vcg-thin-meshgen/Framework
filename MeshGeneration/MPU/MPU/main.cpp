#include <vector>
#include "mpu.hpp"

int main()
{
	// load data points and normal vectors into these vectors
	std::vector<float> pts;
	std::vector<float> pt_norms;
	// maximal octree depth
	unsigned max_depth = 10;
	// N_min
	unsigned n_min = 30;

	mpu::MpuOctree mpu(pts, pt_norms, max_depth, n_min);

	// evaluate the scalar value at (x, y, z)
	float x(0.0f), y(0.0f), z(0.0f);
	// error bound \epsilon
	float epsilon(0.1f);
	float val = mpu.value(x, y, z, epsilon);
	
	return 0;
}