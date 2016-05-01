#ifndef MESHGEN_HPP
#define MESHGEN_HPP

#include <vector>
#include <Eigen\Dense>

#include "utils.hpp"
#include "helper_math.hpp"

namespace meshgen
{
	using namespace Eigen::Vector3f;
	using namespace utils;

	
	void generateMesh(std::vector<float3>& vertices, std::vector<uint3>& triangles, 
				const Array3D<float>& segData, const float3& xyzMin, const float3& xyzMax,
				float isoVal, const uint3& meshGridSize3D, float mpuMaxError,
				uint maxIter = 12U, unsigned mpuMaxDepth = 10U, unsigned mpuNmin = 30U);

	void extractSurfacePts(std::vector<float>& surfacePts, std::vector<float>& ptNorms,
							const std::vector<float3>& verts, const std::vector<uint3>& tris);

	template <typename IMP>
	void sampleGrid(Array3D<float>& scalarGrid, IMP& implicit,
					uint3 size3D, float3 xyzMin, float3 xyzMax, float maxError)
	{
		float3 xyzStep = xyzMax - xyzMin;
		xyzStep.x /= size3D.x;
		xyzStep.y /= size3D.y;
		xyzStep.z /= size3D.z;

		float x, y, z;
		for (unsigned k = 0; k < size3D.z; ++k)
		{
			z = xyzMin.z + k * xyzStep.z;
			for (unsigned j = 0; j < size3D.y; ++j)
			{
				y = xyzMin.y + j * xyzStep.y;
				for (unsigned i = 0; i < size3D.x; ++i)
				{
					x = xyzMin.x + i * xyzStep.x;
					scalarGrid(i, j, k) = implicit.value(x, y, z, maxError);
				}
			}
		}
	}
}; // namespace meshgen
#endif