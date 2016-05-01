#include "meshgen.hpp"

#include "dmc.hpp"
#include "mpu.hpp"

namespace meshgen
{
	void extractSurfacePts(std::vector<float>& surfacePts, std::vector<float>& ptNorms,
							const std::vector<float3>& verts, const std::vector<uint3>& tris)
	{
		surfacePts.clear();
		ptNorms.clear();

		for (const uint3& triangle : tris)
		{
			float3 va = verts[triangle.x];
			float3 vb = verts[triangle.y];
			float3 vc = verts[triangle.z];

			float3 pt = va + vb + vc;
			pt /= 3.0f;

			float3 norm = cross((vb - va), (vc - va));
			float nLen = length(norm);

			if (nLen > 1e-10)
			{
				norm /= nLen;

				surfacePts.push_back(pt.x);
				surfacePts.push_back(pt.y);
				surfacePts.push_back(pt.z);

				ptNorms.push_back(norm.x);
				ptNorms.push_back(norm.y);
				ptNorms.push_back(norm.z);
			}
		}
	}

	void generateMesh(std::vector<float3>& vertices, std::vector<uint3>& triangles, const Array3D<float>& segData, 
					const float3& xyzMin, const float3& xyzMax, float isoVal, 
					const uint3& meshGridSize3D, float mpuMaxError,
					uint maxIter, unsigned mpuMaxDepth, unsigned mpuNmin)
	{
		dmc::setup_device_luts();
		// Surface Point Extraction
		std::vector<float3> segVerts;
		std::vector<uint3> segTris;
		dmc::run_dmc(segVerts, segTris, segData, xyzMin, xyzMax, isoVal, maxIter);

		std::vector<float> surfacePts, ptNorms,
		extractSurfacePts(surfacePts, ptNorms, segVerts, segTris);
		// MPU and resampling
		mpu::MpuOctree mpuImplicit(surfacePts, ptNorms, mpuMaxDepth, mpuNmin);

		Array3D<float> meshScalarGrid(meshGridSize3D, 0.0f);
		sampleGrid(meshScalarGrid, mpuImplicit, meshGridSize3D, xyzMin, xyzMax, maxError);

		// Final mesh generation
		dmc::run_dmc(vertices, triangle, meshScalarGrid, xyzMin, xyzMax, isoVal, 1);

		// shutdown the DMC algorithm
		dmc::cleanup_device_luts();
	}
}; // namespace meshgen;