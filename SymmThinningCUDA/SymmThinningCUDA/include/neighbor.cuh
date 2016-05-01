#ifndef NEIGHBOR_CUH
#define NEIGHBOR_CUH

// This file provides the utility to retrieve neighborhood offsets and to create
// neighborhood masks.
//
// Users do not need to touch this file.

#include <stdint.h>

#include "cuda_includes.h"
#include "thinning_base.cuh"

namespace thin
{
	namespace nb
	{
		namespace tp = thin::_private;

		// A 3D voxel has 26 neighborhood voxels. The existences of each of the
		// surronding voxel can be encoded in a specific bit in a Bitmap. NbMaskType is
		// the implementation type of such Bitmap.
		typedef uint32_t NbMaskType;

		const unsigned NUM_NB_OFFSETS = 26U;
		// Initialize the device side resources for this module to work.
		void initDevice();
		// Release the device side resources.
		void shutdownDevice();

		// A neighborhood offset is a vector in R^3, where the addition of that with a
		// voxel at (i,j,k) will be the coordinate that is the neighborhood of (i,j,k).
		//
		// For a voxel at (i,j,k) to have a voxel at (i',j',k') as its neighborhood, it
		// will satisfy the requirement that:
		// |i - i'| <= 1 and |j - j'| <= 1 and |k - k'| <= 1
		// That means the set of all neighborhood offsets is:
		// {-1, 0, 1} X {-1, 0, 1} X {-1, 0, 1} \ {(0, 0, 0)}

		// Fetch a neighborhood offset by @nbOffsetIdx.
		//
		// [pre-condition]: 0 <= @nbOffsetIdx < @NUM_NB_OFFSETS
		__device__ OffsetIjkType fetchNbOffset(uint8_t nbOffsetIdx);
		// Fetch the index of the passed in neighborhood offset @nbOffs. This function
		// is directly coupled with fetchNbOffset() and the internal device textures for
		// storing neighboroffsets.
		//
		// [invariant] index = fetchIndexOfNbOffset(fetchNbOffset(index))
		__device__ uint8_t fetchIndexOfNbOffset(const OffsetIjkType& nbOffs);

		// Generate the neighborhood bitmap
		//
		// [in] @vxIjk: the (i,j,k) discrete coordinate of the voxel
		// [in] @d_nbIjkArr: the array of all the neighborhood voxels of @vxIjk
		// [in] @nbArrSize: size of @d_nbIjkArr
		// [in] @size3D: the 3D size of the entire voxel complex
		__device__ NbMaskType
		generateNbMask(IjkType vxIjk, const IjkType* d_nbIjkArr, const unsigned nbArrSize, const IjkType& size3D);
	}; // namespace thin::nb;
}; // namespace thin;

#endif