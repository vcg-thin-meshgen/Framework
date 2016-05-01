#ifndef THINNING_BASE_H
#define THINNING_BASE_H

#include <stdint.h>
#include <cassert>

#include "cuda_texture_types.h"					// texture
#include "texture_fetch_functions.h"			// tex1Dfetch

#include "cuda_includes.h"
namespace thin
{
	// Below shows how the vertices are indexed within the same voxel.
    //
    //           4 ------------------------ 7
    //          /|                         /|
    //         / |                        / |
    //        /  |                       /  |
    //       /   |                      /   |
    //      /    |                     /    |
    //     /     |                    /     |
    //    /      |                   /      |
    //   5 ------------------------ 6       |
    //   |       |                  |       |        z
    //   |       |                  |       |        |
    //   |       0 -----------------|------ 3        |___ y
    //   |      /                   |      /        /
    //   |     /                    |     /        x
    //   |    /                     |    /
    //   |   /                      |   /
    //   |  /                       |  /
    //   | /                        | /
    //   |/                         |/
    //   1 ------------------------ 2
    //
    //
    // Below shows how the edges are indexed within the same voxel.
    //
    //           * ---------- 11 ---------- *
    //          /|                         /|
    //         / |                        / |
    //        /  |                       /  |
    //      08   |                     10   |
    //      /    |                     /    |
    //     /    04                    /     07
    //    /      |                   /      |
    //   * ---------- 09 ---------- *       |
    //   |       |                  |       |        z
    //   |       |                  |       |        |
    //   |       * ---------- 03 ---|------ *        |___ y
    //   |      /                   |      /        /
    //  05     /                    |     /        x
    //   |    /                     |    /
    //   |   0                      |   02
    //   |  /                       |  /
    //   | /                        | /
    //   |/                         |/
    //   * ---------- 01 ---------- *
    //
    // This is how the faces are indexed within the same voxel:
    //   -Z pointed face: 0
    //   -Y pointed face: 1
    //   +X pointed face: 2
    //   +Y pointed face: 3
    //   -X pointed face: 4
    //   +Z pointed face: 5

	typedef uint8_t DFaceIndexType;
	
	typedef unsigned ObjIdType;
	typedef unsigned ArrIndexType;
	
	typedef uint3 IjkType;

	typedef int8_t OffsetCompType;
	// CUDA texture reference only supports T, T2 and T4. NO T3!
	typedef char4 OffsetIjkType;

	__host__ __device__ inline bool
	less(const IjkType& lhs, const IjkType& rhs)
	{
		if (lhs.z != rhs.z) return lhs.z < rhs.z;
		else if (lhs.y != rhs.y) return lhs.y < rhs.y;
		else return lhs.x < rhs.x;
	}

	__host__ __device__ inline bool isEqual(const IjkType& lhs, const IjkType& rhs)
	{
		return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
	}

	__host__ __device__ inline IjkType makeIjk(unsigned i, unsigned j, unsigned k)
	{
		return make_uint3(i, j, k);
	}

	__host__ __device__ inline OffsetIjkType 
	makeOffsetIjk(OffsetCompType i, OffsetCompType j, OffsetCompType k)
	{
		return make_char4(i, j, k, ~0);
	}

	const unsigned INVALID_UINT = 0xFFFFFFFF;

	// The set operations used in the symmetric isthmus-based thinning algorithm are
	// implemented using bits manipulation. We call this data type Recording Bits. Since
	// the number of sets involved is deterministic, we can use different bit of an integer
	// to indicate whether a voxel is in the set. 
	typedef unsigned char RecBitsType;

	namespace _private
    {
		typedef texture<uint8_t, cudaTextureType1D, cudaReadModeElementType> Uint8TexType;
		typedef texture<int8_t, cudaTextureType1D, cudaReadModeElementType> Int8TexType;
		typedef texture<OffsetIjkType, cudaTextureType1D, cudaReadModeElementType> OffsetIjkTexType;

        template <typename UINT>
        __host__ __device__ inline void _setBit(UINT& bits, uint8_t n)
        {
            unsigned mask = 1U << n;
            bits |= mask;
        }
        
        template <typename UINT>
        __host__ __device__ inline void _clearBit(UINT& bits, uint8_t n)
        {
            unsigned mask = 1U << n;
            bits &= (~mask);
        }
        
        template <typename UINT>
        __host__ __device__ inline uint8_t _readBit(UINT bits, uint8_t n)
        {
            return (bits >> n) & 1U;
        }
        
        template <typename UINT>
        __host__ __device__ inline uint8_t _countNumSetBits(UINT bits, uint8_t size)
        {
            uint8_t num = 0;
            
            for (unsigned n = 0; n < size; ++n)
            {
                num += _readBit(bits, n);
            }
            
            return num;
        }
        
		__host__ __device__ inline bool 
		_isInBoundary(const IjkType& ijk, int8_t offsI, int8_t offsJ, int8_t offsK, const IjkType& size3D, IjkType& resultIjk)
		{
			auto checker = [](unsigned coord, int8_t offs, unsigned max, unsigned& result)
			{
				bool flag = false;
				if (offs < 0)
				{
					flag = (coord >= (uint8_t)(-offs));
				}
				else
				{
					flag = coord + offs < max;
				}

				result = flag * (coord + offs) + (!flag) * INVALID_UINT;
				return flag;
			};

			return checker(ijk.x, offsI, size3D.x, resultIjk.x) &&
					checker(ijk.y, offsJ, size3D.y, resultIjk.y) &&
					checker(ijk.z, offsK, size3D.z, resultIjk.z);
		}

		__host__ __device__ inline bool 
		_isInBoundary(const IjkType& ijk, const OffsetIjkType& offsIjk, const IjkType& size3D, IjkType& resultIjk)
		{
			return _isInBoundary(ijk, offsIjk.x, offsIjk.y, offsIjk.z, size3D, resultIjk);
		}
    }; // namespace thin::_private;

}; // namespace thin
#endif