#include <memory>			// std::unique_ptr

#include "neighbor.cuh"

namespace thin
{
	namespace nb
	{
		namespace _private
		{
			// The array where each entry is a neighborhood offset. The entries are ordered
			// so that neighborhood voxels adjacent by a vertex come first, then those
			// adjacent by an edge and finally by a literal face.
			const OffsetIjkType h_nbOffsetArr[NUM_NB_OFFSETS] = 
			{
				// vertex, starting from 0, range to 7, inclusive
				makeOffsetIjk(-1, -1, -1),
				makeOffsetIjk( 1, -1, -1),
				makeOffsetIjk( 1,  1, -1),
				makeOffsetIjk(-1,  1, -1),
				makeOffsetIjk(-1, -1,  1),
				makeOffsetIjk( 1, -1,  1),
				makeOffsetIjk( 1,  1,  1),
				makeOffsetIjk(-1,  1,  1),
				// edge, starting from 8, range to 19, inclusive
				makeOffsetIjk( 0, -1, -1),
				makeOffsetIjk( 1,  0, -1),
				makeOffsetIjk( 0,  1, -1),
				makeOffsetIjk(-1,  0, -1),
				makeOffsetIjk(-1, -1,  0),
				makeOffsetIjk( 1, -1,  0),
				makeOffsetIjk( 1,  1,  0),
				makeOffsetIjk(-1,  1,  0),
				makeOffsetIjk( 0, -1,  1),
				makeOffsetIjk( 1,  0,  1),
				makeOffsetIjk( 0,  1,  1),
				makeOffsetIjk(-1,  0,  1),
				// face, starting from 20, range to 25, inclusive
				makeOffsetIjk( 0,  0, -1),
				makeOffsetIjk( 0, -1,  0),
				makeOffsetIjk( 1,  0,  0),
				makeOffsetIjk( 0,  1,  0),
				makeOffsetIjk(-1,  0,  0),
				makeOffsetIjk( 0,  0,  1)
			};

			// The array where each entry is the index of the neighborhood offset in
			// @h_nbOffsetArr. The order of the entries in this array is coupled with that
			// in @h_nbOffsetArr.
			const uint8_t h_nbFlatIjkToIndexLut[] = 
			{
				0,	// -1,-1,-1	--flat-> 0: 0
				8,	// 0,-1,-1	--flat-> 1: 8
				1,	// 1,-1,-1	--flat-> 2: 1
				11, // -1,0,-1	--flat-> 3: 11
				20,	// 0,0,-1	--flat-> 4: 20
				9,	// 1,0,-1	--flat-> 5: 9
				3,	// -1,1,-1	--flat-> 6: 3
				10,	// 0,1,-1	--flat-> 7: 10
				2,	// 1,1,-1	--flat-> 8: 2
				12,	// -1,-1,0	--flat-> 9: 12
				21,	// 0,-1,0	--flat-> 10: 21
				13,	// 1,-1,0	--flat-> 11: 13
				24,	// -1,0,0	--flat-> 12: 24
				0xff, // special case for 0,0,0	--flat-> 13: INVALID!
				22,	// 1,0,0	--flat-> 14: 22
				15,	// -1,1,0	--flat-> 15: 15
				23,	// 0,1,0	--flat-> 16: 23
				14,	// 1,1,0	--flat-> 17: 14
				4,	// -1,-1,1	--flat-> 18: 4
				16,	// 0,-1,1	--flat-> 19: 16
				5,	// 1,-1,1	--flat-> 20: 5
				19,	// -1,0,1	--flat-> 21: 19
				25,	// 0,0,1	--flat-> 22: 25
				17,	// 1,0,1	--flat-> 23: 17
				7,	// -1,1,1	--flat-> 24: 7
				18,	// 0,1,1	--flat-> 25: 18
				6	// 1,1,1	--flat-> 26: 6
			};

			// A singleton class that unions all the device pointers for texture reference.
			class DevArrPtrs
			{
			public:
				static DevArrPtrs* instance()
				{
					if (!m_instance)
					{
						m_instance = std::unique_ptr<DevArrPtrs>(new DevArrPtrs);
					}
					return m_instance.get();
				}

				
				OffsetIjkType* d_nbOffsetArr;
				uint8_t* d_nbFlatIjkToIndexLut;

			private:
				static std::unique_ptr<DevArrPtrs> m_instance;
			};

			std::unique_ptr<DevArrPtrs> DevArrPtrs::m_instance = nullptr;
			// Neighborhood offsets device texture reference object.
			tp::OffsetIjkTexType nbOffsetTex;
			// Neighborhood offset index device texture reference object.
			tp::Uint8TexType nbFlatIjkToIndexTex;
			// Initialize the device texture references of this module.
			void _initDeviceTex(OffsetIjkType** d_nbOffsetArr,  uint8_t** d_nbFlatIjkToIndexLut)
			{
				const cudaChannelFormatDesc char4Desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned);
				const cudaChannelFormatDesc uint8Desc = cudaCreateChannelDesc(8 * sizeof(OffsetCompType), 0, 0, 0, cudaChannelFormatKindUnsigned);

				checkCudaErrors(cudaMalloc(d_nbOffsetArr, sizeof(OffsetIjkType) * NUM_NB_OFFSETS));
				checkCudaErrors(cudaMemcpy(*d_nbOffsetArr, h_nbOffsetArr, sizeof(OffsetIjkType) * NUM_NB_OFFSETS, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, nbOffsetTex, *d_nbOffsetArr, char4Desc, sizeof(OffsetIjkType) * NUM_NB_OFFSETS));

				checkCudaErrors(cudaMalloc(d_nbFlatIjkToIndexLut, sizeof(uint8_t) * 27U));
				checkCudaErrors(cudaMemcpy(*d_nbFlatIjkToIndexLut, h_nbFlatIjkToIndexLut, sizeof(uint8_t) * 27U, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, nbFlatIjkToIndexTex, *d_nbFlatIjkToIndexLut, uint8Desc, sizeof(uint8_t) * 27U));
			}
			// Unbinds the GPU texture references and frees the device memory.
			void _clearDeviceTex(OffsetIjkType* d_nbOffsetArr, uint8_t* d_nbFlatIjkToIndexLut)
			{
				checkCudaErrors(cudaFree(d_nbOffsetArr));
				checkCudaErrors(cudaFree(d_nbFlatIjkToIndexLut));
			}
			// Flat the neighborhood offset into a scalar.
			__device__ uint8_t inline _flatNbOffset(const OffsetIjkType& offs)
			{
				// The bounding box of the neighborhood offsets is always (3,3,3).
				// FlatIjk formular, given offsIjk = (x, y, z),
				// flatIjk = (x + 1) + (y + 1) * 3 + (z + 1) * 3 * 3 = x +3y + 9z + 13,
				// which is always >= 0
				return 13U + offs.x + 3U * offs.y + 9U * offs.z;
			}

			__device__ uint8_t inline _fetchNbOffsIndexByFlat(uint8_t flatOffs)
			{
				return tex1Dfetch(nbFlatIjkToIndexTex, flatOffs);
			}
		}; // namespace thin::nb::_private;

		namespace np = thin::nb::_private;

		void initDevice()
		{
			np::DevArrPtrs* ptrs = np::DevArrPtrs::instance();

			np::_initDeviceTex(&(ptrs->d_nbOffsetArr), &(ptrs->d_nbFlatIjkToIndexLut));
		}

		void shutdownDevice()
		{
			np::DevArrPtrs* ptrs = np::DevArrPtrs::instance();

			np::_clearDeviceTex(ptrs->d_nbOffsetArr, ptrs->d_nbFlatIjkToIndexLut);
		}

		__device__ OffsetIjkType fetchNbOffset(uint8_t nbOffsetIdx)
		{
			return tex1Dfetch(np::nbOffsetTex, nbOffsetIdx);
		}

		__device__ uint8_t fetchIndexOfNbOffset(const OffsetIjkType& nbOffs)
		{
			return np::_fetchNbOffsIndexByFlat(np::_flatNbOffset(nbOffs));
		}

		__device__ NbMaskType
		generateNbMask(IjkType vxIjk, const IjkType* d_nbIjkArr, const unsigned nbArrSize, const IjkType& size3D)
		{
			// The algorithm loops through each neighborhood voxel @checkIjk of @vxIjk to
			// see if it is contained in @d_nbIjkArr. If it is, then the bit at the index
			// of the current neighborhood offset is set to 1 in @nbMask.
			NbMaskType nbMask = 0;
		
			for (uint8_t offsetIndex = 0; offsetIndex < NUM_NB_OFFSETS; ++offsetIndex)
			{
				OffsetIjkType offsIjk = fetchNbOffset(offsetIndex);
			
				IjkType checkIjk;
				if (tp::_isInBoundary(vxIjk, offsIjk, size3D, checkIjk))
				{
					for (uint8_t nbIndex = 0; nbIndex < nbArrSize; ++nbIndex)
					{
						if (isEqual(d_nbIjkArr[nbIndex], checkIjk))
						{
							tp::_setBit(nbMask, offsetIndex);
							break;
						}
					}
				}
			}

			return nbMask;
		}
	}; // namespace thin::nb;
}; // namespace thin;