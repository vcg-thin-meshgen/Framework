#include <memory>								// std::unique_ptr

#include "cuda_texture_types.h"					// texture
#include "texture_fetch_functions.h"			// tex1Dfetch

#include "cuda_includes.h"
#include "clique.cuh"


namespace thin
{
	namespace clique
	{
		namespace _private
		{
			// When we are checking if a clique can be formed at a certain d-face, we use
			// the spatial mask template from [1] Section 10.
			//
			// For each dimension d, a default d-face is selected, with the offsets in the
			// template set to work for this d-face. For the rest d-faces of the same
			// dimension, we can apply a linear transformation on the default offsets to
			// obtain the correct template. After enumerating all the cases, we will need 72
			// transformation matrices, which are stored in a 1D texture of size 72 (8 x 9).
			const unsigned matEntryArrSize = 72U;
			const OffsetCompType h_matEntryArr[matEntryArrSize] =
			{
				// identity
				// D3::m_mat, D2::m_mat_X, D1::m_mat_1, D0::m_mat_0
				1, 0, 0, 0, 1, 0, 0, 0, 1,                  // 0
				// rotate +90 degrees around Z axis
				// D2::m_mat_Y, D1::m_mat_2
				0, -1, 0, 1, 0, 0, 0, 0, 1,                 // 9
				// rotate -90 degrees around Y axis
				// D2::m_mat_Z, 
				0, 0, -1, 0, 1, 0, 1, 0, 0,                 // 18
				// roate -90 degrees around X axis
				// D1::m_mat_5, D0::m_mat_5
				1, 0, 0, 0, 0, 1, 0, -1, 0,                 // 27
				// rotate +90 degrees around X axis
				// D1::m_mat_6, D0::m_mat_2
				1, 0, 0, 0, 0, -1, 0, 1, 0,                 // 36
				// rotate 180 degrees around X axis
				// D1::m_mat_9, D0::m_mat_6
				1, 0, 0, 0, -1, 0, 0, 0, -1,                // 45
				// rotate +90 degrees around Z axis first to obtain X'Y'Z',
				// then rotate 180 degrees around X' axis
				// D1::m_mat_10
				0, 1, 0, 1, 0, 0, 0, 0, -1,                 // 54
				// rotate +90 degrees around Y axis
				// D2::m_mat_Z_neg
				0, 0, 1, 0, 1, 0, -1, 0, 0					// 63
			};

			// Dim 3, Voxel
			//
			// number of voxels to form the essential 3-clique
			const uint8_t D3ecArrSize = 0;
			// number of core voxels to form 3-clique
			const uint8_t D3coreArrSize = 0;
			// number of neighborhood offsets of 3-clique
			const uint8_t D3nbOffsIndexArrSize = 26U;
			// the beginning index in the device texture reference for essential clique
			const uint8_t D3ecTexBegin = 0;
			// the beginning index in the device texture reference for core voxels
			const uint8_t D3coreTexBegin = 0;
			// the beginning index in the device texture reference for the indices of the
			// neighborhood offsets
			const uint8_t D3nbOffsIndexTexBegin = 0;
			
			// Dim 2, Face
			//
			// number of voxels to form the essential 2-clique
			const uint8_t D2ecArrSize = 1U;
			// number of core voxels to form 2-clique
			const uint8_t D2coreArrSize = 2U;
			// number of neighborhood offsets of 2-clique
			const uint8_t D2nbOffsIndexArrSize = 16U;
			// the beginning index in the device texture reference for essential clique
			const uint8_t D2ecTexBegin = D3ecTexBegin + D3ecArrSize;
			// the beginning index in the device texture reference for core voxels
			const uint8_t D2coreTexBegin = D3coreTexBegin + D3coreArrSize;
			// the beginning index in the device texture reference for the indices of the
			// neighborhood offsets
			const uint8_t D2nbOffsIndexTexBegin = D3nbOffsIndexTexBegin + D3nbOffsIndexArrSize;

			// Dim 1, Edge
			//
			// number of voxels to form the essential 1-clique
			const uint8_t D1ecArrSize = 3U;
			// number of core voxels to form 1-clique
			const uint8_t D1coreArrSize = 5U;
			// number of neighborhood offsets of 1-clique
			const uint8_t D1nbOffsIndexArrSize = 8U;
			// the beginning index in the device texture reference for essential clique
			const uint8_t D1ecTexBegin = D2ecTexBegin + D2ecArrSize;
			// the beginning index in the device texture reference for core voxels
			const uint8_t D1coreTexBegin = D2coreTexBegin + D2coreArrSize;
			// the beginning index in the device texture reference for the indices of the
			// neighborhood offsets
			const uint8_t D1nbOffsIndexTexBegin = D2nbOffsIndexTexBegin + D2nbOffsIndexArrSize;

			// Dim 0, Vertex
			//
			// number of voxels to form the essential 0-clique
			const uint8_t D0ecArrSize = 7U;
			// number of core voxels to form 0-clique
			const uint8_t D0coreArrSize = 11U;
			// number of neighborhood offsets of 0-clique
			const uint8_t D0nbOffsIndexArrSize = 0;
			// the beginning index in the device texture reference for essential clique
			const uint8_t D0ecTexBegin = D1ecTexBegin + D1ecArrSize;
			// the beginning index in the device texture reference for core voxels
			const uint8_t D0coreTexBegin = D1coreTexBegin + D1coreArrSize;
			// the beginning index in the device texture reference for the indices of the
			// neighborhood offsets
			const uint8_t D0nbOffsIndexTexBegin = D1nbOffsIndexTexBegin + D1nbOffsIndexArrSize;

			// Essential Clique array size
			const unsigned EC_ARR_SIZE = D3ecArrSize + D2ecArrSize + D1ecArrSize + D0ecArrSize;
			// Core clique voxel array size
			const unsigned CORE_CLQ_ARR_SIZE = D3coreArrSize + D2coreArrSize + D1coreArrSize + D0coreArrSize;
			// Indices of the neighborhood offsets array size
			const unsigned NB_OFFS_IDX_ARR_SIZE = D3nbOffsIndexArrSize + D2nbOffsIndexArrSize + D1nbOffsIndexArrSize + D0nbOffsIndexArrSize;
			const unsigned NB_OFFS_ARR_SIZE = 26U;
			
			// Essential clique array
			const OffsetIjkType h_ecOffsetArr[EC_ARR_SIZE] =
			{
				// Dim 3 does not have any ec
				// Dim 2
				makeOffsetIjk(1,0,0),
				// Dim 1
				makeOffsetIjk(1,0,0),      makeOffsetIjk(1,0,-1),     makeOffsetIjk(0,0,-1),
				// Dim 0
				makeOffsetIjk(0,-1,-1),	makeOffsetIjk(0,0,-1),	makeOffsetIjk(1,-1,-1),	makeOffsetIjk(1,0,-1),
				makeOffsetIjk(0,-1,0),	makeOffsetIjk(1,-1,0),	makeOffsetIjk(1,0,0)
			};

			// Core clique voxel array.
			//
			// For each dimension d, the associated entry is a list of tuples with variable
			// length. For this to work on the device side, there is a -1 inserted between
			// every two tuples, indicating the end of the previous one.
			const int8_t h_coreCliqueArr[CORE_CLQ_ARR_SIZE] = 
			{
				// Dim 3 does not have any core clique index
				// Dim 2
				0 + D2ecTexBegin, -1,
				// Dim 1
				1 + D1ecTexBegin, -1,
				0 + D1ecTexBegin, 2 + D1ecTexBegin, -1,
				// Dim 0
				2 + D0ecTexBegin, -1,
				0 + D0ecTexBegin, 3 + D0ecTexBegin, -1,
				0 + D0ecTexBegin, 5 + D0ecTexBegin, -1,
				3 + D0ecTexBegin, 5 + D0ecTexBegin, -1
			};

			// Indices of neighborhood offsets array
			const uint8_t h_nbOffsIndexArr[NB_OFFS_IDX_ARR_SIZE] = 
			{
				// Dim 3 nb offset indices
				0, 11, 3, 8, 20, 10, 1, 9, 2, 12, 24, 15, 21, 23, 13, 22, 14, 4, 19, 7, 16, 25, 18, 5, 17, 6,
				// Dim 2 nb offset indices
				8, 20, 10, 1, 9, 2, 21, 23, 13, 14, 16, 25, 18, 5, 17, 6,
				// Dim 1 nb offset indices
				8, 10, 1, 2, 21, 23, 13, 14
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

				OffsetCompType* d_matEntryArr;

				OffsetIjkType* d_ecOffsetArr;
				int8_t* d_coreCliqueArr;
				uint8_t* d_nbOffsIndexArr;
				// OffsetIjkType* d_nbOffsetArr;
				// uint8_t* d_nbFlatIjkToIndexLut;

			private:
				static std::unique_ptr<DevArrPtrs> m_instance;
			};

			std::unique_ptr<DevArrPtrs> DevArrPtrs::m_instance = nullptr;

			tp::Int8TexType matEntryTex;
			
			tp::OffsetIjkTexType ecOffsetTex;
			tp::Int8TexType coreCliqueTex;
			tp::Uint8TexType nbOffsIndexTex;

			// Initialize the device texture references of this module.
			void 
			_initDeviceTex(OffsetCompType** d_matEntryArr, OffsetIjkType** d_ecOffsetArr, int8_t** d_coreCliqueArr, uint8_t** d_nbOffsIndexArr)
			{
				const cudaChannelFormatDesc int8Desc = cudaCreateChannelDesc(8 * sizeof(OffsetCompType), 0, 0, 0, cudaChannelFormatKindSigned);
				const cudaChannelFormatDesc uint8Desc = cudaCreateChannelDesc(8 * sizeof(OffsetCompType), 0, 0, 0, cudaChannelFormatKindUnsigned);
				const cudaChannelFormatDesc char4Desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned);

				checkCudaErrors(cudaMalloc(d_matEntryArr, sizeof(OffsetCompType) * matEntryArrSize));
				checkCudaErrors(cudaMemcpy(*d_matEntryArr, h_matEntryArr, sizeof(OffsetCompType) * matEntryArrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, matEntryTex, *d_matEntryArr, int8Desc, sizeof(OffsetCompType) * matEntryArrSize));
				
				checkCudaErrors(cudaMalloc(d_ecOffsetArr, sizeof(OffsetIjkType) * EC_ARR_SIZE));
				checkCudaErrors(cudaMemcpy(*d_ecOffsetArr, h_ecOffsetArr, sizeof(OffsetIjkType) * EC_ARR_SIZE, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, ecOffsetTex, *d_ecOffsetArr, char4Desc, sizeof(OffsetIjkType) * EC_ARR_SIZE));
				
				checkCudaErrors(cudaMalloc(d_coreCliqueArr, sizeof(int8_t) * CORE_CLQ_ARR_SIZE));
				checkCudaErrors(cudaMemcpy(*d_coreCliqueArr, h_coreCliqueArr, sizeof(int8_t) * CORE_CLQ_ARR_SIZE, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, coreCliqueTex, *d_coreCliqueArr, int8Desc, sizeof(int8_t) * CORE_CLQ_ARR_SIZE));

				checkCudaErrors(cudaMalloc(d_nbOffsIndexArr, sizeof(uint8_t) * NB_OFFS_IDX_ARR_SIZE));
				checkCudaErrors(cudaMemcpy(*d_nbOffsIndexArr, h_nbOffsIndexArr, sizeof(uint8_t) * NB_OFFS_IDX_ARR_SIZE, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaBindTexture(0, nbOffsIndexTex, *d_nbOffsIndexArr, uint8Desc, sizeof(uint8_t) * NB_OFFS_IDX_ARR_SIZE));
			}

			// Unbinds the GPU texture references and frees the device memory.
			void 
			_clearDeviceTex(OffsetCompType* d_matEntryArr, OffsetIjkType* d_ecOffsetArr, int8_t* d_coreCliqueArr, uint8_t* d_nbOffsIndexArr)
			{
				checkCudaErrors(cudaFree(d_matEntryArr));
				checkCudaErrors(cudaFree(d_ecOffsetArr));
				checkCudaErrors(cudaFree(d_coreCliqueArr));
				checkCudaErrors(cudaFree(d_nbOffsIndexArr));
			}
            
			__device__ OffsetIjkType _fetchEcOffset(uint8_t ecOffsetIdx)
			{
				return tex1Dfetch(ecOffsetTex, ecOffsetIdx);
			}

			__device__ int8_t _fetchCoreClique(uint8_t coreCliqueIdx)
			{
				return tex1Dfetch(coreCliqueTex, coreCliqueIdx);
			}

			__device__ uint8_t _fetchNbOffsIndex(uint8_t nbOffsIndexIter)
			{
				return tex1Dfetch(nbOffsIndexTex, nbOffsIndexIter);
			}

			__device__ OffsetCompType _fetchMatEntry(uint8_t matEntryIdx)
			{
				return tex1Dfetch(matEntryTex, matEntryIdx);
			}
			// Compute the linear transformation on @offs using the matrix whose whose
			// entries in @matEntryTex starts from @matTexBegin. The computed result is
			// stored in @result.
			__device__ void _transform(uint8_t matTexBegin, const OffsetIjkType& offs, OffsetIjkType& result)
			{
				result.x = _fetchMatEntry(matTexBegin + 0) * offs.x + _fetchMatEntry(matTexBegin + 1) * offs.y + _fetchMatEntry(matTexBegin + 2) * offs.z;
				result.y = _fetchMatEntry(matTexBegin + 3) * offs.x + _fetchMatEntry(matTexBegin + 4) * offs.y + _fetchMatEntry(matTexBegin + 5) * offs.z;
				result.z = _fetchMatEntry(matTexBegin + 6) * offs.x + _fetchMatEntry(matTexBegin + 7) * offs.y + _fetchMatEntry(matTexBegin + 8) * offs.z;
			}

			// Dim3CliquePolicy
			__device__ uint8_t Dim3CliquePolicy::numFaceTokens()
			{
				return 1U;
			}

			__host__ __device__ uint8_t Dim3CliquePolicy::numEcVoxels()
			{
				return 1U;
			}

			__device__ uint8_t Dim3CliquePolicy::ecOffsetArrBegin()
			{
				return D3ecTexBegin;
			}
			__device__ uint8_t Dim3CliquePolicy::ecOffsetArrEnd()
			{
				return D3ecTexBegin + D3ecArrSize;
			}

			__device__ uint8_t Dim3CliquePolicy::coreCliqueArrBegin()
			{
				return D3coreTexBegin;
			}
			__device__ uint8_t Dim3CliquePolicy::coreCliqueArrEnd()
			{
				return D3coreTexBegin + D3coreArrSize;
			}

			__device__ uint8_t Dim3CliquePolicy::nbOffsIndexArrBegin()
			{
				return D3nbOffsIndexTexBegin;
			}
			__device__ uint8_t Dim3CliquePolicy::nbOffsIndexArrEnd()
			{
				return D3nbOffsIndexTexBegin + D3nbOffsIndexArrSize;
			}

			__device__ uint8_t Dim3CliquePolicy::matBeginByFaceToken(FaceTokenType faceToken)
			{
				return faceToken == 0 ? 0 : 0xff;
			}

			// Dim2CliquePolicy
			__device__ uint8_t Dim2CliquePolicy::numFaceTokens()
			{
				// return 3U;
				return 4U;
			}

			__host__ __device__ uint8_t Dim2CliquePolicy::numEcVoxels()
			{
				return 2U;
			}
		
			__device__ uint8_t Dim2CliquePolicy::ecOffsetArrBegin()
			{
				return D2ecTexBegin;
			}
			__device__ uint8_t Dim2CliquePolicy::ecOffsetArrEnd()
			{
				return D2ecTexBegin + D2ecArrSize;
			}

			__device__ uint8_t Dim2CliquePolicy::coreCliqueArrBegin()
			{
				return D2coreTexBegin;
			}
			__device__ uint8_t Dim2CliquePolicy::coreCliqueArrEnd()
			{
				return D2coreTexBegin + D2coreArrSize;
			}

			__device__ uint8_t Dim2CliquePolicy::nbOffsIndexArrBegin()
			{
				return D2nbOffsIndexTexBegin;
			}
			__device__ uint8_t Dim2CliquePolicy::nbOffsIndexArrEnd()
			{
				return D2nbOffsIndexTexBegin + D2nbOffsIndexArrSize;
			}

			__device__ uint8_t Dim2CliquePolicy::matBeginByFaceToken(FaceTokenType faceToken)
			{
				switch (faceToken)
				{
				case D2_FACE_X:
					// default case is Face_X
					return 0;
				case D2_FACE_Y:
					// rotate +90 degrees around Z axis
					return 9U;
				case D2_FACE_Z:
					// rotate -90 degrees around Y axis
					return 18U;
				case D2_FACE_Z_NEG:
					// rotate +90 degrees around Y axis
					return 63U;
				default:
					return 0xff;
				}
			}

			// Dim1CliquePolicy
			
			__device__ uint8_t Dim1CliquePolicy::numFaceTokens()
			{
				return 6U;
			}

			__host__ __device__ uint8_t Dim1CliquePolicy::numEcVoxels()
			{
				return 4U;
			}

			__device__ uint8_t Dim1CliquePolicy::ecOffsetArrBegin()
			{
				return D1ecTexBegin;
			}
			__device__ uint8_t Dim1CliquePolicy::ecOffsetArrEnd()
			{
				return D1ecTexBegin + D1ecArrSize;
			}

			__device__ uint8_t Dim1CliquePolicy::coreCliqueArrBegin()
			{
				return D1coreTexBegin;
			}
			__device__ uint8_t Dim1CliquePolicy::coreCliqueArrEnd()
			{
				return D1coreTexBegin + D1coreArrSize;
			}

			__device__ uint8_t Dim1CliquePolicy::nbOffsIndexArrBegin()
			{
				return D1nbOffsIndexTexBegin;
			}
			__device__ uint8_t Dim1CliquePolicy::nbOffsIndexArrEnd()
			{
				return D1nbOffsIndexTexBegin + D1nbOffsIndexArrSize;
			}

			__device__ uint8_t Dim1CliquePolicy::matBeginByFaceToken(FaceTokenType faceToken)
			{
				switch (faceToken)
				{
				case D1_EDGE_1:
					// default case is edge 1
					return 0;
				case D1_EDGE_2:
					// rotate +90 degrees around Z axis
					return 9U;
				case D1_EDGE_5:
					// roate -90 degrees around X axis
					return 27U;
				case D1_EDGE_6:
					// rotate +90 degrees around X axis
					return 36U;
				case D1_EDGE_9:
					// rotate 180 degrees around X axis
					return 45U;
				case D1_EDGE_10:
					// rotate +90 degrees around Z axis first to obtain X'Y'Z',
					// then rotate 180 degrees around X' axis
					return 54U;
				default:
					return 0xff;
				}
				
			}
			
			// Dim0CliquePolicy
			__device__ uint8_t Dim0CliquePolicy::numFaceTokens()
			{
				return 4U;
			}

			__host__ __device__ uint8_t Dim0CliquePolicy::numEcVoxels()
			{
				return 8U;
			}

			__device__ uint8_t Dim0CliquePolicy::ecOffsetArrBegin()
			{
				return D0ecTexBegin;
			}
			__device__ uint8_t Dim0CliquePolicy::ecOffsetArrEnd()
			{
				return D0ecTexBegin + D0ecArrSize;
			}

			__device__ uint8_t Dim0CliquePolicy::coreCliqueArrBegin()
			{
				return D0coreTexBegin;
			}
			__device__ uint8_t Dim0CliquePolicy::coreCliqueArrEnd()
			{
				return D0coreTexBegin + D0coreArrSize;
			}

			__device__ uint8_t Dim0CliquePolicy::nbOffsIndexArrBegin()
			{
				return D0nbOffsIndexTexBegin;
			}
			__device__ uint8_t Dim0CliquePolicy::nbOffsIndexArrEnd()
			{
				return D0nbOffsIndexTexBegin + D0nbOffsIndexArrSize;
			}

			__device__ uint8_t Dim0CliquePolicy::matBeginByFaceToken(FaceTokenType faceToken)
			{
				switch (faceToken)
				{
				case D0_VERTEX_1:
					// default case is vertex 1
					return 0;
				case D0_VERTEX_2:
					// rotate +90 degrees around X axis
					return 36U;
				case D0_VERTEX_5:
					// rotate -90 degrees around X axis
					return 27U;
				case D0_VERTEX_6:
					// rotate 180 degrees around X axis
					return 45U;
				default:
					return 0xff;
				}
			}

			// Binary search of the 3D discrete coordinate, @targetIjk, in @compactIjkArr.
			// This is necessary due to few libraries on the device side.
			//
			// [precondition] @comapctIjkArr is sorted in ascending order.
			__device__ ArrIndexType 
			_binSearch(const IjkType* compactIjkArr, const IjkType& targetIjk, ArrIndexType lo, ArrIndexType hi)
			{
				ArrIndexType mid;
				while (lo < hi)
				{
					mid = lo + ((hi - lo) >> 1);
					if (isEqual(compactIjkArr[mid], targetIjk))
					{
						return mid;
					}
					else if (less(compactIjkArr[mid], targetIjk))
					{
						lo = mid + 1;
					}
					else
					{
						hi = mid;
					}
				}
				return INVALID_UINT;
			}

			// Find the 3D discrete coordinate, @targetIjk, in @compactIjkArr, if the
			// reference 3D coordinate, @refIjk, and its index in @compactIjkArr, @refIndex,
			// are known.
			//
			// [precondition] @comapctIjkArr is sorted.
			// [precondition] @compactIjkArr[@refIndex] == @refIjk
			__device__ ArrIndexType
			_findIndexOfIjk(const IjkType* compactIjkArr, const unsigned arrSize, const IjkType& targetIjk,
							  const IjkType& refIjk, const ArrIndexType refIndex)
			{
				if (isEqual(targetIjk, refIjk))
				{
					return refIndex;
				}
				else if (less(targetIjk, refIjk))
				{
					return _binSearch(compactIjkArr, targetIjk, 0, refIndex);
				}
				else
				{
					return _binSearch(compactIjkArr, targetIjk, refIndex + 1, arrSize);
				}
			}
			/*
			// Find the targetIjk 3D discrete coordinate in compactIjkArr, if the reference 
			// 3D coord refIjk and its index in compactIjkArr, refIndex, are Unknown. The 
			// function will have to search for the entire array.
			// [precondition]: comapctIjkArr is sorted
			__device__ ArrIndexType
			_findIndexOfIjk(const IjkType* compactIjkArr, const unsigned arrSize, const IjkType targetIjk)
			{
				return _binSearch(compactIjkArr, targetIjk, 0, arrSize);
			}
			*/
			// Find if the target 3D discrete coordinate, @refIjk + @offs, exists in
			// @compactIjkArr.
			//
			// [precondition] comapctIjkArr is sorted
			// [precondition] @compactIjkArr[@refIndex] == @refIjk
			// [postcondition] If returns is true, then @compactIjkArr[@foundIndex] ==
			// @refIjk + @offs.
			__device__ bool 
			_find(ArrIndexType& foundIndex, const IjkType* compactIjkArr, const unsigned arrSize, const IjkType& refIjk, 
					const ArrIndexType refIndex, const OffsetIjkType& offs, const IjkType& size3D)
			{
                IjkType targetIjk;
                
				if (!tp::_isInBoundary(refIjk, offs, size3D, targetIjk))
				{
					return false;
				}

				foundIndex = _findIndexOfIjk(compactIjkArr, arrSize, targetIjk, refIjk, refIndex);
                return foundIndex != INVALID_UINT;
            }
			/*
			__device__ bool 
			_find(ArrIndexType& foundIndex, const IjkType* compactIjkArr, const unsigned arrSize, const IjkType& refIjk,
				const ArrIndexType refIndex, const IjkType& size3D)
            {
                return _find(foundIndex, compactIjkArr, arrSize, refIjk, refIndex, makeOffsetIjk(0, 0, 0), size3D);
            }
			*/

			// This function adds an additional feature on top of the _find() function. Even
			// if @refIjk + @offs is indeed found in @compactIjkArr, we need to check the
			// @nthBit of the corresponding entry in @recBitsArr, and only when that bit is
			// set does the function return true.
			__device__ bool 
			_findInRecBitsArr(ArrIndexType& foundIndex, const IjkType* compactIjkArr, const RecBitsType* recBitsArr,
				const uint8_t nthBit, const unsigned arrSize, const IjkType& refIjk, const ArrIndexType refIndex,
				const OffsetIjkType& offs, const IjkType& size3D)
            {
				if (!_find(foundIndex, compactIjkArr, arrSize, refIjk, refIndex, offs, size3D))
				{
					return false;
				}
                
                return tp::_readBit(recBitsArr[foundIndex], nthBit) == 1;
            }

			__device__ bool 
			_findInX(ArrIndexType& foundIndex, const IjkType* compactIjkArr, const RecBitsType* recBitsArr,
					const unsigned arrSize, const IjkType& refIjk, const ArrIndexType refIndex, 
					const OffsetIjkType& offs, const IjkType& size3D)
			{
				using namespace details;
				return _findInRecBitsArr(foundIndex, compactIjkArr, recBitsArr, REC_BIT_X, arrSize,
										refIjk, refIndex, offs, size3D);
			}
			
			__device__ bool
			// _findInX(ArrIndexType& foundIndex, const ThinningData& thinData, const IjkType& refIjk, 
			_findInX(ArrIndexType& foundIndex, const details::DevDataPack& thinData, const IjkType& refIjk, 
					const ArrIndexType refIndex, const OffsetIjkType& offs)
			{
				bool found = _findInX(foundIndex, thinData.compactIjkArr, thinData.recBitsArr, thinData.arrSize,
									refIjk, refIndex, offs, thinData.size3D());

				if (found && thinData.useVoxelID())
				{
					found &= (thinData.voxelIdArr[foundIndex] == thinData.voxelIdArr[refIndex]);
				}

				return found;
			}

			__device__ bool 
			_containsInX(const IjkType* compactIjkArr, const RecBitsType* recBitsArr,
					const unsigned arrSize, const IjkType& refIjk, const ArrIndexType refIndex, 
					const OffsetIjkType& offs, const IjkType& size3D)
            {
                ArrIndexType foundIndex;
                return _findInX(foundIndex, compactIjkArr, recBitsArr, arrSize, refIjk, refIndex, offs, size3D);
            }

			__device__ bool 
			// _containsInX(const ThinningData& thinData, const IjkType& refIjk, const ArrIndexType refIndex, const OffsetIjkType& offs)
			_containsInX(const details::DevDataPack& thinData, const IjkType& refIjk, const ArrIndexType refIndex, const OffsetIjkType& offs)
            {
                ArrIndexType foundIndex;
                return _findInX(foundIndex, thinData, refIjk, refIndex, offs);
            }

			__device__ bool _isInNbBoundary(const OffsetIjkType& ijk, const OffsetIjkType& offsIjk, OffsetIjkType& resultIjk)
			{
				auto checker = [](OffsetCompType coord, OffsetCompType offs, OffsetCompType& result)
				{
					result = coord + offs;
					bool flag = (-1 <= result) && (result <= 1);
					result = flag * result + (1 - flag) * 0xff;
					return flag;
				};

				return checker(ijk.x, offsIjk.x, resultIjk.x) && 
						checker(ijk.y, offsIjk.y, resultIjk.y) &&
						checker(ijk.z, offsIjk.z, resultIjk.z);
			}

			__device__ nb::NbMaskType _genNbMaskFromCliqueNbMask(nb::NbMaskType cliqueNbMask, uint8_t nthNb)
			{
				nb::NbMaskType nbMask = 0;
				
				OffsetIjkType curIjk = nb::fetchNbOffset(nthNb);

				for (uint8_t nbOffsetIdx = 0; nbOffsetIdx < NB_OFFS_ARR_SIZE; ++nbOffsetIdx)
				{
					OffsetIjkType offsIjk = nb::fetchNbOffset(nbOffsetIdx);
			
					OffsetIjkType targetIjk;
					
					if (_isInNbBoundary(curIjk, offsIjk, targetIjk))
					{
						uint8_t targetNthBit = nb::fetchIndexOfNbOffset(targetIjk);

						if (tp::_readBit(cliqueNbMask, targetNthBit))
						{
							tp::_setBit(nbMask, nbOffsetIdx);
						}
					}
				}
				return nbMask;
			}

			__global__ void _assignKern(RecBitsType* recBitsArr, const unsigned arrSize, const uint8_t SRC, const uint8_t DST)
			{
				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if (index >= arrSize) return;

				if (tp::_readBit(recBitsArr[index], SRC))
				{
					tp::_setBit(recBitsArr[index], DST);
				}
				else
				{
					tp::_clearBit(recBitsArr[index], DST);
				}
			}

			__global__ void _unionKern(RecBitsType* srcRecBitsArr, RecBitsType* dstRecBitsArr, const unsigned arrSize,
						const uint8_t SRC, const uint8_t DST)
			{
				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if (index >= arrSize) return;

				if (tp::_readBit(srcRecBitsArr[index], SRC))
				{
					tp::_setBit(dstRecBitsArr[index], DST);
				}
			}

			__global__ void _clearKern(RecBitsType* recBitsArr, const unsigned arrSize, const uint8_t BIT)
			{
				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if (index >= arrSize) return;

				tp::_clearBit(recBitsArr[index], BIT);
			}

			class _BitPred
			{
			public:
				__host__ __device__ _BitPred(uint8_t bit) : m_bit(bit) { }
				__host__ __device__ bool operator()(const RecBitsType bits) const
				{
					return tp::_readBit(bits, m_bit) == 1;
				}

			private:
				uint8_t m_bit;
			};

			unsigned _countBit(RecBitsType* d_recBitsArr, const unsigned arrSize, const uint8_t BIT)
			{
				_BitPred pred(BIT);
				return thrust::count_if(thrust::device, d_recBitsArr, d_recBitsArr + arrSize, pred);
			}

			__global__ void _flagActiveKern(ArrIndexType* flagArr, const RecBitsType* recBitsArr, const unsigned arrSize)
			{
				using namespace details;

				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;
				if (index >= arrSize) return;

				flagArr[index] = tp::_readBit(recBitsArr[index], REC_BIT_X) 
								|| tp::_readBit(recBitsArr[index], REC_BIT_K);
			}

			template <typename T>
			__global__ void
			_compactArrsKern(T* dstArr, const T* srcArr, const ArrIndexType* flagArr, const ArrIndexType* flagScanArr, const unsigned arrSize)
			{
				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if ((index >= arrSize) || (flagArr[index] == 0)) return;
				
				ArrIndexType newIndex = flagScanArr[index];
				dstArr[newIndex] = srcArr[index];
			}

			__global__ void _updateBirthKern(unsigned* birthArr, const RecBitsType* recBitsArr, const unsigned arrSize, const unsigned iter)
			{
				using namespace details;

				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if (index >= arrSize) return;         

				if (tp::_readBit(recBitsArr[index], REC_BIT_Z) && (birthArr[index] == 0))
				{
					birthArr[index] = iter;
				}
			}

			__global__ void _unionKsetByBirth(RecBitsType* recBitsArr, const unsigned* birthArr, const unsigned arrSize, const unsigned iter, const unsigned p)
			{
				using namespace details;

				ArrIndexType index = blockIdx.y * gridDim.x + blockIdx.x;
				index = index * blockDim.x + threadIdx.x;

				if (index >= arrSize) return;

				if (tp::_readBit(recBitsArr[index], REC_BIT_Y) && birthArr[index] && (iter + 1U - birthArr[index] >= p))
				{
					tp::_setBit(recBitsArr[index], REC_BIT_K);
				}
			}

			unsigned _flagVoxelsInXorK(ArrIndexType* d_flagArr, ArrIndexType* d_flagScanArr, const RecBitsType* d_recBitsArr, 
										const unsigned arrSize, const dim3& blocksDim, const dim3& threadsDim)
			{
				// Find out the active voxels in X or K after one iteration of thinning
				_flagActiveKern<<<blocksDim, threadsDim>>>(d_flagArr, d_recBitsArr, arrSize);
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());

				// Scan the flags array
				thrust::exclusive_scan(thrust::device, d_flagArr, d_flagArr + arrSize, d_flagScanArr);

				unsigned lastFlagArrElem, lastFlagScanArrElem;
				checkCudaErrors(cudaMemcpy(&lastFlagArrElem, d_flagArr + arrSize - 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(&lastFlagScanArrElem, d_flagScanArr + arrSize - 1, sizeof(unsigned), cudaMemcpyDeviceToHost));
				// New array size will be flagArr[-1] + flagScanArr[-1], since flagScanArr is an exclusive scan.
				unsigned newArrSize = lastFlagArrElem + lastFlagScanArrElem;

				return newArrSize;
			}

			// unsigned _shrinkArrs(ThinningData& thinData, const dim3& blocksDim, const dim3& threadsDim)
			unsigned _shrinkArrs(details::DevDataPack& thinData, const dim3& blocksDim, const dim3& threadsDim)
			{
				ArrIndexType* d_flagArr;
				checkCudaErrors(cudaMalloc(&d_flagArr, sizeof(ArrIndexType) * thinData.arrSize));
				checkCudaErrors(cudaMemset(d_flagArr, 0, sizeof(ArrIndexType) * thinData.arrSize));
				
				ArrIndexType* d_flagScanArr;
				checkCudaErrors(cudaMalloc(&d_flagScanArr, sizeof(ArrIndexType) * thinData.arrSize));
				checkCudaErrors(cudaMemset(d_flagScanArr, 0, sizeof(ArrIndexType) * thinData.arrSize));

				unsigned newArrSize = _flagVoxelsInXorK(d_flagArr, d_flagScanArr, thinData.recBitsArr, thinData.arrSize, blocksDim, threadsDim);
				
				// Create two new arrays to stgore the active voxels information by 
				// performing a scatter operation on the original two arrays.
				IjkType* d_dstIjkArr;
				checkCudaErrors(cudaMalloc(&d_dstIjkArr, sizeof(IjkType) * newArrSize));

				_compactArrsKern<<<blocksDim, threadsDim>>>(d_dstIjkArr, thinData.compactIjkArr, d_flagArr, d_flagScanArr, thinData.arrSize);
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				// Free the unused device memory. Notice that the ORIGINAL voxel arrays
				// are being freed!
				checkCudaErrors(cudaFree(thinData.compactIjkArr));
				// Store the address of the device memory
				thinData.compactIjkArr = d_dstIjkArr;

				RecBitsType* d_dstRecBitsArr;
				checkCudaErrors(cudaMalloc(&d_dstRecBitsArr, sizeof(RecBitsType) * newArrSize));
				
				_compactArrsKern<<<blocksDim, threadsDim>>>(d_dstRecBitsArr, thinData.recBitsArr, d_flagArr, d_flagScanArr, thinData.arrSize);
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(thinData.recBitsArr));
				thinData.recBitsArr = d_dstRecBitsArr;

				if (thinData.birthArr)
				{
					unsigned* d_dstBirthArr;
					checkCudaErrors(cudaMalloc(&d_dstBirthArr, sizeof(unsigned) * newArrSize));

					_compactArrsKern<<<blocksDim, threadsDim>>>(d_dstBirthArr, thinData.birthArr, d_flagArr, d_flagScanArr, thinData.arrSize);
					cudaDeviceSynchronize();
					checkCudaErrors(cudaGetLastError());
					checkCudaErrors(cudaFree(thinData.birthArr));

					thinData.birthArr = d_dstBirthArr;
				}

				if (thinData.useVoxelID())
				{
					ObjIdType* d_dstVoxelIdArr;
					checkCudaErrors(cudaMalloc(&d_dstVoxelIdArr, sizeof(ObjIdType) * newArrSize));
					
					_compactArrsKern<<<blocksDim, threadsDim>>>(d_dstVoxelIdArr, thinData.voxelIdArr, d_flagArr, d_flagScanArr, thinData.arrSize);
					cudaDeviceSynchronize();
					checkCudaErrors(cudaGetLastError());
					checkCudaErrors(cudaFree(thinData.voxelIdArr));

					thinData.voxelIdArr = d_dstVoxelIdArr;
				}
				
				checkCudaErrors(cudaFree(d_flagArr));
				checkCudaErrors(cudaFree(d_flagScanArr));
				
				return newArrSize;
			}
		}; // namespace thin::clique::_private;

		void initDevice()
		{
			cp::DevArrPtrs* ptrs = cp::DevArrPtrs::instance();
			cp::_initDeviceTex(&(ptrs->d_matEntryArr), &(ptrs->d_ecOffsetArr), &(ptrs->d_coreCliqueArr), &(ptrs->d_nbOffsIndexArr));
		}

		void shutdownDevice()
		{
			cp::DevArrPtrs* ptrs = cp::DevArrPtrs::instance();
			cp::_clearDeviceTex(ptrs->d_matEntryArr, ptrs->d_ecOffsetArr, ptrs->d_coreCliqueArr, ptrs->d_nbOffsIndexArr);
		}

		// void crucialIsthmus(ThinningData& thinData, const dim3& blocksDim, const dim3& threadsDim)
		void crucialIsthmus(details::DevDataPack& thinData, const dim3& blocksDim, const dim3& threadsDim)
		{
			using namespace details;

			cp::_assignKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_K, REC_BIT_Y);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			cp::_clearKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Z);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			// clear A and B set
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			// Find 3-cliques that are crucial for <X, K>
			dimCrucialIsthmus<D3CliqueChecker>(thinData, blocksDim, threadsDim);
			// clear A and B set
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			// Find 2-cliques that are crucial for <X, K>
			dimCrucialIsthmus<D2CliqueChecker>(thinData, blocksDim, threadsDim);
			// clear A and B set
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			// Find 1-cliques that are crucial for <X, K>
			dimCrucialIsthmus<D1CliqueChecker>(thinData, blocksDim, threadsDim);
			// clear A and B set
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			// Find 0-cliques that are crucial for <X, K>
			dimCrucialIsthmus<D0CliqueChecker>(thinData, blocksDim, threadsDim);
			// clear A and B set
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
		}
	}; // namespace thin::clique;
}; // namespace thin;