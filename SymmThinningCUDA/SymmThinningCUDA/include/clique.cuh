#ifndef CLIQUE_CUH
#define CLIQUE_CUH

// This file defines the CliqueChecker<T> class. It also provides the 
// utilities to detect all the d-cliques that are critical for X.
//
// Users do not need to touch this file.

#include <iostream>
#include <vector>
#include <algorithm>					// std::sort

#include <thrust/execution_policy.h>	// thrust::device
#include <thrust/scan.h>				// thrust::exclusive_scan
#include <thrust/count.h>				// thrust::count_if
#include <thrust/reduce.h>				// thrust::reduce

#include "cuda_includes.h"
#include "thinning_base.cuh"
#include "thinning_details.cuh"
#include "attachment.cuh"
#include "neighbor.cuh"

namespace thin
{
	namespace clique
	{
		namespace tp = thin::_private;
		
		// For each d dimension clique, there is a corresponding d-dim policy, which is
		// used to setup the parameters of the generic clique checker or provide
		// different concrete implementation through a unified iternface.
		//
		// To detect voxels form either an Essential Clique (ec) or Neighbor (nb) of a
		// clique of different dimensions, the paper introduced the spatial masks, which
		// are a list of neighborhood offsets. By checking if the complex X contains
		// voxelIjk + ecOffset (nbOffset), we can determine which voxels in X belongs to
		// the ec (nb) of voxelIjk.
		//
		// It is OK if not all the voxels are covered by the ec mask. However, among the
		// voxels being masked, some are quite important, without which the clique
		// cannot even be formed. Such voxels are the core ones for a clique. For
		// example, to detect if a clique exists at edge 1 for a voxel, either one voxel
		// which is (1,0,-1) offset from the voxel, or two voxels which are (1,0,0) and
		// (0,0,-1) offset from the voxel, respectively, should exist. Once the order of
		// the ec offsets are determined, we can use a list of indices, where each index
		// points to an offset in ec offsets, to check whether all the core voxels
		// exist. Notice that the number of such list is not necessarily to be 1.
		//
		// Each voxel checks several d-face, d in {0,1,2,3}. For the same d dimension,
		// we can get the offset for different d-faces by a linear transformation. The
		// d-face token is used to specify which linear transformation matrix to use.
		typedef uint8_t FaceTokenType;
		// ------ constants definition ------
		// Face token constants
		// For dim 3, there is only one 3-face
		const FaceTokenType D3_VOXEL = 0;
		// For dim 2, there are 6 2-faces. Checking 3 of them is enough to cover all the 2-cliques for each voxel.
		// For chunk thinning, we need to check 4 2-faces!
		const FaceTokenType D2_FACE_X = 0, D2_FACE_Y = 1U, D2_FACE_Z = 2U, D2_FACE_Z_NEG = 3U;
		// For dim 1, there are 12 1-faces. Checking 6 of them is enough to cover all the 1-cliques for each voxel.
		const FaceTokenType D1_EDGE_1 = 0, D1_EDGE_2 = 1U, D1_EDGE_5 = 2U, D1_EDGE_6 = 3U, D1_EDGE_9 = 4U, D1_EDGE_10 = 5U;
		// For dim 0, there are 8 2-faces. Checking 4 of them is enough to cover all the 0-cliques for each voxel.
		const FaceTokenType D0_VERTEX_1 = 0, D0_VERTEX_2 = 1U, D0_VERTEX_5 = 2U, D0_VERTEX_6 = 3U;

		// Clique Critical Type constant
		const uint8_t CLQ_REGULAR = 0, CLQ_CRITICAL = 1U, CLQ_1_CRITICAL = 2U;

		namespace _private
		{
			// Policy for 3-clique checker.
			class Dim3CliquePolicy
			{
			public:
				// Return the number of face tokens for dim 3.
				__device__ static uint8_t numFaceTokens();

				__host__ __device__ static uint8_t numEcVoxels();

				// The beginning index of the Essential Clique spatial mask for dim 3.
				__device__ static uint8_t ecOffsetArrBegin();
				// The end index of the Essential Clique spatial mask for dim 3.
				__device__ static uint8_t ecOffsetArrEnd();

				// The beginning index of the Core Clique Tuple for dim 3.
				__device__ static uint8_t coreCliqueArrBegin();
				// The end index of the Core Clique Tuple for dim 3.
				__device__ static uint8_t coreCliqueArrEnd();

				// The beginning index of the Neighbor Voxel spatial mask for a 3-clique.
				__device__ static uint8_t nbOffsIndexArrBegin();
				// The end index of the Neighbor Voxel spatial mask for a 3-clique.
				__device__ static uint8_t nbOffsIndexArrEnd();

				// Return the beginning index of the transformation matrix according to the
				// given face token.
				__device__ static uint8_t matBeginByFaceToken(FaceTokenType);
			};
    
			// Policy for 2-clique checker.
			class Dim2CliquePolicy
			{
			public:
				// Return the number of face tokens for dim 2.
				__device__ static uint8_t numFaceTokens();

				__host__ __device__ static uint8_t numEcVoxels();

				// The beginning index of the Essential Clique spatial mask for dim 2.
				__device__ static uint8_t ecOffsetArrBegin();
				// The end index of the Essential Clique spatial mask for dim 2.
				__device__ static uint8_t ecOffsetArrEnd();

				// The beginning index of the Core Clique Tuple for dim 2.
				__device__ static uint8_t coreCliqueArrBegin();
				// The end index of the Core Clique Tuple for dim 2.
				__device__ static uint8_t coreCliqueArrEnd();

				// The beginning index of the Neighbor Voxel spatial mask for a 2-clique.
				__device__ static uint8_t nbOffsIndexArrBegin();
				// The end index of the Neighbor Voxel spatial mask for a 2-clique.
				__device__ static uint8_t nbOffsIndexArrEnd();
				
				// Return the beginning index of the transformation matrix according to the
				// given face token.
				__device__ static uint8_t matBeginByFaceToken(FaceTokenType);
			};
    
			// Policy for 1-clique checker.
			class Dim1CliquePolicy
			{
			public:
				// Return the number of face tokens for dim 1.
				__device__ static uint8_t numFaceTokens();

				__host__ __device__ static uint8_t numEcVoxels();

				// The beginning index of the Essential Clique spatial mask for dim 1.
				__device__ static uint8_t ecOffsetArrBegin();
				// The end index of the Essential Clique spatial mask for dim 1.
				__device__ static uint8_t ecOffsetArrEnd();

				// The beginning index of the Core Clique Tuple for dim 1.
				__device__ static uint8_t coreCliqueArrBegin();
				// The end index of the Core Clique Tuple for dim 1.
				__device__ static uint8_t coreCliqueArrEnd();

				// The beginning index of the Neighbor Voxel spatial mask for a 1-clique.
				__device__ static uint8_t nbOffsIndexArrBegin();
				// The end index of the Neighbor Voxel spatial mask for a 1-clique.
				__device__ static uint8_t nbOffsIndexArrEnd();
				
				// Return the beginning index of the transformation matrix according to the
				// given face token.
				__device__ static uint8_t matBeginByFaceToken(FaceTokenType);
			};
    
			// Policy for 0-clique checker.
			class Dim0CliquePolicy
			{
			public:
				// Return the number of face tokens for dim 0.
				__device__ static uint8_t numFaceTokens();

				__host__ __device__ static uint8_t numEcVoxels();

				// The beginning index of the Essential Clique spatial mask for dim 0.
				__device__ static uint8_t ecOffsetArrBegin();
				// The end index of the Essential Clique spatial mask for dim 0.
				__device__ static uint8_t ecOffsetArrEnd();

				// The beginning index of the Core Clique Tuple for dim 0.
				__device__ static uint8_t coreCliqueArrBegin();
				// The end index of the Core Clique Tuple for dim 0.
				__device__ static uint8_t coreCliqueArrEnd();

				// The beginning index of the Neighbor Voxel spatial mask for a 0-clique.
				__device__ static uint8_t nbOffsIndexArrBegin();
				// The end index of the Neighbor Voxel spatial mask for a 0-clique.
				__device__ static uint8_t nbOffsIndexArrEnd();
				
				// Return the beginning index of the transformation matrix according to the
				// given face token.
				__device__ static uint8_t matBeginByFaceToken(FaceTokenType);
			};

			// Feth an Essential Clique offset by the given @ecOffsetIdx.
			__device__ OffsetIjkType _fetchEcOffset(uint8_t ecOffsetIdx);

			// Fetch an index of essential clique offset by the given @coreCliqueIdx.
			__device__ int8_t _fetchCoreClique(uint8_t coreCliqueIdx);

			// Fetch the index of the Neighborhood Offset by the given @nbOffsIndexIter.
			__device__ uint8_t _fetchNbOffsIndex(uint8_t nbOffsIndexIter);

			// Apply the linear transformation on an offset using the matrix speficied by
			// @matTexBegin.
			//
			// [param] @result: stores the result of the transformation on @offs.
			__device__ void _transform(uint8_t matTexBegin, const OffsetIjkType& offs, OffsetIjkType& result);

			// Find if @refIjk + @offs is in @thinData.
			//
			// [precondition] @thinData is sorted in ascending order.
			// [postcondition] If the result is true, @foundIndex will store the index of
			// that voxel.
			// [explain] Internally, this is a binary search. The @refIndex is the index of
			// @refIjk in @thinData, which help reduces the searching size by half on
			// average.
			__device__ bool
			_findInX(ArrIndexType& foundIndex, const details::DevDataPack& thinData, const IjkType& refIjk, 
					const ArrIndexType refIndex, const OffsetIjkType& offs);

			// Check to see if @refIjk + @offs is in @thinData.
			//
			// [precondition] @thinData is sorted in ascending order.
			// [postcondition] If the result is true, @foundIndex will store the index of
			// that voxel.
			// [explain] Internally, this is a binary search. The @refIndex is the index of
			// @refIjk in @thinData, which help reduces the searching size by half on
			// average.
			__device__ bool
			_containsInX(const details::DevDataPack& thinData, const IjkType& refIjk, const ArrIndexType refIndex, const OffsetIjkType& offs);
			// Generate the neighbor mask for ONE voxel denoted by @nthNb, using the
			// information encoded in @cliqueNbMask. Since the neighbor voxels for any
			// d-clique (d = 0 - 3) is a subset of the 26 neighbor voxels of a single voxel,
			// it is enough to use just a 4-byte uint to encode the information of which of
			// the 26 voxels are the neighbor of a d-clique. Then again for each bit in
			// @cliqueNbMask that is 1, we can test whether it is simple or not just by
			// using the information in @cliqueNbMask.
			__device__ nb::NbMaskType _genNbMaskFromCliqueNbMask(nb::NbMaskType cliqueNbMask, uint8_t nthNb);

			// Kernel function to find all d-cliques that are criticle for X.
			//
			// [precondition] @thinData is sorted in ascending order.
			// [precondition] @thinData.procBeginIndex < @thinData.procEndIndex <=
			// @thinData.arrSize
			template <typename Checker>
			__global__ void 
			_findDCriticalCliqueKern(details::DevDataPack thinData)
			{
				using namespace details;

				extern __shared__ ArrIndexType sh_ecIndexArr[];

				ArrIndexType vxIndex = blockIdx.y * gridDim.x + blockIdx.x;
				vxIndex = vxIndex * blockDim.x + threadIdx.x;
				if ((vxIndex < thinData.procBeginIndex) || (vxIndex >= thinData.procEndIndex) || (vxIndex >= thinData.arrSize)) return;
				// if (vxIndex >= thinData.arrSize) return;

				IjkType vxIjk = thinData.compactIjkArr[vxIndex];
				ArrIndexType* ecIndexArrBegin = sh_ecIndexArr + threadIdx.x * Checker::numEcVoxels();

				for (FaceTokenType faceToken = 0; faceToken < Checker::numFaceTokens(); ++faceToken)
				{
					if (Checker::canFormClique(vxIjk, vxIndex, thinData, faceToken))
					{
						uint8_t numEcVoxel = Checker::findEssentialClique(vxIjk, vxIndex, thinData, faceToken, ecIndexArrBegin);
						// It is required that the set of ALL voxels in ecVoxelSet is the subset of X / K.
						// In another word, if there exists a voxel x in ecVoxelSet s.t. x is in K, then
						// this requirement is NOT satisfied.
						bool includedInYset = false;
						for (uint8_t ecIdx = 0; ecIdx < numEcVoxel; ++ecIdx)
						{
							if (tp::_readBit(thinData.recBitsArr[ ecIndexArrBegin[ecIdx] ], REC_BIT_Y))
							{
								includedInYset = true;
								break;
							}
						}

						if (!includedInYset)
						{
							// Requirement is satisfied, at this point we can check if the clique is critical.
							uint8_t criticalResult = Checker::checkCliqueCritical(vxIjk, vxIndex, thinData, faceToken);
							
							if (criticalResult != CLQ_REGULAR)
							{
								// if it's critical, union ecVoxelSet into allCcVoxelSet
								// for (unsigned ecFlatIndex : ecFlatIndexVec)
								for (uint8_t ecIdx = 0; ecIdx < numEcVoxel; ++ecIdx)
								{
									tp::_setBit(thinData.A_recBitsArr[ ecIndexArrBegin[ecIdx] ], REC_BIT_A);
								}
							}

							if (criticalResult == CLQ_1_CRITICAL)
							{
								for (uint8_t ecIdx = 0; ecIdx < numEcVoxel; ++ecIdx)
								{
									tp::_setBit(thinData.B_recBitsArr[ ecIndexArrBegin[ecIdx] ], REC_BIT_B);
								}
							}
						}
					}
				}
			}

			// Kernel function to assign the value at the SRC-th bit to the DST-th bit.
			__global__ void _assignKern(RecBitsType* recBitsArr, const unsigned arrSize, const uint8_t SRC, const uint8_t DST);

			__global__ void _unionKern(RecBitsType* srcRecBitsArr, RecBitsType* dstRecBitsArr, const unsigned arrSize,
						const uint8_t SRC, const uint8_t DST);

			__global__ void _clearKern(RecBitsType* recBitsArr, const unsigned arrSize, const uint8_t BIT);

			unsigned _countBit(RecBitsType* d_recBitsArr, const unsigned arrSize, const uint8_t BIT);

			__global__ void _updateBirthKern(unsigned* birthArr, const RecBitsType* recBitsArr, const unsigned arrSize, const unsigned iter);

			__global__ void _unionKsetByBirth(RecBitsType* recBitsArr, const unsigned* birthArr, const unsigned arrSize, const unsigned iter, const unsigned p);

			unsigned _shrinkArrs(details::DevDataPack& thinData, const dim3& blocksDim, const dim3& threadsDim);
		}; // namespace thin::clique::_private;

		// For each voxel, we don't need to check all its 6 faces + 12 edges + 8 vertices
		// to see if a 2-clique (1-clique, 0-clique, resp) can be formed at that 2-face
		// (1-face, 0-face, resp), as this would result in many duplicate checks.
		//
		// Instead, by symmetry we can reduce the number of clique checks by half. That is,
		// by checking face 2, 3 and 5; edge 1, 2, 5, 6, 9 and 10 and vertex 1, 2, 5, 6 it
		// is enough to find all the cliques in the complex. These tokens are designed to
		// specify which d-face will be checked to see if a d-clique (d = 0, 1, 2 or 3) could
		// be formed at current voxel.
		//
		// Each voxel is itself a 3-clique, we cannot eliminate this check.
    
		// Each clique is Regular or Critical for complex X. And if it is critical, it can
		// be 1-critical, 2-critical or (2+)-critical. In this project, however, we are
		// only concerned in identifying 1-critical cliques, or 1-isthmuses, since the
		// preservation of such cliques will result in the curvilinear skeleton.
		// * The (2+)-critical clique is useful for finding surface skeleton.
		
		namespace cp = thin::clique::_private;

		template <typename DimPolicy>
		class CliqueChecker
		{
		public:

			__device__ static uint8_t numFaceTokens()
			{
				return DimPolicy::numFaceTokens();
			}

			__host__ __device__ static uint8_t numEcVoxels()
			{
				return DimPolicy::numEcVoxels();
			}

			// Check if a d-clique can be formed at vxIjk.
			__device__ static bool 
			canFormClique(const IjkType& vxIjk, ArrIndexType vxIndex, const details::DevDataPack& thinData, const FaceTokenType faceToken)
			{
				if (DimPolicy::ecOffsetArrEnd() == 0) return true;
				// Data betwen DimPolicy::coreCliqueArrBegin() and DimPolicy::coreCliqueArrEnd()
				// has the following format:
				// [X_1, X_2, ... X_m1, -1, X_(m1+1), X_(m1+2), ..., X_m2, -1, ..., X_mk, -1]
				//
				// Each X_i is a valid index to the Essential Clique offset array. Every two 
				// core clique index tuples are separated by -1, with the array ending by another -1. 
				uint8_t coreCliqueIdx = DimPolicy::coreCliqueArrBegin();
				while (coreCliqueIdx < DimPolicy::coreCliqueArrEnd())
				{
					// Fetch the ec offset index, ccInEcIndex, from the core clique index tuple texture.
					int8_t ccInEcIndex = cp::_fetchCoreClique(coreCliqueIdx);

					if (ccInEcIndex == -1)
					{
						// If this branch is reached, that means vxIjk added with all the ec offsets
						// in the current core clique index tuple exist in compactIjkArr, then 
						// a d-clique can be formed at vxIjk's d-face indicated by faceToken.
						return true;
					}

					OffsetIjkType tranOffs;
					cp::_transform(DimPolicy::matBeginByFaceToken(faceToken), cp::_fetchEcOffset(ccInEcIndex), tranOffs);

					if (!cp::_containsInX(thinData, vxIjk, vxIndex, tranOffs))
					{
						// vxIjk added with the current ec offset does not exist in compactIjkArr. It's
						// pointless to continue checking this core clique tuple, skip to -1 immediately.
						while (cp::_fetchCoreClique(coreCliqueIdx) != -1)
						{
							++coreCliqueIdx;
						}
					}

					++coreCliqueIdx;
				}

				return false;
			}
        
			// Find the voxels that form the Essential Clique at @vxIjk.
			//
			// [precondition] @vxIjk indeed forms a d-clique in grid3D at the
			// d-face indicated by faceToken. This is the user's responsibility.
			//
			// [return]: number of voxels in this EC.
			//
			// [postcondition] memory from foundEcIndexArr to foundEcIndexArr + [return]
			// stores the found indices of the voxels forming this Ec.
			__device__ static uint8_t 
			findEssentialClique(const IjkType& vxIjk, ArrIndexType vxIndex, const details::DevDataPack& thinData, const FaceTokenType faceToken, ArrIndexType* foundEcIndexArr)
			{
				uint8_t numEcVoxel = 0;
				uint8_t ecOffsetIdx = DimPolicy::ecOffsetArrBegin();
			
				while (ecOffsetIdx < DimPolicy::ecOffsetArrEnd())
				{
					OffsetIjkType tranOffs;
					cp::_transform(DimPolicy::matBeginByFaceToken(faceToken), cp::_fetchEcOffset(ecOffsetIdx), tranOffs);

					ArrIndexType foundIndex;
					
					if (cp::_findInX(foundIndex, thinData, vxIjk, vxIndex, tranOffs))
					{
						*foundEcIndexArr = foundIndex;
						++foundEcIndexArr;
						++numEcVoxel;
					}

					++ecOffsetIdx;
				}
				// add self into the essential clique voxel array
				*foundEcIndexArr = vxIndex;
				++numEcVoxel;

				return numEcVoxel;
			}
		       
			// Find the voxels that form the Neighbor of the essential clique at voxelIjk.
			//
			// [precondition]: voxelIjk indeed forms an essential d-clique in grid3D at
			// the d-face indicated by faceToken. This is the user's responsibility.
			//
			// [output]: A thin::nb::NbMaskType mask. This is because the voxels of the
			// neighbor for an EC is surely a subset of those for a single voxel. NbMask
			// is designed to store the neighbor voxels for just one voxel, hence it's
			// re-usable for Clique neighbor.
			__device__ static nb::NbMaskType 
			findCliqueNeighbor(const IjkType& vxIjk, ArrIndexType vxIndex, const details::DevDataPack& thinData, const FaceTokenType faceToken)
			{
				nb::NbMaskType cliqueNbMask = 0;
            
				uint8_t nbOffsIndexIter = DimPolicy::nbOffsIndexArrBegin();
				uint8_t nbOffsetIdx;

				while (nbOffsIndexIter < DimPolicy::nbOffsIndexArrEnd())
				{
					nbOffsetIdx = cp::_fetchNbOffsIndex(nbOffsIndexIter);

					OffsetIjkType tranOffs;
					cp::_transform(DimPolicy::matBeginByFaceToken(faceToken), nb::fetchNbOffset(nbOffsetIdx), tranOffs);
					
					// if (cp::_containsInX(thinData.compactIjkArr, thinData.recBitsArr, thinData.arrSize, vxIjk, vxIndex, tranOffs, thinData.size3D))
					if (cp::_containsInX(thinData, vxIjk, vxIndex, tranOffs))
					{
						tp::_setBit(cliqueNbMask, nbOffsetIdx);
					}

					++nbOffsIndexIter;
				}

				return cliqueNbMask;
			}
		
			// Check if the essential clique at @vxIjk (with the face at @faceToken
			// being shared) is Regular/Critical. If it is Critical, then if it is
			// 1-Critical.
			//
			// [precondition] @vxIjk indeed forms a d-clique in @thinData at the
			// d-face indicated by @faceToken. This is the user's responsibility.
			// [note] 2-Critical and (2+)-Critical are ignored in this project.
			__device__ static uint8_t 
			checkCliqueCritical(const IjkType& vxIjk, ArrIndexType vxIndex, const details::DevDataPack& thinData, const FaceTokenType faceToken)
		
			{
				// Find out the voxels that are the neighbor of the essential clique
				nb::NbMaskType cliqueNbMask = findCliqueNeighbor(vxIjk, vxIndex, thinData, faceToken);
				// Count the number of nb voxels.
				uint8_t lastIterSize = tp::_countNumSetBits(cliqueNbMask, nb::NUM_NB_OFFSETS);

				while (lastIterSize)
				{
					// Greedy algorithm by randomly pick up a Simple Point and remove it until
					// no sumch point coudl be found.
					for (uint8_t nthNb = 0; nthNb < nb::NUM_NB_OFFSETS; ++nthNb)
					{
						if (tp::_readBit(cliqueNbMask, nthNb))
						{
							nb::NbMaskType nbMask = cp::_genNbMaskFromCliqueNbMask(cliqueNbMask, nthNb);
							
							attach::Attachment attach = attach::generateAttachment(nbMask);
							// If attach is collapsible, then the n-th voxel in cliqueNbMask is Simple.
							// PAY attention to which neighbor mask is used, and which bit is cleared!
							if (attach.isCollapsible())
							{
								tp::_clearBit(cliqueNbMask, nthNb);
								break;
							}
						}
					}

					uint8_t curIterSize = tp::_countNumSetBits(cliqueNbMask, nb::NUM_NB_OFFSETS); // cp::NUM_NB_VOXELS

					if (curIterSize == lastIterSize) break;
					lastIterSize = curIterSize;
				}

				switch (lastIterSize)
				{
				case 1U:
					// Reducible to one voxel
					return CLQ_REGULAR;
				case 2U:
					// Reducible to two disconnected voxels, or a 0-surface
					return CLQ_1_CRITICAL;
				default:
					// Reducible to other type we don't care
					return CLQ_CRITICAL;
				}
			}
		};
		// Clique Checker for 3D
		using D3CliqueChecker = CliqueChecker<cp::Dim3CliquePolicy>;
		// Clique Checker for 2D
		using D2CliqueChecker = CliqueChecker<cp::Dim2CliquePolicy>;
		// Clique Checker for 1D
		using D1CliqueChecker = CliqueChecker<cp::Dim1CliquePolicy>;
		// Clique Checker for 0D
		using D0CliqueChecker = CliqueChecker<cp::Dim0CliquePolicy>;

		// Initialize the device side resources for this module to work.
		void initDevice();
		// Release the device side resources for this module to work.
		void shutdownDevice();

		// Run the crucial isthmus algorithm for a particular dimension.
		//
		// [param] blocksDim, threadsDim: CUDA kernel grid/block setup.
		template <typename Checker>
		void dimCrucialIsthmus(details::DevDataPack& thinData, const dim3& blocksDim, const dim3& threadsDim)
		{
			using namespace details;

			if(!_isDeviceInited())
			{
				std::cerr << "Thinning module is not initialized on device side" << std::endl;
				exit(1);
			}

			cp::_findDCriticalCliqueKern<Checker><<<blocksDim, threadsDim, Checker::numEcVoxels() * threadsDim.x * sizeof(ArrIndexType)>>>(thinData);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			
			cp::_unionKern<<<blocksDim, threadsDim>>>(thinData.A_recBitsArr, thinData.recBitsArr, thinData.arrSize, REC_BIT_A, REC_BIT_Y);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			cp::_unionKern<<<blocksDim, threadsDim>>>(thinData.B_recBitsArr, thinData.recBitsArr, thinData.arrSize, REC_BIT_B, REC_BIT_Z);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
		}

		void crucialIsthmus(details::DevDataPack& thinData, const dim3& blocksDim, const dim3& threadsDim);
	}; // namespace thin::clique;

}; // namespace thin;

#endif
