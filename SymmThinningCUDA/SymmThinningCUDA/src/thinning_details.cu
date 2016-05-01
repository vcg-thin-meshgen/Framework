#include <thrust/transform.h>				// thrust::transform
#include <thrust/execution_policy.h>

#include "thinning_details.cuh"

namespace thin
{
	namespace details
	{
		DevDataPack::DevDataPack(const DevDataPack::InitParams& params)
			: compactIjkArr(nullptr), voxelIdArr(nullptr), recBitsArr(nullptr), A_recBitsArr(nullptr), B_recBitsArr(nullptr)
			, birthArr(nullptr), arrSize(params.arrSize), procBeginIndex(0), procEndIndex(params.arrSize)
			, m_size3D(params.size3D), m_useVoxelID(params.useVoxelID), m_useBirth(params.useBirth) { }

		void DevDataPack::alloc()
		{
			checkCudaErrors(cudaMalloc(&recBitsArr, sizeof(RecBitsType) * arrSize));
			checkCudaErrors(cudaMalloc(&A_recBitsArr, sizeof(RecBitsType) * arrSize));
			checkCudaErrors(cudaMalloc(&B_recBitsArr, sizeof(RecBitsType) * arrSize));

			checkCudaErrors(cudaMalloc(&compactIjkArr, sizeof(IjkType) * arrSize));

			if (m_useBirth)
			{
				checkCudaErrors(cudaMalloc(&birthArr, sizeof(unsigned) * arrSize));
			}

			if (m_useVoxelID)
			{
				checkCudaErrors(cudaMalloc(&voxelIdArr, sizeof(ObjIdType) * arrSize));
			}
		}

		void DevDataPack::dispose()
		{
			checkCudaErrors(cudaFree(compactIjkArr));
			checkCudaErrors(cudaFree(voxelIdArr));
			checkCudaErrors(cudaFree(recBitsArr));
			checkCudaErrors(cudaFree(A_recBitsArr));
			checkCudaErrors(cudaFree(B_recBitsArr));
			checkCudaErrors(cudaFree(birthArr));
		}

		void HostDataPack::clear()
		{
			compactIjkVec.clear();
			voxelIdVec.clear();
			recBitsVec.clear();
			birthVec.clear();	
		}

		void HostDataPack::resize(unsigned size)
		{
			compactIjkVec.resize(size);
			voxelIdVec.resize(size);
			recBitsVec.resize(size);
			birthVec.resize(size);
		}

		void HostDataPack::swap(HostDataPack& rhs)
		{
			compactIjkVec.swap(rhs.compactIjkVec);
			voxelIdVec.swap(rhs.voxelIdVec);
			recBitsVec.swap(rhs.recBitsVec);
			birthVec.swap(rhs.birthVec);
		}
		// When copy thinning data from host to device, the recording bits on host side
		// store set X/K/Y/Z/A/B, yet the recording bits on device side only store set
		// X/K/Y/Z (with bit A and B stored in another two independent arrays). This
		// functor is used for retaining the bit information of X/K/Y/Z.
		class RetainRecBitsFunctor
		{
		public:
			__device__ RecBitsType operator() (const RecBitsType& bits) const
			{
				return bits & 0x0f;
			}
		};
		// On host side, the recording bit stores set A's information at HOST_REC_BIT_A
		// bit, while on device side, the recording bit stores it at REC_BIT_A bit in an
		// individual array. This functor is used for scattering the bit data of set A
		// from HOST_REC_BIT_A bit to REC_BIT_A bit.
		class ScatterRecBitAFunctor
		{
		public:
			__device__ RecBitsType operator() (const RecBitsType& bits) const
			{
				return (bits >> HOST_REC_BIT_A) & 1U;
			}
		};
		// On host side, the recording bit stores set B's information at HOST_REC_BIT_B
		// bit, while on device side, the recording bit stores it at REC_BIT_B bit in an
		// individual array. This functor is used for scattering the bit data of set B
		// from HOST_REC_BIT_B bit to REC_BIT_B bit.
		class ScatterRecBitBFunctor
		{
		public:
			__device__ RecBitsType operator() (const RecBitsType& bits) const
			{
				return (bits >> HOST_REC_BIT_B) & 1U;
			}
		};

		void _copyThinningDataToDevice(DevDataPack& d_thinData, const HostDataPack& h_thinData)
		{
			const unsigned arrSize = d_thinData.arrSize;
			assert(arrSize == h_thinData.compactIjkVec.size());

			checkCudaErrors(cudaMemcpy(d_thinData.compactIjkArr, h_thinData.compactIjkVec.data(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
			if (d_thinData.useVoxelID())
			{
				checkCudaErrors(cudaMemcpy(d_thinData.voxelIdArr, h_thinData.voxelIdVec.data(), sizeof(ObjIdType) * arrSize, cudaMemcpyHostToDevice));
			}
			// On Host side, recording bits B and A are stored along with Z, Y, K and X using a single unsigned char.
			// On Device side, recording bit B and A are stored in two separated arrays, however. Therefore, we need
			// to separate out the recording bit information from host side to device side.
			//
			// Step 1, copy all the recording bits data from host side to device side
			checkCudaErrors(cudaMemcpy(d_thinData.recBitsArr, h_thinData.recBitsVec.data(), sizeof(RecBitsType) * arrSize, cudaMemcpyHostToDevice));
			// Step 2, copy the recording bit A's data to its proper array
			thrust::transform(thrust::device, d_thinData.recBitsArr, d_thinData.recBitsArr + arrSize, d_thinData.A_recBitsArr, ScatterRecBitAFunctor());
			// Step 3, copy the recording bit B's data to its proper array
			thrust::transform(thrust::device, d_thinData.recBitsArr, d_thinData.recBitsArr + arrSize, d_thinData.B_recBitsArr, ScatterRecBitBFunctor());
			// Step 4, clear recording bits A and B's data on device side.
			thrust::transform(thrust::device, d_thinData.recBitsArr, d_thinData.recBitsArr + arrSize, d_thinData.recBitsArr, RetainRecBitsFunctor());

			if (d_thinData.useBirth())
			{
				checkCudaErrors(cudaMemcpy(d_thinData.birthArr, h_thinData.birthVec.data(), sizeof(unsigned) * arrSize, cudaMemcpyHostToDevice));
			}
		}
		// The functor that performs the reverse operation of ScatterRecBitAFunctor.
		class CompactRecBitAFunctor
		{
		public:
			__device__ RecBitsType operator() (const RecBitsType& bits, const RecBitsType& bitA) const
			{
				return bits | (bitA << HOST_REC_BIT_A);
			}
		};
		// The functor that performs the reverse operation of ScatterRecBitBFunctor.
		class CompactRecBitBFunctor
		{
		public:
			__device__ RecBitsType operator() (const RecBitsType& bits, const RecBitsType& bitB) const
			{
				return bits | (bitB << HOST_REC_BIT_B);
			}
		};

		void _copyThinningDataToHost(HostDataPack& h_thinData, const DevDataPack& d_thinData)
		{
			const unsigned arrSize = d_thinData.arrSize;
			assert(arrSize == h_thinData.compactIjkVec.size());

			h_thinData.clear();
			h_thinData.resize(arrSize);

			checkCudaErrors(cudaMemcpy(h_thinData.compactIjkVec.data(), d_thinData.compactIjkArr, sizeof(IjkType) * arrSize, cudaMemcpyDeviceToHost));
			if (d_thinData.useVoxelID())
			{
				checkCudaErrors(cudaMemcpy(h_thinData.voxelIdVec.data(), d_thinData.voxelIdArr, sizeof(ObjIdType) * arrSize, cudaMemcpyDeviceToHost));
			}
			// Since the device side separates recording bits A and B from X/K/Y/Z, we
			// need to pack them together before copying to the host side.
			RecBitsType* d_packedRecBitsArr;
			checkCudaErrors(cudaMalloc(&d_packedRecBitsArr, sizeof(RecBitsType) * arrSize));
			// 1. Copy the recording bits X/K/Y/Z from the device side
			checkCudaErrors(cudaMemcpy(d_packedRecBitsArr, d_thinData.recBitsArr, sizeof(RecBitsType) * arrSize, cudaMemcpyDeviceToDevice));
			// 2. Pack the recording bit A together with X/K/Y/Z from the device side.
			thrust::transform(thrust::device, d_packedRecBitsArr, d_packedRecBitsArr + arrSize, d_thinData.A_recBitsArr, d_packedRecBitsArr, CompactRecBitAFunctor());
			// 3. Pack the recording bit B together with X/K/Y/Z/A from the device side.
			thrust::transform(thrust::device, d_packedRecBitsArr, d_packedRecBitsArr + arrSize, d_thinData.B_recBitsArr, d_packedRecBitsArr, CompactRecBitBFunctor());
			// 4. Copy the packed recording bits to host side.
			checkCudaErrors(cudaMemcpy(h_thinData.recBitsVec.data(), d_packedRecBitsArr, sizeof(RecBitsType) * arrSize, cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaFree(d_packedRecBitsArr));

			if (d_thinData.useBirth())
			{
				checkCudaErrors(cudaMemcpy(h_thinData.birthVec.data(), d_thinData.birthArr, sizeof(unsigned) * arrSize, cudaMemcpyDeviceToHost));
			}
		}

		static bool _isInited = false;
		bool _isDeviceInited() { return _isInited; }
		void _setDeviceInited() { _isInited = true; }
	}; // namespace thin::details;
}; // namespace thin;