#ifndef THINNING_DETAILS_CUH
#define THINNING_DETAILS_CUH

// This file provides some shared utilities for the thinning algorithm that
// belongs to the implementation details.
//
// Users do not need to touch this file.

#include <vector>

#include "cuda_includes.h"
#include "thinning_base.cuh"

namespace thin
{
	namespace details
	{
		// Here, set X is the raw voxel complex, set K being the constraints set, set Y
		// D(X, K) and set Z I(X, K, k=1). These four sets can be stored using one
		// single array of ints, with the last four bits of each entry being an
		// indicator of each set.
		const uint8_t REC_BIT_X = 0;
		const uint8_t REC_BIT_K = 1;
		const uint8_t REC_BIT_Y = 2;
		const uint8_t REC_BIT_Z = 3;
		// However, set A and B must be stored in two independent array, otherwise we
		// will encounter race condition due to the nature of the thinning algorithm.
		const uint8_t REC_BIT_A = 0;
		const uint8_t REC_BIT_B = 0;
		// On the host side, set A and B could be packed with X/K/Y/Z bits together for
		// data r/w between RAM and disk.
		const uint8_t HOST_REC_BIT_A = 4U ;
		const uint8_t HOST_REC_BIT_B = HOST_REC_BIT_A + 1U;
		// Thinning data pack on the device side.
		class DevDataPack
		{
		public:
			// Intializing parameters class
			struct InitParams
			{
				// Array size
				unsigned arrSize;
				// 3D size of the entire voxel complex
				IjkType size3D;
				// Flag of whether the thinning algorithm uses voxel ID.
				bool useVoxelID;
				// Flag of whether the thinning algorithm uses birth date.
				bool useBirth;
			};

			DevDataPack(const InitParams& params);
			
			// Cannot use destructor to free device memory because we will
			// pass ThinningData instance by value between host and kernel code.
			// This will result in calling the d'tor of the local copy.
			//
			// ~DevDataPack();

			// Allocate the necessary amount of device memory for the device arrays.
			void alloc();
			// Free the allocated device memory.
			void dispose();

			__host__ __device__ inline const IjkType& size3D() const { return m_size3D; }

			__host__ __device__ inline bool useVoxelID() const { return m_useVoxelID; }
			__host__ __device__ inline bool useBirth() const { return m_useBirth; }

			IjkType* compactIjkArr;
			ObjIdType* voxelIdArr;
			RecBitsType* recBitsArr;
			RecBitsType* A_recBitsArr;
			RecBitsType* B_recBitsArr;
			unsigned* birthArr;

			unsigned arrSize;
			unsigned procBeginIndex;
			unsigned procEndIndex;

		private:
			IjkType m_size3D;

			bool m_useVoxelID;
			bool m_useBirth;
		};
		// Thinning data pack on the host side.
		class HostDataPack
		{
		public:
			std::vector<IjkType> compactIjkVec;
			std::vector<ObjIdType> voxelIdVec;
			std::vector<RecBitsType> recBitsVec;
			std::vector<unsigned> birthVec;
        
			void clear();

			void resize(unsigned size);
        
			void swap(HostDataPack& rhs);
		};
		// Copy the data from host side to device side.
		void _copyThinningDataToDevice(DevDataPack& d_thinData, const HostDataPack& h_thinData);
		// Copy the data from device side to host side.
		void _copyThinningDataToHost(HostDataPack& h_thinData, const DevDataPack& d_thinData);
		// Check whether the entire thinning module is initialized
		bool _isDeviceInited();
		void _setDeviceInited();
	}; // namespace thin::details
}; // namespace thin;

#endif