#include "thinning.h"
#include "cuda_includes.h"
#include "neighbor.cuh"
#include "attachment.cuh"
#include "clique.cuh"
#include "thinning_details.cuh"

namespace thin
{
	void initDevice()
	{
		nb::initDevice();
		attach::initDevice();
		clique::initDevice();

		details::_setDeviceInited();
	}

	void shutdownDevice()
	{
		clique::shutdownDevice();
		attach::shutdownDevice();
		nb::shutdownDevice();
	}

	static unsigned _numThreads = 128U;

	void setNumThreadsPerBlock(unsigned num) { _numThreads = num; }

	unsigned numThreadsPerBlock() { return _numThreads; }

	void isthmusSymmetricThinning(const std::vector<IjkType>& compactIjkVec,/* const std::vector<ObjIdType>& voxelIdVec,*/ std::vector<IjkType>& D_XK, const IjkType& size3D, int maxIter)
	{
		// using namespace clique;
		using namespace details;
		namespace cp = clique::_private;

		DevDataPack::InitParams packInitParams;
		packInitParams.arrSize = compactIjkVec.size();
		packInitParams.size3D = size3D;
		packInitParams.useBirth = false;
		packInitParams.useVoxelID = false;
		DevDataPack thinData(packInitParams);
		thinData.alloc();

		checkCudaErrors(cudaMemset(thinData.recBitsArr, 0x01, sizeof(RecBitsType) * thinData.arrSize));
		checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
		checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
        
		// IjkType* d_compactIjkArr;
		// checkCudaErrors(cudaMalloc(&(thinData.compactIjkArr), sizeof(IjkType) * thinData.arrSize));
		checkCudaErrors(cudaMemcpy(thinData.compactIjkArr, compactIjkVec.data(), sizeof(IjkType) * thinData.arrSize, cudaMemcpyHostToDevice));

		unsigned curIter = 1;
		unsigned lastIterSize = thinData.arrSize;
        
		dim3 threadsDim(_numThreads, 1U, 1U);
		dim3 blocksDim((thinData.arrSize + threadsDim.x - 1U) / threadsDim.x, 1U, 1U);

		while (blocksDim.x > 32768U)
		{
			blocksDim.x /= 2;
			blocksDim.y *= 2;
		}

		while ((maxIter < 0) || (maxIter > 0 && curIter <= maxIter))
		{
			std::cout << "Current iteration: " << curIter 
					<< ", size: " << lastIterSize << std::endl;
            
			clique::crucialIsthmus(thinData, blocksDim, threadsDim);
            
			unsigned curIterSize = cp::_countBit(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y);

			if (curIterSize == lastIterSize) break;

			cp::_assignKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y, REC_BIT_X);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			cp::_unionKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.recBitsArr, thinData.arrSize, REC_BIT_Z, REC_BIT_K);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			thinData.arrSize = cp::_shrinkArrs(thinData, blocksDim, threadsDim);
			assert(thinData.arrSize == curIterSize);

			// To-Do:
			// 1. clean up the d_A/B_recBitsArr accordingly
			// 2. re-calculate blocksDim
			checkCudaErrors(cudaFree(thinData.A_recBitsArr));
			checkCudaErrors(cudaFree(thinData.B_recBitsArr));
			
			checkCudaErrors(cudaMalloc(&(thinData.A_recBitsArr), sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMalloc(&(thinData.B_recBitsArr), sizeof(RecBitsType) * thinData.arrSize));
        
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			blocksDim.x = (thinData.arrSize + threadsDim.x - 1U) / threadsDim.x;
			blocksDim.y = 1U;
			while (blocksDim.x > 32768U)
			{
				blocksDim.x /= 2;
				blocksDim.y *= 2;
			}

			lastIterSize = curIterSize;
			++curIter;
		}

		D_XK.clear();
		D_XK.resize(thinData.arrSize);
		checkCudaErrors(cudaMemcpy(D_XK.data(), thinData.compactIjkArr, sizeof(IjkType) * thinData.arrSize, cudaMemcpyDeviceToHost));

		thinData.dispose();
	}

	void persistenceIsthmusThinningCore(details::DevDataPack& thinData, unsigned curIter, unsigned p, int maxIter)
	{
		using namespace details;
		namespace cp = clique::_private;

		unsigned lastIterSize = thinData.arrSize;
        
		dim3 threadsDim(_numThreads, 1U, 1U);
		dim3 blocksDim((thinData.arrSize + threadsDim.x - 1U) / threadsDim.x, 1U, 1U);

		while (blocksDim.x > 32768U)
		{
			blocksDim.x /= 2;
			blocksDim.y *= 2;
		}

		while ((maxIter < 0) || (maxIter > 0 && curIter <= maxIter))
		{
			std::cout << "Current iteration: " << curIter 
					<< ", size: " << lastIterSize << std::endl;
            
			// crucialIsthmus(grid3D, Kset, D_XK, I_XK1);
			// crucialIsthmusCUDA(compactFlatIjkVec, flatMngr, recBitsVec, numThreads);
			clique::crucialIsthmus(thinData, blocksDim, threadsDim);
            
			unsigned curIterSize = cp::_countBit(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y);
			if (curIterSize == lastIterSize) break;

			cp::_assignKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y, REC_BIT_X);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			
			cp::_updateBirthKern<<<blocksDim, threadsDim>>>(thinData.birthArr, thinData.recBitsArr, thinData.arrSize, curIter);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			cp::_unionKsetByBirth<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.birthArr, thinData.arrSize, curIter, p);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			thinData.arrSize = cp::_shrinkArrs(thinData, blocksDim, threadsDim);
			assert(thinData.arrSize == curIterSize);

			// To-Do:
			// 1. clean up the d_A/B_recBitsArr accordingly
			// 2. re-calculate blocksDim
			checkCudaErrors(cudaFree(thinData.A_recBitsArr));
			checkCudaErrors(cudaFree(thinData.B_recBitsArr));
			
			checkCudaErrors(cudaMalloc(&(thinData.A_recBitsArr), sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMalloc(&(thinData.B_recBitsArr), sizeof(RecBitsType) * thinData.arrSize));
        
			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			
			blocksDim.x = (thinData.arrSize + threadsDim.x - 1U) / threadsDim.x;
			blocksDim.y = 1U;
			while (blocksDim.x > 32768U)
			{
				blocksDim.x /= 2;
				blocksDim.y *= 2;
			}

			lastIterSize = curIterSize;
			++curIter;
		}
	}

	void persistenceIsthmusThinning(const std::vector<IjkType>& compactIjkVec, const std::vector<ObjIdType>& voxelIdVec, std::vector<IjkType>& D_XK, 
									const IjkType& size3D, unsigned p, int maxIter)
	{
		// using namespace clique;
		using namespace details;
		namespace cp = clique::_private;

		// ThinningData thinData(compactIjkVec.size(), size3D);
		DevDataPack::InitParams packInitParams;
		packInitParams.arrSize = compactIjkVec.size();
		packInitParams.size3D = size3D;
		packInitParams.useBirth = true;
		packInitParams.useVoxelID = voxelIdVec.size() > 0;
		DevDataPack thinData(packInitParams);

        thinData.alloc();
		
		checkCudaErrors(cudaMemset(thinData.recBitsArr, 0x01, sizeof(RecBitsType) * thinData.arrSize));
		checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
		checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
        
		checkCudaErrors(cudaMemcpy(thinData.compactIjkArr, compactIjkVec.data(), sizeof(IjkType) * thinData.arrSize, cudaMemcpyHostToDevice));

		if (thinData.useVoxelID())
		{
			checkCudaErrors(cudaMemcpy(thinData.voxelIdArr, voxelIdVec.data(), sizeof(ObjIdType) * thinData.arrSize, cudaMemcpyHostToDevice));
		}
		
		checkCudaErrors(cudaMemset(thinData.birthArr, 0, sizeof(unsigned) * thinData.arrSize));

		unsigned curIter = 0;
		persistenceIsthmusThinningCore(thinData, curIter, p, maxIter);
        
		D_XK.clear();
		D_XK.resize(thinData.arrSize);
		checkCudaErrors(cudaMemcpy(D_XK.data(), thinData.compactIjkArr, sizeof(IjkType) * thinData.arrSize, cudaMemcpyDeviceToHost));

		thinData.dispose();
	}

	void persistenceIsthmusThinning(const std::vector<IjkType>& compactIjkVec, std::vector<IjkType>& D_XK, const IjkType& size3D, unsigned p, int maxIter)
	{
		std::vector<ObjIdType> fakeVoxelIdVec;
		persistenceIsthmusThinning(compactIjkVec, fakeVoxelIdVec, D_XK, size3D, p, maxIter);
	}

	void oneChunkThinning(details::DevDataPack& thinData, unsigned curIter, unsigned dim, 
		unsigned p, const dim3& blocksDim, const dim3& threadsDim)
	{
		using namespace thin::clique;
		namespace cp = thin::clique::_private;
		using namespace details;

		if (dim == 3U)
		{
			// Y <- K
			cp::_assignKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_K, REC_BIT_Y);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			// Z <- {}
			cp::_clearKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Z);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			dimCrucialIsthmus<D3CliqueChecker>(thinData, blocksDim, threadsDim);
		}
		else if (dim == 2U)
		{
			dimCrucialIsthmus<D2CliqueChecker>(thinData, blocksDim, threadsDim);
		}
		else if (dim == 1U)
		{
			dimCrucialIsthmus<D1CliqueChecker>(thinData, blocksDim, threadsDim);
		}
		else if (dim == 0)
		{
			dimCrucialIsthmus<D0CliqueChecker>(thinData, blocksDim, threadsDim);

			cp::_assignKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y, REC_BIT_X);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			
			cp::_updateBirthKern<<<blocksDim, threadsDim>>>(thinData.birthArr, thinData.recBitsArr, thinData.arrSize, curIter);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			cp::_unionKsetByBirth<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.birthArr, thinData.arrSize, curIter, p);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			cp::_clearKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Y);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			cp::_clearKern<<<blocksDim, threadsDim>>>(thinData.recBitsArr, thinData.arrSize, REC_BIT_Z);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaMemset(thinData.A_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
			checkCudaErrors(cudaMemset(thinData.B_recBitsArr, 0, sizeof(RecBitsType) * thinData.arrSize));
		}
	}
}; // namespace thin;