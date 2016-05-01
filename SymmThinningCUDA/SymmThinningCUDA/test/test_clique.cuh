#ifndef TEST_CLIQUE_CUH
#define TEST_CLIQUE_CUH

#ifdef COMPILE_TEST

#include "cuda_includes.h"

#include "thinning_base.cuh"
#include "clique.cuh"
#include "neighbor.cuh"

#include "for_test.h"

namespace thin
{
	namespace test_clique
	{
		namespace 
		{
			using namespace thin::clique;
			using namespace thin::clique::_private;
			
			template <typename CHECKER>
			__global__ void 
			kern(IjkType vxIjk, ArrIndexType vxIndex, const IjkType* compactIjkArr, const RecBitsType* recBitsArr,
				const unsigned arrSize, IjkType size3D, FaceTokenType faceToken, bool expectClique, uint8_t expectCritical, unsigned numTest)
			{
				printf("test: %d\n", numTest);

				bool canClq = CHECKER::canFormClique(vxIjk, vxIndex, compactIjkArr, recBitsArr, arrSize, size3D, faceToken);
				
				printf("  >> can form clique: %d, shoule be: %d\n", canClq, expectClique);
				
				if (!expectClique) return;

				auto critical = CHECKER::checkCliqueCritical(vxIjk, vxIndex, compactIjkArr, recBitsArr, arrSize, size3D, faceToken);
				printf("  >> critical: %d, should be: %d\n", critical, expectCritical);
			}

			void test0()
			{
				const IjkType size3D = make_uint3(2,3,3);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(1,0,0);
				grid3D.add(0,1,1);
				grid3D.add(1,1,1);
				grid3D.add(0,2,1);
				grid3D.add(0,1,2);
				grid3D.add(1,1,2);
				grid3D.add(1,2,2);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(0,1,1);
				const ArrIndexType vxIndex = 2;
				FaceTokenType faceToken = D2_FACE_X;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_1_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D2CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 1);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test1()
			{
				const IjkType size3D = make_uint3(2,3,3);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(1,0,0);
				grid3D.add(0,1,0);
				grid3D.add(0,1,1);
				grid3D.add(1,1,1);
				grid3D.add(0,2,1);
				grid3D.add(0,1,2);
				grid3D.add(1,1,2);
				grid3D.add(1,2,2);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(0,1,1);
				const ArrIndexType vxIndex = 3;
				FaceTokenType faceToken = D2_FACE_X;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_REGULAR;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D2CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 2);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test2()
			{
				const IjkType size3D = make_uint3(3,3,3);
				MyGrid grid3D;
				grid3D.add(0,2,0);
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(1,2,0);
				grid3D.add(2,1,0);
				grid3D.add(0,0,1);
				grid3D.add(1,0,1);
				grid3D.add(1,1,1);
				grid3D.add(1,2,1);
				grid3D.add(0,1,2);
				grid3D.add(1,0,2);
				grid3D.add(1,1,2);
				grid3D.add(1,2,2);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(1,1,1);
				const ArrIndexType vxIndex = 7;
				FaceTokenType faceToken = D3_VOXEL;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D3CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 3);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test3()
			{
				const IjkType size3D = make_uint3(2,3,2);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(1,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,0,1);
				grid3D.add(1,1,1);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(0,1,0);
				const ArrIndexType vxIndex = 1;
				FaceTokenType faceToken = D1_EDGE_9;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_1_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D1CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 4);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test4()
			{
				const IjkType size3D = make_uint3(3,3,3);
				MyGrid grid3D;
				grid3D.add(0,1,0);
				grid3D.add(1,0,1);
				grid3D.add(2,0,1);
				grid3D.add(1,1,1);
				grid3D.add(1,0,2);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(1,0,1);
				const ArrIndexType vxIndex = 1;
				FaceTokenType faceToken = D2_FACE_Y;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_1_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D2CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 5);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test5()
			{
				const IjkType size3D = make_uint3(3,3,2);
				MyGrid grid3D;
				grid3D.add(1,0,0);
				grid3D.add(2,0,0);
				grid3D.add(0,1,0);
				grid3D.add(1,1,0);
				grid3D.add(1,0,1);

				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(1,0,0);
				const ArrIndexType vxIndex = 0;
				FaceTokenType faceToken = D2_FACE_Y;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_REGULAR;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D2CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 6);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test6()
			{
				const IjkType size3D = make_uint3(2,2,2);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,1);
				grid3D.add(1,1,1);
				
				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(0,0,0);
				const ArrIndexType vxIndex = 0;
				FaceTokenType faceToken = D0_VERTEX_6;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = true;
				const uint8_t expectCritical = CLQ_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D0CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 7);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}

			void test7()
			{
				const IjkType size3D = make_uint3(2,2,2);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,1);
				
				// grid3D's elements needs to be sorted. However, we
				// cannot sort automatically, otherwise we lose vxIndex
				// grid3D.sort();

				const IjkType vxIjk = make_uint3(0,0,0);
				const ArrIndexType vxIndex = 0;
				FaceTokenType faceToken = D0_VERTEX_6;

				IjkType* d_compactIjkArr;
				RecBitsType* d_recBitsArr;
				unsigned arrSize = grid3D.size();

				const bool expectClique = false;
				const uint8_t expectCritical = CLQ_CRITICAL;

				checkCudaErrors(cudaMalloc(&d_compactIjkArr, sizeof(IjkType) * arrSize));
				checkCudaErrors(cudaMemcpy(d_compactIjkArr, grid3D.begin(), sizeof(IjkType) * arrSize, cudaMemcpyHostToDevice));
				checkCudaErrors(cudaMalloc(&d_recBitsArr, sizeof(RecBitsType) * arrSize));
				checkCudaErrors(cudaMemset(d_recBitsArr, 0x01, sizeof(RecBitsType) * arrSize));

				kern<D0CliqueChecker><<<1, 1>>>(vxIjk, vxIndex, d_compactIjkArr, d_recBitsArr, arrSize, size3D, faceToken, expectClique, expectCritical, 8);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_compactIjkArr));
				checkCudaErrors(cudaFree(d_recBitsArr));
			}
		}; // namespace thin::test_clique::anonymous

		void runTests()
		{
			nb::initDevice();
			attach::initDevice();
			clique::initDevice();

			test0();
			test1();
			test2();
			test3();
			test4();
			test5();
			test6();
			test7();

			clique::shutdownDevice();
			attach::shutdownDevice();
			nb::shutdownDevice();
		}
	}; // namespace thin::test_clique
}; // namespace thin
#endif

#endif