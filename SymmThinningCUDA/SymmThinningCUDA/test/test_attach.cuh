#ifndef TEST_ATTACH_CUH
#define TEST_ATTACH_CUH

#ifdef COMPILE_TEST
#include "cuda_includes.h"

#include "thinning_base.cuh"
#include "attachment.cuh"
#include "neighbor.cuh"

#include "for_test.h"

namespace thin
{
	namespace test_attach
	{
		namespace 
		{
			__global__ void 
			kern(IjkType vxIjk, const thin::IjkType* d_nbIjkArr, const unsigned nbArrSize, 
				IjkType size3D, unsigned expect, unsigned numTest)
			{
				nb::NbMaskType nbMask = nb::generateNbMask(vxIjk, d_nbIjkArr, nbArrSize, size3D);
				attach::Attachment attach = attach::generateAttachment(nbMask);
				printf("test: %d, collapsible: %d, shoule be: %d\n", numTest, attach.isCollapsible(), expect);
			}

			void test0()
			{
				const IjkType size3D = make_uint3(3,3,3);
				MyGrid grid3D;
				grid3D.add(1, 0, 1);
				grid3D.add(1, 2, 1);
				grid3D.add(0, 1, 1);
				grid3D.add(2, 1, 1);
				grid3D.add(1, 1, 1);

				const IjkType vxIjk = make_uint3(1,1,1);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 1);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test1()
			{
				const IjkType size3D = make_uint3(2,3,1);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(1,2,0);

				const IjkType vxIjk = make_uint3(0,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 1);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test2()
			{
				const IjkType size3D = make_uint3(2,3,1);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(1,1,0);

				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 2);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test3()
			{
				const IjkType size3D = make_uint3(3,3,1);
				MyGrid grid3D;
				grid3D.add(0,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,1,0);
				grid3D.add(1,2,0);
				grid3D.add(2,2,0);

				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 3);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test4()
			{
				const IjkType size3D = make_uint3(3,3,1);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(1,2,0);
				grid3D.add(2,0,0);
				grid3D.add(2,1,0);
				grid3D.add(2,2,0);
        

				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 4);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}
			
			void test5()
			{
				const IjkType size3D = make_uint3(2,3,1);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,0,0);
				grid3D.add(1,2,0);
        

				const IjkType vxIjk = make_uint3(0,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 5);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}
			
			void test6()
			{
				const IjkType size3D = make_uint3(3,3,1);
				MyGrid grid3D;
				grid3D.add(1,1,0);
        
				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 6);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test7()
			{
				const IjkType size3D = make_uint3(2,3,3);
				MyGrid grid3D;
				grid3D.add(1, 0, 0);
				grid3D.add(1, 1, 0);
				grid3D.add(1, 2, 0);
				grid3D.add(0, 0, 1);
				grid3D.add(0, 2, 1);
				grid3D.add(1, 0, 1);
				grid3D.add(1, 2, 1);
				grid3D.add(0, 0, 2);
				grid3D.add(0, 1, 2);
				grid3D.add(0, 2, 2);

				const IjkType vxIjk = make_uint3(0,1,2);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 7);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test8()
			{
				const IjkType size3D = make_uint3(3,3,3);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(0,2,0);
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(1,2,0);
				grid3D.add(2,0,0);
				grid3D.add(2,1,0);
				grid3D.add(2,2,0);
        
				grid3D.add(0,0,1);
				grid3D.add(0,1,1);
				grid3D.add(0,2,1);
				grid3D.add(1,0,1);
				grid3D.add(1,1,1);
				grid3D.add(1,2,1);
				grid3D.add(2,0,1);
				grid3D.add(2,1,1);
				grid3D.add(2,2,1);
        
				grid3D.add(0,0,2);
				grid3D.add(0,1,2);
				grid3D.add(0,2,2);
				grid3D.add(1,0,2);
				grid3D.add(1,1,2);
				grid3D.add(1,2,2);
				grid3D.add(2,0,2);
				grid3D.add(2,1,2);
				grid3D.add(2,2,2);

				const IjkType vxIjk = make_uint3(1,1,1);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 10);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test9()
			{
				const IjkType size3D = make_uint3(3,1,1);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(1,0,0);
				grid3D.add(2,0,0);

				const IjkType vxIjk = make_uint3(1,0,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 9);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test10()
			{
				const IjkType size3D = make_uint3(3,2,2);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(1,0,0);
				grid3D.add(2,0,0);
				grid3D.add(1,1,1);

				const IjkType vxIjk = make_uint3(1,0,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 10);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}
			
			void test11()
			{
				const IjkType size3D = make_uint3(2,2,2);
				MyGrid grid3D;
				grid3D.add(0,0,0);
				grid3D.add(0,1,0);
				grid3D.add(1,0,0);
				grid3D.add(0,0,1);

				const IjkType vxIjk = make_uint3(0,0,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 11);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test12()
			{
				const IjkType size3D = make_uint3(3,3,3);
				MyGrid grid3D;
				grid3D.add(1,1,0);
				grid3D.add(1,0,1);
				grid3D.add(1,1,1);
				grid3D.add(2,1,1);
				grid3D.add(0,2,2);

				const IjkType vxIjk = make_uint3(1,1,1);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 12);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test13()
			{
				const IjkType size3D = make_uint3(3,3,2);
				MyGrid grid3D;
				grid3D.add(1,1,0);
				grid3D.add(1,0,1);
				grid3D.add(1,1,1);
				grid3D.add(2,1,1);
				grid3D.add(0,2,1);

				const IjkType vxIjk = make_uint3(1,1,1);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 1;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 13);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test14()
			{
				const IjkType size3D = make_uint3(2,3,2);
				MyGrid grid3D;
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(0,2,1);

				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 14);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}

			void test15()
			{
				const IjkType size3D = make_uint3(2,3,2);
				MyGrid grid3D;
				grid3D.add(1,0,0);
				grid3D.add(1,1,0);
				grid3D.add(0,2,0);

				const IjkType vxIjk = make_uint3(1,1,0);

				IjkType* d_nbArr;
				unsigned nbArrSize = grid3D.size();

				const unsigned expect = 0;

				checkCudaErrors(cudaMalloc(&d_nbArr, sizeof(IjkType) * nbArrSize));
				checkCudaErrors(cudaMemcpy(d_nbArr, grid3D.begin(), sizeof(IjkType) * nbArrSize, cudaMemcpyHostToDevice));

				kern<<<1, 1>>>(vxIjk, d_nbArr, nbArrSize, size3D, expect, 15);
				
				cudaDeviceSynchronize();
				checkCudaErrors(cudaGetLastError());
				
				checkCudaErrors(cudaFree(d_nbArr));
			}
		}; // namespace thin::test_attach::anonymous;

		void runTests()
		{
			nb::initDevice();
			attach::initDevice();

			test0();
			test1();
			test2();
			test3();
			test4();
			test5();
			test6();
			test7();
			test8();
			test9();
			test10();
			test11();
			test12();
			test13();
			test14();
			test15();

			attach::shutdownDevice();
			nb::shutdownDevice();
		}
	}; // namespace thin::test_attach
}; // namespace thin;
#endif

#endif