#include <vector>
#include <iostream>

#include "thinning.h"
#include "h5_io.h"


int main(int argc, char *argv[])
{
	
	unsigned width = 256U, height = 256U;
	unsigned numSlices = 256U;
	thin::IjkType size3D = thin::makeIjk(width, height, numSlices);
	unsigned maxNumVoxelsPerChunk = 100000U;

	h5_io::H5SliceIoManager sliceIoMngr("", "oldH5", "newH5", width, height, numSlices, "chunkMap.txt", maxNumVoxelsPerChunk);
	
	unsigned curIter = 0, curDim = 3U, maxIter = 80U;
	unsigned p = 10U;

	thin::initDevice();
	thin::setNumThreadsPerBlock(196U);

	thin::chunkwiseThinning(sliceIoMngr, size3D, curIter, curDim, p, maxIter);
	thin::shutdownDevice();


	return 0;
}
