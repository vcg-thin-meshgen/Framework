#include <vector>
#include <iostream>

#include "thinning.h"
#include "h5_io.h"

int main(int argc, char *argv[])
{
	// number of slices in total
	unsigned numSlices;
	// number of slices in each chunk
	unsigned chunkSize;
	// total number of chunks
	unsigned numChunks; // = (numSlices + chunkSize - 1U) / chunkSize;
	// volumetric data size
	thin::IjkType size3D;
	// current thinning iteration and dimension
	unsigned curIter, curDim;
	// maximal thinning iteration
	unsigned maxIter;
	// persistence threshold
	unsigned p;
	
	// Chunk I/O manager using the HDF5 file format
	h5_io::H5SliceIoManager h5SliceIoMngr("", "oldGroup", "newGroup", size3D.x, size3D.y, chunkSize, chunkSize);
	
	// initiliaze the thinning framework
	thin::initDevice();
	thin::setNumThreadsPerBlock(196U);
	// run chunk-wise thinning
	thin::fullThinning(thin::MatchRawIOPolicy(), h5SliceIoMngr, chunkSize, numSlices, size3D, curIter, curDim, maxIter, p);
	// shutdown the thinning framework
	thin::shutdownDevice();

	return 0;
}
