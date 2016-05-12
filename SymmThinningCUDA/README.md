# Skeletonization

This project implements the symmetric parallel thinning algorithm with persistence [1] that supports a chunk-wise thinnign scheme. The project is implemeneted in C++11/CUDA 7.5. `thinning.h` header file provides all the necessary functions of the algorithm.

### Data Types

All the data types are defined in `thinning_base.h`.

- `thin::IjkType`: This type is used to store the discrete coordinate of a voxel. It is the alias of `uint3` provided by CUDA.
- `thin::ObjIdType`: This type is used to store the object ID. It is the alias of `uint32_t`. 
- `thin::RecBitsType`: Because each voxel is associated with some intermediate states (A voxel might be added to several sets: X/Y/K/A/B [1]) during the thinning process, we record the states in different bits of the object of this type. The type name is short for Recording Bits. It is the alias of `unsigned char`. 

	The last four bits of a `RecBitsType` object is used to store if the associated voxel is in set X, K, Y, Z, respectively:
	
	```
	// RecBitType
	
	|----- 3 -----|----- 2 -----|----- 1 -----|----- 0 -----|
	|   in set Z  |   in set Y  |   in set K  |   in set X  |
	```
	
	Set X denotes the original voxel set, therefore, **the initialize of the intermediate state for each voxel should always be 1**. (For the complete meaning of set X, K, Y and Z, please refer to Algorithm 1 and 2 in [1]).

### Functions

-
```
// thinning.h
void
thin::persistenceIsthmusThinning (
	const std::vector<thin::IjkType>& voxelIjkVec,
	const std::vector<ObjIdType>& voxelIdVec,
	std::vector<IjkType>& D_XK,
	const IjkType& size3D,
	unsigned p,
	int maxIter
);
```

This function is the implementation of the original parallel persistence thinning algorithm [1]. `voxelIjkVec` stores the raw voxel complex dataset, `voxelIdVec` stores the object ID of each voxel. `D_XK` stores the output result, which is the set of voxels preserved as the skeleton. `size3D` is the three-dimensional size of the voxel dataset. `p` is the persistence threshold and `maxIter` the maximum iteration number.

*precondition*: `voxelIjkVec` is sorted in ascending order in terms of the flat coordinate `i + j * size3D.x + k * size3D.x * size3D.y`.

*precondition*: If there are multiple objects in the dataset, then `voxelIjkVec.size() == voxelIdVec.size()`. Otherwise `voxelIdVec.size() == 0`, and all the voxels in `voxelIjkVec` are treated as belonging to a single object.

-
```
// thinning.h

template <typename MNGR>
void chunkwiseThinning(
	MNGR& mngr,
	const IjkType& size3D,
	unsigned curIter,
	unsigned dim,
	unsigned maxIter,
	unsigned p
);
```
This function is the implementation of the chunk-wise parallel thinning algorithm. `mngr` is the manager object that controls the transfer of the slice data between the algorithm and the external storage. `size3D` is the three-dimensional size of the voxel dataset. `curIter` is the current thinning iteration and `dim` the current dimension (rank) for thinning. `p` is the persistence threshold and `maxIter` the maximum iteration number.

Currently, we have implemented a slice manager, `h5_io::H5SliceIoManager`, that uses HDF5 file format to store the data. However, we have made the type of the manager object a template to enable the users to choose any other file format they prefer, as long as the interfaces are correctly implemented.

### Slice Manager

The role of a slice manager is to transfer the slice data between RAM and the external storage. Before the beginning of the thinning algorithm, it loads a chunk of slices into the RAM. Upon the completion of the thinning algorithm on this chunk, it writes the data back to the external storage.

The slice manager can be viewed as a state machine, since it only manages one slice of data inside the RAM at any given time. Below lists some of the most important interface for the slice manager:

- `void load(unsigned k)`: This function loads the `k`-th slice into the RAM.
- `void alloc(unsigned k)`: This function allocates an empty slice data inside the RAM.
- `void dump()`: This function writes the current slice data back to the external storage.
- `void numSlices()`: This function returns the number of the slices the dataset contains. 
- `void numChunks()`: This function returns the number of the chunks the dataset are divided into.
- `std::pair<unsigned, unsigned> sliceRange(unsigned i)`: This function returns a pair of the beginning and the ending index of the slices for the `i`-th chunk.
- `void storeID(unsigned x, unsigned y, unsigned ID)`: This function stores the object ID of the voxel at `(x, y)`.
- `void storeBirth(unsigned x, unsigned y, unsigned b)`: This function stores the birth (for persistence filtering) of the voxel at `(x, y)`
- `void storeRecBits(unsigned x, unsigned y, RecBitsType bits)`: This functions stores the meta intermediate states of the voxel at `(x, y)`. 
- `void beginOneThinningStep()`: This function can be used as an intialization step before each thinning step.
- `void endOneThinningStep()`: This function can be used as a postprocessing step after each thinning step.
- `void swapGroup()`: This function swaps the dataset contained in the two groups, so that the dataset in the output group will become the input in the next iteration. (Recall that the chunk-wise thinning algorithm stores the output result of a chunk in a separate group from the input.)

-

`h5_io::H5SliceIoManager` compacts multiple slices into one HDF5 file to reduce the IO time. Relative index is used for each slice within one HDF5 file. For instance, if there are $5$ slices in one chunk, then the datasets in this file are named as $0, 1, 2, 3, 4$. 

If a slice is not empty, then the assoicated dataset will be a $3 \times N$ matrix, where $N$ is the number of voxels in the dataset. The first row stores the flat coordinate `x + y * width`, the second row stores the object ID and the third one stores the combination of the birth and the intermediate states for each voxel. Otherwise if the slice if empty, **it is still associated with a dataset of size $1\times 1$**, which is used as a placeholder.

```
# sample output of a nonempty slice "0"
# in "1.h5" after calling "h5dump"

HDF5 "1.h5" {
GROUP "/" {
	DATASET "0" {
		DATATYPE H5T_STD_U32LE
		DATASPACE SIMPLE { ( 3, 10 ) / ( 3, 10 ) }
		DATA {
		(0,0) : # -- flat coordinate --
		(0,6) : # -- flat coordinate --
		(1,0) : # -- object ID --
		(1,6) : # -- object ID --
		(2,0) : # -- (birth << 8) | intermediate states --
		(2,6) : # -- (birth << 8) | intermediate states --
		}
	}
	# data for the rest slices
}
}

# sample output of an empty slice "2"
# in "0.h5" after calling "h5dump"

HDF5 "0.h5" {
GROUP "/" {
	DATASET "0" # data for slice 0
	DATASET "1" # data for slice 1
	DATASET "2" {
		DATATYPE H5T_STD_U32LE
		DATASPACE SIMPLE { ( 1, 1 ) / ( 1, 1 ) }
		DATA {
		(0,0) : 0
		}
	}
}
}
```

During the thinning process, the number of voxels in each slice will keep decreasing. To balance the work load in each chunk, the manager will constantly merge smaller chunks into one HDF5 file so that the number of voxels among different chunks stays approximately the same. Because of this feature, the manager should be able to know how many chunks there are in the dataset, and which slices each chunk covers ahead of the thinning. As a result, an additional configuration file used to describe the information about each chunk is **stored in the same folder with the actual dataset in the old group**. It is a `.txt` file, where each line stores a pair of number, specifying the beginning and the ending index of the slices at this chunk. The number of the chunks is equal to the total number of lines in the file.

```
# Sample chunk configuration file

0 12
12 17
17 20

# The first chunk covers slices [0, 12)
# The second chunk covers slices [12, 17)
# The third chunk covers slices [17, 20)
# Hence, there are a total of 3 chunks and 20 slices in the dataset.
```

```
// @prefix: Dataset chunks are indexed sequentially starting from 0. 
//		This string is the prefix of the index of the chunk filename
//		(the suffix is always ".h5"). As an example, if the filename 
//		follows the pattern of "data_N.h5", where "N" is the index, 
//		then the prefix string should be "data_".
// @oldGroup: Relative path to the old group.
// @newGroup: Relative path to the new group.
// @w: Width of each slice.
// @h: Height of each slice.
// @numSlices: Total number of slices of the dataset.
// @mapFilename: Filename of the chunk range mapping file, 
//		should be inside the folder specified by @oldGroup.
// @maxNumVoxels: An approximate upperbound of the number of voxels 
//		each chunk should have. Voxel data of multiple slices will 
//		be stored in one chunk until the number of voxels in the 
//		chunk exceeds this number.

h5_io::H5SliceIoManager(
	const std::string& prefix,
	const std::string& oldGroup,
	const std::string& newGroup,
	unsigned w,
	unsigned h,
	unsigned numSlices,
	const std::string& mapFilename,
	unsigned maxNumVoxels
);
```

### Sample Code

```
# Assuming that there are 256 slices separated in 16 chunks.
# The resolution of each slice is 256 x 256.
#
# File structure
<folder>
	|
	|-- "main.cpp"
	|
	|-- "old"
	|	  |
	|	  |-- "data_0.h5"
	|	  |-- "data_1.h5"
	|	  |-- ...
	|	  |-- "data_15.h5"
	|	  |-- "chunkMap.txt" # chunk mapping file is in the same folder as chunk data file. 
	|
	|-- "new"
```

```
// main.cpp
#include "thinning.h"
#include "h5_io.h"
int main()
{
	unsigend width = 256U height = 256U;
	unsigned numSlices = 256U;
	thin::IjkType size3D = thin::makeIjk(width, height, numSlices);
	unsigned maxNumVoxelsPerChunk = 1000000U;
	
	h5_io::H5SliceManager sliceMngr(
			"data_", "old", "new", 
			width, height, numSlices, 
			"chunkMap.txt", maxNumVoxelsPerChunk);
	// Each thinning step is determined by both the current iteration
	// and the current dimension: (curIter, curDim).
	//
	// (curIter, 3) -> (curIter, 2) -> (curIter, 1) -> (curIter, 0) ->
	// (curIter + 1, 3) -> ... -> (maxIter - 1, 0)
	unsigned curIter = 0, curDim = 3U, maxIter = 50U;
	unsigned p = 10U;
	
	thin::initDevice();
	thin::chunkwiseThinning(sliceMngr, size3D, curIter, curDim, p, maxIter);
	thin::shutdownDevice();
	
	return 0;
}
```
### Reference

[1] Bertrand, Gilles, and Michel Couprie. "Isthmus based parallel and symmetric 3D thinning algorithms." Graphical Models 80 (2015): 1-15.