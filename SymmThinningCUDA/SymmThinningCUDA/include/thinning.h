#ifndef THINNING_H
#define THINNING_H

#include <vector>
#include <thread>

#include "thinning_base.cuh"
#include "clique.cuh"

namespace thin
{
	// Initialize the device side resources for this module to work.
	void initDevice();
	// Release the device side resources for this module to work.
	void shutdownDevice();

	void setNumThreadsPerBlock(unsigned num);
	unsigned numThreadsPerBlock();
	// Run the original symmetric parallel thinning algorithm on the dataset in
	// `compactIjkVec`. The output crucial voxels are stored in `D_XK`.
	void isthmusSymmetricThinning(const std::vector<IjkType>& compactIjkVec, std::vector<IjkType>& D_XK, const IjkType& size3D, int maxIter = -1);

	// persistence thinning algorithm core
	void persistenceIsthmusThinningCore(details::DevDataPack& thinData, unsigned curIter, unsigned p, int maxIter);
	// Run the symmetric parallel thinning algorithm with persistence on the dataset
	// in `compactIjkVec`. `voxelIdVec` will store the object ID for each
	// corresponding voxel. The output crucial voxels are stored in `D_XK`.
	//
	// [param] size3D: the size in each dimension x, y and z
	// [param] p: the threshold of persistence
	// [param] maxIter: maximal number of thinning iterations. Set -1 means infinity.
	void persistenceIsthmusThinning(const std::vector<IjkType>& compactIjkVec, const std::vector<ObjIdType>& voxelIdVec, std::vector<IjkType>& D_XK, const IjkType& size3D, unsigned p = 25U, int maxIter = -1);
	// Run the symmetric parallel thinning algorithm with persistence on the dataset
	// in `compactIjkVec`. These voxels will be assumed to belong to a single
	// object. The output crucial voxels are stored in `D_XK`.
	//
	// [param] size3D: the size in each dimension x, y and z
	// [param] p: the threshold of persistence
	// [param] maxIter: maximal number of thinning iterations. Set -1 means infinity.
	void persistenceIsthmusThinning(const std::vector<IjkType>& compactIjkVec, std::vector<IjkType>& D_XK, const IjkType& size3D, unsigned p = 25U, int maxIter = -1);


	// Using the chunk-wise thinning requires the user to provide a chunk IO manager
	// object that implements the required interface:
	//
	// Interface of chunk IO manager:
	//
	//   - void dump()
	//      This function dumps the data of the slice currently managed to the disk.
	//
	//   - bool load(unsinged i)
	//      This function loads the `i`-th slice from disk into RAM. It is the
	//      user's choice to dump the data of the previous slice being managed,
	//      cache it and delay the dumping or other behavior.
	//      [postcondition] The manager now manages the data for `i`-th slice.
	//
	//   - void alloc(unsigned i)
	//      This function allocates an empty data space for the `i`-th slice. It is
	//      the user's choice to dump the data of the previous slice being managed,
	//      cache it and delay the dumping or other behavior.
	//      [postcondition] The manager now managers the data for `i`-th slice.
	//
	//	 - unsigned numSlices() const
	//		Return the total number of slices in the dataset.
	//
	//	 - unsigned numChunks() const
	//		Return the total number of chunks the dataset has been divided into.
	//
	//	 - std::pair<unsigned, unsigned> sliceRange(unsigned i) const
	//		Return a pair of the beginning and the ending index of the slices
	//		covered by the @i-th chunk.
	//
	//   - unsigned slice() const
	//      Return the index of the slice currently being managed.
	//
	//   - void storeID(unsigned x, unsigned y, unsigned ID)
	//      Stores the object ID `ID` of the voxel at coordinate (`x`, `y`) in the
	//      current slice. This function does not need the z coordinate because it
	//      can be infered from the index of the current slice.
	//
	//   - void storeBirth(unsigned x, unsigned y, unsigned birth)
	//      Stores the birth date `birth` of the voxel at coordinate (`x`, `y`).
	//
	//   - void swapGroup() const
	//      Swaps the contents in new and old group folder on disk.
	//
	//  ///// GeneralIoPolicy Only /////
	//
	//   - void storeIn_K(unsigned x, unsigned y, uint8_t flag)
	//      Stores whether the voxel at (`x`, `y`) is in set K or not, indicated by
	//      `flag`.
	//
	//   - void storeIn_DXK(unsigned x, unsigned y, uint8_t flag)
	//      Stores whether the voxel at (`x`, `y`) is in set D(X, K) or not,
	//      indicated by `flag`.
	//
	//   - void storeIn_IXK(unsigned x, unsigned y, uint8_t flag)
	//      Stores whether the voxel at (`x`, `y`) is in set I(X, K, k=1) or not,
	//      indicated by `flag`.
	//
	//   - void storeIn_A(unsigned x, unsigned y, uint8_t flag)
	//      Stores whether the voxel at (`x`, `y`) is in set A or not, indicated by
	//      `flag`.
	//
	//   - void storeIn_B(unsigned x, unsigned y, uint8_t flag)
	//      Stores whether the voxel at (`x`, `y`) is in set B or not, indicated by
	//      `flag`.
	//
	//  ///// MatchRawIoPolicy Only /////
	//
	//   - void storeRecBits(unsigned x, unsigned y, uint8_t bits)
	//      Stores the recording bits `bits` indicating whether the voxel at (`x`,
	//      `y`) is in each of the set K, D(X, K), I(X, K, 1), A and B.
	//
	//  ///// Unique Interface Ends /////
	//
	//   - Range cached() const
	//      Returns a range object that iterates through the existing voxels in the
	//      current slice. This requires the manager to declare/define a Range class
	//     and an Iterator class. The interface of the Iterator also depends on the
	//     which policy this manager is designated to use.
	//
	//  Interface of Iterator
	//
	//   - unsigned x() const
	//      Returns the x component of the coordinate of the voxel this iterator
	//      points to.
	//
	//   - unsigned y() const
	//      Returns the y component of the coordinate of the voxel this iterator
	//      points to.
	//
	//   - unsigned ID() const
	//      Returns the object ID of the voxel this iterator points to.
	//
	//   - unsigned birth() const
	//      Returns the birth date of the voxel this iterator points to.
	//
	//   ///// GeneralIoPolicy Only /////
	//
	//   - void in_K() const
	//      Returns if the voxel this iterator points to is in set K.
	//
	//   - void in_DXK() const
	//      Returns if the voxel this iterator points to is in set D(X, K).
	//
	//   - void in_IXK() const
	//      Returns if the voxel this iterator points to is in set I(X, K, k=1).
	//
	//   - void in_A() const
	//      Returns if the voxel this iterator points to is in set A.
	//
	//   - void in_B() const
	//      Returns if the voxel this iterator points to is in set B.
	//
	// ///// MatchRawIoPolicy Only /////
	//
	//   - void recBits() const
	//      Returns the recording bits of the voxel this iterator points to.
	//
	//  ///// Unique Interface Ends /////
	//
	// The general IO policy could be used for most of the chunk IO managers as it
	// does not place any restrictions on how the user is going to implement the
	// chunk IO manager, as long as it provides the correct interface to
	// retrieve/store 3D coordinate, object ID, birth data of a voxel and which
	// set this voxel is currently in.
	struct GeneralIOPolicy { };
	// The raw matching IO policy assumes that the user's implementation of chunk IO
	// manager matches the underlying raw representation of recording bits used in
	// this thinning module. The IO manager still needs to provide the interface to
	// retrieve/store 3D coordinate, object ID and birth data. Instead of providing
	// an interface for every set that this voxel might be in, only a unioned
	// interface for the recording bits is required.
    struct MatchRawIOPolicy { };

	// This namespace provides the function template for interacting with chunk IO
	// manager.
	namespace chunk
	{
		namespace _private
		{
			namespace tp = thin::_private;
			// Load the data of the slice currently cached by `mngr` to `h_thinData`. This
			// function is invoked when using `GeneralIOPolicy`
			template <typename MNGR>
			void _load(const GeneralIOPolicy&, details::HostDataPack& h_thinData, const MNGR& mngr)
			{
				using namespace details;

				for (const auto& vxIter : mngr.cached())
				{
					h_thinData.compactIjkVec.push_back(makeIjk(vxIter.x(), vxIter.y(), mngr.slice()));
					h_thinData.voxelIdVec.push_back(vxIter.ID());

					RecBitsType bits;
					if (vxIter.in_K()) tp::_setBit(bits, REC_BIT_K);
					if (vxIter.in_DXK()) tp::_setBit(bits, REC_BIT_Y);
					if (vxIter.in_IXK()) tp::_setBit(bits, REC_BIT_Z);
					if (vxIter.in_A()) tp::_setBit(bits, HOST_REC_BIT_A);
					if (vxIter.in_B()) tp::_setBit(bits, HOST_REC_BIT_B);
					h_thinData.recBitsVec.push_back(bits);

					h_thinData.birthVec.push_back(vxIter.birth());
				}
			}
			// Load the data of the slice currently cached by `mngr` to `h_thinData`. This
			// function is invoked when using `MatchRawIOPolicy`.
			template <typename MNGR>
			void _load(const MatchRawIOPolicy&, details::HostDataPack& h_thinData, const MNGR& mngr)
			{
				for (const auto& vxIter : mngr.cached())
				{
					h_thinData.compactIjkVec.push_back(makeIjk(vxIter.x(), vxIter.y(), mngr.slice()));
					h_thinData.voxelIdVec.push_back(vxIter.ID());
					h_thinData.recBitsVec.push_back(vxIter.recBits());
					h_thinData.birthVec.push_back(vxIter.birth());
				}
			}
			// Load the slices from `beginSlice` to `endSlice` to `h_thinData` using the io
			// manager `mngr`. This function dispatches to the correct load function
			// according to the templated POLICY.
			template <typename POLICY, typename MNGR>
			void _loadSlices(const POLICY& policy, details::HostDataPack& h_thinData, MNGR& mngr, const unsigned beginSlice,
							 const unsigned endSlice)
			{
				for (unsigned sliceK = beginSlice; sliceK < endSlice; ++sliceK)
				{
					mngr.load(sliceK);
					_load(policy, h_thinData, mngr);
				}
			}
			// Load the entire chunk, which contains `chunkSize` slices, beginning at
			// `beginSlice` to `h_thinData`. `procBeginIndex`/`procEndIndex` stores the
			// beginning/ending index of the voxels in `h_thinData` for the thinning
			// algorithm to process. The remaining voxels are used as reference and not
			// touched in the thinning process.
			template <typename POLICY, typename MNGR>
			void _loadChunk(const POLICY& policy, details::HostDataPack& h_thinData, MNGR& mngr,
							unsigned beginSlice, const unsigned chunkSize, const unsigned numSlices,
							unsigned& procBeginIndex, unsigned& procEndIndex)
			{
				// mngr.beginLoadChunk();

				h_thinData.clear();
				// Compute the begin/end index of the BOTTOM reference slice.
				unsigned refBeginSlice = beginSlice < 2U ? 0 : beginSlice - 2U;
				unsigned refEndSlice = beginSlice;
        
				_loadSlices(policy, h_thinData, mngr, refBeginSlice, refEndSlice);
        
				procBeginIndex = (unsigned)h_thinData.compactIjkVec.size();
				// Compute the end index of the slice for thinning.
				unsigned endSlice = beginSlice + chunkSize;
				endSlice = endSlice < numSlices ? endSlice : numSlices;
				// Load all those slices for thinning.
				_loadSlices(policy, h_thinData, mngr, beginSlice, endSlice);
        
				procEndIndex = (unsigned)h_thinData.compactIjkVec.size();
				// Compute the begin/end index of the TOP reference slice
				refBeginSlice = endSlice;
				refEndSlice = refBeginSlice + 2U;
				refEndSlice = refEndSlice < numSlices ? refEndSlice : numSlices;
				// Load the two top reference slices
				_loadSlices(policy, h_thinData, mngr, refBeginSlice, refEndSlice);

				// mngr.endLoadChunk();

				assert(h_thinData.compactIjkVec.size() == h_thinData.voxelIdVec.size());
				assert(h_thinData.compactIjkVec.size() == h_thinData.recBitsVec.size());
				assert(h_thinData.compactIjkVec.size() == h_thinData.birthVec.size());
			}

			template <typename MNGR>
			void _loadChunkI(details::HostDataPack& h_thinData, MNGR& mngr,
							unsigned chunkI, const unsigned numSlices,
							unsigned& procBeginIndex, unsigned& procEndIndex)
			{
				std::pair<unsigned, unsigned> chunkRange = mngr.chunkRange(chunkI);
				unsigned chunkSize = chunkRange.second - chunkRange.first;

				_loadChunk(MatchRawIOPolicy(), h_thinData, mngr, chunkRange.first, chunkSize, numSlices, procBeginIndex, procEndIndex);
			}

			// Stores the data of the voxel in `h_thinData` pointed by `index` into the
			// slice current being managed by `mngr`. This function is invoked when using
			// `GeneralIOPolicy`.
			template <typename MNGR>
			void _store(const GeneralIOPolicy&, const details::HostDataPack& h_thinData,
						MNGR& mngr, unsigned index)
			{
				using namespace details;

				unsigned x, y;
				x = h_thinData.compactIjkVec[index].x;
				y = h_thinData.compactIjkVec[index].y;
        
				mngr.storeID(x, y, h_thinData.voxelIdVec[index]);
        
				RecBitsType bits = h_thinData.recBitsVec[index];
				mngr.storeIn_K(x, y, tp::_readBit(bits, REC_BIT_K));
				mngr.storeIn_DXK(x, y, tp::_readBit(bits, REC_BIT_Y));
				mngr.storeIn_IXK(x, y, tp::_readBit(bits, REC_BIT_Z));
				mngr.storeIn_A(x, y, tp::_readBit(bits, HOST_REC_BIT_A));
				mngr.storeIn_B(x, y, tp::_readBit(bits, HOST_REC_BIT_B));
        
				mngr.storeBirth(x, y, h_thinData.birthVec[index]);
			}
			// Stores the data of the voxel in `h_thinData` pointed by `index` into the
			// slice current being managed by `mngr`. This function is invoked when using
			// `MatchRawIOPolicy`.
			template <typename MNGR>
			void _store(const MatchRawIOPolicy&, const details::HostDataPack& h_thinData,
						MNGR& mngr, unsigned index)
			{
				unsigned x, y;
				x = h_thinData.compactIjkVec[index].x;
				y = h_thinData.compactIjkVec[index].y;
        
				mngr.storeID(x, y, h_thinData.voxelIdVec[index]);
				mngr.storeRecBits(x, y, h_thinData.recBitsVec[index]);
				mngr.storeBirth(x, y, h_thinData.birthVec[index]);
			}
			// Dump the chunk of slices to disk. This chunk includes voxels from
			// `procBeginIndex` and `procEndIndex` in `h_thinData`.
			template <typename POLICY, typename MNGR>
			void _dumpChunk(const POLICY& policy, const details::HostDataPack& h_thinData, MNGR& mngr,
							const unsigned procBeginIndex, const unsigned procEndIndex)
			{
				// mngr.beginDumpChunk();

				unsigned index = procBeginIndex;
				while (index < procEndIndex)
				{
					unsigned curSlice = h_thinData.compactIjkVec[index].z;
					mngr.alloc(curSlice);
					// The loop condition ensures that all the voxels in the loop belongs to the
					// same slice.
					while (index < procEndIndex && (h_thinData.compactIjkVec[index].z == curSlice))
					{
						if (h_thinData.recBitsVec[index])
						{
							_store(policy, h_thinData, mngr, index);
						}
						++index;
					}
            
					mngr.dump();
				}

				// mngr.endDumpChunk();
			}

			template <typename MNGR>
			inline void _dumpChunk(const details::HostDataPack& h_thinData, MNGR& mngr,
							const unsigned procBeginIndex, const unsigned procEndIndex)
			{
				_dumpChunk(MatchRawIOPolicy(), h_thinData, mngr, procBeginIndex, procEndIndex);
			}
		}; // namespace thin::chunk::_private;
	}; // namespace thin::chunk;
	
	// thinning algorithm one a single chunk.
	// [param] curIter: current iteration
	// [param] dim: current dimension (rank) of the cliques to detect
	void oneChunkThinning(details::DevDataPack& thinData, unsigned curIter, unsigned dim, 
								unsigned p, const dim3& blocksDim, const dim3& threadsDim);

	// Run the thinning on all the chunks sequentially at iteration `curIter` for
	// dimension `dim`.
	// template <typename POLICY, typename MNGR>
	template <typename MNGR>
    static void allChunksThinningOneStep(MNGR& mngr, const IjkType& size3D, unsigned curIter, unsigned dim, unsigned p)
    {
		using namespace details;
		namespace chp = chunk::_private;
        
		unsigned numChunks = mngr.numChunks();
		if (numChunks == 0) return;
        
		unsigned numSlices = mngr.numSlices();

		HostDataPack h_thinData;
		HostDataPack h_thinDataBuffer;
        
        // unsigned beginSlice = 0;
        bool hasOutput = false;
        
        unsigned procBeginIndex, procEndIndex;
        unsigned lastProcBeginIndex, lastProcEndIndex;
        unsigned nextProcBeginIndex, nextProcEndIndex;
        
		mngr.beginOneThinningStep();
        // chp::_loadChunk(policy, h_thinData, mngr, beginSlice, chunkSize, numSlices, procBeginIndex, procEndIndex);
        chp::_loadChunkI(h_thinData, mngr, 0, numSlices, procBeginIndex, procEndIndex);

		std::cout << "Cur iter: " << curIter << ", dim: " << dim << std::endl;
        for (unsigned chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx)
        {
            // beginSlice = chunkIdx * chunkSize;
            std::thread ioThread([&]()
            {
                if (hasOutput)
                {
                    chp::_dumpChunk(h_thinDataBuffer, mngr, lastProcBeginIndex, lastProcEndIndex);
                }
                // chp::_loadChunk(policy, h_thinDataBuffer, mngr, beginSlice + chunkSize, chunkSize, numSlices,
                //           nextProcBeginIndex, nextProcEndIndex);

				if (chunkIdx + 1U < numChunks)
				{
					chp::_loadChunkI(h_thinDataBuffer, mngr, chunkIdx + 1U, numSlices, nextProcBeginIndex, nextProcEndIndex);
				}
            });
            
			DevDataPack::InitParams packInitParams;
			packInitParams.arrSize = h_thinData.compactIjkVec.size();
			packInitParams.size3D = size3D;
			packInitParams.useBirth = true;
			packInitParams.useVoxelID = true;

			DevDataPack thinData(packInitParams);
			thinData.alloc();
			
			_copyThinningDataToDevice(thinData, h_thinData);
			
			thinData.procBeginIndex = procBeginIndex;
			thinData.procEndIndex = procEndIndex;

			std::cout << "  chunk: " << chunkIdx << ", size: " << thinData.arrSize 
					<< ", proc begin: " << procBeginIndex << ", proc end: " << procEndIndex << std::endl;

			dim3 threadsDim(numThreadsPerBlock(), 1U, 1U);
			dim3 blocksDim((thinData.arrSize + threadsDim.x - 1U) / threadsDim.x, 1U, 1U);

			while (blocksDim.x > 32768U)
			{
				blocksDim.x /= 2U;
				blocksDim.y *= 2U;
			}

            oneChunkThinning(thinData, curIter, dim, p, blocksDim, threadsDim);
            _copyThinningDataToHost(h_thinData, thinData);
			thinData.dispose();

            hasOutput = true;
            ioThread.join();
            
            h_thinData.swap(h_thinDataBuffer);
            
            lastProcBeginIndex = procBeginIndex;
            lastProcEndIndex = procEndIndex;
            
            procBeginIndex = nextProcBeginIndex;
            procEndIndex = nextProcEndIndex;
        }
        
        chp::_dumpChunk(h_thinDataBuffer, mngr, lastProcBeginIndex, lastProcEndIndex);
        
		mngr.endOneThinningStep();
        mngr.swapGroup();
    }

	// template <typename POLICY, typename MNGR>
	// void chunkwiseThinning(const POLICY& policy, MNGR& mngr, const unsigned numSlices,
	template <typename MNGR>
	void chunkwiseThinning(MNGR& mngr, const IjkType& size3D, unsigned curIter, unsigned dim, unsigned p, unsigned maxIter)
	{
		bool ramThinning = (mngr.numChunks() == 1);
		while ((curIter < maxIter) && (!ramThinning))
		{
			while (true)
			{
				// fullThinningOnce(policy, mngr, chunkSize, numSlices, size3D, curIter, dim, p);
				allChunksThinningOneStep(mngr, size3D, curIter, dim, p);

				if (dim == 0)
				{
					break;
				}
				--dim;
			}

			dim = 3U;
			++curIter;
			
			// stop using chunk-wise thinning
			ramThinning = (mngr.numChunks() == 1);
		}
		// run thinning using RAM
		if (ramThinning)
		{
			std::cout << "using RAM at iter: " << curIter << std::endl;

			using namespace details;
			namespace chp = chunk::_private;
			HostDataPack h_thinData;
			unsigned procBeginIndex, procEndIndex;

			mngr.beginOneThinningStep();
			chp::_loadChunkI(h_thinData, mngr, 0, mngr.numSlices(), procBeginIndex, procEndIndex);

			DevDataPack::InitParams packInitParams;
			packInitParams.arrSize = h_thinData.compactIjkVec.size();
			packInitParams.size3D = size3D;
			packInitParams.useBirth = true;
			packInitParams.useVoxelID = true;

			DevDataPack thinData(packInitParams);
			thinData.alloc();
			
			_copyThinningDataToDevice(thinData, h_thinData);
			persistenceIsthmusThinningCore(thinData, curIter, p, maxIter);

			h_thinData.clear();
			procEndIndex = thinData.arrSize;
			h_thinData.resize(thinData.arrSize);
			_copyThinningDataToHost(h_thinData, thinData);

			thinData.dispose();
			chp::_dumpChunk(h_thinData, mngr, procBeginIndex, procEndIndex);

			mngr.endOneThinningStep();
			mngr.swapGroup();
		}
	}

	// Run the thinning on all the chunks sequentially at iteration `curIter` for
	// dimension `dim`.
	/*template <typename POLICY, typename MNGR>
    static void fullThinningOnce(const POLICY& policy, MNGR& mngr,
                          const unsigned chunkSize, const unsigned numSlices,
                          const IjkType& size3D, unsigned curIter, unsigned dim, unsigned p)
    {
		using namespace details;
		namespace chp = chunk::_private;

        unsigned numChunks = (numSlices + chunkSize - 1U) / chunkSize;
        
		HostDataPack h_thinData;
		HostDataPack h_thinDataBuffer;
        
        unsigned beginSlice = 0;
        bool hasOutput = false;
        
        unsigned procBeginIndex, procEndIndex;
        unsigned lastProcBeginIndex, lastProcEndIndex;
        unsigned nextProcBeginIndex, nextProcEndIndex;
        
        chp::_loadChunk(policy, h_thinData, mngr, beginSlice, chunkSize, numSlices, procBeginIndex, procEndIndex);
        
		std::cout << "Cur iter: " << curIter << ", dim: " << dim << std::endl;
        for (unsigned chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx)
        {
            beginSlice = chunkIdx * chunkSize;
            std::thread ioThread([&]()
            {
                if (hasOutput)
                {
                    chp::_dumpChunk(policy, h_thinDataBuffer, mngr, lastProcBeginIndex, lastProcEndIndex);
                }
                chp::_loadChunk(policy, h_thinDataBuffer, mngr, beginSlice + chunkSize, chunkSize, numSlices,
                           nextProcBeginIndex, nextProcEndIndex);
            });
            
			DevDataPack::InitParams packInitParams;
			packInitParams.arrSize = h_thinData.compactIjkVec.size();
			packInitParams.size3D = size3D;
			packInitParams.useBirth = true;
			packInitParams.useVoxelID = true;

			DevDataPack thinData(packInitParams);
			thinData.alloc();
			
			_copyThinningDataToDevice(thinData, h_thinData);
			
			thinData.procBeginIndex = procBeginIndex;
			thinData.procEndIndex = procEndIndex;

			std::cout << "  chunk: " << chunkIdx << ", size: " << thinData.arrSize 
					<< ", proc begin: " << procBeginIndex << ", proc end: " << procEndIndex << std::endl;

			dim3 threadsDim(numThreadsPerBlock(), 1U, 1U);
			dim3 blocksDim((thinData.arrSize + threadsDim.x - 1U) / threadsDim.x, 1U, 1U);

			while (blocksDim.x > 32768U)
			{
				blocksDim.x /= 2U;
				blocksDim.y *= 2U;
			}

            oneChunkThinning(thinData, curIter, dim, p, blocksDim, threadsDim);
            _copyThinningDataToHost(h_thinData, thinData);
			thinData.dispose();

            hasOutput = true;
            ioThread.join();
            
            h_thinData.swap(h_thinDataBuffer);
            
            lastProcBeginIndex = procBeginIndex;
            lastProcEndIndex = procEndIndex;
            
            procBeginIndex = nextProcBeginIndex;
            procEndIndex = nextProcEndIndex;
        }
        
        chp::_dumpChunk(policy, h_thinDataBuffer, mngr, lastProcBeginIndex, lastProcEndIndex);
        
        mngr.swapGroup();
    }*/
}; // namespace thin;
#endif