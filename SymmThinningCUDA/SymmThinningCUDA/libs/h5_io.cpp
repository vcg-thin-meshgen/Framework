#include "h5_io.h"
#include "io_shared.h"

namespace h5_io
{
	H5SliceIoManager::H5SliceIoManager(const std::string& prefix, const std::string& oldG, const std::string& newG,
					unsigned w, unsigned h, unsigned loadChkSz, unsigned dumpChkSz)
					: m_filePrefix(prefix), m_oldGroupname(oldG), m_newGroupname(newG)
					, m_width(w), m_height(h), m_curSlice(~0)
					, m_curChunk(~0), m_curChunkBeginSlice(0), m_curChunkEndSlice(0)
					, m_loadChunkSize(loadChkSz), m_dumpChunkSize(dumpChkSz), m_isH5FileOpen(false)
					, m_isLoading(false), m_isDumping(false)
	{
		_clear();
		_resetDumpedDatasetFlags();
	}

	bool H5SliceIoManager::load(unsigned slice)
	{
		m_curSlice = slice;

		// If either the H5 file is not yet open or the passed in
		// slice is out of the range of current chunk
		if (!(m_isH5FileOpen && _isSliceInRange(slice)))
		{
			// 1. close current H5 file
			_closeH5File();
			// 2. update curChunkSliceBegin/End
			_updateChunkRange(m_loadChunkSize);
			// 3. open the correct H5 file storing the current chunk data
			std::string h5filename = _getH5Filename(m_oldGroupname);
			_openH5File(h5filename, H5F_ACC_RDONLY);
		}
		// load data directly from current H5 file
		_loadFromDataset();

		return true;
	}

	void H5SliceIoManager::alloc(unsigned slice)
	{
		m_curSlice = slice;

		_clear();
	}

    void H5SliceIoManager::dump()
	{
		// If either the H5 file is not yet open or the passed in
		// slice is out of the range of current chunk
		if (!(m_isH5FileOpen && _isSliceInRange(m_curSlice)))
		{
			// 1. close current H5 file
			_dumpAndCloseH5File();

			_updateChunkRange(m_dumpChunkSize);

			std::string h5filename = _getH5Filename(m_newGroupname);
			_openH5File(h5filename, H5F_ACC_TRUNC);
		}
		// store the current slice data into a new dataset
		_writeToDataset();
	}

	void H5SliceIoManager::beginLoadChunk()
	{
		if (m_isDumping) throw "cannot start loading while dumping";
		if (m_isLoading) throw "cannot start a new loading process";

		_closeH5File();
		// _resetChunkSliceRange();

		m_isLoading = true;
	}

	void H5SliceIoManager::endLoadChunk()
	{
		_closeH5File();
		// _resetChunkSliceRange();

		m_isLoading = false;
	}

	void H5SliceIoManager::beginDumpChunk()
	{
		if (m_isLoading) throw "cannot start dumping while loading";
		if (m_isDumping) throw "cannot start a new dumping process";

		m_isDumping = true;
		// It is possible that we may want to start dumping while 
		// the H5 file is still open for loading process. Therefore, 
		// we need to close the file first.
		_closeH5File();
		// Again, because we may want to start dumping while the H5
		// file was previously opened for loading, we need to reset
		// the slice range information of the current chunk.
		// _resetChunkSliceRange();

		// Reset
		_resetDumpedDatasetFlags();
	}

	void H5SliceIoManager::endDumpChunk()
	{
		// For empty slices that have been skipped, we need to create a dummy slice
		// so that the Dataset still exists in the H5 file!
		for (unsigned localSliceIndex = 0; localSliceIndex < m_dumpedDatasetFlags.size(); ++localSliceIndex)
		{
			if (m_dumpedDatasetFlags[localSliceIndex] == 0)
			{
				unsigned emptyGlobalSliceIndex = localSliceIndex + m_curChunkBeginSlice;
				alloc(emptyGlobalSliceIndex);
				dump();
				m_dumpedDatasetFlags[localSliceIndex] = 1U;
			}
		}

		m_isDumping = false;
		// make the actual dump happen here
		_dumpAndCloseH5File();

		_resetDumpedDatasetFlags();
		// _resetChunkSliceRange();
	}

    void H5SliceIoManager::storeID(unsigned x, unsigned y, unsigned ID)
	{
		unsigned flattenIjk = _getFlattenIjk(x, y);

		if (m_flattenIjkVec.size() && (flattenIjk <= m_flattenIjkVec.back()))
		{
			throw "not following ascending order";
		}
		// piggyback the operation of storing flatten ijk in storeID function
		m_flattenIjkVec.push_back(flattenIjk);

		m_voxelIdVec.push_back(ID);
	}

    void H5SliceIoManager::storeBirth(unsigned, unsigned, unsigned birth)
	{
		m_birthVec.push_back(birth);
	}
        
    void H5SliceIoManager::storeRecBits(unsigned, unsigned, RecBitsType bits)
	{
		m_recBitsVec.push_back(bits);
	}
        
    void H5SliceIoManager::swapGroup() const
	{ 
		io_shared::swapGroupFiles(m_oldGroupname, m_newGroupname);
	}

	void H5SliceIoManager::_clear() 
	{
		m_flattenIjkVec.clear();
		m_voxelIdVec.clear();
		m_recBitsVec.clear();
		m_birthVec.clear();
	}

	void H5SliceIoManager::_updateChunkRange(unsigned chunkSize)
	{
		if (m_isH5FileOpen) throw "cannot update chunk range while H5 file is still open";

		m_curChunk = m_curSlice / chunkSize;
		m_curChunkBeginSlice = m_curChunk * chunkSize;
		m_curChunkEndSlice = m_curChunkBeginSlice + chunkSize;
	}

	void H5SliceIoManager::_openH5File(const std::string& filename, unsigned flag)
	{
		if (m_isH5FileOpen) throw "cannot open a new H5 file while the current is still open";

		m_curH5File = H5::H5File(filename, flag);
		m_isH5FileOpen = true;
	}
		
	void H5SliceIoManager::_closeH5File()
	{
		if (m_isH5FileOpen)
		{
			m_curH5File.close();
			m_isH5FileOpen = false;

			_resetChunkSliceRange();
		}
	}

	std::string H5SliceIoManager::_getH5Filename(const std::string& group) const
	{
		std::stringstream ss;
		ss << group << "\\" << m_filePrefix << m_curChunk << ".h5";
		return ss.str();
	}

	void H5SliceIoManager::_loadFromDataset()
	{
		using namespace H5;

		_clear();

		std::string dsetName = _getDatasetName();
		DataSet dataset;
		try
		{
			dataset = m_curH5File.openDataSet(dsetName);
		}
		catch( DataSetIException error )
		{
			error.printError();
		}
		// To save us the trouble from detecting whether a dataset exists,
		// For empty slice, we let the dimension to be 1x1.
		// For all the rest slices that do contain voxels, the dimension
		// should always be Nx3, where N denotes the number of voxels.
		//
		// Three rows of data:
		// first row	: voxel flattend ijk
		// second row	: voxel object ID
		// third row	: voxel birth and recording bits
		//					[31 - 24] bits: unused | [23 - 8] bits: birth | [7 - 0] bits: recording bits
		DataSpace dataspace = dataset.getSpace();
		hsize_t dims[2U];
		dataspace.getSimpleExtentDims(dims);

		if (dims[0] == 3U)
		{
			unsigned numVoxels = (unsigned)dims[1U];
			unsigned size1D = (unsigned)dims[0] * numVoxels;
			// load all the data at once
			std::vector<unsigned> tmpDataVec(size1D, 0);
			dataset.read(tmpDataVec.data(), PredType::STD_U32LE);
			// copy the flatten ijk part
			m_flattenIjkVec.resize(numVoxels, 0);
			std::copy(tmpDataVec.begin(), tmpDataVec.begin() + numVoxels, m_flattenIjkVec.begin());
			// copy the voxel object ID part
			m_voxelIdVec.resize(numVoxels, 0);
			std::copy(tmpDataVec.begin() + numVoxels, tmpDataVec.begin() + 2U * numVoxels, m_voxelIdVec.begin());
			// copy the packed birth and recording bits part
			m_recBitsVec.resize(numVoxels, 0);
			m_birthVec.resize(numVoxels, 0);
			// unpack birth and recording bits
			unsigned offs = 2U * numVoxels;
			for (unsigned i = 0; i < numVoxels; ++i)
			{
				m_recBitsVec[i] = (RecBitsType)(tmpDataVec[i + offs] & 0xff);
				m_birthVec[i] = (tmpDataVec[i + offs] >> 8U) & 0xffff;
			}
		}

		dataspace.close();
		dataset.close();
	}

	void H5SliceIoManager::_writeToDataset()
	{
		using namespace H5;

		hsize_t dims[2U];
		std::vector<unsigned> tmpDataVec;
			
		if (m_flattenIjkVec.size() == 0)
		{
			// empty slice, make dim = (1, 1)
			dims[0] = 1U; dims[1U] = 1U;
			tmpDataVec.push_back(0);
		}
		else
		{
			unsigned numVoxels = m_flattenIjkVec.size();

			dims[0] = 3U; dims[1U] = numVoxels;
			tmpDataVec.resize(numVoxels * 3U, 0);

			std::copy(m_flattenIjkVec.begin(), m_flattenIjkVec.end(), tmpDataVec.begin());
			std::copy(m_voxelIdVec.begin(), m_voxelIdVec.end(), tmpDataVec.begin() + numVoxels);
			// pack birth and recording bits together
			unsigned offs = 2U * numVoxels;
			for (unsigned i = 0; i < numVoxels; ++i)
			{
				tmpDataVec[i + offs] = (unsigned)m_recBitsVec[i];
				tmpDataVec[i + offs] |= (m_birthVec[i] << 8U);
			}
		}
		// rank is always 2
		DataSpace dataspace(2U, dims);
		IntType datatype(PredType::STD_U32LE);
		// write to dataset
		std::string dsetName = _getDatasetName();
		DataSet dataset;
		try
		{
			dataset = m_curH5File.createDataSet(dsetName, datatype, dataspace);
		}
		catch( DataSetIException error )
		{
			error.printError();

		}
		dataset.write(tmpDataVec.data(), datatype);

		dataset.close();
		datatype.close();
		dataspace.close();
			
		unsigned localSliceIndex = _getLocalSliceIndex();
		m_dumpedDatasetFlags[localSliceIndex] = 1U;

		// std::cout << "dumped local slice: " << localSliceIndex << std::endl;
		// for (unsigned ii = 0; ii < m_dumpedDatasetFlags.size() ; ++ii)
		// {
		//	std::cout << "  [" << ii << "] " << m_dumpedDatasetFlags[ii] << std::endl;
		// }
		// std::cout << std::endl;

		_clear();
	}
}; // namespace h5_io;