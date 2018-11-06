#ifndef NH5_SLICE_IO_H
#define NH5_SLICE_IO_H

#include <vector>
#include <unordered_map>
#include <algorithm>				// std::copy
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>

#include <H5Cpp.h>

#include "io_shared.h"

namespace h5_io
{
	class H5SliceIoManager
	{
	public:
		typedef unsigned ObjIDType;
		typedef unsigned char RecBitsType;

        H5SliceIoManager(const std::string& prefix, const std::string& oldGroup, const std::string& newGroup,
						unsigned w, unsigned h, unsigned numSlices, const std::string& mapFilename, unsigned maxNumVx)
		: m_filePrefix(prefix)
		, m_oldGroupname(oldGroup)
		, m_newGroupname(newGroup)
		, m_chunkRangeFilename(mapFilename)
		, m_width(w)
		, m_height(h)
		, m_numSlices(numSlices)
		, m_maxNumVoxelsInChunk(maxNumVx)
		, m_curSlice(~0)
		, m_curLoadChunk(~0)
		, m_curLoadChunkBeginSlice(~0)
		, m_curLoadChunkEndSlice(0)
		, m_isLoadFileOpen(false)
		, m_curDumpChunk(~0)
		, m_curDumpChunkBeginSlice(~0)
		, m_isDumpFileOpen(false)
		, m_lastDumpedSlice(~0)
		, m_numVoxelsDumpedInChunk(0) 
		, m_originalNumSlices(0)
		{ 
			m_loadChunkRangeMap.clear();

			std::stringstream ss;
			ss << m_oldGroupname << "\\" << m_chunkRangeFilename;
			
			std::ifstream fh(ss.str());
			std::string line;
			unsigned lo, hi;
			while (std::getline(fh, line))
			{
				std::stringstream lss(line);
				lss >> lo >> hi;
				m_loadChunkRangeMap.push_back(std::make_pair(lo, hi));
			}

			m_originalNumSlices = m_loadChunkRangeMap.back().second;
		}
        
        class Iterator;
        class Range;
        // Load a slice of data into RAM, which is the currently managed slice.
        bool load(unsigned slice)
		{
			m_curSlice = slice;

			if (!_isSliceInRange(m_curSlice))
			{
				_updateChunkRange(m_curSlice);
				// 3. open the correct H5 file storing the current chunk data
				std::string h5filename = _getH5Filename(m_oldGroupname, m_curLoadChunk);
				_openLoadH5File(h5filename);
			}

			_loadFromDataset();

			return true;
		}
		// Allocate an empty slice, whose index will be @slice.
        void alloc(unsigned slice)
		{
			m_curSlice = slice;

			_clear();
		}
		// Dump the currently managed onto the disk.
		// [invariant] It is IMPORTANT to know that the dump operation is NOT RANDOM ACCESSED.
		// Instead, it must be used in an INCREMENTAL way, meaning that once slice `k` is dumped,
		// no slice with an index <= k can be dumped, until @swapGroup() is called. The usage
		// is like the following: we allocate an empty slice using @alloc(), write the data
		// for this slice into the RAM, then @dump() the data and allocate RAM for the next slice.
        void dump()
		{
			if (!m_isDumpFileOpen)
			{
				if (m_curDumpChunk != (~0)) throw "wrong curDumpChunk";
				m_curDumpChunk = 0;
				m_curDumpChunkBeginSlice = 0;
				// m_lastDumpedSlice = 0;
				m_numVoxelsDumpedInChunk = 0;
				
				std::string h5filename = _getH5Filename(m_newGroupname, m_curDumpChunk);
				_createDumpH5File(h5filename);
				m_isDumpFileOpen = true;
			}
			// Althoug slices between (@m_lastDumpedSlices, m_curSlice) do not exist
			// we need to store an empty slice for each of them to conform to the 
			// protocal used by our HDF5 dataset format.
			unsigned m_curSliceCopy = m_curSlice;
			m_curSlice = (m_lastDumpedSlice == (~0)) ? 0 : (m_lastDumpedSlice + 1U);
			for (; m_curSlice < m_curSliceCopy; ++m_curSlice)
			{
				// force an empty slice, therefore ignore the data currently stored.
				_writeToDataset(true);
			}
			// save this slice
			m_curSlice = m_curSliceCopy;
			_writeToDataset(false);

			m_numVoxelsDumpedInChunk += m_flattenIjkVec.size();
			m_lastDumpedSlice = m_curSlice;
			_clear();

			// It is only when we have stored an amount of voxels >= @m_maxNumVoxelsInChunk in 
			// the RAM will we make the actual dump happen.
			if (m_numVoxelsDumpedInChunk >= m_maxNumVoxelsInChunk)
			{
				// dumpy current data
				m_dumpH5File.close();
				m_isDumpFileOpen = false;
				// update dump range map
				m_dumpChunkRangeMap.push_back(std::make_pair(m_curDumpChunkBeginSlice, m_lastDumpedSlice + 1U));
				m_curDumpChunk += 1U;
				m_curDumpChunkBeginSlice = m_lastDumpedSlice + 1U;
				m_numVoxelsDumpedInChunk = 0;
				// create the next chunk to dump
				std::string h5filename = _getH5Filename(m_newGroupname, m_curDumpChunk);
				_createDumpH5File(h5filename);
				m_isDumpFileOpen = true;
			}
		}

		inline unsigned numSlices() const { return m_numSlices; }
		inline unsigned numChunks() const { return m_loadChunkRangeMap.size(); }
		inline std::pair<unsigned, unsigned> chunkRange(unsigned i) const { return m_loadChunkRangeMap[i]; }

        inline Range cached() const { return Range(this); }
        inline unsigned slice() const { return m_curSlice; }

        void storeID(unsigned x, unsigned y, unsigned ID)
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

        void storeBirth(unsigned, unsigned, unsigned birth)
		{
			m_birthVec.push_back(birth);
		}
        void storeRecBits(unsigned, unsigned, RecBitsType bits)
		{
			m_recBitsVec.push_back(bits);
		}
		
		void beginOneThinningStep() { }
		void endOneThinningStep()
		{
			_clear();
			if (m_curSlice < (m_originalNumSlices - 1))
			{
				m_curSlice = m_originalNumSlices - 1;
				dump();
			}

			if (m_isLoadFileOpen) m_loadH5File.close();
			if (m_isDumpFileOpen) m_dumpH5File.close();

			_dumpRangeMap();

			m_curSlice = ~0;
			m_curLoadChunk = ~0;
			m_curLoadChunkBeginSlice = ~0;
			m_curLoadChunkEndSlice = 0;
		
			m_isLoadFileOpen = false;

			m_curDumpChunk = ~0;
			m_curDumpChunkBeginSlice = ~0;

			m_isDumpFileOpen = false;

			m_lastDumpedSlice = ~0;
			m_numVoxelsDumpedInChunk = 0;
		}

        void swapGroup()
		{ 
			io_shared::swapGroupFiles(m_oldGroupname, m_newGroupname);

			m_loadChunkRangeMap.swap(m_dumpChunkRangeMap);
			m_dumpChunkRangeMap.clear();
		}
        
        class Iterator
        {
        public:
            Iterator();
            
            const Iterator& operator*() const { return *this; }
            
            inline unsigned x() const
            {
				unsigned flattenIjk = m_mngr->m_flattenIjkVec[m_index];
				return flattenIjk % (m_mngr->m_width);
            }
            
            inline unsigned y() const
            {
				unsigned flattenIjk = m_mngr->m_flattenIjkVec[m_index];
				return flattenIjk / (m_mngr->m_width);
            }
            
            inline unsigned ID() const
            {
				unsigned id = m_mngr->m_voxelIdVec[m_index];
                return id;
            }
            
            inline RecBitsType recBits() const
            {
				RecBitsType bits = m_mngr->m_recBitsVec[m_index];
				return bits;
            }
            
            inline unsigned birth() const
            {
				unsigned b = m_mngr->m_birthVec[m_index];
                return b;
            }
            
            Iterator& operator++()
            {     
				if (m_index < m_mngr->m_flattenIjkVec.size())
				{
					++m_index;
				}
                return *this;
            }
            
            Iterator operator++(int)
            {
                Iterator tmp(*this);
                ++(*this);
                return tmp;
            }
            
            bool operator==(const Iterator& rhs) const
            {
                return (m_mngr == rhs.m_mngr) && (m_index == rhs.m_index);
            }
            
            bool operator!=(const Iterator& rhs) const
            {
                return !(this->operator==(rhs));
            }
            
        private:
            friend class Range;

			Iterator(const H5SliceIoManager* mngr, unsigned index) : m_mngr(mngr), m_index(index) { }

			const H5SliceIoManager* m_mngr;
			unsigned m_index;
        }; // class H5SliceIoManager::Iterator
        
        class Range
        {
        public:
            Range() { }
            
			inline Iterator begin() const { return Iterator(m_mngr, 0); }
			inline Iterator end() const { return Iterator(m_mngr, (unsigned)(m_mngr->m_flattenIjkVec.size())); }
        private:
            friend class H5SliceIoManager;

			Range(const H5SliceIoManager* mngr) : m_mngr(mngr) { }
			const H5SliceIoManager* m_mngr;
        }; // class H5SliceIoManager::Range
        
    private:
		inline unsigned _getFlattenIjk(unsigned x, unsigned y)
		{
			// return x + (y + m_curSlice * m_height) * m_width;
			// Do not need to store the third dimension
			return x + y * m_width;
		}
		
		void _clear()
		{
			m_flattenIjkVec.clear();
			m_voxelIdVec.clear();
			m_recBitsVec.clear();
			m_birthVec.clear();
		}

		inline bool _isSliceInRange(unsigned slice) const
		{
			if (!m_isLoadFileOpen) return false;

			return (m_curLoadChunkBeginSlice <= slice) && (slice < m_curLoadChunkEndSlice);
		}

		inline void _resetChunkSliceRange()
		{
			m_curLoadChunk = ~0;
			m_curLoadChunkBeginSlice = ~0;
			m_curLoadChunkEndSlice = 0;
		}

		void _updateChunkRange(unsigned slice)
		{
			for (unsigned i = 0; i < m_loadChunkRangeMap.size(); ++i)
			{
				const std::pair<unsigned, unsigned>& range = m_loadChunkRangeMap[i];
				if ((range.first <= slice) && (slice < range.second))
				{
					m_curLoadChunk = i;
					m_curLoadChunkBeginSlice = range.first;
					m_curLoadChunkEndSlice = range.second;

					break;
				}
			}
		}

		void _openLoadH5File(const std::string& filename, unsigned flag = H5F_ACC_RDONLY)
		{
			if (m_isLoadFileOpen) 
			{
				m_loadH5File.close();
			}

			m_isLoadFileOpen = true;

			m_loadH5File = H5::H5File(filename, flag);
		}

		void _createDumpH5File(const std::string& filename, unsigned flag = H5F_ACC_TRUNC)
		{
			if (m_isDumpFileOpen) 
			{
				m_dumpH5File.close();
			}

			m_isDumpFileOpen = true;

			m_dumpH5File = H5::H5File(filename, flag);
		}
		
		std::string _getH5Filename(const std::string& group, unsigned chunk) const
		{
			std::stringstream ss;
			ss << group << "\\" << m_filePrefix << chunk << ".h5";
			return ss.str();
		}


		inline std::string _getDatasetName(unsigned chunkBegin) const
		{
			std::stringstream ss;
			unsigned localSliceIndex = m_curSlice - chunkBegin;
			ss << localSliceIndex;
			return ss.str();
		}
		
		void _dumpRangeMap()
		{
			// if (m_dumpChunkRangeMap.size() == m_curDumpChunk)
			// if (m_numVoxelsDumpedInChunk)
			if (m_curDumpChunkBeginSlice < (m_lastDumpedSlice + 1U))
			{
				// update dump range
				m_dumpChunkRangeMap.push_back(std::make_pair(m_curDumpChunkBeginSlice, m_lastDumpedSlice + 1U));
			}

			std::stringstream ss;
			ss << m_newGroupname << "\\" << m_chunkRangeFilename;
			std::string filename(ss.str());

			std::ofstream fh(filename);
			for (auto& p : m_dumpChunkRangeMap)
			{
				fh << p.first << " " << p.second << std::endl;
			}
		}

		void _loadFromDataset()
		{
			using namespace H5;

			_clear();

			std::string dsetName = _getDatasetName(m_curLoadChunkBeginSlice);
			DataSet dataset;
			try
			{
				dataset = m_loadH5File.openDataSet(dsetName);
			}
			catch( DataSetIException error )
			{
				error.printErrorStack();
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

		void _writeToDataset(bool enforceEmpty)
		{
			using namespace H5;

			hsize_t dims[2U];
			std::vector<unsigned> tmpDataVec;
			
			// if (m_flattenIjkVec.size() == 0)
			if (enforceEmpty || (m_flattenIjkVec.size() == 0))
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
			std::string dsetName = _getDatasetName(m_curDumpChunkBeginSlice);
			DataSet dataset;
			try
			{
				dataset = m_dumpH5File.createDataSet(dsetName, datatype, dataspace);
			}
			catch( DataSetIException error )
			{
				error.printErrorStack();

			}
			dataset.write(tmpDataVec.data(), datatype);

			dataset.close();
			datatype.close();
			dataspace.close();

			// _clear();
		}

		std::vector<unsigned> m_flattenIjkVec;
        std::vector<ObjIDType> m_voxelIdVec;
		std::vector<RecBitsType> m_recBitsVec;
		std::vector<unsigned> m_birthVec;

		std::string m_filePrefix;
		std::string m_oldGroupname;
		std::string m_newGroupname;
		std::string m_chunkRangeFilename;

		unsigned m_width;
		unsigned m_height;
		unsigned m_numSlices;
		unsigned m_maxNumVoxelsInChunk;

		unsigned m_curSlice;

		unsigned m_curLoadChunk;
		unsigned m_curLoadChunkBeginSlice;
		unsigned m_curLoadChunkEndSlice;
		
		bool m_isLoadFileOpen;
		H5::H5File m_loadH5File;
		std::vector< std::pair<unsigned, unsigned> > m_loadChunkRangeMap;

		unsigned m_curDumpChunk;
		unsigned m_curDumpChunkBeginSlice;

		bool m_isDumpFileOpen;
		H5::H5File m_dumpH5File;
		std::vector< std::pair<unsigned, unsigned> > m_dumpChunkRangeMap;

		unsigned m_lastDumpedSlice;
		unsigned m_numVoxelsDumpedInChunk;

		unsigned m_originalNumSlices;
	}; 
}; // namespace h5_io

#endif
