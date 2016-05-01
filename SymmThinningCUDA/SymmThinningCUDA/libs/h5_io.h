#ifndef H5_SLICE_IO_H
#define H5_SLICE_IO_H

#include <vector>
#include <algorithm>				// std::copy
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>

#include <H5Cpp.h>

namespace h5_io
{
	class H5SliceIoManager
	{
	public:
		typedef unsigned ObjIDType;
		typedef unsigned char RecBitsType;

        H5SliceIoManager(const std::string& prefix, const std::string& oldG, const std::string& newG,
						unsigned w, unsigned h, unsigned loadChkSz, unsigned dumpChkSz);
        
        class Iterator;
        class Range;
        
        bool load(unsigned slice);
        void alloc(unsigned slice);
        void dump();

		void beginLoadChunk();
		void endLoadChunk();

		void beginDumpChunk();
		void endDumpChunk();
        
        inline Range cached() const { return Range(this); }
        inline unsigned slice() const { return m_curSlice; }

        void storeID(unsigned x, unsigned y, unsigned ID);
        void storeBirth(unsigned, unsigned, unsigned birth);
        void storeRecBits(unsigned, unsigned, RecBitsType bits);
        void swapGroup() const;
        
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

		inline void _resetDumpedDatasetFlags()
		{
			m_dumpedDatasetFlags.clear();
			m_dumpedDatasetFlags.resize(m_dumpChunkSize, 0);
		}
		
		void _clear();

		inline bool _isSliceInRange(unsigned slice) const
		{
			return (m_curChunkBeginSlice <= slice) && (slice < m_curChunkEndSlice);
		}

		inline void _resetChunkSliceRange()
		{
			m_curChunk = ~0;
			m_curChunkBeginSlice = 0;
			m_curChunkEndSlice = 0;
		}

		void _updateChunkRange(unsigned chunkSize);

		void _openH5File(const std::string& filename, unsigned flag = H5F_ACC_TRUNC);
		
		void _closeH5File();
		inline void _dumpAndCloseH5File() { _closeH5File(); }

		std::string _getH5Filename(const std::string& group) const;

		inline unsigned _getLocalSliceIndex() const
		{
			unsigned index = m_curSlice - m_curChunkBeginSlice;
			return index;
		}

		inline std::string _getDatasetName() const
		{
			std::stringstream ss;
			unsigned localSliceIndex = _getLocalSliceIndex();
			ss << localSliceIndex;
			return ss.str();
		}

		void _loadFromDataset();
		void _writeToDataset();

		std::vector<unsigned> m_flattenIjkVec;
        std::vector<ObjIDType> m_voxelIdVec;
		std::vector<RecBitsType> m_recBitsVec;
		std::vector<unsigned> m_birthVec;

		std::string m_filePrefix;
		std::string m_oldGroupname;
		std::string m_newGroupname;

		unsigned m_width;
		unsigned m_height;

		unsigned m_curSlice;

		unsigned m_curChunk;
		unsigned m_curChunkBeginSlice;
		unsigned m_curChunkEndSlice;
		
		unsigned m_loadChunkSize;
		unsigned m_dumpChunkSize;
		std::vector<unsigned char> m_dumpedDatasetFlags;

		bool m_isH5FileOpen;
		H5::H5File m_curH5File;

		bool m_isLoading;
		bool m_isDumping;
	}; 
}; // namespace h5_io
#endif
