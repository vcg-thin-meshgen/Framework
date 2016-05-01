//
//  png_io.h
//  SymmThinning
//
//  Created by Ye Kuang on 3/4/16.
//  Copyright © 2016 Ye Kuang. All rights reserved.
//

#ifndef png_io_h
#define png_io_h

#include <vector>
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>

namespace png_io
{
    
#if defined(WIN32) || defined(_WIN32)
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif
    

#define USE_RGB

    class PngSliceIoManager
    {
    public:
		typedef unsigned char RecBitsType;

        PngSliceIoManager(const std::string& prefix, unsigned w, unsigned h,
                          const std::string& oldG, const std::string& newG, unsigned offs = 0);
        
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
        void storeBirth(unsigned x, unsigned y, unsigned birth);
        void storeRecBits(unsigned x, unsigned y, RecBitsType bits);
        
        void swapGroup() const;
        
        class Iterator
        {
        public:
            Iterator() : m_mngr(nullptr), m_index(0) { }
            
            const Iterator& operator*() const { return *this; }
            
            inline unsigned x() const
            {
#ifdef USE_RGB
                return (m_index / 3U) % (m_mngr->m_imWidth);
#else
                return m_index % (m_mngr->m_imWidth);
#endif
            }
            
            inline unsigned y() const
            {
#ifdef USE_RGB
                return (m_index / 3U) / (m_mngr->m_imWidth);
#else
                return m_index / (m_mngr->m_imWidth);
#endif
            }
            
            inline unsigned ID() const
            {
#ifdef USE_RGB
                unsigned lb = (m_mngr->m_idVec[m_index] << 16U);
                lb |= (m_mngr->m_idVec[m_index + 1U] << 8U);
                lb |= m_mngr->m_idVec[m_index + 2U];
#else         
				unsigned lb = (unsigned)m_mngr->m_idVec[m_index];
#endif           
                return lb;
            }
            
            inline RecBitsType recBits() const
            {
#ifdef USE_RGB
                return m_mngr->m_recBitsVec[m_index + 2U];
#else
                return m_mngr->m_recBitsVec[m_index * 3U + 2];
#endif
            }
            
            unsigned birth() const
            {
#ifdef USE_RGB
				unsigned b = 0;
				b |= m_mngr->m_recBitsVec[m_index + 1U];
#else
				unsigned b = 0;
				b |= m_mngr->m_recBitsVec[m_index * 3U + 1U];
#endif
                
                return b;
            }
            
            Iterator& operator++()
            {
#ifdef USE_RGB
                m_index += 3U;
#else
                ++m_index;
#endif
                _fix();
                
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
            Iterator(const PngSliceIoManager* mngr, unsigned index)
            : m_mngr(mngr), m_index(index) { _fix(); }
            
			inline bool _noVoxel() const 
			{
#ifdef USE_RGB
				return ((m_mngr->m_idVec[m_index] == 0) && 
						(m_mngr->m_idVec[m_index + 1U] == 0) && 
						(m_mngr->m_idVec[m_index + 2U] == 0)) ||
						(m_mngr->m_recBitsVec[m_index + 2U] & 0x03 == 0);
#else
				return (m_mngr->m_idVec[m_index] == 0);
#endif
			}

            void _fix()
            {
                // while ((m_index < m_mngr->m_recBitsVec.size()) && (m_mngr->m_recBitsVec[m_index + 2U] == 0))
                while ((m_index < m_mngr->m_idVec.size()) && _noVoxel())
                {
#ifdef USE_RGB
                    m_index += 3U;
#else
                    ++m_index;
#endif
                }
                
                // if (m_index >= m_mngr->m_recBitsVec.size())
                if (m_index >= m_mngr->m_idVec.size())
                {
                    // m_index = (unsigned)m_mngr->m_recBitsVec.size();
                    m_index = (unsigned)m_mngr->m_idVec.size();
                }
            }
                             
            const PngSliceIoManager* m_mngr;
            unsigned m_index;
        }; // class PngSliceIoManager::Iterator
        
        class Range
        {
        public:
            Range() : m_mngr(nullptr) { }
            
            inline Iterator begin() const { return Iterator(m_mngr, 0); }
            inline Iterator end() const { return Iterator(m_mngr, (unsigned)m_mngr->m_idVec.size()); }
        private:
            friend class PngSliceIoManager;
            Range(const PngSliceIoManager* mngr) : m_mngr(mngr) { }
            
            const PngSliceIoManager* m_mngr;
        }; // class PngSliceIoManager::Range
        
    private:
        inline std::string _getName(const std::string& group, const std::string& suffix) const
        {
            std::stringstream ss;
            ss << group << PATH_SEPARATOR << m_filePrefix << m_curSlice + m_sliceOffset << suffix;
            return ss.str();
        }
        
        inline std::string _getPngFilename(const std::string& group) const
        {
            return _getName(group, ".png");
        }
        
        inline std::string _getRecBitsFilename(const std::string& group) const
        {
            return _getName(group, "_recBits.png");
        }
        
        inline void _clear()
        {
            m_hasVoxel = false;
            
            m_idVec.clear();
            m_recBitsVec.clear();
        }
        
        std::string m_filePrefix;
        unsigned m_imWidth, m_imHeight;
        
        std::string m_oldGroupname;
        std::string m_newGroupname;
        
        unsigned m_sliceOffset;
        
        unsigned m_curSlice;
        
        bool m_hasVoxel;
        // this is the raw png data for slice, in RGB RGB RGB ... format
        std::vector<unsigned char> m_idVec;
        // this is the raw png data for rec bits and birth for each voxel,
        // in RGB RGB RGB ... format.
        std::vector<unsigned char> m_recBitsVec;
        
    };
}; // namespace png_io;

#endif /* png_io_h */
