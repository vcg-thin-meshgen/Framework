//
//  png_io.cpp
//  SymmThinning
//
//  Created by Ye Kuang on 3/4/16.
//  Copyright © 2016 Ye Kuang. All rights reserved.
//

#include <stdio.h>
#include <fstream>

#include "png_io.h"
#include "lodepng.h"
#include "io_shared.h"

namespace png_io
{
    PngSliceIoManager::PngSliceIoManager(const std::string& prefix, unsigned w, unsigned h,
                                         const std::string& oldG, const std::string& newG, unsigned offs)
    : m_filePrefix(prefix), m_imWidth(w), m_imHeight(h)
    , m_oldGroupname(oldG), m_newGroupname(newG), m_sliceOffset(offs)
    {
        _clear();
    }
    
    bool PngSliceIoManager::load(unsigned slice)
    {
        m_curSlice = slice;
        std::string filename(_getPngFilename(m_oldGroupname));
        
        _clear();
        
        unsigned w, h;
#ifdef USE_RGB
        unsigned error = lodepng::decode(m_idVec, w, h, filename, LCT_RGB);
#else
		unsigned error = lodepng::decode(m_idVec, w, h, filename, LCT_GREY);
#endif
        if (error)
        {
            _clear();
            return false;
        }
        
        assert((w == m_imWidth) && (h == m_imHeight));
        for (unsigned idx = 0; idx < m_idVec.size(); ++idx)
        {
            if (m_idVec[idx])
            {
                m_hasVoxel = true;
                break;
            }
        }
        
        // recording bits filename
        filename = _getRecBitsFilename(m_oldGroupname);
        
        //if (isFileExist(filename))
		if (io_shared::isFileExist(filename))
        {
            // If this recording bits file exists, then we use the data in this file.
            lodepng::decode(m_recBitsVec, w, h, filename, LCT_RGB);
        }
        else
        {
            // Otherwise, we create the recording bits according to the voxel data
            unsigned idx = 0;

#ifdef USE_RGB
			m_recBitsVec.resize(m_idVec.size(), 0);
#else
            m_recBitsVec.resize(m_idVec.size() * 3, 0);
#endif
			
            while (idx < m_idVec.size())
            {
                // if ID exists
#ifdef USE_RGB
				if (m_idVec[idx] || m_idVec[idx + 1U] || m_idVec[idx + 2U])
#else
                if (m_idVec[idx])
#endif
                {
                    // We use R and G channel to store the birthData, and B channel for the
                    // recording bits. + 2 is the offset of B channel.

#ifdef USE_RGB
					m_recBitsVec[idx + 2U] = 1U;
#else
                    m_recBitsVec[idx * 3 + 2U] = 1U;
#endif
                }
                
#ifdef USE_RGB
                idx += 3U;
#else
                ++idx;
#endif
            }
        }
        
        return true;
    }
    
    void PngSliceIoManager::alloc(unsigned slice)
    {
        m_curSlice = slice;
        _clear();
        
#ifdef USE_RGB
		unsigned size = m_imWidth * m_imHeight * 3U;
        m_idVec.resize(size, 0);
        m_recBitsVec.resize(size, 0);
#else
        unsigned size = m_imWidth * m_imHeight;
        m_idVec.resize(size, 0);
        m_recBitsVec.resize(size * 3, 0);
#endif
    }
    
    void PngSliceIoManager::dump()
    {
        if (m_hasVoxel)
        {
            std::string filename(_getPngFilename(m_newGroupname));
#ifdef USE_RGB
			lodepng::encode(filename, m_idVec, m_imWidth, m_imHeight, LCT_RGB);
#else
            lodepng::encode(filename, m_idVec, m_imWidth, m_imHeight, LCT_GREY);
#endif
            
            filename = _getRecBitsFilename(m_newGroupname);
            lodepng::encode(filename, m_recBitsVec, m_imWidth, m_imHeight, LCT_RGB);
        }
    }
    
    void PngSliceIoManager::storeID(unsigned x, unsigned y, unsigned ID)
    {
#ifdef USE_RGB
        unsigned idx = (x + y * m_imWidth) * 3U;
        
        m_idVec[idx] = (ID >> 16U) & 0xff;
        m_idVec[idx + 1U] = (ID >> 8U) & 0xff;
        m_idVec[idx + 2U] = ID & 0xff;
#else   
        unsigned idx = x + y * m_imWidth;
        m_idVec[idx] = ID;
#endif   
        m_hasVoxel = true;
    }
    
    void PngSliceIoManager::storeBirth(unsigned x, unsigned y, unsigned birth)
    {
        unsigned idx = (x + y * m_imWidth) * 3U;
        // m_recBitsVec[idx] = (birth >> 8) & 0xff;
		m_recBitsVec[idx] = 100;
        m_recBitsVec[idx + 1] = birth & 0xff;
        
        m_hasVoxel = true;
    }
    
    void PngSliceIoManager::storeRecBits(unsigned x, unsigned y, RecBitsType bits)
    {
        unsigned idx = (x + y * m_imWidth) * 3U;
        m_recBitsVec[idx + 2U] = bits;
        
        m_hasVoxel = true;
    }

	void PngSliceIoManager::swapGroup() const
    {
		io_shared::swapGroupFiles(m_oldGroupname, m_newGroupname);
    }
}; // namespace png_io