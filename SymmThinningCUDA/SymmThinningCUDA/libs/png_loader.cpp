//
//  png_loader.cpp
//  MarchingTetrahedra
//
//  Created by Ye Kuang on 11/19/15.
//  Copyright © 2015 Ye Kuang. All rights reserved.
//

#include "png_loader.h"

#include <sstream>
#include <iostream>

#include "lodepng.h"

namespace png_load
{
    PngLoader::PngLoader(const std::string& file_prefix, unsigned num_slices, const std::string& file_suffix)
    : m_file_prefix(file_prefix)
    , m_num_slices(num_slices)
    , m_file_suffix(file_suffix) { }
        
    void PngLoader::load(unsigned slice_k, std::vector<data_type>& png_data, unsigned& width, unsigned& height, LodePNGColorType colorType) const
    {
        std::stringstream ss;
        ss << m_file_prefix << slice_k << m_file_suffix;
        std::string filename(ss.str());
            
        png_data.clear();
        unsigned error = lodepng::decode(png_data, width, height, filename, colorType);
        
        //if there's an error, display it
        if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
    
    void PngLoader::load(unsigned slice_k, std::vector<data_type>& png_data, LodePNGColorType colorType) const
    {
        unsigned width, height;
        load(slice_k, png_data, width, height, colorType);
    }
    
    std::vector<PngLoader::data_type> PngLoader::load(unsigned slice_k, LodePNGColorType colorType) const
    {
        std::vector<data_type> png_data;
        unsigned width, height;
        load(slice_k, png_data, width, height, colorType);
        return png_data;
    }
    
}; // namespace png_load