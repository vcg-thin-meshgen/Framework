#ifndef IO_SHARED_H
#define IO_SHARED_H

#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>

namespace io_shared
{
	inline bool isFileExist(const std::string& filename)
    {
        std::ifstream infile(filename.c_str());
        return infile.good();
    }

	void swapGroupFiles(const std::string& oldGroupname, const std::string& newGroupname);
}; // namespace io_shared

#endif