#include "io_shared.h"

#include <iostream>

namespace io_shared
{
	void swapGroupFiles(const std::string& oldGroupname, const std::string& newGroupname)
	{ 
		namespace fs = boost::filesystem;
		fs::path oldPath(oldGroupname), newPath(newGroupname);
            
		assert(fs::is_directory(oldPath) && fs::is_directory(newPath));
            
		std::stringstream ss;
		ss << oldGroupname << "_tmp";
		fs::rename(newPath, ss.str());
            
		// remove all the files in the "old" group
		fs::directory_iterator end;
		for(fs::directory_iterator it(oldPath); it != end; ++it)
		{
			try
			{
				if(fs::is_regular_file(it->status()))
				{
					fs::remove(it->path());
				}
			}
			catch(const std::exception &ex)
			{
				std::cerr << ex.what() << std::endl;
				// ex;
			}
		}
            
		fs::rename(oldPath, newPath);
		fs::rename(ss.str(), oldPath);
	}
}; // namespace io_shared;