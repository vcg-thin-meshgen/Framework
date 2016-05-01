#ifndef BINVOX_H
#define BINVOX_H

//
// This example program reads a .binvox file and writes
// an ASCII version of the same file called "voxels.txt"
//
// 0 = empty voxel
// 1 = filled voxel
// A newline is output after every "dim" voxels (depth = height = width = dim)
//
// Note that this ASCII version is not supported by "viewvox" and "thinvox"
//
// The x-axis is the most significant axis, then the z-axis, then the y-axis.
//

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>

namespace binvox
{
	typedef unsigned char byte;
	
	struct BinvoxHeader
	{
	public:
		unsigned wxh() const;

		unsigned version;
		unsigned depth, height, width;
		float tx, ty, tz;
		float scale;
	};

	int readBinvox(const std::string& filespec, std::vector<byte>& voxels, BinvoxHeader& bh);

	inline unsigned getIndex(unsigned x, unsigned y, unsigned z, const BinvoxHeader& bh)
	{
		unsigned index = x * bh.wxh() + z * bh.width + y;  // wxh = width * height = d * d
		return index;
	} 
};

#endif