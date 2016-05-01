#include "binvox.h"

namespace binvox
{
	unsigned BinvoxHeader::wxh() const
	{
		static unsigned _wxh = width * height;
		return _wxh;
	}

	int readBinvox(const std::string& filespec, std::vector<byte>& voxels, BinvoxHeader& bh)
	{
		using namespace std;

		ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);
		//
		// read header
		//
		string line;
		*input >> line;  // #binvox
		if (line.compare("#binvox") != 0) 
		{
			cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
			delete input;
			return 0;
		}
		*input >> bh.version;
		cout << "reading binvox version " << bh.version << endl;

		bh.depth = ~0;
		int done = 0;
		while(input->good() && !done) 
		{
			*input >> line;
			if (line.compare("data") == 0) done = 1;
			else if (line.compare("dim") == 0) 
			{
				*input >> bh.depth >> bh.height >> bh.width;
			}
			else if (line.compare("translate") == 0) 
			{
				*input >> bh.tx >> bh.ty >> bh.tz;
			}
			else if (line.compare("scale") == 0) 
			{
				*input >> bh.scale;
			}
			else 
			{
				cout << "  unrecognized keyword [" << line << "], skipping" << endl;
				char c;
				do 
				{  // skip until end of line
					c = input->get();
				} while(input->good() && (c != '\n'));

			}
		}
		
		if (!done) 
		{
			cout << "  error reading header" << endl;
			return 0;
		}
		if (bh.depth == (~0)) 
		{
			cout << "  missing dimensions in header" << endl;
			return 0;
		}

		unsigned size = bh.width * bh.height * bh.depth;
		// voxels = new byte[size];

		voxels.clear();
		voxels.resize(size);

		//
		// read voxel data
		//
		byte value;
		byte count;
		int index = 0;
		int end_index = 0;
		int nr_voxels = 0;
  
		input->unsetf(ios::skipws);  // need to read every byte now (!)
		*input >> value;  // read the linefeed char

		while((end_index < size) && input->good()) {
		*input >> value >> count;

		if (input->good()) {
			end_index = index + count;
			if (end_index > size) return 0;
			for(int i=index; i < end_index; i++) voxels[i] = value;
      
			if (value) nr_voxels += count;
			index = end_index;
		}  // if file still ok
    
		}  // while

		input->close();
		delete input;
		cout << "  read " << nr_voxels << " voxels" << endl;

		return 1;
	}
}