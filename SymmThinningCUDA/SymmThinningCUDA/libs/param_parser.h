#ifndef PARAM_PARSER_H
#define PARAM_PARSER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

namespace pp
{
	struct ParamPack
	{
		std::string inputFilePrefix;
		std::string outputFilename;
		unsigned numSlices;
		unsigned sliceOffset;
		std::vector<unsigned> labelVec;

		bool usePersistence;
		unsigned p;
		int maxIter;
		unsigned numThreads;

		unsigned curIter;
		unsigned dim;
	};

	ParamPack loadParams()
	{
		ParamPack params;
		std::ifstream input("config.txt", std::ios::in);

		std::string line;
		while (input.good())
		{
			input >> line;
			if (line.compare("inputPrefix") == 0)
			{
				input >> params.inputFilePrefix;
			}
			else if (line.compare("outputFile") == 0)
			{
				input >> params.outputFilename;
			}
			else if (line.compare("numSlices") == 0)
			{
				input >> line;
				params.numSlices = std::stoul(line);
			}
			else if (line.compare("sliceOffset") == 0)
			{
				input >> line;
				params.sliceOffset = std::stoul(line);
			}
			else if (line.compare("label") == 0)
			{
				input >> line;
				params.labelVec.push_back(std::stoul(line));
			}
			else if (line.compare("usePersistence") == 0)
			{
				input >> line;
				params.usePersistence = std::stoi(line);
			}
			else if (line.compare("p") == 0)
			{
				input >> line;
				params.p = std::stoul(line);
			}
			else if (line.compare("maxIter") == 0)
			{
				input >> line;
				params.maxIter = std::stoi(line);
			}
			else if (line.compare("numThreads") == 0)
			{
				input >> line;
				params.numThreads = std::stoul(line);
			}
			else if (line.compare("curIter") == 0)
			{
				input >> line;
				params.curIter = std::stoul(line);
			}
			else if (line.compare("dim") == 0)
			{
				input >> line;
				params.dim = std::stoul(line);
			}
			else 
			{
				std::cerr << "  unrecognized keyword [" << line << "], skipping" << std::endl;
				char c;
				do 
				{  // skip until end of line
					c = input.get();
				} while(input.good() && (c != '\n'));

			}
		}

		return params;
	}
}; // namespace pp;
#endif