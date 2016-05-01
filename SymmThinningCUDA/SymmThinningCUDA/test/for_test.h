#ifndef FOR_TEST_H
#define FOR_TEST_H

#include <vector>
#include <algorithm>

#include "thinning_base.cuh"

class MyGrid
{
public:

	void add(int i, int j, int k)
	{
		_vec.push_back(make_uint3(i, j, k));
	}

	void sort()
	{
		std::sort(_vec.begin(), _vec.end(), thin::less);
	}

	const uint3* begin() const
	{
		return _vec.data();
	}

	unsigned size() const { return _vec.size(); }

private:
	unsigned _sI, _sIj;
	std::vector<uint3> _vec;
};

#endif