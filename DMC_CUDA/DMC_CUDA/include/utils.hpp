#ifndef utils_h
#define utils_h

#include <vector>
#include <iostream>
#include <cmath>
#include <functional>   // std::less, std::greater

#include "cuda_includes.h"

namespace utils
{
    // small 3D vector wrapper implemented using std::vector<T>
    template <typename T>
    class Array3D
    {
    public:
        typedef T value_type;
        typedef typename std::vector<value_type>::iterator iterator;
        typedef typename std::vector<value_type>::const_iterator const_iterator;
        
        Array3D(const uint3& size3D, const T& value = T())
        : m_dim_x(size3D.x)
        , m_dim_y(size3D.y)
        , m_dim_z(size3D.z)
        , m_dim_xy(size3D.x * size3D.y)
        , m_data(std::vector<value_type>(m_dim_xy * m_dim_z, value)) { }

        Array3D(unsigned dim_x, unsigned dim_y, unsigned dim_z, const T& value = T())
        : m_dim_x(dim_x)
        , m_dim_y(dim_y)
        , m_dim_z(dim_z)
        , m_dim_xy(dim_x * dim_y)
        , m_data(std::vector<value_type>(m_dim_xy * dim_z, value)) { }
        
        const T& operator()(unsigned x, unsigned y, unsigned z) const
        {
            return m_data[z * m_dim_xy + y * m_dim_x + x];
        }
        
        T& operator()(unsigned x, unsigned y, unsigned z)
        {
            return m_data[z * m_dim_xy + y * m_dim_x + x];
        }
        
        unsigned dim_x() const { return m_dim_x; }
        unsigned dim_y() const { return m_dim_y; }
        unsigned dim_z() const { return m_dim_z; }
        
        iterator begin()                { return m_data.begin(); }
        iterator end()                  { return m_data.end(); }
        const_iterator cbegin() const   { return m_data.cbegin(); }
        const_iterator cend() const     { return m_data.cend(); }

		T* data() { return m_data.data(); }
		const T* data() const { return m_data.data(); }
		size_t size() const { return dim_x() * dim_y() * dim_z(); }

    private:
        unsigned m_dim_x;
        unsigned m_dim_y;
        unsigned m_dim_z;
        unsigned m_dim_xy;
        
        std::vector<value_type> m_data;
    };
    
    __host__ __device__ inline void 
	index3D_to_1D(unsigned i, unsigned j, unsigned k,
					unsigned num_voxels_i, unsigned num_voxels_j, unsigned& index1D)
    {
        index1D = (k * num_voxels_j + j) * num_voxels_i + i;
    }

	__host__ __device__ inline unsigned
	index3D_to_1D(unsigned i, unsigned j, unsigned k,
					unsigned num_voxels_i, unsigned num_voxels_j)
    {
        unsigned index1D;
		index3D_to_1D(i, j, k, num_voxels_i, num_voxels_j, index1D);
		return index1D;
    }
    
    __host__ __device__ inline void 
	index3D_to_1D(const uint3& index3D, const uint3& num_voxels_dim, unsigned& index1D)
    {
        index3D_to_1D(index3D.x, index3D.y, index3D.z, num_voxels_dim.x, num_voxels_dim.y, index1D);
    }

	__host__ __device__ inline unsigned
	index3D_to_1D(const uint3& index3D, const uint3& num_voxels_dim)
	{
		unsigned index1D;
		index3D_to_1D(index3D, num_voxels_dim, index1D);
		return index1D;
	}
    
    __host__ __device__ inline void 
	index1D_to_3D(unsigned index1D, const uint3& num_voxels_dim, uint3& index3D)
    {
        unsigned num_voxels_xy = num_voxels_dim.x * num_voxels_dim.y;
        
        index3D.z = index1D / num_voxels_xy;
        index1D = index1D % num_voxels_xy;
        index3D.y = index1D / num_voxels_dim.x;
        index1D = index1D % num_voxels_dim.x;
        index3D.x = index1D;
    }

	__host__ __device__ inline uint3
	index1D_to_3D(unsigned index1D, const uint3& num_voxels_dim)
	{
		uint3 index3D;
		index1D_to_3D(index1D, num_voxels_dim, index3D);
		return index3D;
	}
    
    __host__ __device__ inline float 
	ijk_to_xyz(unsigned i, unsigned size, float f_range, float f_min)
    {
        return (float)i / size * f_range + f_min;
    }
    
	__host__ __device__ inline float2 xy(const float3& f) { return make_float2(f.x, f.y); }

	__host__ __device__ inline float2 yz(const float3& f) { return make_float2(f.y, f.z); }

	__host__ __device__ inline float2 xz(const float3& f) { return make_float2(f.x, f.z); }

    template <size_t, size_t, typename Op, typename T>
    size_t argmin_impl(size_t min_index, const T&, const Op&) { return min_index; }
    
    template <size_t Index, size_t Size, typename Op, typename T, typename ...Ts>
    size_t argmin_impl(size_t min_index, const T& cur_min, const Op& op, const T& head, const Ts&... rest)
    {
        if (op(head, cur_min))
        {
            return argmin_impl<Index + 1, Size>(Index, head, op, rest...);
        }
        return argmin_impl<Index + 1, Size>(min_index, cur_min, op, rest...);
    }
    
    template <typename T, typename ...Ts>
    size_t argmin(const T& head, const Ts&... rest)
    {
        return argmin_impl<0, sizeof...(Ts)>(0, head, std::less<T>(), head, rest...);
    }
    
    template <typename T, typename ...Ts>
    size_t argmax(const T& head, const Ts&... rest)
    {
        return argmin_impl<0, sizeof...(Ts)>(0, head, std::greater<T>(), head, rest...);
    }
}; // namespace utils

std::ostream& operator<<(std::ostream& os, const float3& f);

std::ostream& operator<<(std::ostream& os, const uint3& u);

#endif /* utils_h */
