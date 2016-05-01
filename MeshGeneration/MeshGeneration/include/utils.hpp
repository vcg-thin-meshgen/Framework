#ifndef utils_h
#define utils_h

#include <vector>
#include <iostream>
#include <cmath>
#include <functional>   // std::less, std::greater

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
   
}; // namespace utils


#endif /* utils_h */
