#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <Eigen/Dense>
#include <cmath>

namespace transform
{
    using Eigen::Matrix4f;
    using Eigen::MatrixXf;
    using Eigen::Vector3f;
    using Eigen::Vector4f;
    using Eigen::VectorXf;

    inline float norm(float x, float y, float z)
    {
        return sqrt(x * x + y * y + z * z);
    }

	inline float norm(const Vector3f& v)
	{
		return norm(v(0), v(1), v(2));
	}

    /*
     * Generate a rotation matrix.
     */
    Matrix4f rotate(const Vector3f& axis, const float radian);

    /*
     * Generate a translation matrix.
     */
    Matrix4f translate(const Vector3f& xyz);

    /*
     * Generate a transform matrix to the new coordinate system where 'origin'
     * is its origin and its local z axis aligned with 'new_z'.
     */
    Matrix4f to_local_coordinate(const Vector3f& origin, const Vector3f& new_z);

    /*
     * Base method to apply a 4x4 matrix transformation on a 3x1 vector. 'fourth_num' determines
     * the fourth element of the homogenous vector.
     */
    Vector3f apply_transform_base(const Matrix4f& M, const Vector3f& xyz, float fourth_num);

    inline Vector3f apply_transform_pt(const Matrix4f& M, const Vector3f& pt)
    {
        return apply_transform_base(M, pt, 1.0f);
    }

	inline Vector3f apply_transform_pt(const Matrix4f& M, float x, float y, float z)
	{
		Vector3f pt;
		pt << x, y, z;
		return apply_transform_pt(M, pt);
	}

    inline Vector3f apply_transform_vec(const Matrix4f& M, const Vector3f& vec)
    {
        return apply_transform_base(M, vec, 0.0f);
    }

	inline Vector3f apply_transform_vec(const Matrix4f& M, float x, float y, float z)
	{
		Vector3f vec;
		vec << x, y, z;
		return apply_transform_vec(M, vec);
	}
}; // namespace transform

#endif