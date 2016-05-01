#include "math_utils.hpp"

namespace transform
{
	Matrix4f rotate(const Vector3f& axis, const float radian)
    {
        Vector3f normalized_axis = axis.normalized();
        
        float cos_theta = cos(radian);
        float sin_theta = sin(radian);

        float x = normalized_axis.x();
        float y = normalized_axis.y();
        float z = normalized_axis.z();

        Matrix4f rot_mat;
        rot_mat << x * x * ( 1.0f - cos_theta ) + cos_theta,     y * x * ( 1.0f - cos_theta ) - z * sin_theta, z * x * ( 1.0f - cos_theta ) + y * sin_theta, 0.0f,
                   x * y * ( 1.0f - cos_theta ) + z * sin_theta, y * y * ( 1.0f - cos_theta ) + cos_theta,     z * y * ( 1.0f - cos_theta ) - x * sin_theta, 0.0f,
                   x * z * ( 1.0f - cos_theta ) - y * sin_theta, y * z * ( 1.0f - cos_theta ) + x * sin_theta, z * z * ( 1.0f - cos_theta ) + cos_theta,     0.0f,
                  0.0f,                                          0.0f,                                         0.0f,                                         1.0f;
        return rot_mat;
    }

	Matrix4f translate(const Vector3f& xyz)
    {
        Matrix4f tran_mat;
        tran_mat << 1.0f, 0.0f, 0.0f, xyz(0),
                    0.0f, 1.0f, 0.0f, xyz(1),
                    0.0f, 0.0f, 1.0f, xyz(2),
                    0.0f, 0.0f, 0.0f, 1.0f;
        return tran_mat;
    }

    Matrix4f to_local_coordinate(const Vector3f& origin, const Vector3f& new_z)
    {
        // transform global_z to new_z direction with a translation of origin
        Vector3f global_z(0.0f, 0.0f, 1.0f);
        Vector3f normalized_new_z = new_z.normalized();
        
        auto rot_axis = global_z.cross(normalized_new_z);
        float rot_radian = acos(global_z.dot(normalized_new_z));

        auto mat = translate(origin) * rotate(rot_axis, rot_radian);
        return mat;
    }

    Vector3f apply_transform_base(const Matrix4f& M, const Vector3f& xyz, float fourth_num)
    {
        Vector4f hom_xyz;
        hom_xyz.head<3>() = xyz;
        hom_xyz(3) = fourth_num;

        hom_xyz = M * hom_xyz;
        return hom_xyz.head<3>();
    }
}; // namespace transform