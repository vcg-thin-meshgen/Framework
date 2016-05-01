#ifndef QUADRATIC_UTILS_HPP
#define QUADRATIC_UTILS_HPP

#include <cstdint>          // int32_t
#include <cmath>            // sqrt()
#include <cassert>
#include <vector>
#include <limits>
#include <memory>

#include <Eigen/Dense>
#include <flann/flann.hpp>

#include "math_utils.hpp"

namespace
{
    const unsigned AUX_NEAR = 6;
    const float TWO_SQRT_3 = sqrt(3.0f) * 2.0f;
}; // namespace

namespace mpu
{
    using Eigen::MatrixXf;
    using Eigen::Vector3f;
    using Eigen::VectorXf;
    typedef int32_t IndexType; // use int type as -1 may be used to indicate invalid.

    using transform::norm;
    /*
     * Support radius = alpha * sqrt(3) * 2 * half_edge_len
     */
    inline float support_radius(float half_edge_len, float alpha = 0.75f)
    {
        return alpha * half_edge_len * TWO_SQRT_3;
    }

    float B_spline(float x);

    /*
     * Weight function class that uses quadratic B-spline.
     */
    class WFn
    {
    public:
        WFn(float center_x, float center_y, float center_z, float radius)
        : p_center_x(center_x)
        , p_center_y(center_y)
        , p_center_z(center_z)
        , p_radius(radius) { }

        WFn(const Vector3f& center, float radius)
        : p_center_x(center(0))
        , p_center_y(center(1))
        , p_center_z(center(2))
        , p_radius(radius) { }

        inline float operator()(float x, float y, float z) const
        {
            float t = 1.5f * norm(x - p_center_x,
                                   y - p_center_y,
                                   z - p_center_z) / p_radius;
            return B_spline(t);
        }

        inline float operator()(const Vector3f& xyz) const
        {
            return this->operator()(xyz(0), xyz(1), xyz(2));
        }

    private:
        float p_center_x;
        float p_center_y;
        float p_center_z;
        float p_radius;
    };

    /*
     * Expand the xyz coordinate to be the general Q function's coefficient base.
     * [x^2, y^2, z^2, x * y, y * z, x * z, x, y, z, 1.0f]
     */
    inline VectorXf expand_xyz(float x, float y, float z)
    {
        VectorXf expanded(10);
        expanded << x * x, y * y, z * z,
                    x * y, y * z, x * z,
                    x,     y,     z,    1.0f;
        return expanded;
    }

    inline VectorXf expand_xyz(const Vector3f& xyz)
    {
        return expand_xyz(xyz(0), xyz(1), xyz(2));
    }

    /*
     * Expand the xyz coordinate to be the bivariate Q function's coefficient base.
     * [x^2, y^2, x * y, x, y, 1.0f]
     */
    inline VectorXf expand_xyz_biv(float local_x, float local_y, float)
    {
        VectorXf expanded(6);
        expanded << local_x * local_x, local_y * local_y, local_x * local_y,
                    local_x,           local_y,           1.0f;
        return expanded;
    }

    inline VectorXf expand_xyz_biv(const Vector3f& local_xyz)
    {
        return expand_xyz_biv(local_xyz(0), local_xyz(1), local_xyz(2));
    }

    /*
     * Initialized all the auxiliary points given the geometric info of the octree cell.
     */
	void init_aux_pts(const Vector3f& center, float half_edge_len, std::vector<float>& aux_pts);

    //void init_aux_pts(const Vector3f& center, float half_edge_len, std::vector<Vector3f>& aux_pts)
    //{
    //    std::vector<Vector3f> ops = 
    //    {
    //        {-1.0f, -1.0f, -1.0f},
    //        {+1.0f, -1.0f, -1.0f},
    //        {-1.0f, +1.0f, -1.0f},
    //        {-1.0f, -1.0f, +1.0f},
    //        {+1.0f, +1.0f, -1.0f},
    //        {+1.0f, -1.0f, +1.0f},
    //        {-1.0f, +1.0f, +1.0f},
    //        {+1.0f, +1.0f, +1.0f},
    //    };

    //    // push in eight corners
    //    for (unsigned i = 0; i < ops.size(); ++i)
    //    {
    //        Vector3f aux = ops[i] * half_edge_len + center;
    //        aux_pts.push_back(aux);
    //    }
    //    // push in the center
    //    aux_pts.push_back(center);
    //}

    /*
     * Find the 6 nearest points in the 'pts' set for the given auxiliary point in brute force.
     */
    //void find_aux_near_brute(const Vector3f& aux_pt, 
    //                        const std::vector<Vector3f> pts,
    //                        std::vector<IndexType>& aux_near_indices,
    //                        std::vector<float>& aux_near_dists)
    //{
    //    aux_near_indices.clear();
    //    aux_near_indices.assign(AUX_NEAR, -1);
    //    aux_near_dists.clear();
    //    aux_near_dists.assign(AUX_NEAR, std::numeric_limits<float>::max());

    //    for (unsigned i = 0; i < pts.size(); ++i)
    //    {
    //        auto p = pts[i];
    //        float d = (aux_pt - p).norm();

    //        IndexType insert = -1;
    //        for (unsigned j = 0; j < AUX_NEAR; ++j)
    //        {
    //            if (aux_near_dists[j] > d) insert = j;
    //            else break;
    //        }

    //        if (insert < 0) continue;
    //        for (unsigned j = 0; j < insert; ++j)
    //        {
    //            aux_near_indices[j] = aux_near_indices[j + 1];
    //            aux_near_dists[j] = aux_near_dists[j + 1];
    //        }

    //        aux_near_indices[insert] = i;
    //        aux_near_dists[insert] = d;
    //    }
    //}

    //std::shared_ptr<float> vec_pts_to_raw_array(const std::vector<Vector3f>& pts)
    //{
    //     this is completely rubbish. Should use a better data structure to store points so that
    //     they don't need to be copied. TOO EXPENSIVE!
    //    unsigned i(0);
    //    std::shared_ptr<float> raw_array(new float[pts.size() * 3], std::default_delete<float[]>());

    //    for (const auto& pt : pts)
    //    {
    //        raw_array.get()[i] = pt(0);
    //        raw_array.get()[i + 1] = pt(1);
    //        raw_array.get()[i + 2] = pt(2);
    //        i += 3;
    //    }
    //    return raw_array;
    //}

    typedef flann::L2_3D<float> L2Type;
    typedef flann::Index<L2Type> FLANNIndexType;

    /*
     * Find the 6 nearest points in the 'pts' set for the given auxiliary point using KDTree.
     * Cannot use self defined L2_3D_Sqrt as this will result imprecision in the distance.
     */
	void find_aux_near_kd(const float *aux_pt, 
                        const std::vector<float>& pts,
                        const FLANNIndexType& kd_index,
                        std::vector<IndexType>& aux_near_indices,
                        std::vector<float>& aux_near_dists);

    //void find_aux_near_kd(const Vector3f& aux_pt, 
    //                    const std::vector<Vector3f>& pts,
    //                    const FLANNIndexType& kd_index,
    //                    std::vector<IndexType>& aux_near_indices,
    //                    std::vector<float>& aux_near_dists)
    //{
    //    aux_near_indices.clear();
    //    aux_near_dists.clear();

    //    flann::Matrix<float> query(const_cast<float*>(aux_pt.data()), 1, 3);
    //    std::vector<std::vector<int>> query_indices;
    //    std::vector<std::vector<FLANNIndexType::DistanceType>> query_distances;
    //    // kd-tree kNN search
    //    kd_index.knnSearch(query, query_indices, query_distances, AUX_NEAR, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));

    //    for (unsigned i = 0; i < AUX_NEAR; ++i)
    //    {
    //        aux_near_indices.push_back(query_indices[0][i]);
    //        aux_near_dists.push_back(query_distances[0][i]);
    //    }
    //}

    /*
     * Test to see if the auxiliary point is valid given the corresponding point set.
     */
    bool is_aux_pt_valid(const float *aux_pt, 
                        const std::vector<float>& pts,
                        const std::vector<float>& pt_norms,
                        const FLANNIndexType& kd_index,
                        float& aux_d);
    //bool is_aux_pt_valid(const Vector3f& aux_pt, 
    //                    const std::vector<Vector3f>& pts,
    //                    const std::vector<Vector3f>& pt_norms,
    //                    const FLANNIndexType& kd_index,
    //                    float& aux_d)
    //{
    //    std::vector<IndexType> aux_near_indices;
    //    std::vector<float> aux_near_dists;

    //    //find_aux_near_brute(aux_pt, pts, aux_near_indices, aux_near_dists);
    //    find_aux_near_kd(aux_pt, pts, kd_index, aux_near_indices, aux_near_dists);

    //    assert(aux_near_indices.size() == AUX_NEAR);
    //    assert(aux_near_indices.size() == aux_near_dists.size());

    //    aux_d = 0.0f;
    //    float mult = 0.0f, first_mult = 0.0f;
    //    bool is_first = true;
    //    for (auto idx : aux_near_indices)
    //    {
    //        mult = pt_norms[idx].dot(aux_pt - pts[idx]);
    //        aux_d += mult;

    //        if (is_first)
    //        {
    //            first_mult = mult;
    //            is_first = false;
    //        }
    //        else if (first_mult * mult <= 0.0f)
    //        {
    //            // make it invalid
    //            aux_d = std::numeric_limits<float>::min();
    //            return false;
    //        }
    //    }

    //    aux_d /= aux_near_indices.size();
    //    return true;
    //}

    /*
     * Find all the valid auxiliary points given an octree cell's configuration and 
     * the points contained in that cell.
     */
	 void find_valid_aux_pts(const Vector3f& center, float half_edge_len,
                            const std::vector<float>& pts,
                            const std::vector<float>& pt_norms,
                            std::vector<float>& aux_pts,
                            std::vector<float>& aux_ds);
    // void find_valid_aux_pts(const Vector3f& center, float half_edge_len,
    //                        const std::vector<Vector3f>& pts,
    //                        const std::vector<Vector3f>& pt_norms,
    //                        std::vector<Vector3f>& aux_pts,
    //                        std::vector<float>& aux_ds)
    //{
    //    aux_pts.clear();
    //    aux_ds.clear();

    //    std::vector<Vector3f> tmp_aux_pts; // tentative auxiliary points
    //    init_aux_pts(center, half_edge_len, tmp_aux_pts);

    //    // build a local kd-tree
    //    std::shared_ptr<float> raw_array = vec_pts_to_raw_array(pts);
    //    flann::Matrix<float> pts_mat(raw_array.get(), pts.size(), 3);
    //    // could utilize CUDA here by changing the KDTreeIndexParams.
    //    FLANNIndexType kd_index(pts_mat, flann::KDTreeSingleIndexParams());
    //    kd_index.buildIndex();

    //    for (const Vector3f& aux_pt : tmp_aux_pts)
    //    {
    //        float aux_d;
    //        bool valid = is_aux_pt_valid(aux_pt, pts, pt_norms, kd_index, aux_d);
    //        if (valid)
    //        {
    //            aux_pts.push_back(aux_pt);
    //            aux_ds.push_back(aux_d);
    //        }
    //    }
    // }
    
    /*
     * Calculate the averaged weighted normal.
     */
	Vector3f calc_average_normal(const std::vector<float>& pts, const std::vector<float>& pt_norms, const WFn& w_fn);
    //Vector3f calc_average_normal(const std::vector<Vector3f>& pts, const std::vector<Vector3f>& pt_norms, const WFn& w_fn)
    //{
    //    Vector3f ave_norm(Vector3f::Zero());
    //    float w(0.0f), total_w(0.0f);

    //    for (unsigned i = 0; i < pts.size(); ++i)
    //    {
    //        const Vector3f& pt = pts[i];
    //        const Vector3f& norm = pt_norms[i];
    //        w = w_fn(pt);
    //        ave_norm += w * norm;
    //        total_w += w;
    //    }
    //    ave_norm /= total_w;
    //    return ave_norm;
    //}
}; // namespace mpu

#endif