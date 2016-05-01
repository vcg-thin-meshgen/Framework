#ifndef MPU_H
#define MPU_H

#include <utility>
#include <vector>
#include <cassert>
#include <iostream>

#include "math_utils.hpp"
#include "quadratics.hpp"

namespace mpu
{
    class QFnBase
    {
    public:
        virtual ~QFnBase() = default;

        virtual float value(float x, float y, float z) const = 0;
        virtual Vector3f gradient(float x, float y, float z) const = 0;

    };

    class QFn : public QFnBase
    {
    public:
        QFn(const VectorXf& coeff) : m_coeff(coeff) { }

        float value(float x, float y, float z) const override
        {
            return eval_Q(x, y, z, m_coeff); 
        }

        Vector3f gradient(float x, float y, float z) const override
        {
            return eval_Q_grad(x, y, z, m_coeff);
        }

    private:
        VectorXf m_coeff;
    };

	class QBivFn : public QFnBase
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        QBivFn(const VectorXf& coeff, const Matrix4f& M_inv) 
		: m_M_inv(M_inv)
		, m_coeff(coeff) { }

        float value(float x, float y, float z) const override
        {
			return eval_Q_biv(x, y, z, m_coeff, m_M_inv);
        }

        Vector3f gradient(float x, float y, float z) const override
        {
            return eval_Q_biv_grad(x, y, z, m_coeff, m_M_inv);
        }

    private:
        VectorXf m_coeff;
		Matrix4f m_M_inv;
	};

    class BoundingBox
    {
    public:
        typedef Vector3f point_type;

        BoundingBox()
        : m_min(point_type::Zero())
        , m_max(point_type::Zero()) { }

        BoundingBox(const point_type& p1, const point_type& p2)
        : m_min(p1)
        , m_max(p1)
        {
            *this |= p2;
        }

        inline BoundingBox& operator|=(const point_type& p)
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (p(i) < m_min(i)) m_min(i) = p(i);
                if (p(i) > m_max(i)) m_max(i) = p(i);
            }
            return *this;
        }

        inline const point_type& min() const { return m_min; }
        
        inline const point_type& max() const { return m_max; }
        
        inline point_type center() const { return 0.5f * (m_min + m_max); }
        
        inline float edge_size(unsigned axis) const { return m_max(axis) - m_min(axis); }

		inline bool contains(const float* p) const
		{
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (p[i] < m_min(i) || p[i] > m_max(i))
                {
                    return false;
                }
            }
            return true;
		}

		inline bool contains(const point_type& p) const 
        {
			return contains(p.data());
        }
    private:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        const unsigned DIM = 3;

        point_type m_min;
        point_type m_max;
    };

    inline float max_edge_size(const BoundingBox& bbox)
    {
        float max_val = -1.0f;
        float cur_val;
        for (unsigned i = 0; i < 3; ++i)
        {
            cur_val = bbox.edge_size(i);
            max_val = max_val > cur_val ? max_val : cur_val;
        }
        return max_val;
    }

    struct MpuEvalResult
    {
    public:
        float SwQ = 0.0f;
        float Sw = 0.0f;

		unsigned count = 0;
    };

	class MpuOctree;

    class MpuOctreeNode
    {
    public:
        typedef unsigned level_type;

        MpuOctreeNode(const MpuOctree *tree, const BoundingBox& bbox, level_type lv);

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline level_type level() const { return m_level; }

        inline bool is_leaf() const { return m_children.size() == 0; }

        void eval_value(float x, float y,float z, MpuEvalResult& eval_result, float max_error);
        
    private:
		const MpuOctree *m_octree;
		BoundingBox m_bbox;
        Vector3f m_center;
		float m_half_edge_len;
        float m_radius;
        level_type m_level;

        std::vector<MpuOctreeNode> m_children;

        WFn m_w_fn;
        std::shared_ptr<QFnBase> m_q_fn;
		bool m_checked;
		float m_error;

		void compute_coeff();
		
		void subdivide();
        /*friend MpuOctreeNode _build_mpu_octree_node(const std::vector<Vector3f>& pts, const std::vector<Vector3f>& pt_norms,
                                                const FLANNIndexType& kd_index, const BoundingBox& bbox,
                                                float max_error, unsigned level, unsigned max_level, 
                                                unsigned num_min_pts, unsigned num_max_pts, bool& oct_node_valid);*/
    };

    class MpuOctree
    {
    public:
        MpuOctree(const std::vector<float>& pts, const std::vector<float>& pt_norms,
			unsigned max_lv, unsigned num_min_pts, float lambda = 0.1f);
        
		inline const std::vector<float>& pts() const { return m_pts; }
		inline const std::vector<float>& pt_norms() const { return m_pt_norms; }
		inline unsigned max_level() const { return m_max_level; }
		inline unsigned num_min_pts() const { return m_num_min_pts; }
		inline float lambda() const { return m_lambda; }
		inline const FLANNIndexType& kd_index() const { return (*m_kdtree); }
		inline const BoundingBox& bbox() const { return m_bbox; }

        float value(float x, float y, float z, float max_err);
        
    private:
		std::vector<float> m_pts;
		std::vector<float> m_pt_norms;
		unsigned m_max_level;
		unsigned m_num_min_pts;
		float m_lambda;
		BoundingBox m_bbox;
		
		std::shared_ptr<FLANNIndexType> m_kdtree;
        std::shared_ptr<MpuOctreeNode> m_root;
    };

    // unsigned count = 0;

    //MpuOctreeNode _build_mpu_octree_node(const std::vector<Vector3f>& pts, const std::vector<Vector3f>& pt_norms,
    //                                    const FLANNIndexType& kd_index, const BoundingBox& bbox,
    //                                    float max_error, unsigned level, unsigned max_level, 
    //                                    unsigned num_min_pts, unsigned num_max_pts, bool& oct_node_valid)
    //{
    //    oct_node_valid = true;

    //    if (level >= max_level)
    //    {
    //        std::cout << "exceeds max level" << std::endl;

    //        oct_node_valid = false;
    //        return {};
    //    }

    //    Vector3f center = bbox.center();
    //    float half_edge_size = 0.5f * max_edge_size(bbox);

    //    float radius = support_radius(half_edge_size);
    //    bool need_extend_radius = false;

    //    MpuOctreeNode mpu_oct_node(bbox, level, radius);

    //    flann::Matrix<float> query(const_cast<float*>(center.data()), 1, 3);
    //    std::vector< std::vector<int> > query_indices;
    //    std::vector< std::vector<FLANNIndexType::DistanceType> > query_distances;
    //    // kd-tree radius query
    //    kd_index.radiusSearch(query, query_indices, query_distances, radius*radius, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
    //    bool too_many_pts = query_indices[0].size() >= num_max_pts;

    //    if (query_indices[0].size() < num_min_pts)
    //    {
    //        need_extend_radius = true;
    //        query_indices.clear();
    //        query_distances.clear();

    //        kd_index.knnSearch(query, query_indices, query_distances, num_min_pts, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));

    //        radius = query_distances[0].back();
    //        mpu_oct_node.m_radius = radius;
    //    }

    //    std::vector<Vector3f> supp_pts;
    //    std::vector<Vector3f> supp_norms;
    //    bool is_empty_inside = true;
    //    for (int supp_index : query_indices[0])
    //    {
    //        supp_pts.push_back(pts[supp_index]);
    //        supp_norms.push_back(pt_norms[supp_index]);

    //        if (bbox.contains(supp_pts.back()))
    //        {
    //            is_empty_inside = false;
    //        }
    //    }

    //    if (is_empty_inside)
    //    {
    //        // std::cout << "empty inside" << std::endl;
    //        oct_node_valid = false;
    //        return mpu_oct_node;
    //    }

    //    ++count;

    //    WFn w_fn(center, radius);
    //    std::shared_ptr<QFnBase> q_fn(nullptr);
    //    bool will_calc_coeff = (!too_many_pts) || (level == max_level - 1);
    //    if (will_calc_coeff)
    //    {
    //        VectorXf coeff(calc_Q_coeff(center, half_edge_size, supp_pts, supp_norms, w_fn));
    //        q_fn.reset(new QFn(coeff));

    //        float cell_error = 10.0f;
    //        if (cell_error < max_error || need_extend_radius || (level == max_level - 1))
    //        {
    //            mpu_oct_node.m_w_fn = w_fn;
    //            mpu_oct_node.m_q_fn.swap(q_fn);

    //            /*std::cout << "cell_error < max_error: " << (cell_error < max_error) << std::endl;
    //            std::cout << "need_extend_radius: " << need_extend_radius << std::endl;
    //            std::cout << "level == max_level - 1: " << (level == max_level - 1) << std::endl;*/
    //            oct_node_valid = true;
    //            return mpu_oct_node;
    //        }
    //    }

    //    float b_min_x = bbox.min()(0), b_min_y = bbox.min()(1), b_min_z = bbox.min()(2);
    //    float b_max_x = bbox.max()(0), b_max_y = bbox.max()(1), b_max_z = bbox.max()(2);
    //    
    //    std::vector<Vector3f> corner_pts =
    //    {
    //        {b_min_x, b_min_y, b_min_z},
    //        {b_max_x, b_min_y, b_min_z},
    //        {b_min_x, b_max_y, b_min_z},
    //        {b_max_x, b_max_y, b_min_z},
    //        {b_min_x, b_min_y, b_max_z},
    //        {b_max_x, b_min_y, b_max_z},
    //        {b_min_x, b_max_y, b_max_z},
    //        {b_max_x, b_max_y, b_max_z}
    //    };

    //    for (const auto& corner_pt : corner_pts)
    //    {
    //        BoundingBox child_bbox(center, corner_pt);
    //        bool child_oct_node_valid = true;
    //        auto child_oct_node = _build_mpu_octree_node(pts, pt_norms, kd_index, 
    //                                                    child_bbox, max_error, 
    //                                                    level + 1, max_level, 
    //                                                    num_min_pts, num_max_pts, child_oct_node_valid);
    //        if (child_oct_node_valid)
    //        {
    //            mpu_oct_node.m_children.push_back(child_oct_node);
    //        }
    //    }

    //    if (count % 100 == 0)
    //        std::cout << "MPU Octree node " << count << std::endl;

    //    assert(mpu_oct_node.m_children.size());

    //    oct_node_valid = true;
    //    return mpu_oct_node;
    //}

    //MpuOctreeNode build_mpu_octree_node(const std::vector<Vector3f>& pts, const std::vector<Vector3f>& pt_norms,
    //                                    const BoundingBox& bbox, float max_error, unsigned max_level, 
    //                                    unsigned num_min_pts, unsigned num_max_pts, bool& oct_node_valid)
    //{
    //    // build a local kd-tree
    //    std::shared_ptr<float> raw_array = vec_pts_to_raw_array(pts);
    //    flann::Matrix<float> pts_mat(raw_array.get(), pts.size(), 3);
    //    //for (unsigned i = 0; i < pts_mat.rows; ++i)
    //    //{
    //    //    std::cout << *((float*)pts_mat[i]) << " " << *((float*)pts_mat[i]+1) << " " << *((float*)pts_mat[i]+2) << std::endl;
    //    //}
    //    // could utilize CUDA here by changing the KDTreeIndexParams.
    //    FLANNIndexType kd_index(pts_mat, flann::KDTreeSingleIndexParams());
    //    kd_index.buildIndex();

    //    //float center[] = {0.0, 0.0, 0.0};
    //    //flann::Matrix<float> query(center, 1, 3);
    //    //std::vector< std::vector<int> > query_indices;
    //    //std::vector< std::vector<float> > query_distances;
    //    //std::cout << "query found radius pts: " << kd_index.radiusSearch(query, query_indices, query_distances, 27, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED)) << std::endl;;

    //    auto node = _build_mpu_octree_node(pts, pt_norms, kd_index, bbox, max_error, 0, max_level, 
    //                                    num_min_pts, num_max_pts, oct_node_valid);

    //    return node;
    //}
}; // namespace mpu

#endif