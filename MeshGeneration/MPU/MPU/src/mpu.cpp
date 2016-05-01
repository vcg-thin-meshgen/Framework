#include "mpu.hpp"

namespace mpu
{
	MpuOctreeNode::MpuOctreeNode(const MpuOctree *tree, const BoundingBox& bbox, level_type lv)
    : m_octree(tree)
	, m_bbox(bbox)
	, m_center(bbox.center())
	, m_half_edge_len(0.5f * max_edge_size(bbox))
	, m_radius(0.0f)
    , m_level(lv)
    , m_w_fn(Vector3f::Zero(), 0.0f)
    , m_q_fn(nullptr) 
	, m_checked(false)
	, m_error(100000000.0f) 
	{ 
		m_radius = support_radius(m_half_edge_len);
	}

	void MpuOctreeNode::eval_value(float x, float y,float z, MpuEvalResult& eval_result, float max_error)
    {
        namespace tr = transform;

		float dist = tr::norm(x - m_center(0), y - m_center(1), z - m_center(2));
		if (dist > m_radius)
            return;

		if (!m_checked)
		{
			compute_coeff();
		}
		
		if ((m_error >= max_error) && (m_level < m_octree->max_level()))
		{
			if (m_q_fn) 
			{
				m_q_fn.reset();
			}

			if (is_leaf())
			{
				subdivide();
			}

			for (auto& child : m_children)
            {
                child.eval_value(x, y, z, eval_result, max_error);
            }
		}
		else
		{
			if (!m_q_fn)
			{
				compute_coeff();
			}

			float w = m_w_fn(x, y, z);
			float Q = m_q_fn->value(x, y, z);

			eval_result.SwQ += w * Q;
            eval_result.Sw  += w;
			eval_result.count += 1;
		}
    }

	bool is_node_empty(float* center, float radius, const FLANNIndexType& kd_index)
	{
		flann::Matrix<float> query(center, 1, 3);
        std::vector< std::vector<int> > query_indices;
        std::vector< std::vector<FLANNIndexType::DistanceType> > query_distances;

		kd_index.radiusSearch(query, query_indices, query_distances, radius * radius, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
		return query_indices[0].size() == 0;
	}

	void MpuOctreeNode::compute_coeff()
	{
		m_checked = true;

		float radius = support_radius(m_half_edge_len);
		m_radius = radius;

		flann::Matrix<float> query(const_cast<float*>(m_center.data()), 1, 3);
        std::vector< std::vector<int> > query_indices;
        std::vector< std::vector<FLANNIndexType::DistanceType> > query_distances;
        // kd-tree radius query
        m_octree->kd_index().radiusSearch(query, query_indices, query_distances, m_radius * m_radius, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));

        while (query_indices[0].size() < m_octree->num_min_pts())
        {
            query_indices.clear();
            query_distances.clear();

			m_radius += m_octree->lambda() * radius;
			m_octree->kd_index().radiusSearch(query, query_indices, query_distances, m_radius * m_radius, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
        }
		// copy to support pts in local region
		std::vector<float> supp_pts;
        std::vector<float> supp_norms;
		
		const std::vector<float>& pts = m_octree->pts();
		const std::vector<float>& pt_norms = m_octree->pt_norms();

        for (int q_index : query_indices[0])
        {
			unsigned supp_index_base = q_index * 3U;

            supp_pts.push_back(pts[supp_index_base]);
			supp_pts.push_back(pts[supp_index_base + 1U]);
			supp_pts.push_back(pts[supp_index_base + 2U]);

            supp_norms.push_back(pt_norms[supp_index_base]);
			supp_norms.push_back(pt_norms[supp_index_base + 1U]);
			supp_norms.push_back(pt_norms[supp_index_base + 2U]);
        }

		// check which quadratic function to use
		m_w_fn = WFn(m_center, m_radius);
		Vector3f avg_norm = calc_average_normal(supp_pts, supp_norms, m_w_fn);
		bool greater_half_pi = false;
		for (unsigned n_i = 0; n_i < supp_norms.size(); n_i += 3U)
		{
			if ((avg_norm(0) * supp_norms[n_i] + avg_norm(1) * supp_norms[n_i + 1U] + avg_norm(2) * supp_norms[n_i + 2U]) < 0)
			{
				greater_half_pi = true;
				break;
			}
		}
		// compute quadratic function
        std::shared_ptr<QFnBase> q_fn(nullptr);
		if (greater_half_pi)
		{
			VectorXf coeff(calc_Q_coeff(m_center, m_half_edge_len, supp_pts, supp_norms, m_w_fn));
			m_q_fn.reset(new QFn(coeff));
		}
		else
		{
			Matrix4f M = to_local_coordinate(m_center, avg_norm);
			Matrix4f M_inv = M.inverse();
			VectorXf coeff(calc_Q_biv_coeff(supp_pts, m_w_fn, M_inv));
			
			m_q_fn.reset(new QBivFn(coeff, M_inv));
		}
		// compute fitting error
		namespace tr = transform;
		float curmax_err = -100.0f;
		for (unsigned i = 0; i < supp_pts.size(); i += 3U)
		{
			float x = supp_pts[i], y = supp_pts[i + 1U], z = supp_pts[i + 2U];
			if (tr::norm(x - m_center(0), y - m_center(1), z - m_center(2)) < m_radius)
			{
				Vector3f grad = m_q_fn->gradient(x, y, z);
				float err = abs(m_q_fn->value(x, y, z)) / abs(tr::norm(m_q_fn->gradient(x, y, z)));
				if (err > curmax_err) curmax_err = err;
			}
		}

		m_error = curmax_err;
	}

	void MpuOctreeNode::subdivide()
	{
		float b_min_x = m_bbox.min()(0), b_min_y = m_bbox.min()(1), b_min_z = m_bbox.min()(2);
        float b_max_x = m_bbox.max()(0), b_max_y = m_bbox.max()(1), b_max_z = m_bbox.max()(2);
        
        std::vector<Vector3f> corner_pts =
        {
            {b_min_x, b_min_y, b_min_z},
            {b_max_x, b_min_y, b_min_z},
            {b_min_x, b_max_y, b_min_z},
            {b_max_x, b_max_y, b_min_z},
            {b_min_x, b_min_y, b_max_z},
            {b_max_x, b_min_y, b_max_z},
            {b_min_x, b_max_y, b_max_z},
            {b_max_x, b_max_y, b_max_z}
        };

        for (const auto& corner_pt : corner_pts)
        {
            BoundingBox child_bbox(m_center, corner_pt);
			Vector3f child_center(child_bbox.center());
			float child_radius = support_radius(0.5f * max_edge_size(child_bbox));

            if (!is_node_empty(child_center.data(), child_radius, m_octree->kd_index()))
			{
				MpuOctreeNode child_node(m_octree, child_bbox, m_level + 1);
				m_children.push_back(child_node);
			}
        }
	}

	MpuOctree::MpuOctree(const std::vector<float>& pts, const std::vector<float>& pt_norms,
			unsigned max_lv, unsigned num_min_pts, float lambda) 
			: m_pts(pts)
			, m_pt_norms(pt_norms)
			, m_max_level(max_lv)
			, m_num_min_pts(num_min_pts)
			, m_lambda(lambda)
	{
        
		assert((m_pts.size() % 3) == 0);
        flann::Matrix<float> pts_mat(m_pts.data(), m_pts.size() / 3U, 3U);
        
        // could utilize CUDA here by changing the KDTreeIndexParams.
        m_kdtree.reset(new FLANNIndexType(pts_mat, flann::KDTreeIndexParams(1)));
        m_kdtree->buildIndex();

		// BoundingBox root_bbox;
		for (unsigned i = 0; i < m_pts.size(); i += 3U)
		{
			Vector3f pt;
			pt << m_pts[i], m_pts[i + 1U], m_pts[i + 2U];
			m_bbox |= pt;
		}

		m_root.reset(new MpuOctreeNode(const_cast<MpuOctree*>(this), m_bbox, 0));
	}

	float MpuOctree::value(float x, float y, float z, float max_err)
    {
        MpuEvalResult eval_result;
        m_root->eval_value(x, y, z, eval_result, max_err);
           
		if (eval_result.count == 0)
		{
			return 1000.0f;
		}
        else if (eval_result.Sw <= 1e-10f)
        {
			float log_swq = log(abs(eval_result.SwQ));
			float log_sw = log(eval_result.Sw);
			float log_res = log_swq - log_sw;
			float sign = (eval_result.SwQ < 0) ? -1.0f : 1.0f;
			
			if (log_res > 20.0f)
			{
				return 1000.0f;
			}
			else if (log_res < -10.0f)
			{
				return 0.0f;
			}
			else
			{
				return exp(log_res) * sign;
			}
        }

        return eval_result.SwQ / eval_result.Sw;
    }
}; // namespace mpu;