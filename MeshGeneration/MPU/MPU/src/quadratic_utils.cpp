#include "quadratic_utils.hpp"

namespace mpu
{
	float B_spline(float x)
    {
        float abs_x = fabs(x);
        if (abs_x <= 0.5f) return 0.75f - abs_x * abs_x;
        else if (abs_x <= 1.5f) return 0.5f * (abs_x - 1.5f) * (abs_x - 1.5f);
        return 0.0f;
    }

	void init_aux_pts(const Vector3f& center, float half_edge_len, std::vector<float>& aux_pts)
    {
		aux_pts.clear();

        std::vector<float> ops = 
        {
            -1.0f, -1.0f, -1.0f,
            +1.0f, -1.0f, -1.0f,
            -1.0f, +1.0f, -1.0f,
            -1.0f, -1.0f, +1.0f,
            +1.0f, +1.0f, -1.0f,
            +1.0f, -1.0f, +1.0f,
            -1.0f, +1.0f, +1.0f,
            +1.0f, +1.0f, +1.0f,
			0.0f, 0.0f, 0.0f
        };

        // push in eight corners and one center
		float aux;
        for (unsigned i = 0; i < ops.size(); i += 3U)
        {
			aux = ops[i] * half_edge_len + center(0);
			aux_pts.push_back(aux);

			aux = ops[i + 1] * half_edge_len + center(1);
			aux_pts.push_back(aux);

			aux = ops[i + 2] * half_edge_len + center(2);
			aux_pts.push_back(aux);
        }
    }

	void find_aux_near_kd(const float *aux_pt, 
                        const std::vector<float>& pts,
                        const FLANNIndexType& kd_index,
                        std::vector<IndexType>& aux_near_indices,
                        std::vector<float>& aux_near_dists)
    {
        aux_near_indices.clear();
        aux_near_dists.clear();

        flann::Matrix<float> query(const_cast<float*>(aux_pt), 1, 3);
        std::vector< std::vector<int> > query_indices;
        std::vector< std::vector<FLANNIndexType::DistanceType> > query_distances;
        // kd-tree kNN search
        kd_index.knnSearch(query, query_indices, query_distances, AUX_NEAR, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));

        for (unsigned i = 0; i < AUX_NEAR; ++i)
        {
            aux_near_indices.push_back(query_indices[0][i]);
            aux_near_dists.push_back(query_distances[0][i]);
        }
    }

    bool is_aux_pt_valid(const float *aux_pt, 
                        const std::vector<float>& pts,
                        const std::vector<float>& pt_norms,
                        const FLANNIndexType& kd_index,
                        float& aux_d)
    {
        std::vector<IndexType> aux_near_indices;
        std::vector<float> aux_near_dists;

        //find_aux_near_brute(aux_pt, pts, aux_near_indices, aux_near_dists);
        find_aux_near_kd(aux_pt, pts, kd_index, aux_near_indices, aux_near_dists);

        assert(aux_near_indices.size() == AUX_NEAR);
        assert(aux_near_indices.size() == aux_near_dists.size());

        aux_d = 0.0f;
        float mult = 0.0f, first_mult = 0.0f;
        bool is_first = true;

		Vector3f aux_pt_v;
		aux_pt_v << aux_pt[0], aux_pt[1], aux_pt[2];
        for (auto q_idx : aux_near_indices)
        {
			unsigned index_base = q_idx * 3U;

			Vector3f pt_v;
			pt_v << pts[index_base], pts[index_base + 1U], pts[index_base + 2U];
			Vector3f pt_norm_v;
			pt_norm_v << pt_norms[index_base], pt_norms[index_base + 1U], pt_norms[index_base + 2U];

            mult = pt_norm_v.dot(aux_pt_v - pt_v);
            aux_d += mult;

            if (is_first)
            {
                first_mult = mult;
                is_first = false;
            }
            else if (first_mult * mult <= 0.0f)
            {
                // make it invalid
                aux_d = std::numeric_limits<float>::min();
                return false;
            }
        }

        aux_d /= aux_near_indices.size();
        return true;
    }

	void find_valid_aux_pts(const Vector3f& center, float half_edge_len,
                            const std::vector<float>& pts,
                            const std::vector<float>& pt_norms,
                            std::vector<float>& aux_pts,
                            std::vector<float>& aux_ds)
    {
        aux_pts.clear();
        aux_ds.clear();

		assert((pts.size() % 3) == 0);
		
        std::vector<float> tmp_aux_pts; // tentative auxiliary points
        init_aux_pts(center, half_edge_len, tmp_aux_pts);

		assert(tmp_aux_pts.size() == 27);
        // build a local kd-tree
        
        flann::Matrix<float> pts_mat(const_cast<float *>(pts.data()), pts.size() / 3, 3);
        // could utilize CUDA here by changing the KDTreeIndexParams.
        FLANNIndexType kd_index(pts_mat, flann::KDTreeSingleIndexParams());
        kd_index.buildIndex();

        for (unsigned i = 0; i < tmp_aux_pts.size(); i += 3U)
        {
			float *aux_pt = tmp_aux_pts.data() + i;
            float aux_d;
            bool valid = is_aux_pt_valid(aux_pt, pts, pt_norms, kd_index, aux_d);
            if (valid)
            {
                aux_pts.push_back(aux_pt[0]);
				aux_pts.push_back(aux_pt[1]);
				aux_pts.push_back(aux_pt[2]);

                aux_ds.push_back(aux_d);
            }
        }
    }

	Vector3f calc_average_normal(const std::vector<float>& pts, const std::vector<float>& pt_norms, const WFn& w_fn)
    {
        Vector3f ave_norm(Vector3f::Zero());
        float w(0.0f), total_w(0.0f);

        for (unsigned i = 0; i < pts.size(); i += 3U)
        {
            w = w_fn(pts[i], pts[i + 1U], pts[i + 2U]);
			Vector3f norm;
			norm << pt_norms[i], pt_norms[i + 1U], pt_norms[i + 2U];
            ave_norm += w * norm;
            total_w += w;
        }

        ave_norm /= total_w;
        return ave_norm;
    }
}; // namespace mpu;