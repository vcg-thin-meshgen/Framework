#include "quadratics.hpp"

namespace mpu
{
    float eval_Q(float x, float y, float z, const VectorXf& coeff)
    {
        auto expanded = expand_xyz(x, y, z);
        float val = expanded.dot(coeff);
        return val;
    }

    float eval_Q_biv(const Vector3f& xyz, const VectorXf& coeff, const Matrix4f& M_inv)
    {
        Vector3f local_xyz = apply_transform_pt(M_inv, xyz);
        float local_z = local_xyz(0);

        auto expanded = expand_xyz_biv(local_xyz);
        float val = local_z - expanded.dot(coeff);
        return val;
    }

    Vector3f eval_Q_grad(float x, float y, float z, const VectorXf& coeff)
    {
        float grad_x = 2.0f * coeff(0) * x + coeff(3) * y + coeff(5) * z + coeff(6);
        float grad_y = 2.0f * coeff(1) * y + coeff(3) * x + coeff(4) * z + coeff(7);
        float grad_z = 2.0f * coeff(2) * z + coeff(4) * y + coeff(5) * x + coeff(8);

        Vector3f grad(grad_x, grad_y, grad_z);
        return grad;
    }

    Vector3f eval_Q_biv_grad(const Vector3f& xyz, const VectorXf& coeff, const Matrix4f& M_inv)
    {
        Vector3f local_xyz = apply_transform_pt(M_inv, xyz);
        float local_x(local_xyz(0)), local_y(local_xyz(1)), local_z(local_xyz(2));

        float dQ_dxl = -2.0f * coeff(0) * local_x - coeff(2) * local_y - coeff(3);
        float dQ_dyl = -2.0f * coeff(1) * local_y - coeff(2) * local_x - coeff(4);

        float grad_x = dQ_dxl * M_inv(0,0) + dQ_dyl * M_inv(1,0) + M_inv(2,0);
        float grad_y = dQ_dxl * M_inv(0,1) + dQ_dyl * M_inv(1,1) + M_inv(2,1);
        float grad_z = dQ_dxl * M_inv(0,2) + dQ_dyl * M_inv(1,2) + M_inv(2,2);

        Vector3f grad(grad_x, grad_y, grad_z);
        return grad;
    }

	VectorXf calc_Q_coeff(const Vector3f& center, float half_edge_len, std::vector<float>& pts, 
		std::vector<float>& pt_norms, const WFn& w_fn)
    {
        const unsigned dim = 10;
        MatrixXf A(MatrixXf::Zero(dim, dim));
        VectorXf b(VectorXf::Zero(dim));

        float w(0.0f), total_w(0.0f);
		assert(pts.size() == pt_norms.size());
		for (unsigned ii = 0; ii < pts.size(); ii += 3) 
		{
			w = w_fn(pts[ii], pts[ii + 1U], pts[ii + 2U]);
            total_w += w;

            auto expanded = expand_xyz(pts[ii], pts[ii + 1U], pts[ii + 2U]);
            for (unsigned j = 0; j < dim; ++j)
            {
                A.row(j) += w * expanded(j) * expanded;
            }
		}

        A /= total_w;

        std::vector<float> aux_pts;
        std::vector<float> aux_ds;
        find_valid_aux_pts(center, half_edge_len, pts, pt_norms, aux_pts, aux_ds);

		assert(aux_pts.size() == (aux_ds.size() * 3U));
        if (aux_pts.size() == 0) throw "auxiliary points size is 0";

        w = 1.0f / (float)(aux_pts.size());
        float d_i(0.0f);

        for (unsigned i = 0; i < aux_ds.size(); ++i)
        {
            // const Vector3f& q = aux_pts[i];
            unsigned i_3x = i * 3U;
			auto expanded = expand_xyz(aux_pts[i_3x], aux_pts[i_3x + 1U], aux_pts[i_3x + 2U]);
			d_i = aux_ds[i];

            for (unsigned j = 0; j < dim; ++j)
            {
                A.row(j) += w * expanded(j) * expanded;
                b(j) += w * d_i * expanded(j);
            }
        }

        VectorXf coeff = A.colPivHouseholderQr().solve(b);
        return coeff;
    }

	VectorXf calc_Q_biv_coeff(std::vector<float>& pts, const WFn& w_fn, const Matrix4f& M_inv) 
    {
		assert((pts.size() % 3) == 0);
        const int dim = 6;
        MatrixXf A(MatrixXf::Zero(dim, dim));
        VectorXf b(VectorXf::Zero(dim));

        float w(0.0f), total_w(0.0f), local_z(0.0f);
        Vector3f local_xyz;

        // for (const auto& pt : pts)
		for (unsigned ii = 0; ii < pts.size(); ii += 3U)
        {
            w = w_fn(pts[ii], pts[ii + 1U], pts[ii + 2U]);
            total_w += w;

            local_xyz = apply_transform_pt(M_inv, pts[ii], pts[ii + 1U], pts[ii + 2U]);
            local_z = local_xyz(2);
            auto expanded = expand_xyz_biv(local_xyz);
            for (unsigned j = 0; j < dim; ++j)
            {
                A.row(j) += w * expanded * expanded(j);
                b(j) += w * local_z * expanded(j);
            }
        }
        // shouln't affect anything
        // A /= total_w;
        // b /= total_w;

        VectorXf coeff = A.colPivHouseholderQr().solve(b);
        return coeff;
    }




	float eval_obj_Q(const VectorXf& coeff, 
                    const std::vector<Vector3f>& pts,
                    const WFn& w_fn,
                    const std::vector<Vector3f>& aux_pts,
                    const std::vector<float>& aux_ds)
    {
        float obj_val1(0.0f), w(0.0f), total_w(0.0f), Q(0.0f);
        for (const auto& pt : pts)
        {
            w = w_fn(pt);
            Q = eval_Q(pt, coeff);
            obj_val1 += w * Q * Q;
            total_w += w;
        }
        obj_val1 /= total_w;

        float obj_val2(0.0f), d_i(0.0f);
        size_t m(aux_pts.size());
        if (m)
        {
            for (unsigned i = 0; i < m; ++i)
            {
                const Vector3f& aux_pt = aux_pts[i];
                d_i = aux_ds[i];
                Q = eval_Q(aux_pt, coeff);
                obj_val2 += (Q - d_i) * (Q - d_i);
            }
            obj_val2 /= (float)(m);
        }

        float obj_val = obj_val1 + obj_val2;
        return obj_val;
    }

    float eval_obj_Q_biv(const VectorXf& coeff,
                        const std::vector<Vector3f>& pts,
                        const WFn& w_fn,
                        const Matrix4f& M_inv)
    {
        float obj_val(0.0f), w(0.0f), total_w(0.0f), Q(0.0f);
        for (const auto& pt : pts)
        {
            w = w_fn(pt);
            Q = eval_Q_biv(pt, coeff, M_inv);
            obj_val += w * Q * Q;
            total_w += w;
        }
        obj_val /= total_w;
        
        return obj_val;
    }

    VectorXf eval_obj_Q_grad(const VectorXf& coeff, 
                            const std::vector<Vector3f>& pts,
                            const WFn& w_fn,
                            const std::vector<Vector3f>& aux_pts,
                            const std::vector<float>& aux_ds)
    {
        VectorXf obj_grad1(VectorXf::Zero(coeff.size()));
        float w(0.0f), total_w(0.0f), Q(0.0f);

        for (const auto& pt : pts)
        {
            w = w_fn(pt);
            Q = eval_Q(pt, coeff);
            obj_grad1 += 2.0f * w * Q * expand_xyz(pt);
            total_w += w;
        }
        obj_grad1 /= total_w;

        VectorXf obj_grad2(VectorXf::Zero(coeff.size()));
        float d_i(0.0f);
        size_t m(aux_pts.size());
        if (m)
        {
            for (unsigned i = 0; i < m; ++i)
            {
                const Vector3f& aux_pt = aux_pts[i];
                d_i = aux_ds[i];
                Q = eval_Q(aux_pt, coeff);
                obj_grad2 += 2.0f * (Q - d_i) * expand_xyz(aux_pt);
            }
            obj_grad2 /= (float)(m);
        }

        VectorXf obj_grad = obj_grad1 + obj_grad2;
        return obj_grad;
    }

    VectorXf eval_obj_Q_biv_grad(const VectorXf& coeff,
                                const std::vector<Vector3f>& pts,
                                const WFn& w_fn,
                                const Matrix4f& M_inv)
    {
        VectorXf obj_grad(VectorXf::Zero(coeff.size()));
        float w(0.0f), total_w(0.0f), Q(0.0f);
        Vector3f local_xyz;

        for (const auto& pt : pts)
        {
            w = w_fn(pt);
            Q = eval_Q_biv(pt, coeff, M_inv);

            local_xyz = apply_transform_pt(M_inv, pt);
            obj_grad += -2.0f * w * Q * expand_xyz_biv(local_xyz);
            total_w += w;
        }
        obj_grad /= total_w;

        return obj_grad;
    }
}; // namespace mpu;