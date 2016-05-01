#ifndef QUADRATIC_HPP
#define QUADRATIC_HPP

#include <vector>
#include "math_utils.hpp"
#include "quadratic_utils.hpp"

namespace mpu
{
    using Eigen::Matrix4f;
    using namespace transform;

    /*
     * Evaluate the general Q(x, y, z)
     * coeff: [a_0 ~ a_9], length is 10
     * a0 x^2 + a1 y^2 + a2 z^2 + a3 x*y + a4 y*z + a5 x*z + a6 x + a7 y + a8 z + a9
     */
    float eval_Q(float x, float y, float z, const VectorXf& coeff);

    inline float eval_Q(const Vector3f& xyz, const VectorXf& coeff)
    {
        return eval_Q(xyz(0), xyz(1), xyz(2), coeff);
    }

    /*
     * Evaluate the bivariate Q(x, y, z)
     * coeff: [a_0 ~ a_5], length is 6
     * local_z - (a0 local_x^2 + a1 local_y^2 + a2 local_x * local_y + a3 local_x + a4 local_y + a5)
     */
    float eval_Q_biv(const Vector3f& xyz, const VectorXf& coeff, const Matrix4f& M_inv);

    inline float eval_Q_biv(float x, float y, float z, const VectorXf& coeff, const Matrix4f& M_inv)
    {
        Vector3f xyz(x, y, z);
        return eval_Q_biv(xyz, coeff, M_inv);
    }

    /*
     * Evaluate the gradient of the general Q: [dQ/dx, dQ/dy, dQ/dz]
     */
    Vector3f eval_Q_grad(float x, float y, float z, const VectorXf& coeff);

    inline Vector3f eval_Q_grad(const Vector3f& xyz, const VectorXf& coeff)
    {
        return eval_Q_grad(xyz(0), xyz(1), xyz(2), coeff);
    }

    /*
     * Evaluate the gradient of the bivariate Q: [dQ/dx, dQ/dy, dQ/dz]
     */
    Vector3f eval_Q_biv_grad(const Vector3f& xyz, const VectorXf& coeff, const Matrix4f& M_inv);

	inline Vector3f eval_Q_biv_grad(float x, float y, float z, const VectorXf& coeff, const Matrix4f& M_inv)
    {
        Vector3f xyz(x, y, z);
        return eval_Q_biv_grad(xyz, coeff, M_inv);
    }

    /*
     * Calculate the coefficient of general Q function.
     */
	VectorXf calc_Q_coeff(const Vector3f& center, float half_edge_len, std::vector<float>& pts, 
		std::vector<float>& pt_norms, const WFn& w_fn);
    //VectorXf calc_Q_coeff(const Vector3f& center, float half_edge_len,
    //                    const std::vector<Vector3f>& pts, 
    //                    const std::vector<Vector3f>& pt_norms,
    //                    const WFn& w_fn)
    //{
    //    const unsigned dim = 10;
    //    MatrixXf A(MatrixXf::Zero(dim, dim));
    //    VectorXf b(VectorXf::Zero(dim));

    //    float w(0.0f), total_w(0.0f);

    //    for (const auto& pt : pts)
    //    {
    //        w = w_fn(pt);
    //        total_w += w;

    //        auto expanded = expand_xyz(pt);
    //        for (unsigned j = 0; j < dim; ++j)
    //        {
    //            A.row(j) += w * expanded(j) * expanded;
    //        }
    //    }
    //    A /= total_w;

    //    std::vector<Vector3f> aux_pts;
    //    std::vector<float> aux_ds;
    //    find_valid_aux_pts(center, half_edge_len, pts, pt_norms, aux_pts, aux_ds);

    //    if (aux_pts.size() == 0) throw "auxiliary points size is 0";
    //    w = 1.0f / (float)(aux_pts.size());
    //    float d_i(0.0f);

    //    for (unsigned i = 0; i < aux_pts.size(); ++i)
    //    {
    //        const Vector3f& q = aux_pts[i];
    //        d_i = aux_ds[i];
    //        auto expanded = expand_xyz(q);

    //        for (unsigned j = 0; j < dim; ++j)
    //        {
    //            A.row(j) += w * expanded(j) * expanded;
    //            b(j) += w * d_i * expanded(j);
    //        }
    //    }

    //    VectorXf coeff = A.colPivHouseholderQr().solve(b);
    //    return coeff;
    //}
	
	VectorXf calc_Q_biv_coeff(std::vector<float>& pts, const WFn& w_fn, const Matrix4f& M_inv);
    //VectorXf calc_Q_biv_coeff(const std::vector<Vector3f>& pts,
    //                        const WFn& w_fn,
    //                        const Matrix4f& M_inv) 
    //{
    //    const int dim = 6;
    //    MatrixXf A(MatrixXf::Zero(dim, dim));
    //    VectorXf b(VectorXf::Zero(dim));

    //    float w(0.0f), total_w(0.0f), local_z(0.0f);
    //    Vector3f local_xyz;

    //    for (const auto& pt : pts)
    //    {
    //        w = w_fn(pt);
    //        total_w += w;

    //        local_xyz = apply_transform_pt(M_inv, pt);
    //        local_z = local_xyz(2);
    //        auto expanded = expand_xyz_biv(local_xyz);
    //        for (unsigned j = 0; j < dim; ++j)
    //        {
    //            A.row(j) += w * expanded * expanded(j);
    //            b(j) += w * local_z * expanded(j);
    //        }
    //    }
    //    // shouln't affect anything
    //    // A /= total_w;
    //    // b /= total_w;

    //    VectorXf coeff = A.colPivHouseholderQr().solve(b);
    //    return coeff;
    //}

//// test using numerical 

    /*
     * Evaluate the object function composed of the general Q.
     * obj_fn = \sum {(w(p_i)) * Q(p_i)^2} / (\sum w(p_i)) + 1 / m * \sum (Q(q_i) - d_i)^2
     */
    float eval_obj_Q(const VectorXf& coeff, 
                    const std::vector<Vector3f>& pts,
                    const WFn& w_fn,
                    const std::vector<Vector3f>& aux_pts,
                    const std::vector<float>& aux_ds);

    /*
     * Evaluate the object function composed of the general Q.
     * obj_fn = \sum {(w(p_i)) * Q(p_i)^2} / (\sum w(p_i)) + 1 / m * \sum (Q(q_i) - d_i)^2
     */
    float eval_obj_Q_biv(const VectorXf& coeff,
                        const std::vector<Vector3f>& pts,
                        const WFn& w_fn,
                        const Matrix4f& M_inv);

    /*
     * Evaluate the gradient of the object function composed of the general Q w.r.t. the 'coeff'
     */
    VectorXf eval_obj_Q_grad(const VectorXf& coeff, 
                            const std::vector<Vector3f>& pts,
                            const WFn& w_fn,
                            const std::vector<Vector3f>& aux_pts,
                            const std::vector<float>& aux_ds);

    /*
     * Evaluate the gradient of the object function composed of the bivariate Q w.r.t. the 'coeff'
     */
    VectorXf eval_obj_Q_biv_grad(const VectorXf& coeff,
                                const std::vector<Vector3f>& pts,
                                const WFn& w_fn,
                                const Matrix4f& M_inv);

    class ObjQFn;
    class ObjQbivFn;

    struct ObjQParams
    {
    public:
        /*
         * Using non-const reference 'p', 'ap', 'ad' on purpose. Enforcing these data to have
         * a longer life.
         */
        ObjQParams(std::vector<Vector3f>& p, const WFn& w, 
                    std::vector<Vector3f>& ap, std::vector<float>& ad)
        : pts(p), w_fn(w), aux_pts(ap), aux_ds(ad) { }

    private:
        friend class ObjQFn;

        const std::vector<Vector3f>& pts;
        WFn w_fn;
        const std::vector<Vector3f>& aux_pts;
        const std::vector<float>& aux_ds;
    };

    struct ObjQbivParams
    {
    public:
        ObjQbivParams(std::vector<Vector3f>& p, const WFn& w, const Matrix4f& m)
        : pts(p), w_fn(w), M_inv(m) { }

    private:
        friend class ObjQbivFn;

        const std::vector<Vector3f>& pts;
        WFn w_fn;
        Matrix4f M_inv;
    };

    class ObjFnBase
    {
    public:
        virtual ~ObjFnBase() { }

        virtual float operator()(const VectorXf& coeff) const = 0;
        virtual VectorXf eval_grad(const VectorXf& coeff) const = 0;
    };

    class ObjQFn : public ObjFnBase
    {
    public:
        ObjQFn(const ObjQParams& params) : m_params(params) { }

        inline float operator()(const VectorXf& coeff) const override
        {
            return eval_obj_Q(coeff, m_params.pts, m_params.w_fn, 
                                m_params.aux_pts, m_params.aux_ds);
        }

        inline VectorXf eval_grad(const VectorXf& coeff) const override
        {
            return eval_obj_Q_grad(coeff, m_params.pts, m_params.w_fn, 
                                    m_params.aux_pts, m_params.aux_ds);
        }

    private:
        ObjQParams m_params;
    };

    class ObjQbivFn : public ObjFnBase
    {
    public:
        ObjQbivFn(const ObjQbivParams& params) : m_params(params) { }

        inline float operator()(const VectorXf& coeff) const override
        {
            return eval_obj_Q_biv(coeff, m_params.pts, m_params.w_fn, m_params.M_inv);
        }

        inline VectorXf eval_grad(const VectorXf& coeff) const override
        {
            return eval_obj_Q_biv_grad(coeff, m_params.pts, m_params.w_fn, m_params.M_inv);
        }

    private:
        ObjQbivParams m_params;
    };

    template <typename OBJ_FN>
    VectorXf numerical_eval_grad(const OBJ_FN& obj_fn, const VectorXf& args, float epsilon=1e-5f)
    {
        int args_size = args.size();
        VectorXf grad(VectorXf::Zero(args_size));

        for (unsigned i = 0; i < args_size; ++i)
        {
            VectorXf args_l(args);
            VectorXf args_r(args);

            args_l(i) -= epsilon; args_r(i) += epsilon;

            float val_l = obj_fn(args_l);
            float val_r = obj_fn(args_r);
            grad(i) = (val_r - val_l) / (2.0f * epsilon);
        }

        return grad;
    }
}; // namespace mpu

#endif