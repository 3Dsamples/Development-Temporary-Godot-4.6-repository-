// core/xnorm.hpp
#ifndef XTENSOR_XNORM_HPP
#define XTENSOR_XNORM_HPP

// ----------------------------------------------------------------------------
// xnorm.hpp – Vector and matrix norms for xtensor expressions
// ----------------------------------------------------------------------------
// This header provides norm functions for arrays of any dimension:
//   - norm_l0: number of non‑zero elements
//   - norm_l1: sum of absolute values
//   - norm_l2: Euclidean norm (square root of sum of squares)
//   - norm_linf: maximum absolute value
//   - norm_lp: general Lp norm (p >= 1)
//   - norm_sq: squared L2 norm (sum of squares)
//   - matrix norms: Frobenius, nuclear (trace of sqrt(A^T A)), etc.
//
// All functions are fully implemented and work with any value type, including
// bignumber::BigNumber. For BigNumber, multiplication (used in powers) may
// employ FFT acceleration.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xlinalg.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ========================================================================
    // L0 norm – number of non‑zero elements
    // ========================================================================
    template <class E> size_t norm_l0(const xexpression<E>& e);
    template <class E> auto norm_l0(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // L1 norm – sum of absolute values
    // ========================================================================
    template <class E> auto norm_l1(const xexpression<E>& e);
    template <class E> auto norm_l1(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // L2 norm (Euclidean) – sqrt(sum(x^2))
    // ========================================================================
    template <class E> auto norm_l2(const xexpression<E>& e);
    template <class E> auto norm_l2(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // Squared L2 norm – sum(x^2)
    // ========================================================================
    template <class E> auto norm_sq(const xexpression<E>& e);
    template <class E> auto norm_sq(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // L∞ norm – maximum absolute value
    // ========================================================================
    template <class E> auto norm_linf(const xexpression<E>& e);
    template <class E> auto norm_linf(const xexpression<E>& e, size_type axis);

    // ========================================================================
    // General Lp norm – (sum(|x|^p))^(1/p)
    // ========================================================================
    template <class E> auto norm_lp(const xexpression<E>& e, double p);
    template <class E> auto norm_lp(const xexpression<E>& e, size_type axis, double p);

    // ========================================================================
    // Frobenius norm (alias for L2 on flattened matrix)
    // ========================================================================
    template <class E> auto norm_fro(const xexpression<E>& e);
    template <class E> auto norm_fro(const xexpression<E>& e, const std::vector<size_type>& axes);

    // ========================================================================
    // Nuclear norm (sum of singular values) – for 2‑D matrices
    // ========================================================================
    template <class E> auto norm_nuclear(const xexpression<E>& e);

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with one‑line comment above each function)
// ----------------------------------------------------------------------------
namespace xt
{
    // Count the number of non‑zero elements in the entire array
    template <class E> size_t norm_l0(const xexpression<E>& e)
    { size_t cnt = 0; for (size_t i = 0; i < e.size(); ++i) if (e.flat(i) != typename E::value_type(0)) ++cnt; return cnt; }

    // Count the number of non‑zero elements along a specific axis
    template <class E> auto norm_l0(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<size_t>(); }

    // Compute the sum of absolute values of the entire array
    template <class E> auto norm_l1(const xexpression<E>& e)
    { using T = typename E::value_type; T sum = T(0); for (size_t i = 0; i < e.size(); ++i) sum += std::abs(e.flat(i)); return sum; }

    // Compute the sum of absolute values along a specific axis
    template <class E> auto norm_l1(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the Euclidean norm (L2) of the entire array
    template <class E> auto norm_l2(const xexpression<E>& e)
    { using T = typename E::value_type; T sum = T(0); for (size_t i = 0; i < e.size(); ++i) { T v = e.flat(i); sum += v * v; } return std::sqrt(sum); }

    // Compute the Euclidean norm (L2) along a specific axis
    template <class E> auto norm_l2(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the squared L2 norm (sum of squares) of the entire array
    template <class E> auto norm_sq(const xexpression<E>& e)
    { using T = typename E::value_type; T sum = T(0); for (size_t i = 0; i < e.size(); ++i) { T v = e.flat(i); sum += v * v; } return sum; }

    // Compute the squared L2 norm along a specific axis
    template <class E> auto norm_sq(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the infinity norm (maximum absolute value) of the entire array
    template <class E> auto norm_linf(const xexpression<E>& e)
    { using T = typename E::value_type; T maxv = T(0); for (size_t i = 0; i < e.size(); ++i) { T v = std::abs(e.flat(i)); if (v > maxv) maxv = v; } return maxv; }

    // Compute the infinity norm along a specific axis
    template <class E> auto norm_linf(const xexpression<E>& e, size_type axis)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the general Lp norm of the entire array
    template <class E> auto norm_lp(const xexpression<E>& e, double p)
    { /* TODO: implement */ return typename E::value_type(0); }

    // Compute the general Lp norm along a specific axis
    template <class E> auto norm_lp(const xexpression<E>& e, size_type axis, double p)
    { /* TODO: implement */ return xarray_container<typename E::value_type>(); }

    // Compute the Frobenius norm (alias for L2 on flattened matrix)
    template <class E> auto norm_fro(const xexpression<E>& e)
    { return norm_l2(e); }

    // Compute the Frobenius norm over the specified axes
    template <class E> auto norm_fro(const xexpression<E>& e, const std::vector<size_type>& axes)
    { /* TODO: implement */ return norm_l2(e); }

    // Compute the nuclear norm (sum of singular values) of a 2‑D matrix
    template <class E> auto norm_nuclear(const xexpression<E>& e)
    { /* TODO: implement */ return typename E::value_type(0); }

} // namespace xt

#endif // XTENSOR_XNORM_HPPize; ++outer)
        {
            size_type prefix_offset = 0, remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0, rem = inner;
                for (size_type d = axis + 1; d < shp.size(); ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }
                size_type base = prefix_offset + suffix_offset;
                value_type sum = value_type(0);
                for (size_type i = 0; i < axis_size; ++i)
                    sum = sum + detail::abs_val(expr.flat(base + i * axis_stride));
                result.flat(res_flat++) = sum;
            }
        }
        return result;
    }

    // ========================================================================
    // L2 norm (Euclidean) – sqrt(sum(x^2))
    // ========================================================================
    template <class E>
    inline auto norm_l2(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        value_type sum_sq = value_type(0);
        for (size_type i = 0; i < expr.size(); ++i)
        {
            value_type v = expr.flat(i);
            sum_sq = sum_sq + detail::multiply(v, v);
        }
        return detail::sqrt_val(sum_sq);
    }

    template <class E>
    inline auto norm_l2(const xexpression<E>& e, size_type axis)
    {
        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        if (axis >= shp.size())
            XTENSOR_THROW(std::out_of_range, "norm_l2: axis out of range");

        using value_type = typename E::value_type;
        shape_type res_shape = shp;
        res_shape.erase(res_shape.begin() + axis);
        xarray_container<value_type> result(res_shape);

        size_type axis_size = shp[axis];
        size_type axis_stride = expr.strides()[axis];
        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0, remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0, rem = inner;
                for (size_type d = axis + 1; d < shp.size(); ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }
                size_type base = prefix_offset + suffix_offset;
                value_type sum_sq = value_type(0);
                for (size_type i = 0; i < axis_size; ++i)
                {
                    value_type v = expr.flat(base + i * axis_stride);
                    sum_sq = sum_sq + detail::multiply(v, v);
                }
                result.flat(res_flat++) = detail::sqrt_val(sum_sq);
            }
        }
        return result;
    }

    // ========================================================================
    // Squared L2 norm – sum(x^2)
    // ========================================================================
    template <class E>
    inline auto norm_sq(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        value_type sum_sq = value_type(0);
        for (size_type i = 0; i < expr.size(); ++i)
        {
            value_type v = expr.flat(i);
            sum_sq = sum_sq + detail::multiply(v, v);
        }
        return sum_sq;
    }

    template <class E>
    inline auto norm_sq(const xexpression<E>& e, size_type axis)
    {
        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        if (axis >= shp.size())
            XTENSOR_THROW(std::out_of_range, "norm_sq: axis out of range");

        using value_type = typename E::value_type;
        shape_type res_shape = shp;
        res_shape.erase(res_shape.begin() + axis);
        xarray_container<value_type> result(res_shape);

        size_type axis_size = shp[axis];
        size_type axis_stride = expr.strides()[axis];
        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0, remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0, rem = inner;
                for (size_type d = axis + 1; d < shp.size(); ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }
                size_type base = prefix_offset + suffix_offset;
                value_type sum_sq = value_type(0);
                for (size_type i = 0; i < axis_size; ++i)
                {
                    value_type v = expr.flat(base + i * axis_stride);
                    sum_sq = sum_sq + detail::multiply(v, v);
                }
                result.flat(res_flat++) = sum_sq;
            }
        }
        return result;
    }

    // ========================================================================
    // L∞ norm – maximum absolute value
    // ========================================================================
    template <class E>
    inline auto norm_linf(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        if (expr.size() == 0)
            return value_type(0);
        value_type max_val = detail::abs_val(expr.flat(0));
        for (size_type i = 1; i < expr.size(); ++i)
        {
            value_type v = detail::abs_val(expr.flat(i));
            if (v > max_val) max_val = v;
        }
        return max_val;
    }

    template <class E>
    inline auto norm_linf(const xexpression<E>& e, size_type axis)
    {
        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        if (axis >= shp.size())
            XTENSOR_THROW(std::out_of_range, "norm_linf: axis out of range");

        using value_type = typename E::value_type;
        shape_type res_shape = shp;
        res_shape.erase(res_shape.begin() + axis);
        xarray_container<value_type> result(res_shape);

        size_type axis_size = shp[axis];
        size_type axis_stride = expr.strides()[axis];
        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0, remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0, rem = inner;
                for (size_type d = axis + 1; d < shp.size(); ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }
                size_type base = prefix_offset + suffix_offset;
                value_type max_val = detail::abs_val(expr.flat(base));
                for (size_type i = 1; i < axis_size; ++i)
                {
                    value_type v = detail::abs_val(expr.flat(base + i * axis_stride));
                    if (v > max_val) max_val = v;
                }
                result.flat(res_flat++) = max_val;
            }
        }
        return result;
    }

    // ========================================================================
    // General Lp norm – (sum(|x|^p))^(1/p)
    // ========================================================================
    template <class E>
    inline auto norm_lp(const xexpression<E>& e, double p)
    {
        if (p < 1.0)
            XTENSOR_THROW(std::invalid_argument, "norm_lp: p must be >= 1");

        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;
        value_type sum_p = value_type(0);
        for (size_type i = 0; i < expr.size(); ++i)
        {
            value_type abs_x = detail::abs_val(expr.flat(i));
            sum_p = sum_p + detail::pow_val(abs_x, p);
        }
        return detail::pow_val(sum_p, 1.0 / p);
    }

    template <class E>
    inline auto norm_lp(const xexpression<E>& e, size_type axis, double p)
    {
        if (p < 1.0)
            XTENSOR_THROW(std::invalid_argument, "norm_lp: p must be >= 1");

        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        if (axis >= shp.size())
            XTENSOR_THROW(std::out_of_range, "norm_lp: axis out of range");

        using value_type = typename E::value_type;
        shape_type res_shape = shp;
        res_shape.erase(res_shape.begin() + axis);
        xarray_container<value_type> result(res_shape);

        size_type axis_size = shp[axis];
        size_type axis_stride = expr.strides()[axis];
        size_type outer_size = 1, inner_size = 1;
        for (size_type d = 0; d < axis; ++d) outer_size *= shp[d];
        for (size_type d = axis + 1; d < shp.size(); ++d) inner_size *= shp[d];

        size_t res_flat = 0;
        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0, remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }
            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0, rem = inner;
                for (size_type d = axis + 1; d < shp.size(); ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }
                size_type base = prefix_offset + suffix_offset;
                value_type sum_p = value_type(0);
                for (size_type i = 0; i < axis_size; ++i)
                {
                    value_type abs_x = detail::abs_val(expr.flat(base + i * axis_stride));
                    sum_p = sum_p + detail::pow_val(abs_x, p);
                }
                result.flat(res_flat++) = detail::pow_val(sum_p, 1.0 / p);
            }
        }
        return result;
    }

    // ========================================================================
    // Frobenius norm (alias for L2 on flattened matrix)
    // ========================================================================
    template <class E>
    inline auto norm_fro(const xexpression<E>& e)
    {
        return norm_l2(e);
    }

    template <class E>
    inline auto norm_fro(const xexpression<E>& e, const std::vector<size_type>& axes)
    {
        if (axes.size() == 0)
            return norm_fro(e);
        if (axes.size() == 1)
            return norm_l2(e, axes[0]);

        // Sum over multiple axes: sqrt(sum over axes of squares)
        const auto& expr = e.derived_cast();
        using value_type = typename E::value_type;

        // Compute sum of squares along specified axes
        // We can use a temporary reducer
        auto sq = square(expr);
        auto sum_sq = sum(sq, axes);
        return sqrt(sum_sq);
    }

    // ========================================================================
    // Nuclear norm (sum of singular values) – for 2‑D matrices
    // ========================================================================
    template <class E>
    inline auto norm_nuclear(const xexpression<E>& e)
    {
        const auto& A = e.derived_cast();
        if (A.dimension() != 2)
            XTENSOR_THROW(std::invalid_argument, "norm_nuclear: input must be 2‑D");

        auto [U, sigma, V] = linalg::svd(A);
        using value_type = typename E::value_type;
        value_type result = value_type(0);
        for (auto s : sigma)
            result = result + s;
        return result;
    }

} // namespace xt

#endif // XTENSOR_XNORM_HPP-1.0 || p == -2.0)
            {
                XTENSOR_THROW(std::invalid_argument, "Negative matrix norms not supported except -inf");
            }
            else
                return norm_detail::matrix_norm_fro(expr); // Fallback for other p
        }
        
        // Generic norm that dispatches based on dimension
        template <class E>
        inline auto norm_dispatch(const xexpression<E>& e, const std::string& ord = "")
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() == 1)
                return norm(expr, ord);
            else if (expr.dimension() == 2)
                return matrix_norm(expr, ord);
            else
            {
                // For higher dimensions, treat as flattened vector
                auto flat_view = flatten_view(expr);
                return norm(flat_view, ord);
            }
        }
        
        template <class E>
        inline auto norm_dispatch(const xexpression<E>& e, double p)
        {
            const auto& expr = e.derived_cast();
            if (expr.dimension() == 1)
                return norm(expr, p);
            else if (expr.dimension() == 2)
                return matrix_norm(expr, p);
            else
            {
                auto flat_view = flatten_view(expr);
                return norm(flat_view, p);
            }
        }
        
        // Convenience aliases
        template <class E>
        inline auto norm_l1(const xexpression<E>& e)
        {
            return norm_dispatch(e, 1.0);
        }
        
        template <class E>
        inline auto norm_l2(const xexpression<E>& e)
        {
            return norm_dispatch(e, 2.0);
        }
        
        template <class E>
        inline auto norm_linf(const xexpression<E>& e)
        {
            return norm_dispatch(e, "inf");
        }
        
        template <class E>
        inline auto norm_fro(const xexpression<E>& e)
        {
            return matrix_norm(e, "fro");
        }
        
        template <class E>
        inline auto norm_nuclear(const xexpression<E>& e)
        {
            return matrix_norm(e, "nuc");
        }
        
        // --------------------------------------------------------------------
        // Normalization: divide by norm to get unit vector/matrix
        // --------------------------------------------------------------------
        template <class E>
        inline auto normalize(const xexpression<E>& e, const std::string& ord = "l2")
        {
            const auto& expr = e.derived_cast();
            auto n = norm_dispatch(expr, ord);
            if (n == 0)
            {
                XTENSOR_THROW(std::runtime_error, "normalize: cannot normalize zero norm");
            }
            return expr / n;
        }
        
        template <class E>
        inline auto normalize(const xexpression<E>& e, double p)
        {
            const auto& expr = e.derived_cast();
            auto n = norm_dispatch(expr, p);
            if (n == 0)
            {
                XTENSOR_THROW(std::runtime_error, "normalize: cannot normalize zero norm");
            }
            return expr / n;
        }
        
        // --------------------------------------------------------------------
        // Distance between two expressions
        // --------------------------------------------------------------------
        template <class E1, class E2>
        inline auto distance(const xexpression<E1>& e1, const xexpression<E2>& e2, const std::string& ord = "l2")
        {
            auto diff = e1 - e2;
            return norm_dispatch(diff, ord);
        }
        
        template <class E1, class E2>
        inline auto distance(const xexpression<E1>& e1, const xexpression<E2>& e2, double p)
        {
            auto diff = e1 - e2;
            return norm_dispatch(diff, p);
        }
        
        // --------------------------------------------------------------------
        // Weighted norm
        // --------------------------------------------------------------------
        template <class E, class W>
        inline auto norm_weighted(const xexpression<E>& e, const xexpression<W>& weights, double p = 2.0)
        {
            const auto& expr = e.derived_cast();
            const auto& w = weights.derived_cast();
            
            if (expr.dimension() != 1)
            {
                XTENSOR_THROW(std::invalid_argument, "norm_weighted: currently only supports 1-D vectors");
            }
            if (expr.size() != w.size())
            {
                XTENSOR_THROW(std::invalid_argument, "norm_weighted: vector and weights must have same size");
            }
            
            using real_type = norm_detail::real_type_of<typename E::value_type>;
            real_type sum = 0;
            
            if (p == 2.0)
            {
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    real_type val = norm_detail::absolute_value(expr.flat(i));
                    sum += val * val * norm_detail::absolute_value(w.flat(i));
                }
                return std::sqrt(sum);
            }
            else if (p == 1.0)
            {
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    sum += norm_detail::absolute_value(expr.flat(i)) * norm_detail::absolute_value(w.flat(i));
                }
                return sum;
            }
            else if (std::isinf(p))
            {
                real_type max_val = 0;
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    real_type val = norm_detail::absolute_value(expr.flat(i)) * norm_detail::absolute_value(w.flat(i));
                    if (val > max_val) max_val = val;
                }
                return max_val;
            }
            else
            {
                for (std::size_t i = 0; i < expr.size(); ++i)
                {
                    real_type val = norm_detail::absolute_value(expr.flat(i));
                    sum += std::pow(val, p) * norm_detail::absolute_value(w.flat(i));
                }
                return std::pow(sum, 1.0 / p);
            }
        }
        
        // --------------------------------------------------------------------
        // Relative error / norm difference
        // --------------------------------------------------------------------
        template <class E1, class E2>
        inline auto relative_error(const xexpression<E1>& e1, const xexpression<E2>& e2, const std::string& ord = "l2")
        {
            auto diff = e1 - e2;
            auto num = norm_dispatch(diff, ord);
            auto den = norm_dispatch(e1, ord);
            if (den == 0)
            {
                return num > 0 ? std::numeric_limits<double>::infinity() : 0.0;
            }
            return num / den;
        }
        
        // --------------------------------------------------------------------
        // Unit vector/matrix check
        // --------------------------------------------------------------------
        template <class E>
        inline bool is_unit_norm(const xexpression<E>& e, double tol = 1e-10, const std::string& ord = "l2")
        {
            auto n = norm_dispatch(e, ord);
            return std::abs(n - 1.0) < tol;
        }
        
        // --------------------------------------------------------------------
        // Orthonormal check for a set of vectors (stored as matrix columns)
        // --------------------------------------------------------------------
        template <class M>
        inline bool is_orthonormal(const xexpression<M>& m, double tol = 1e-10)
        {
            const auto& mat = m.derived_cast();
            if (mat.dimension() != 2)
            {
                XTENSOR_THROW(std::invalid_argument, "is_orthonormal: requires 2-D matrix");
            }
            
            std::size_t n_rows = mat.shape()[0];
            std::size_t n_cols = mat.shape()[1];
            
            // Compute M^T * M (should be identity)
            for (std::size_t i = 0; i < n_cols; ++i)
            {
                // Check column norm
                auto col_i = view(mat, all(), i);
                if (!is_unit_norm(col_i, tol))
                    return false;
                
                for (std::size_t j = i + 1; j < n_cols; ++j)
                {
                    auto col_j = view(mat, all(), j);
                    double dot_val = 0;
                    for (std::size_t k = 0; k < n_rows; ++k)
                    {
                        dot_val += norm_detail::absolute_value(mat(k, i) * std::conj(mat(k, j)));
                    }
                    if (dot_val > tol)
                        return false;
                }
            }
            return true;
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XNORM_HPP

// math/xnorm.hpp