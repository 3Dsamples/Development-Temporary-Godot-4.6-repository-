//File 0022 : core/xnorm.hpp
//Vector and matrix norms (L0, L1, L2, Lp, Frobenius, infinity) with SIMD-accelerated accumulation and axis reduction.
#ifndef XTENSOR_XNORM_HPP
#define XTENSOR_XNORM_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xarray.hpp"
#include "xreducer.hpp"
#include "xeval.hpp"

namespace xt
{
    namespace norm
    {
        /**
         * Compute the L0 pseudo-norm (count of non-zero elements) of an expression.
         */
        template <class E>
        inline auto norm_l0(const E& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto non_zero = xt::abs(e) > value_type(0);
            return xt::sum(non_zero);
        }

        /**
         * Compute the L0 pseudo-norm along a specific axis.
         */
        template <class E>
        inline auto norm_l0(const E& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto non_zero = xt::abs(e) > value_type(0);
            auto sum_f = xt::sum(non_zero, axis);
            return sum_f;
        }

        /**
         * Compute the L1 norm (sum of absolute values) of an expression.
         */
        template <class E>
        inline auto norm_l1(const E& e)
        {
            auto abs_e = xt::abs(e);
            return xt::sum(abs_e)();
        }

        /**
         * Compute the L1 norm along a given axis.
         */
        template <class E>
        inline auto norm_l1(const E& e, std::size_t axis)
        {
            auto abs_e = xt::abs(e);
            return xt::sum(abs_e, axis);
        }

        /**
         * Compute the L1 norm over multiple axes.
         */
        template <class E, class X>
        inline auto norm_l1(const E& e, X&& axes)
        {
            auto abs_e = xt::abs(e);
            return xt::sum(abs_e, std::forward<X>(axes));
        }

        /**
         * Compute the L2 norm (Euclidean length) of an expression.
         */
        template <class E>
        inline auto norm_l2(const E& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto sq = xt::pow(xt::abs(e), value_type(2));
            auto sum_sq = xt::sum(sq)();
            return std::sqrt(sum_sq);
        }

        /**
         * Compute the L2 norm along a specific axis.
         */
        template <class E>
        inline auto norm_l2(const E& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto sq = xt::pow(xt::abs(e), value_type(2));
            auto sum_sq = xt::sum(sq, axis);
            return xt::sqrt(sum_sq);
        }

        /**
         * Compute the L2 norm over multiple axes.
         */
        template <class E, class X>
        inline auto norm_l2(const E& e, X&& axes)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto sq = xt::pow(xt::abs(e), value_type(2));
            auto sum_sq = xt::sum(sq, std::forward<X>(axes));
            return xt::sqrt(sum_sq);
        }

        /**
         * Compute the Lp norm (sum(abs(x)^p)^(1/p)) of an expression.
         */
        template <class E>
        inline auto norm_lp(const E& e, double p)
        {
            using value_type = typename std::decay_t<E>::value_type;
            if (p <= 0.0) throw std::runtime_error("p must be positive for Lp norm.");
            auto abs_vals = xt::abs(e);
            auto powered = xt::pow(abs_vals, static_cast<value_type>(p));
            auto sum_val = xt::sum(powered)();
            return std::pow(sum_val, static_cast<value_type>(1.0 / p));
        }

        /**
         * Compute the Lp norm along an axis.
         */
        template <class E>
        inline auto norm_lp(const E& e, double p, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            if (p <= 0.0) throw std::runtime_error("p must be positive.");
            auto abs_vals = xt::abs(e);
            auto powered = xt::pow(abs_vals, static_cast<value_type>(p));
            auto sum_val = xt::sum(powered, axis);
            return xt::pow(sum_val, static_cast<value_type>(1.0 / p));
        }

        /**
         * Compute the L-infinity norm (maximum absolute value) of an expression.
         */
        template <class E>
        inline auto norm_linf(const E& e)
        {
            auto abs_vals = xt::abs(e);
            return xt::amax(abs_vals)();
        }

        /**
         * Compute the L-infinity norm along an axis.
         */
        template <class E>
        inline auto norm_linf(const E& e, std::size_t axis)
        {
            auto abs_vals = xt::abs(e);
            return xt::amax(abs_vals, axis);
        }

        /**
         * Compute the Frobenius norm (same as L2 on the flattened array) of an expression.
         */
        template <class E>
        inline auto norm_frobenius(const E& e)
        {
            return norm_l2(e);
        }

        /**
         * Compute the Frobenius norm along specified axes (flattening the remaining dimensions).
         */
        template <class E, class X>
        inline auto norm_frobenius(const E& e, X&& axes)
        {
            // For axis-specific Frobenius, treat as L2 over the remaining dimensions? Usually not standard; we'll provide as L2 over flattened rest.
            // We'll just compute L2 norm over the flattened representation along the complementary axes.
            using value_type = typename std::decay_t<E>::value_type;
            auto shape = e.shape();
            // Not trivial; we will just reduce over the given axes using L2 as sum of squares then sqrt.
            return norm_l2(e, std::forward<X>(axes));
        }

        /**
         * Generic norm dispatching based on a string or integer type (e.g., "l2", "fro", "inf").
         */
        template <class E>
        inline auto norm(const E& e, const std::string& type)
        {
            if (type == "l0" || type == "L0")
                return norm_l0(e);
            else if (type == "l1" || type == "L1")
                return norm_l1(e);
            else if (type == "l2" || type == "L2")
                return norm_l2(e);
            else if (type == "linf" || type == "Linf" || type == "inf")
                return norm_linf(e);
            else if (type == "fro" || type == "frobenius")
                return norm_frobenius(e);
            else
                throw std::runtime_error("Unknown norm type: " + type);
        }

        /**
         * Generic norm with axis.
         */
        template <class E, class X>
        inline auto norm(const E& e, const std::string& type, X&& axes)
        {
            if (type == "l0" || type == "L0")
                return norm_l0(e, std::forward<X>(axes));
            else if (type == "l1" || type == "L1")
                return norm_l1(e, std::forward<X>(axes));
            else if (type == "l2" || type == "L2")
                return norm_l2(e, std::forward<X>(axes));
            else if (type == "linf" || type == "Linf" || type == "inf")
                return norm_linf(e, std::forward<X>(axes));
            else if (type == "fro" || type == "frobenius")
                return norm_frobenius(e, std::forward<X>(axes));
            else
                throw std::runtime_error("Unknown norm type: " + type);
        }

        /**
         * Compute the matrix norm induced by vector p-norm (currently only supports p=1, inf, fro).
         */
        template <class E>
        inline auto matrix_norm(const E& mat, const std::string& type)
        {
            auto sh = mat.shape();
            if (sh.size() != 2)
                throw std::runtime_error("matrix_norm requires 2D array.");
            if (type == "fro" || type == "frobenius")
                return norm_frobenius(mat);
            if (type == "1")
            {
                // max column sum
                auto abs_mat = xt::abs(mat);
                auto col_sums = xt::sum(abs_mat, 0); // axis 0 -> columns
                return xt::amax(col_sums)();
            }
            if (type == "inf")
            {
                // max row sum
                auto abs_mat = xt::abs(mat);
                auto row_sums = xt::sum(abs_mat, 1); // axis 1 -> rows
                return xt::amax(row_sums)();
            }
            throw std::runtime_error("Unsupported matrix norm type: " + type);
        }

    } // namespace norm
} // namespace xt

#endif // XTENSOR_XNORM_HPP