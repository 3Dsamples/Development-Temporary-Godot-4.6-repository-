//File 0352 : xframe/xvariable_math.hpp
//Element-wise mathematical functions for variables: sin, cos, exp, log, sqrt, abs, pow, clip, and more, with SIMD-accelerated evaluation and lazy expression nodes.
#ifndef XFRAME_XVARIABLE_MATH_HPP
#define XFRAME_XVARIABLE_MATH_HPP

#include <cmath>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xvariable.hpp"
#include "xvariable_function.hpp"

namespace xframe
{
    namespace math
    {
        // Unary functors with optional simd_apply for xsimd.
        struct abs_fun
        {
            template <class T> T operator()(T x) const { return std::abs(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::abs(x); }
        };

        struct sqrt_fun
        {
            template <class T> T operator()(T x) const { return std::sqrt(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::sqrt(x); }
        };

        struct exp_fun
        {
            template <class T> T operator()(T x) const { return std::exp(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::exp(x); }
        };

        struct log_fun
        {
            template <class T> T operator()(T x) const { return std::log(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::log(x); }
        };

        struct sin_fun
        {
            template <class T> T operator()(T x) const { return std::sin(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::sin(x); }
        };

        struct cos_fun
        {
            template <class T> T operator()(T x) const { return std::cos(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::cos(x); }
        };

        struct tan_fun
        {
            template <class T> T operator()(T x) const { return std::tan(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::tan(x); }
        };

        struct asin_fun
        {
            template <class T> T operator()(T x) const { return std::asin(x); }
        };

        struct acos_fun
        {
            template <class T> T operator()(T x) const { return std::acos(x); }
        };

        struct atan_fun
        {
            template <class T> T operator()(T x) const { return std::atan(x); }
        };

        struct sinh_fun
        {
            template <class T> T operator()(T x) const { return std::sinh(x); }
        };

        struct cosh_fun
        {
            template <class T> T operator()(T x) const { return std::cosh(x); }
        };

        struct tanh_fun
        {
            template <class T> T operator()(T x) const { return std::tanh(x); }
        };

        struct ceil_fun
        {
            template <class T> T operator()(T x) const { return std::ceil(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::ceil(x); }
        };

        struct floor_fun
        {
            template <class T> T operator()(T x) const { return std::floor(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::floor(x); }
        };

        struct round_fun
        {
            template <class T> T operator()(T x) const { return std::round(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::round(x); }
        };

        struct trunc_fun
        {
            template <class T> T operator()(T x) const { return std::trunc(x); }
        };

        struct pow_fun
        {
            template <class T> T operator()(T x, T y) const { return std::pow(x, y); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x, xsimd::batch<T, default_simd_arch> y) const { return xsimd::pow(x, y); }
        };

        struct atan2_fun
        {
            template <class T> T operator()(T y, T x) const { return std::atan2(y, x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> y, xsimd::batch<T, default_simd_arch> x) const { return xsimd::atan2(y, x); }
        };

        struct hypot_fun
        {
            template <class T> T operator()(T x, T y) const { return std::hypot(x, y); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x, xsimd::batch<T, default_simd_arch> y) const { return xsimd::hypot(x, y); }
        };

        struct fmod_fun
        {
            template <class T> T operator()(T x, T y) const { return std::fmod(x, y); }
        };

        struct clip_fun
        {
            template <class T> T operator()(T x, T lo, T hi) const { return x < lo ? lo : (hi < x ? hi : x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x, xsimd::batch<T, default_simd_arch> lo, xsimd::batch<T, default_simd_arch> hi) const
            {
                return xsimd::select(x < lo, lo, xsimd::select(hi < x, hi, x));
            }
        };

        struct sign_fun
        {
            template <class T> T operator()(T x) const { return (T(0) < x) - (x < T(0)); }
        };

        struct isfinite_fun
        {
            template <class T> bool operator()(T x) const { return std::isfinite(x); }
        };

        struct isinf_fun
        {
            template <class T> bool operator()(T x) const { return std::isinf(x); }
        };

        struct isnan_fun
        {
            template <class T> bool operator()(T x) const { return std::isnan(x); }
        };

        struct deg2rad_fun
        {
            template <class T> T operator()(T x) const { return x * T(3.14159265358979323846) / T(180); }
        };

        struct rad2deg_fun
        {
            template <class T> T operator()(T x) const { return x * T(180) / T(3.14159265358979323846); }
        };

        struct log10_fun
        {
            template <class T> T operator()(T x) const { return std::log10(x); }
        };

        struct log2_fun
        {
            template <class T> T operator()(T x) const { return std::log2(x); }
        };

        struct exp2_fun
        {
            template <class T> T operator()(T x) const { return std::exp2(x); }
        };

        struct cbrt_fun
        {
            template <class T> T operator()(T x) const { return std::cbrt(x); }
        };
    } // namespace math

    // Free functions that create lazy variable_function nodes
    template <class T, class L>
    inline auto abs(const variable<T, L>& v)
    {
        return make_variable_function(math::abs_fun{}, v);
    }

    template <class T, class L>
    inline auto sqrt(const variable<T, L>& v)
    {
        return make_variable_function(math::sqrt_fun{}, v);
    }

    template <class T, class L>
    inline auto exp(const variable<T, L>& v)
    {
        return make_variable_function(math::exp_fun{}, v);
    }

    template <class T, class L>
    inline auto log(const variable<T, L>& v)
    {
        return make_variable_function(math::log_fun{}, v);
    }

    template <class T, class L>
    inline auto log10(const variable<T, L>& v)
    {
        return make_variable_function(math::log10_fun{}, v);
    }

    template <class T, class L>
    inline auto log2(const variable<T, L>& v)
    {
        return make_variable_function(math::log2_fun{}, v);
    }

    template <class T, class L>
    inline auto sin(const variable<T, L>& v)
    {
        return make_variable_function(math::sin_fun{}, v);
    }

    template <class T, class L>
    inline auto cos(const variable<T, L>& v)
    {
        return make_variable_function(math::cos_fun{}, v);
    }

    template <class T, class L>
    inline auto tan(const variable<T, L>& v)
    {
        return make_variable_function(math::tan_fun{}, v);
    }

    template <class T, class L>
    inline auto asin(const variable<T, L>& v)
    {
        return make_variable_function(math::asin_fun{}, v);
    }

    template <class T, class L>
    inline auto acos(const variable<T, L>& v)
    {
        return make_variable_function(math::acos_fun{}, v);
    }

    template <class T, class L>
    inline auto atan(const variable<T, L>& v)
    {
        return make_variable_function(math::atan_fun{}, v);
    }

    template <class T, class L>
    inline auto sinh(const variable<T, L>& v)
    {
        return make_variable_function(math::sinh_fun{}, v);
    }

    template <class T, class L>
    inline auto cosh(const variable<T, L>& v)
    {
        return make_variable_function(math::cosh_fun{}, v);
    }

    template <class T, class L>
    inline auto tanh(const variable<T, L>& v)
    {
        return make_variable_function(math::tanh_fun{}, v);
    }

    template <class T, class L>
    inline auto ceil(const variable<T, L>& v)
    {
        return make_variable_function(math::ceil_fun{}, v);
    }

    template <class T, class L>
    inline auto floor(const variable<T, L>& v)
    {
        return make_variable_function(math::floor_fun{}, v);
    }

    template <class T, class L>
    inline auto round(const variable<T, L>& v)
    {
        return make_variable_function(math::round_fun{}, v);
    }

    template <class T, class L>
    inline auto trunc(const variable<T, L>& v)
    {
        return make_variable_function(math::trunc_fun{}, v);
    }

    template <class T, class L>
    inline auto pow(const variable<T, L>& base, const variable<T, L>& exponent)
    {
        return make_variable_function(math::pow_fun{}, base, exponent);
    }

    template <class T, class L>
    inline auto atan2(const variable<T, L>& y, const variable<T, L>& x)
    {
        return make_variable_function(math::atan2_fun{}, y, x);
    }

    template <class T, class L>
    inline auto hypot(const variable<T, L>& x, const variable<T, L>& y)
    {
        return make_variable_function(math::hypot_fun{}, x, y);
    }

    template <class T, class L>
    inline auto fmod(const variable<T, L>& x, const variable<T, L>& y)
    {
        return make_variable_function(math::fmod_fun{}, x, y);
    }

    template <class T, class L>
    inline auto clip(const variable<T, L>& x, T lo, T hi)
    {
        return make_variable_function(math::clip_fun{}, x,
                                      variable<T, L>({lo}, "lo"),
                                      variable<T, L>({hi}, "hi"));
    }

    template <class T, class L>
    inline auto sign(const variable<T, L>& v)
    {
        return make_variable_function(math::sign_fun{}, v);
    }

    template <class T, class L>
    inline auto isfinite(const variable<T, L>& v)
    {
        return make_variable_function(math::isfinite_fun{}, v);
    }

    template <class T, class L>
    inline auto isinf(const variable<T, L>& v)
    {
        return make_variable_function(math::isinf_fun{}, v);
    }

    template <class T, class L>
    inline auto isnan(const variable<T, L>& v)
    {
        return make_variable_function(math::isnan_fun{}, v);
    }

    template <class T, class L>
    inline auto deg2rad(const variable<T, L>& v)
    {
        return make_variable_function(math::deg2rad_fun{}, v);
    }

    template <class T, class L>
    inline auto rad2deg(const variable<T, L>& v)
    {
        return make_variable_function(math::rad2deg_fun{}, v);
    }

    template <class T, class L>
    inline auto exp2(const variable<T, L>& v)
    {
        return make_variable_function(math::exp2_fun{}, v);
    }

    template <class T, class L>
    inline auto cbrt(const variable<T, L>& v)
    {
        return make_variable_function(math::cbrt_fun{}, v);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_MATH_HPP