//File 0315 : xframe/xframe_math.hpp
//Element‑wise mathematical functions for xframe expressions: sin, cos, tan, exp, log, sqrt, abs, pow, clip with SIMD acceleration.
#ifndef XFRAME_MATH_HPP
#define XFRAME_MATH_HPP

#include <cmath>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_function.hpp"
#include "xframe_scalar.hpp"

namespace xframe
{
    namespace math
    {
        // Functor wrappers that delegate to std:: functions, with optional SIMD overloads where available.
        struct abs_f
        {
            template <class T>
            T operator()(T x) const { return std::abs(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::abs(x); }
        };

        struct sqrt_f
        {
            template <class T>
            T operator()(T x) const { return std::sqrt(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::sqrt(x); }
        };

        struct exp_f
        {
            template <class T>
            T operator()(T x) const { return std::exp(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::exp(x); }
        };

        struct log_f
        {
            template <class T>
            T operator()(T x) const { return std::log(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::log(x); }
        };

        struct sin_f
        {
            template <class T>
            T operator()(T x) const { return std::sin(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::sin(x); }
        };

        struct cos_f
        {
            template <class T>
            T operator()(T x) const { return std::cos(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::cos(x); }
        };

        struct tan_f
        {
            template <class T>
            T operator()(T x) const { return std::tan(x); }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::tan(x); }
        };

        struct asin_f
        {
            template <class T> T operator()(T x) const { return std::asin(x); }
        };

        struct acos_f
        {
            template <class T> T operator()(T x) const { return std::acos(x); }
        };

        struct atan_f
        {
            template <class T> T operator()(T x) const { return std::atan(x); }
        };

        struct sinh_f
        {
            template <class T> T operator()(T x) const { return std::sinh(x); }
        };

        struct cosh_f
        {
            template <class T> T operator()(T x) const { return std::cosh(x); }
        };

        struct tanh_f
        {
            template <class T> T operator()(T x) const { return std::tanh(x); }
        };

        struct ceil_f
        {
            template <class T> T operator()(T x) const { return std::ceil(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::ceil(x); }
        };

        struct floor_f
        {
            template <class T> T operator()(T x) const { return std::floor(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::floor(x); }
        };

        struct round_f
        {
            template <class T> T operator()(T x) const { return std::round(x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x) const { return xsimd::round(x); }
        };

        struct trunc_f
        {
            template <class T> T operator()(T x) const { return std::trunc(x); }
        };

        struct pow_f
        {
            template <class T> T operator()(T x, T y) const { return std::pow(x, y); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x,
                                               xsimd::batch<T, default_simd_arch> y) const { return xsimd::pow(x, y); }
        };

        struct atan2_f
        {
            template <class T> T operator()(T y, T x) const { return std::atan2(y, x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> y,
                                               xsimd::batch<T, default_simd_arch> x) const { return xsimd::atan2(y, x); }
        };

        struct hypot_f
        {
            template <class T> T operator()(T x, T y) const { return std::hypot(x, y); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x,
                                               xsimd::batch<T, default_simd_arch> y) const { return xsimd::hypot(x, y); }
        };

        struct fmod_f
        {
            template <class T> T operator()(T x, T y) const { return std::fmod(x, y); }
        };

        struct clip_f
        {
            template <class T> T operator()(T x, T lo, T hi) const { return x < lo ? lo : (hi < x ? hi : x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x,
                                               xsimd::batch<T, default_simd_arch> lo,
                                               xsimd::batch<T, default_simd_arch> hi) const
            {
                return xsimd::select(x < lo, lo, xsimd::select(hi < x, hi, x));
            }
        };

        struct sign_f
        {
            template <class T> T operator()(T x) const { return (T(0) < x) - (x < T(0)); }
        };

        struct isfinite_f
        {
            template <class T> bool operator()(T x) const { return std::isfinite(x); }
        };

        struct isinf_f
        {
            template <class T> bool operator()(T x) const { return std::isinf(x); }
        };

        struct isnan_f
        {
            template <class T> bool operator()(T x) const { return std::isnan(x); }
        };

        struct deg2rad_f
        {
            template <class T> T operator()(T x) const { return x * T(3.14159265358979323846) / T(180); }
        };

        struct rad2deg_f
        {
            template <class T> T operator()(T x) const { return x * T(180) / T(3.14159265358979323846); }
        };

        struct log10_f
        {
            template <class T> T operator()(T x) const { return std::log10(x); }
        };

        struct log2_f
        {
            template <class T> T operator()(T x) const { return std::log2(x); }
        };

        struct exp2_f
        {
            template <class T> T operator()(T x) const { return std::exp2(x); }
        };

        struct cbrt_f
        {
            template <class T> T operator()(T x) const { return std::cbrt(x); }
        };
    } // namespace math

    // Free functions that create xframe_function nodes

    /**
     * Absolute value.
     */
    template <class E>
    inline auto abs(const expression<E>& e)
    {
        return make_xframe_function(math::abs_f{}, e.derived());
    }

    /**
     * Square root.
     */
    template <class E>
    inline auto sqrt(const expression<E>& e)
    {
        return make_xframe_function(math::sqrt_f{}, e.derived());
    }

    /**
     * Exponential.
     */
    template <class E>
    inline auto exp(const expression<E>& e)
    {
        return make_xframe_function(math::exp_f{}, e.derived());
    }

    /**
     * Natural logarithm.
     */
    template <class E>
    inline auto log(const expression<E>& e)
    {
        return make_xframe_function(math::log_f{}, e.derived());
    }

    /**
     * Base‑10 logarithm.
     */
    template <class E>
    inline auto log10(const expression<E>& e)
    {
        return make_xframe_function(math::log10_f{}, e.derived());
    }

    /**
     * Base‑2 logarithm.
     */
    template <class E>
    inline auto log2(const expression<E>& e)
    {
        return make_xframe_function(math::log2_f{}, e.derived());
    }

    /**
     * Sine.
     */
    template <class E>
    inline auto sin(const expression<E>& e)
    {
        return make_xframe_function(math::sin_f{}, e.derived());
    }

    /**
     * Cosine.
     */
    template <class E>
    inline auto cos(const expression<E>& e)
    {
        return make_xframe_function(math::cos_f{}, e.derived());
    }

    /**
     * Tangent.
     */
    template <class E>
    inline auto tan(const expression<E>& e)
    {
        return make_xframe_function(math::tan_f{}, e.derived());
    }

    /**
     * Arcsine.
     */
    template <class E>
    inline auto asin(const expression<E>& e)
    {
        return make_xframe_function(math::asin_f{}, e.derived());
    }

    /**
     * Arccosine.
     */
    template <class E>
    inline auto acos(const expression<E>& e)
    {
        return make_xframe_function(math::acos_f{}, e.derived());
    }

    /**
     * Arctangent.
     */
    template <class E>
    inline auto atan(const expression<E>& e)
    {
        return make_xframe_function(math::atan_f{}, e.derived());
    }

    /**
     * Hyperbolic sine.
     */
    template <class E>
    inline auto sinh(const expression<E>& e)
    {
        return make_xframe_function(math::sinh_f{}, e.derived());
    }

    /**
     * Hyperbolic cosine.
     */
    template <class E>
    inline auto cosh(const expression<E>& e)
    {
        return make_xframe_function(math::cosh_f{}, e.derived());
    }

    /**
     * Hyperbolic tangent.
     */
    template <class E>
    inline auto tanh(const expression<E>& e)
    {
        return make_xframe_function(math::tanh_f{}, e.derived());
    }

    /**
     * Ceiling.
     */
    template <class E>
    inline auto ceil(const expression<E>& e)
    {
        return make_xframe_function(math::ceil_f{}, e.derived());
    }

    /**
     * Floor.
     */
    template <class E>
    inline auto floor(const expression<E>& e)
    {
        return make_xframe_function(math::floor_f{}, e.derived());
    }

    /**
     * Round to nearest integer.
     */
    template <class E>
    inline auto round(const expression<E>& e)
    {
        return make_xframe_function(math::round_f{}, e.derived());
    }

    /**
     * Truncate toward zero.
     */
    template <class E>
    inline auto trunc(const expression<E>& e)
    {
        return make_xframe_function(math::trunc_f{}, e.derived());
    }

    /**
     * Power: base^exponent.
     */
    template <class E1, class E2>
    inline auto pow(const expression<E1>& base, const expression<E2>& exponent)
    {
        return make_xframe_function(math::pow_f{}, base.derived(), exponent.derived());
    }

    /**
     * Two‑argument arctangent: atan2(y, x).
     */
    template <class E1, class E2>
    inline auto atan2(const expression<E1>& y, const expression<E2>& x)
    {
        return make_xframe_function(math::atan2_f{}, y.derived(), x.derived());
    }

    /**
     * Hypotenuse: sqrt(x² + y²).
     */
    template <class E1, class E2>
    inline auto hypot(const expression<E1>& x, const expression<E2>& y)
    {
        return make_xframe_function(math::hypot_f{}, x.derived(), y.derived());
    }

    /**
     * Floating‑point remainder.
     */
    template <class E1, class E2>
    inline auto fmod(const expression<E1>& x, const expression<E2>& y)
    {
        return make_xframe_function(math::fmod_f{}, x.derived(), y.derived());
    }

    /**
     * Clip values to [lo, hi].
     */
    template <class E, class Lo, class Hi>
    inline auto clip(const expression<E>& x, const expression<Lo>& lo, const expression<Hi>& hi)
    {
        return make_xframe_function(math::clip_f{}, x.derived(), lo.derived(), hi.derived());
    }

    /**
     * Sign function: -1 for negative, 0 for zero, +1 for positive.
     */
    template <class E>
    inline auto sign(const expression<E>& e)
    {
        return make_xframe_function(math::sign_f{}, e.derived());
    }

    /**
     * Check if finite.
     */
    template <class E>
    inline auto isfinite(const expression<E>& e)
    {
        return make_xframe_function(math::isfinite_f{}, e.derived());
    }

    /**
     * Check if infinite.
     */
    template <class E>
    inline auto isinf(const expression<E>& e)
    {
        return make_xframe_function(math::isinf_f{}, e.derived());
    }

    /**
     * Check if NaN.
     */
    template <class E>
    inline auto isnan(const expression<E>& e)
    {
        return make_xframe_function(math::isnan_f{}, e.derived());
    }

    /**
     * Convert degrees to radians.
     */
    template <class E>
    inline auto deg2rad(const expression<E>& e)
    {
        return make_xframe_function(math::deg2rad_f{}, e.derived());
    }

    /**
     * Convert radians to degrees.
     */
    template <class E>
    inline auto rad2deg(const expression<E>& e)
    {
        return make_xframe_function(math::rad2deg_f{}, e.derived());
    }

    /**
     * Exponential base 2.
     */
    template <class E>
    inline auto exp2(const expression<E>& e)
    {
        return make_xframe_function(math::exp2_f{}, e.derived());
    }

    /**
     * Cube root.
     */
    template <class E>
    inline auto cbrt(const expression<E>& e)
    {
        return make_xframe_function(math::cbrt_f{}, e.derived());
    }

} // namespace xframe

#endif // XFRAME_MATH_HPP