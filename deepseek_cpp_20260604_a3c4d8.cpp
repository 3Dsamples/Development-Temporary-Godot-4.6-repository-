/****************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht
 * Copyright (c) QuantStack
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 ****************************************************************************/

/**
 * @brief Advanced mathematical and search functions for xtensor expressions.
 *
 * This file provides a comprehensive set of mathematical operations,
 * from basic arithmetic to advanced 2D/3D simulation functions, all
 * designed for high performance, accuracy, and low memory consumption.
 * It is fully rewritten for C++17 and integrates seamlessly with the
 * xtensor expression system.
 */
#ifndef XTENSOR_XMATH_HPP
#define XTENSOR_XMATH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>

#include <xtl/xcomplex.hpp>
#include <xtl/xsequence.hpp>
#include <xtl/xtype_traits.hpp>

#include "../core/xeval.hpp"
#include "../core/xoperation.hpp"
#include "../core/xtensor_config.hpp"
#include "../misc/xmanipulation.hpp"
#include "../reducers/xaccumulator.hpp"
#include "../reducers/xreducer.hpp"
#include "../views/xslice.hpp"
#include "../views/xstrided_view.hpp"

namespace xt
{
    // Forward declarations
    template <class T>
    struct numeric_constants;

    namespace math
    {
        // Standard math functions
        using std::abs;
        using std::fabs;
        using std::acos;
        using std::asin;
        using std::atan;
        using std::cos;
        using std::sin;
        using std::tan;
        using std::acosh;
        using std::asinh;
        using std::atanh;
        using std::cosh;
        using std::sinh;
        using std::tanh;
        using std::cbrt;
        using std::sqrt;
        using std::exp;
        using std::exp2;
        using std::expm1;
        using std::ilogb;
        using std::log;
        using std::log10;
        using std::log1p;
        using std::log2;
        using std::logb;
        using std::ceil;
        using std::floor;
        using std::llround;
        using std::lround;
        using std::nearbyint;
        using std::remainder;
        using std::rint;
        using std::round;
        using std::trunc;
        using std::erf;
        using std::erfc;
        using std::lgamma;
        using std::tgamma;
        using std::arg;
        using std::conj;
        using std::imag;
        using std::real;
        using std::atan2;
        #if !defined(_MSC_VER)
        using std::copysign;
        #endif
        using std::fdim;
        using std::fmax;
        using std::fmin;
        using std::fmod;
        using std::hypot;
        using std::pow;
        using std::fma;

        // Classification functions with bool return type
        template <class T>
        inline std::enable_if_t<std::is_arithmetic_v<T>, bool>
        isinf(const T& t)
        {
            return bool(std::isinf(t));
        }

        template <class T>
        inline std::enable_if_t<std::is_arithmetic_v<T>, bool>
        isnan(const T& t)
        {
            return bool(std::isnan(t));
        }

        template <class T>
        inline std::enable_if_t<std::is_arithmetic_v<T>, bool>
        isfinite(const T& t)
        {
            return bool(std::isfinite(t));
        }

        // Overloads for complex types
        template <class T>
        inline bool isinf(const std::complex<T>& c)
        {
            return std::isinf(std::real(c)) || std::isinf(std::imag(c));
        }

        template <class T>
        inline bool isnan(const std::complex<T>& c)
        {
            return std::isnan(std::real(c)) || std::isnan(std::imag(c));
        }

        template <class T>
        inline bool isfinite(const std::complex<T>& c)
        {
            return !isinf(c) && !isnan(c);
        }

        // Specializations for unsigned types
        constexpr inline unsigned char abs(unsigned char x) { return x; }
        constexpr inline unsigned short abs(unsigned short x) { return x; }
        constexpr inline unsigned int abs(unsigned int x) { return x; }
        constexpr inline unsigned long abs(unsigned long x) { return x; }
        constexpr inline unsigned long long abs(unsigned long long x) { return x; }
    }

    // Numeric constants
    template <class T>
    struct numeric_constants
    {
        static constexpr T PI = T(3.141592653589793238463);
        static constexpr T PI_2 = T(1.57079632679489661923);
        static constexpr T PI_4 = T(0.785398163397448309616);
        static constexpr T D_1_PI = T(0.318309886183790671538);
        static constexpr T D_2_PI = T(0.636619772367581343076);
        static constexpr T D_2_SQRTPI = T(1.12837916709551257390);
        static constexpr T SQRT2 = T(1.41421356237309504880);
        static constexpr T SQRT1_2 = T(0.707106781186547524401);
        static constexpr T E = T(2.71828182845904523536);
        static constexpr T LOG2E = T(1.44269504088896340736);
        static constexpr T LOG10E = T(0.434294481903251827651);
        static constexpr T LN2 = T(0.693147180559945309417);
    };

    /*************************
     * Math Functors
     *************************/

#define XTENSOR_UNARY_MATH_FUNCTOR(NAME)                         \
    struct NAME##_fun                                            \
    {                                                            \
        template <class T>                                       \
        constexpr auto operator()(const T& arg) const            \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg);                                    \
        }                                                        \
        template <class B>                                       \
        constexpr auto simd_apply(const B& arg) const            \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg);                                    \
        }                                                        \
    }

#define XTENSOR_BINARY_MATH_FUNCTOR(NAME)                        \
    struct NAME##_fun                                            \
    {                                                            \
        template <class T1, class T2>                            \
        constexpr auto operator()(const T1& arg1, const T2& arg2) const \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg1, arg2);                             \
        }                                                        \
        template <class B>                                       \
        constexpr auto simd_apply(const B& arg1, const B& arg2) const \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg1, arg2);                             \
        }                                                        \
    }

#define XTENSOR_TERNARY_MATH_FUNCTOR(NAME)                       \
    struct NAME##_fun                                            \
    {                                                            \
        template <class T1, class T2, class T3>                  \
        constexpr auto operator()(const T1& arg1, const T2& arg2, const T3& arg3) const \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg1, arg2, arg3);                       \
        }                                                        \
        template <class B>                                       \
        constexpr auto simd_apply(const B& arg1, const B& arg2, const B& arg3) const \
        {                                                        \
            using math::NAME;                                    \
            return NAME(arg1, arg2, arg3);                       \
        }                                                        \
    }

    namespace math
    {
        XTENSOR_UNARY_MATH_FUNCTOR(abs);
        XTENSOR_UNARY_MATH_FUNCTOR(fabs);
        XTENSOR_BINARY_MATH_FUNCTOR(fmod);
        XTENSOR_BINARY_MATH_FUNCTOR(remainder);
        XTENSOR_TERNARY_MATH_FUNCTOR(fma);
        XTENSOR_BINARY_MATH_FUNCTOR(fmax);
        XTENSOR_BINARY_MATH_FUNCTOR(fmin);
        XTENSOR_BINARY_MATH_FUNCTOR(fdim);
        XTENSOR_UNARY_MATH_FUNCTOR(exp);
        XTENSOR_UNARY_MATH_FUNCTOR(exp2);
        XTENSOR_UNARY_MATH_FUNCTOR(expm1);
        XTENSOR_UNARY_MATH_FUNCTOR(log);
        XTENSOR_UNARY_MATH_FUNCTOR(log10);
        XTENSOR_UNARY_MATH_FUNCTOR(log2);
        XTENSOR_UNARY_MATH_FUNCTOR(log1p);
        XTENSOR_BINARY_MATH_FUNCTOR(pow);
        XTENSOR_UNARY_MATH_FUNCTOR(sqrt);
        XTENSOR_UNARY_MATH_FUNCTOR(cbrt);
        XTENSOR_BINARY_MATH_FUNCTOR(hypot);
        XTENSOR_UNARY_MATH_FUNCTOR(sin);
        XTENSOR_UNARY_MATH_FUNCTOR(cos);
        XTENSOR_UNARY_MATH_FUNCTOR(tan);
        XTENSOR_UNARY_MATH_FUNCTOR(asin);
        XTENSOR_UNARY_MATH_FUNCTOR(acos);
        XTENSOR_UNARY_MATH_FUNCTOR(atan);
        XTENSOR_BINARY_MATH_FUNCTOR(atan2);
        XTENSOR_UNARY_MATH_FUNCTOR(sinh);
        XTENSOR_UNARY_MATH_FUNCTOR(cosh);
        XTENSOR_UNARY_MATH_FUNCTOR(tanh);
        XTENSOR_UNARY_MATH_FUNCTOR(asinh);
        XTENSOR_UNARY_MATH_FUNCTOR(acosh);
        XTENSOR_UNARY_MATH_FUNCTOR(atanh);
        XTENSOR_UNARY_MATH_FUNCTOR(erf);
        XTENSOR_UNARY_MATH_FUNCTOR(erfc);
        XTENSOR_UNARY_MATH_FUNCTOR(tgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(lgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(ceil);
        XTENSOR_UNARY_MATH_FUNCTOR(floor);
        XTENSOR_UNARY_MATH_FUNCTOR(trunc);
        XTENSOR_UNARY_MATH_FUNCTOR(round);
        XTENSOR_UNARY_MATH_FUNCTOR(nearbyint);
        XTENSOR_UNARY_MATH_FUNCTOR(rint);
        XTENSOR_UNARY_MATH_FUNCTOR(isfinite);
        XTENSOR_UNARY_MATH_FUNCTOR(isinf);
        XTENSOR_UNARY_MATH_FUNCTOR(isnan);
        XTENSOR_UNARY_MATH_FUNCTOR(conj);
    }

#undef XTENSOR_UNARY_MATH_FUNCTOR
#undef XTENSOR_BINARY_MATH_FUNCTOR
#undef XTENSOR_TERNARY_MATH_FUNCTOR

    /*************************
     * Reduction Helpers
     *************************/

    namespace detail
    {
        template <class T, class R>
        std::enable_if_t<std::is_arithmetic_v<R>, R> fill_init(T init)
        {
            return R(init);
        }

        template <class T, class R>
        std::enable_if_t<!std::is_arithmetic_v<R>, R> fill_init(T init)
        {
            R result;
            std::fill(std::begin(result), std::end(result), init);
            return result;
        }
    }

#define XTENSOR_REDUCER_FUNCTION(NAME, FUNCTOR, INIT_VALUE_TYPE, INIT)              \
    template <class T = void, class E, class X,                                    \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                \
              XTL_REQUIRES(std::negation<xtl::is_integral<X>>,                     \
                           std::negation<xtl::is_integral<EVS>>)>                  \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS())                              \
    {                                                                               \
        using init_value_type = std::conditional_t<std::is_same_v<T, void>,        \
                                                  INIT_VALUE_TYPE, T>;              \
        using functor_type = FUNCTOR;                                              \
        using init_value_fct = xt::const_value<init_value_type>;                    \
        return xt::reduce(                                                          \
            make_xreducer_functor(functor_type(),                                   \
                                  init_value_fct(detail::fill_init<T, init_value_type>(INIT))), \
            std::forward<E>(e),                                                    \
            std::forward<X>(axes),                                                 \
            es                                                                      \
        );                                                                          \
    }                                                                               \
                                                                                    \
    template <class T = void, class E, class X,                                    \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                \
              XTL_REQUIRES(std::negation<xtl::is_integral<X>>,                     \
                           xtl::is_integral<EVS>)>                                 \
    inline auto NAME(E&& e, X axis, EVS es = EVS())                                \
    {                                                                               \
        return NAME<T>(std::forward<E>(e), {axis}, es);                             \
    }                                                                               \
                                                                                    \
    template <class T = void, class E,                                             \
              class EVS = DEFAULT_STRATEGY_REDUCERS,                                \
              XTL_REQUIRES(xtl::is_integral<EVS>)>                                 \
    inline auto NAME(E&& e, EVS es = EVS())                                        \
    {                                                                               \
        using init_value_type = std::conditional_t<std::is_same_v<T, void>,        \
                                                  INIT_VALUE_TYPE, T>;              \
        using functor_type = FUNCTOR;                                              \
        using init_value_fct = xt::const_value<init_value_type>;                    \
        return xt::reduce(                                                          \
            make_xreducer_functor(functor_type(),                                   \
                                  init_value_fct(detail::fill_init<T, init_value_type>(INIT))), \
            std::forward<E>(e),                                                    \
            es                                                                      \
        );                                                                          \
    }                                                                               \
                                                                                    \
    template <class T = void, class E, class I, std::size_t N,                     \
              class EVS = DEFAULT_STRATEGY_REDUCERS>                               \
    inline auto NAME(E&& e, const I (&axes)[N], EVS es = EVS())                    \
    {                                                                               \
        using init_value_type = std::conditional_t<std::is_same_v<T, void>,        \
                                                  INIT_VALUE_TYPE, T>;              \
        using functor_type = FUNCTOR;                                              \
        using init_value_fct = xt::const_value<init_value_type>;                    \
        return xt::reduce(                                                          \
            make_xreducer_functor(functor_type(),                                   \
                                  init_value_fct(detail::fill_init<T, init_value_type>(INIT))), \
            std::forward<E>(e),                                                    \
            axes,                                                                   \
            es                                                                      \
        );                                                                          \
    }

    /*************************
     * Basic Element-wise Functions
     *************************/

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     */
    template <class E>
    inline auto abs(E&& e) noexcept -> detail::xfunction_type_t<math::abs_fun, E>
    {
        return detail::make_xfunction<math::abs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Absolute value function (floating point).
     */
    template <class E>
    inline auto fabs(E&& e) noexcept -> detail::xfunction_type_t<math::fabs_fun, E>
    {
        return detail::make_xfunction<math::fabs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Remainder of the floating point division operation.
     */
    template <class E1, class E2>
    inline auto fmod(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmod_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmod_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Signed remainder of the division operation.
     */
    template <class E1, class E2>
    inline auto remainder(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::remainder_fun, E1, E2>
    {
        return detail::make_xfunction<math::remainder_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Fused multiply-add operation.
     */
    template <class E1, class E2, class E3>
    inline auto fma(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<math::fma_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::fma_fun>(
            std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum function.
     */
    template <class E1, class E2>
    inline auto fmax(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmax_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmax_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Minimum function.
     */
    template <class E1, class E2>
    inline auto fmin(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmin_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmin_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Positive difference function.
     */
    template <class E1, class E2>
    inline auto fdim(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fdim_fun, E1, E2>
    {
        return detail::make_xfunction<math::fdim_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    // Exponential and power functions
    template <class E>
    inline auto exp(E&& e) noexcept -> detail::xfunction_type_t<math::exp_fun, E>
    {
        return detail::make_xfunction<math::exp_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto exp2(E&& e) noexcept -> detail::xfunction_type_t<math::exp2_fun, E>
    {
        return detail::make_xfunction<math::exp2_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto expm1(E&& e) noexcept -> detail::xfunction_type_t<math::expm1_fun, E>
    {
        return detail::make_xfunction<math::expm1_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto log(E&& e) noexcept -> detail::xfunction_type_t<math::log_fun, E>
    {
        return detail::make_xfunction<math::log_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto log10(E&& e) noexcept -> detail::xfunction_type_t<math::log10_fun, E>
    {
        return detail::make_xfunction<math::log10_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto log2(E&& e) noexcept -> detail::xfunction_type_t<math::log2_fun, E>
    {
        return detail::make_xfunction<math::log2_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto log1p(E&& e) noexcept -> detail::xfunction_type_t<math::log1p_fun, E>
    {
        return detail::make_xfunction<math::log1p_fun>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto pow(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::pow_fun, E1, E2>
    {
        return detail::make_xfunction<math::pow_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E>
    inline auto sqrt(E&& e) noexcept -> detail::xfunction_type_t<math::sqrt_fun, E>
    {
        return detail::make_xfunction<math::sqrt_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto cbrt(E&& e) noexcept -> detail::xfunction_type_t<math::cbrt_fun, E>
    {
        return detail::make_xfunction<math::cbrt_fun>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto hypot(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::hypot_fun, E1, E2>
    {
        return detail::make_xfunction<math::hypot_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    // Trigonometric functions
    template <class E>
    inline auto sin(E&& e) noexcept -> detail::xfunction_type_t<math::sin_fun, E>
    {
        return detail::make_xfunction<math::sin_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto cos(E&& e) noexcept -> detail::xfunction_type_t<math::cos_fun, E>
    {
        return detail::make_xfunction<math::cos_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto tan(E&& e) noexcept -> detail::xfunction_type_t<math::tan_fun, E>
    {
        return detail::make_xfunction<math::tan_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto asin(E&& e) noexcept -> detail::xfunction_type_t<math::asin_fun, E>
    {
        return detail::make_xfunction<math::asin_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto acos(E&& e) noexcept -> detail::xfunction_type_t<math::acos_fun, E>
    {
        return detail::make_xfunction<math::acos_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto atan(E&& e) noexcept -> detail::xfunction_type_t<math::atan_fun, E>
    {
        return detail::make_xfunction<math::atan_fun>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto atan2(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::atan2_fun, E1, E2>
    {
        return detail::make_xfunction<math::atan2_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    // Hyperbolic functions
    template <class E>
    inline auto sinh(E&& e) noexcept -> detail::xfunction_type_t<math::sinh_fun, E>
    {
        return detail::make_xfunction<math::sinh_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto cosh(E&& e) noexcept -> detail::xfunction_type_t<math::cosh_fun, E>
    {
        return detail::make_xfunction<math::cosh_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto tanh(E&& e) noexcept -> detail::xfunction_type_t<math::tanh_fun, E>
    {
        return detail::make_xfunction<math::tanh_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto asinh(E&& e) noexcept -> detail::xfunction_type_t<math::asinh_fun, E>
    {
        return detail::make_xfunction<math::asinh_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto acosh(E&& e) noexcept -> detail::xfunction_type_t<math::acosh_fun, E>
    {
        return detail::make_xfunction<math::acosh_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto atanh(E&& e) noexcept -> detail::xfunction_type_t<math::atanh_fun, E>
    {
        return detail::make_xfunction<math::atanh_fun>(std::forward<E>(e));
    }

    // Error and gamma functions
    template <class E>
    inline auto erf(E&& e) noexcept -> detail::xfunction_type_t<math::erf_fun, E>
    {
        return detail::make_xfunction<math::erf_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto erfc(E&& e) noexcept -> detail::xfunction_type_t<math::erfc_fun, E>
    {
        return detail::make_xfunction<math::erfc_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto tgamma(E&& e) noexcept -> detail::xfunction_type_t<math::tgamma_fun, E>
    {
        return detail::make_xfunction<math::tgamma_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto lgamma(E&& e) noexcept -> detail::xfunction_type_t<math::lgamma_fun, E>
    {
        return detail::make_xfunction<math::lgamma_fun>(std::forward<E>(e));
    }

    // Rounding functions
    template <class E>
    inline auto ceil(E&& e) noexcept -> detail::xfunction_type_t<math::ceil_fun, E>
    {
        return detail::make_xfunction<math::ceil_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto floor(E&& e) noexcept -> detail::xfunction_type_t<math::floor_fun, E>
    {
        return detail::make_xfunction<math::floor_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto trunc(E&& e) noexcept -> detail::xfunction_type_t<math::trunc_fun, E>
    {
        return detail::make_xfunction<math::trunc_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto round(E&& e) noexcept -> detail::xfunction_type_t<math::round_fun, E>
    {
        return detail::make_xfunction<math::round_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto nearbyint(E&& e) noexcept -> detail::xfunction_type_t<math::nearbyint_fun, E>
    {
        return detail::make_xfunction<math::nearbyint_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto rint(E&& e) noexcept -> detail::xfunction_type_t<math::rint_fun, E>
    {
        return detail::make_xfunction<math::rint_fun>(std::forward<E>(e));
    }

    // Classification functions
    template <class E>
    inline auto isfinite(E&& e) noexcept -> detail::xfunction_type_t<math::isfinite_fun, E>
    {
        return detail::make_xfunction<math::isfinite_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto isinf(E&& e) noexcept -> detail::xfunction_type_t<math::isinf_fun, E>
    {
        return detail::make_xfunction<math::isinf_fun>(std::forward<E>(e));
    }

    template <class E>
    inline auto isnan(E&& e) noexcept -> detail::xfunction_type_t<math::isnan_fun, E>
    {
        return detail::make_xfunction<math::isnan_fun>(std::forward<E>(e));
    }

    // Complex functions
    template <class E>
    inline auto conj(E&& e) noexcept -> detail::xfunction_type_t<math::conj_fun, E>
    {
        return detail::make_xfunction<math::conj_fun>(std::forward<E>(e));
    }

    /*************************
     * Advanced Math: Angle Conversion
     *************************/

    namespace math
    {
        struct deg2rad
        {
            template <class A>
            constexpr auto operator()(const A& a) const noexcept
            {
                if constexpr (std::is_integral_v<A>)
                {
                    return a * xt::numeric_constants<double>::PI / 180.0;
                }
                else
                {
                    return a * xt::numeric_constants<A>::PI / A(180.0);
                }
            }
            template <class A>
            constexpr auto simd_apply(const A& a) const noexcept
            {
                if constexpr (std::is_integral_v<A>)
                {
                    return a * xt::numeric_constants<double>::PI / 180.0;
                }
                else
                {
                    return a * xt::numeric_constants<A>::PI / A(180.0);
                }
            }
        };

        struct rad2deg
        {
            template <class A>
            constexpr auto operator()(const A& a) const noexcept
            {
                if constexpr (std::is_integral_v<A>)
                {
                    return a * 180.0 / xt::numeric_constants<double>::PI;
                }
                else
                {
                    return a * A(180.0) / xt::numeric_constants<A>::PI;
                }
            }
            template <class A>
            constexpr auto simd_apply(const A& a) const noexcept
            {
                if constexpr (std::is_integral_v<A>)
                {
                    return a * 180.0 / xt::numeric_constants<double>::PI;
                }
                else
                {
                    return a * A(180.0) / xt::numeric_constants<A>::PI;
                }
            }
        };
    }

    template <class E>
    inline auto deg2rad(E&& e) noexcept -> detail::xfunction_type_t<math::deg2rad, E>
    {
        return detail::make_xfunction<math::deg2rad>(std::forward<E>(e));
    }

    template <class E>
    inline auto radians(E&& e) noexcept -> detail::xfunction_type_t<math::deg2rad, E>
    {
        return detail::make_xfunction<math::deg2rad>(std::forward<E>(e));
    }

    template <class E>
    inline auto rad2deg(E&& e) noexcept -> detail::xfunction_type_t<math::rad2deg, E>
    {
        return detail::make_xfunction<math::rad2deg>(std::forward<E>(e));
    }

    template <class E>
    inline auto degrees(E&& e) noexcept -> detail::xfunction_type_t<math::rad2deg, E>
    {
        return detail::make_xfunction<math::rad2deg>(std::forward<E>(e));
    }

    /*************************
     * Advanced Math: Clamping and Sign
     *************************/

    namespace math
    {
        struct minimum
        {
            template <class A1, class A2>
            constexpr auto operator()(const A1& t1, const A2& t2) const noexcept
            {
                return t1 < t2 ? t1 : t2;
            }
            template <class A1, class A2>
            constexpr auto simd_apply(const A1& t1, const A2& t2) const noexcept
            {
                return xt_simd::select(t1 < t2, t1, t2);
            }
        };

        struct maximum
        {
            template <class A1, class A2>
            constexpr auto operator()(const A1& t1, const A2& t2) const noexcept
            {
                return t1 > t2 ? t1 : t2;
            }
            template <class A1, class A2>
            constexpr auto simd_apply(const A1& t1, const A2& t2) const noexcept
            {
                return xt_simd::select(t1 > t2, t1, t2);
            }
        };

        struct clamp_fun
        {
            template <class A1, class A2, class A3>
            constexpr auto operator()(const A1& v, const A2& lo, const A3& hi) const
            {
                return v < lo ? lo : (hi < v ? hi : v);
            }
            template <class A1, class A2, class A3>
            constexpr auto simd_apply(const A1& v, const A2& lo, const A3& hi) const
            {
                return xt_simd::select(v < lo, lo, xt_simd::select(hi < v, hi, v));
            }
        };

        struct sign_fun
        {
            template <class T>
            constexpr auto operator()(const T& x) const noexcept
            {
                return (T(0) < x) - (x < T(0));
            }
        };
    }

    template <class E1, class E2>
    inline auto maximum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::maximum, E1, E2>
    {
        return detail::make_xfunction<math::maximum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto minimum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::minimum, E1, E2>
    {
        return detail::make_xfunction<math::minimum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2, class E3>
    inline auto clip(E1&& e1, E2&& lo, E3&& hi) noexcept
        -> detail::xfunction_type_t<math::clamp_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::clamp_fun>(
            std::forward<E1>(e1), std::forward<E2>(lo), std::forward<E3>(hi));
    }

    template <class E>
    inline auto sign(E&& e) noexcept -> detail::xfunction_type_t<math::sign_fun, E>
    {
        return detail::make_xfunction<math::sign_fun>(std::forward<E>(e));
    }

    /*************************
     * Reducers
     *************************/

    XTENSOR_REDUCER_FUNCTION(sum, xt::detail::plus, typename std::decay_t<E>::value_type, 0);
    XTENSOR_REDUCER_FUNCTION(prod, xt::detail::multiplies, typename std::decay_t<E>::value_type, 1);
    XTENSOR_REDUCER_FUNCTION(amax, math::maximum, typename std::decay_t<E>::value_type,
                             std::numeric_limits<typename std::decay_t<E>::value_type>::lowest());
    XTENSOR_REDUCER_FUNCTION(amin, math::minimum, typename std::decay_t<E>::value_type,
                             std::numeric_limits<typename std::decay_t<E>::value_type>::max());

#undef XTENSOR_REDUCER_FUNCTION

    /*************************
     * 2D/3D Simulation Math
     *************************/

    namespace simd
    {
        /**
         * @brief Dot product of two containers.
         */
        template <class E1, class E2>
        inline auto dot(const E1& a, const E2& b)
        {
            return (a * b).sum();
        }

        /**
         * @brief Cross product for 3-element vectors.
         */
        template <class E1, class E2>
        inline auto cross(const E1& a, const E2& b)
        {
            return xt::xarray<typename std::common_type_t<typename E1::value_type, typename E2::value_type>>{
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            };
        }

        /**
         * @brief Euclidean norm (L2 norm).
         */
        template <class E>
        inline auto norm(const E& a)
        {
            return std::sqrt(dot(a, a));
        }

        /**
         * @brief Normalize a vector.
         */
        template <class E>
        inline auto normalize(const E& a)
        {
            return a / norm(a);
        }

        /**
         * @brief Linear interpolation.
         */
        template <class T, class U>
        constexpr auto lerp(const T& a, const T& b, U t)
        {
            return a + t * (b - a);
        }

        /**
         * @brief Spherical linear interpolation for 3D vectors.
         */
        template <class T>
        auto slerp(const T& a, const T& b, double t)
        {
            auto omega = std::acos(clip(dot(a, b), -1.0, 1.0));
            auto sin_omega = std::sin(omega);
            if (sin_omega == 0.0)
            {
                return lerp(a, b, t);
            }
            return (std::sin((1.0 - t) * omega) / sin_omega) * a
                   + (std::sin(t * omega) / sin_omega) * b;
        }

        /**
         * @brief Bilinear interpolation for 2D grids.
         */
        template <class T>
        constexpr auto bilerp(T v00, T v10, T v01, T v11, double fx, double fy)
        {
            return lerp(lerp(v00, v10, fx), lerp(v01, v11, fx), fy);
        }

        /**
         * @brief Clamp a value between a minimum and maximum.
         */
        template <class T>
        constexpr T clamp(const T& v, const T& lo, const T& hi)
        {
            return v < lo ? lo : (hi < v ? hi : v);
        }
    }

    /*************************
     * Search Functions
     *************************/

    namespace search
    {
        /**
         * @brief Find the first index of a value in a 1D sorted container.
         * @return Index of the element if found, otherwise -1.
         */
        template <class E, class T>
        inline long binary_search(const E& sorted, const T& value)
        {
            auto it = std::lower_bound(sorted.begin(), sorted.end(), value);
            if (it != sorted.end() && *it == value)
            {
                return std::distance(sorted.begin(), it);
            }
            return -1;
        }

        /**
         * @brief Find the insertion point for a value in a sorted container.
         * @return Index of the first element not less than value.
         */
        template <class E, class T>
        inline long lower_bound(const E& sorted, const T& value)
        {
            auto it = std::lower_bound(sorted.begin(), sorted.end(), value);
            return std::distance(sorted.begin(), it);
        }

        /**
         * @brief Find the insertion point for a value in a sorted container.
         * @return Index of the first element greater than value.
         */
        template <class E, class T>
        inline long upper_bound(const E& sorted, const T& value)
        {
            auto it = std::upper_bound(sorted.begin(), sorted.end(), value);
            return std::distance(sorted.begin(), it);
        }

        /**
         * @brief Find the index of the nearest value in a sorted 1D container.
         * @return Index of the element with the smallest absolute difference.
         */
        template <class E, class T>
        inline long nearest_sorted(const E& sorted, const T& value)
        {
            auto it = std::lower_bound(sorted.begin(), sorted.end(), value);
            if (it == sorted.begin())
            {
                return 0;
            }
            if (it == sorted.end())
            {
                return sorted.size() - 1;
            }
            auto prev = it - 1;
            return (value - *prev) <= (*it - value) ? std::distance(sorted.begin(), prev)
                                                    : std::distance(sorted.begin(), it);
        }

        /**
         * @brief Find the index of the nearest value in an unsorted 1D container.
         * @return Index of the element with the smallest absolute difference.
         */
        template <class E, class T>
        inline long nearest(const E& arr, const T& value)
        {
            long best_idx = -1;
            T best_diff = std::numeric_limits<T>::max();
            for (long i = 0; i < arr.size(); ++i)
            {
                T diff = std::abs(arr[i] - value);
                if (diff < best_diff)
                {
                    best_diff = diff;
                    best_idx = i;
                }
            }
            return best_idx;
        }

        /**
         * @brief Find the indices of all occurrences of a value in a 1D container.
         * @return A vector of indices.
         */
        template <class E, class T>
        inline auto find_all(const E& arr, const T& value)
        {
            std::vector<long> indices;
            for (long i = 0; i < arr.size(); ++i)
            {
                if (arr[i] == value)
                {
                    indices.push_back(i);
                }
            }
            return indices;
        }
    }

    /*************************
     * Numerical Integration
     *************************/

    namespace integrate
    {
        /**
         * @brief Trapezoidal rule integration for uniform grids.
         */
        template <class E>
        inline auto trapz(const E& y, double dx = 1.0)
        {
            using T = typename E::value_type;
            T sum = (y[0] + y[y.size() - 1]) * 0.5;
            for (std::size_t i = 1; i < y.size() - 1; ++i)
            {
                sum += y[i];
            }
            return sum * dx;
        }

        /**
         * @brief Simpson's rule integration for uniform grids (requires odd number of points).
         */
        template <class E>
        inline auto simpson(const E& y, double dx = 1.0)
        {
            using T = typename E::value_type;
            std::size_t n = y.size();
            T sum = y[0] + y[n - 1];
            for (std::size_t i = 1; i < n - 1; i += 2)
            {
                sum += T(4.0) * y[i];
            }
            for (std::size_t i = 2; i < n - 2; i += 2)
            {
                sum += T(2.0) * y[i];
            }
            return sum * dx / 3.0;
        }
    }

    /*************************
     * Filtering
     *************************/

    namespace filter
    {
        /**
         * @brief 1D median filter.
         */
        template <class E>
        inline auto median(const E& signal, std::size_t window_size)
        {
            using T = typename E::value_type;
            auto result = xt::xarray<T>::from_shape(signal.shape());
            std::size_t half = window_size / 2;
            for (std::size_t i = 0; i < signal.size(); ++i)
            {
                std::size_t start = (i < half) ? 0 : i - half;
                std::size_t end = std::min(i + half + 1, signal.size());
                std::vector<T> window(signal.begin() + start, signal.begin() + end);
                std::sort(window.begin(), window.end());
                result[i] = window[window.size() / 2];
            }
            return result;
        }
    }

} // namespace xt

#endif // XTENSOR_XMATH_HPP