//File 0109 : numdot/math.h
//Element-wise mathematical functions with SIMD acceleration, expression integration, and advanced searching utilities.
#ifndef NUMDOT_MATH_H
#define NUMDOT_MATH_H

#include <cmath>
#include <type_traits>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include "config.h"
#include "forward.h"
#include "types.h"
#include "elementwise.h"
#include "array.h"

namespace numdot
{
    namespace math
    {
        // ========== Functor definitions with optional simd_apply ==========
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

        struct fmax_fun
        {
            template <class T> T operator()(T x, T y) const { return x > y ? x : y; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x, xsimd::batch<T, default_simd_arch> y) const { return xsimd::select(x > y, x, y); }
        };

        struct fmin_fun
        {
            template <class T> T operator()(T x, T y) const { return x < y ? x : y; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x, xsimd::batch<T, default_simd_arch> y) const { return xsimd::select(x < y, x, y); }
        };

        struct deg2rad_fun
        {
            template <class T> T operator()(T x) const { return x * constants<T>::pi / T(180); }
        };

        struct rad2deg_fun
        {
            template <class T> T operator()(T x) const { return x * T(180) / constants<T>::pi; }
        };

        struct clip_fun
        {
            template <class T> T operator()(T x, T lo, T hi) const { return x < lo ? lo : (hi < x ? hi : x); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> x,
                                               xsimd::batch<T, default_simd_arch> lo,
                                               xsimd::batch<T, default_simd_arch> hi) const
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
    } // namespace math

    // Free functions that create expression nodes
    template <class E>
    inline auto abs(const expression<E>& e) { return make_function(math::abs_fun{}, e.derived()); }

    template <class E>
    inline auto sqrt(const expression<E>& e) { return make_function(math::sqrt_fun{}, e.derived()); }

    template <class E>
    inline auto exp(const expression<E>& e) { return make_function(math::exp_fun{}, e.derived()); }

    template <class E>
    inline auto log(const expression<E>& e) { return make_function(math::log_fun{}, e.derived()); }

    template <class E>
    inline auto sin(const expression<E>& e) { return make_function(math::sin_fun{}, e.derived()); }

    template <class E>
    inline auto cos(const expression<E>& e) { return make_function(math::cos_fun{}, e.derived()); }

    template <class E>
    inline auto tan(const expression<E>& e) { return make_function(math::tan_fun{}, e.derived()); }

    template <class E>
    inline auto asin(const expression<E>& e) { return make_function(math::asin_fun{}, e.derived()); }

    template <class E>
    inline auto acos(const expression<E>& e) { return make_function(math::acos_fun{}, e.derived()); }

    template <class E>
    inline auto atan(const expression<E>& e) { return make_function(math::atan_fun{}, e.derived()); }

    template <class E>
    inline auto sinh(const expression<E>& e) { return make_function(math::sinh_fun{}, e.derived()); }

    template <class E>
    inline auto cosh(const expression<E>& e) { return make_function(math::cosh_fun{}, e.derived()); }

    template <class E>
    inline auto tanh(const expression<E>& e) { return make_function(math::tanh_fun{}, e.derived()); }

    template <class E>
    inline auto ceil(const expression<E>& e) { return make_function(math::ceil_fun{}, e.derived()); }

    template <class E>
    inline auto floor(const expression<E>& e) { return make_function(math::floor_fun{}, e.derived()); }

    template <class E>
    inline auto round(const expression<E>& e) { return make_function(math::round_fun{}, e.derived()); }

    template <class E>
    inline auto trunc(const expression<E>& e) { return make_function(math::trunc_fun{}, e.derived()); }

    template <class E1, class E2>
    inline auto pow(const expression<E1>& e1, const expression<E2>& e2)
    { return make_function(math::pow_fun{}, e1.derived(), e2.derived()); }

    template <class E1, class E2>
    inline auto atan2(const expression<E1>& y, const expression<E2>& x)
    { return make_function(math::atan2_fun{}, y.derived(), x.derived()); }

    template <class E1, class E2>
    inline auto hypot(const expression<E1>& e1, const expression<E2>& e2)
    { return make_function(math::hypot_fun{}, e1.derived(), e2.derived()); }

    template <class E1, class E2>
    inline auto fmod(const expression<E1>& e1, const expression<E2>& e2)
    { return make_function(math::fmod_fun{}, e1.derived(), e2.derived()); }

    template <class E1, class E2>
    inline auto fmax(const expression<E1>& e1, const expression<E2>& e2)
    { return make_function(math::fmax_fun{}, e1.derived(), e2.derived()); }

    template <class E1, class E2>
    inline auto fmin(const expression<E1>& e1, const expression<E2>& e2)
    { return make_function(math::fmin_fun{}, e1.derived(), e2.derived()); }

    template <class E>
    inline auto deg2rad(const expression<E>& e)
    { return make_function(math::deg2rad_fun{}, e.derived()); }

    template <class E>
    inline auto rad2deg(const expression<E>& e)
    { return make_function(math::rad2deg_fun{}, e.derived()); }

    template <class E>
    inline auto sign(const expression<E>& e)
    { return make_function(math::sign_fun{}, e.derived()); }

    template <class E>
    inline auto isfinite(const expression<E>& e)
    { return make_function(math::isfinite_fun{}, e.derived()); }

    template <class E>
    inline auto isinf(const expression<E>& e)
    { return make_function(math::isinf_fun{}, e.derived()); }

    template <class E>
    inline auto isnan(const expression<E>& e)
    { return make_function(math::isnan_fun{}, e.derived()); }

    template <class E1, class E2, class E3>
    inline auto clip(const expression<E1>& e1, const expression<E2>& lo, const expression<E3>& hi)
    { return make_function(math::clip_fun{}, e1.derived(), lo.derived(), hi.derived()); }

    // ========== Advanced Searching ==========
    namespace search
    {
        /**
         * Binary search in a sorted 1D array.
         * Returns index of element if found, otherwise -1.
         */
        template <class E, class T>
        inline long binary_search(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            auto it = std::lower_bound(arr.data(), arr.data() + arr.size(), value);
            if (it != arr.data() + arr.size() && *it == value)
                return static_cast<long>(std::distance(arr.data(), it));
            return -1;
        }

        /**
         * Lower bound: first index where value could be inserted without breaking order.
         */
        template <class E, class T>
        inline long lower_bound(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            auto it = std::lower_bound(arr.data(), arr.data() + arr.size(), value);
            return static_cast<long>(std::distance(arr.data(), it));
        }

        /**
         * Upper bound: first index where value is strictly greater.
         */
        template <class E, class T>
        inline long upper_bound(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            auto it = std::upper_bound(arr.data(), arr.data() + arr.size(), value);
            return static_cast<long>(std::distance(arr.data(), it));
        }

        /**
         * Find the index of the nearest element to a target value in a sorted array.
         */
        template <class E, class T>
        inline long nearest_sorted(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            auto it = std::lower_bound(arr.data(), arr.data() + arr.size(), value);
            if (it == arr.data()) return 0;
            if (it == arr.data() + arr.size()) return static_cast<long>(arr.size()) - 1;
            auto prev = it - 1;
            return (value - *prev) <= (*it - value) ? std::distance(arr.data(), prev) : std::distance(arr.data(), it);
        }

        /**
         * Find the index of the nearest element (unsorted) – linear scan.
         */
        template <class E, class T>
        inline long nearest(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            long best_idx = -1;
            T best_diff = std::numeric_limits<T>::max();
            for (std::size_t i = 0; i < arr.size(); ++i)
            {
                T diff = std::abs(arr[i] - value);
                if (diff < best_diff) { best_diff = diff; best_idx = static_cast<long>(i); }
            }
            return best_idx;
        }

        /**
         * Find all indices where value occurs.
         */
        template <class E, class T>
        inline auto find_all(const expression<E>& e, const T& value)
        {
            const auto& arr = e.derived();
            std::vector<long> indices;
            for (std::size_t i = 0; i < arr.size(); ++i)
                if (arr[i] == value) indices.push_back(static_cast<long>(i));
            return array<long>(indices.begin(), indices.end());
        }
    } // namespace search

} // namespace numdot

#endif // NUMDOT_MATH_H