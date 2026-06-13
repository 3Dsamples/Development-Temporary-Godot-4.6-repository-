//File 0108 : numdot/reductions.h
//Reduction operations: sum, prod, mean, min, max with axis support, SIMD accumulation, and parallel execution policies for high performance.
#ifndef NUMDOT_REDUCTIONS_H
#define NUMDOT_REDUCTIONS_H

#include <type_traits>
#include <utility>
#include <vector>
#include <cstddef>
#include <numeric>
#include <algorithm>
#include <limits>
#include <thread>
#include <functional>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"
#include "elementwise.h"
#include "array.h"

namespace numdot
{
    namespace detail
    {
        /**
         * SIMD-enabled sequential reduction over a 1D buffer.
         */
        template <class T, class BinaryOp, class Init>
        inline T reduce_1d(const T* data, std::size_t count, BinaryOp op, Init init_val)
        {
            T result = init_val;
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = count / simd_size;
                simd_type vacc(result);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type vdata = simd_type::load_unaligned(data + i * simd_size);
                    vacc = op.simd_apply(vacc, vdata);
                }
                T tmp[simd_size];
                vacc.store_unaligned(tmp);
                for (std::size_t k = 0; k < simd_size; ++k)
                    result = op(result, tmp[k]);
                for (std::size_t i = vec_count * simd_size; i < count; ++i)
                    result = op(result, data[i]);
            }
            else
            {
                for (std::size_t i = 0; i < count; ++i)
                    result = op(result, data[i]);
            }
            return result;
        }

        /**
         * Parallel reduction over a 1D buffer using multiple threads.
         */
        template <class T, class BinaryOp, class Init>
        inline T reduce_1d_parallel(const T* data, std::size_t count, BinaryOp op, Init init_val)
        {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;
            if (count < num_threads * 4096)
                return reduce_1d(data, count, op, init_val);

            std::size_t chunk = (count + num_threads - 1) / num_threads;
            std::vector<T> partials(num_threads, init_val);
            std::vector<std::thread> threads;
            for (unsigned int t = 0; t < num_threads; ++t)
            {
                threads.emplace_back([&, t]() {
                    std::size_t start = t * chunk;
                    std::size_t end = std::min(start + chunk, count);
                    T local = init_val;
                    for (std::size_t i = start; i < end; ++i)
                        local = op(local, data[i]);
                    partials[t] = local;
                });
            }
            for (auto& th : threads) th.join();
            T result = init_val;
            for (auto& p : partials)
                result = op(result, p);
            return result;
        }

        // Functors with simd_apply
        struct plus
        {
            template <class T>
            T operator()(T a, T b) const { return a + b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a + b; }
        };

        struct multiplies
        {
            template <class T>
            T operator()(T a, T b) const { return a * b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a * b; }
        };

        struct maximum
        {
            template <class T>
            T operator()(T a, T b) const { return a > b ? a : b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const
            {
                return xsimd::select(a > b, a, b);
            }
        };

        struct minimum
        {
            template <class T>
            T operator()(T a, T b) const { return a < b ? a : b; }
            template <class T>
            auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const
            {
                return xsimd::select(a < b, a, b);
            }
        };
    }

    /**
     * Sum over all elements (full reduction).
     */
    template <class E>
    inline auto sum(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        const T* data = expr.data();
        std::size_t count = expr.size();
        if (data == nullptr)
        {
            // Expression without contiguous storage; use iterator fallback
            T s = T(0);
            for (std::size_t i = 0; i < count; ++i) s += expr[i];
            return s;
        }
        return detail::reduce_1d_parallel(data, count, detail::plus{}, T(0));
    }

    /**
     * Sum over a specific axis.
     */
    template <class E>
    inline auto sum(const expression<E>& e, std::size_t axis)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        auto shape = expr.shape();
        if (axis >= shape.size()) throw std::out_of_range("sum: axis out of range.");
        std::size_t axis_len = shape[axis];
        std::size_t outer_size = 1;
        for (std::size_t d = 0; d < axis; ++d) outer_size *= shape[d];
        std::size_t inner_size = 1;
        for (std::size_t d = axis + 1; d < shape.size(); ++d) inner_size *= shape[d];
        // Result shape = shape with axis removed
        std::vector<std::size_t> res_shape = shape;
        res_shape.erase(res_shape.begin() + axis);
        array<T> result(res_shape, T(0));
        T* res_data = result.data();
        const T* src_data = expr.data();
        for (std::size_t o = 0; o < outer_size; ++o)
        {
            for (std::size_t i = 0; i < inner_size; ++i)
            {
                T s = T(0);
                for (std::size_t k = 0; k < axis_len; ++k)
                {
                    std::size_t idx = o * axis_len * inner_size + k * inner_size + i;
                    s += src_data[idx];
                }
                res_data[o * inner_size + i] = s;
            }
        }
        return result;
    }

    /**
     * Product over all elements.
     */
    template <class E>
    inline auto prod(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        const T* data = expr.data();
        std::size_t count = expr.size();
        if (data == nullptr)
        {
            T p = T(1);
            for (std::size_t i = 0; i < count; ++i) p *= expr[i];
            return p;
        }
        return detail::reduce_1d_parallel(data, count, detail::multiplies{}, T(1));
    }

    /**
     * Mean over all elements.
     */
    template <class E>
    inline auto mean(const expression<E>& e)
    {
        return sum(e) / static_cast<double>(e.derived().size());
    }

    /**
     * Mean over an axis.
     */
    template <class E>
    inline auto mean(const expression<E>& e, std::size_t axis)
    {
        auto s = sum(e, axis);
        auto len = static_cast<double>(e.derived().shape()[axis]);
        return s / len;
    }

    /**
     * Maximum over all elements.
     */
    template <class E>
    inline auto max(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        const T* data = expr.data();
        std::size_t count = expr.size();
        if (data == nullptr)
        {
            T m = std::numeric_limits<T>::lowest();
            for (std::size_t i = 0; i < count; ++i) { T v = expr[i]; if (v > m) m = v; }
            return m;
        }
        return detail::reduce_1d_parallel(data, count, detail::maximum{}, std::numeric_limits<T>::lowest());
    }

    /**
     * Minimum over all elements.
     */
    template <class E>
    inline auto min(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        const T* data = expr.data();
        std::size_t count = expr.size();
        if (data == nullptr)
        {
            T m = std::numeric_limits<T>::max();
            for (std::size_t i = 0; i < count; ++i) { T v = expr[i]; if (v < m) m = v; }
            return m;
        }
        return detail::reduce_1d_parallel(data, count, detail::minimum{}, std::numeric_limits<T>::max());
    }

    /**
     * Argmax (index of maximum element) for 1D.
     */
    template <class E>
    inline auto argmax(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        std::size_t count = expr.size();
        if (count == 0) throw std::runtime_error("argmax: empty array.");
        T best = expr[0];
        std::size_t idx = 0;
        for (std::size_t i = 1; i < count; ++i)
        {
            T v = expr[i];
            if (v > best) { best = v; idx = i; }
        }
        return idx;
    }

    /**
     * Argmin (index of minimum element) for 1D.
     */
    template <class E>
    inline auto argmin(const expression<E>& e)
    {
        using T = typename E::value_type;
        const auto& expr = e.derived();
        std::size_t count = expr.size();
        if (count == 0) throw std::runtime_error("argmin: empty array.");
        T best = expr[0];
        std::size_t idx = 0;
        for (std::size_t i = 1; i < count; ++i)
        {
            T v = expr[i];
            if (v < best) { best = v; idx = i; }
        }
        return idx;
    }

} // namespace numdot

#endif // NUMDOT_REDUCTIONS_H