//File 0042 : core/xassign.hpp
//Assignment engine with SIMD‑strided loops, layout‑aware dispatching, and full expression tag specializations.
#ifndef XTENSOR_XASSIGN_HPP
#define XTENSOR_XASSIGN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xnoalias.hpp"

namespace xt
{
    /*********************************************
     * assignment tags for dispatching
     *********************************************/
    struct row_major_assign {};
    struct column_major_assign {};
    struct strided_assign {};
    struct trivial_assign {};

    namespace detail
    {
        // Detect contiguous layout
        template <class E>
        bool is_contiguous_row_major(const E& e)
        {
            auto sh = e.shape();
            if (sh.empty()) return true;
            auto st = e.strides();
            std::size_t expected = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sh.size()) - 1; i >= 0; --i)
            {
                if (st[static_cast<std::size_t>(i)] != expected) return false;
                expected *= sh[static_cast<std::size_t>(i)];
            }
            return true;
        }

        template <class E>
        bool is_contiguous_column_major(const E& e)
        {
            auto sh = e.shape();
            if (sh.empty()) return true;
            auto st = e.strides();
            std::size_t expected = 1;
            for (std::size_t i = 0; i < sh.size(); ++i)
            {
                if (st[i] != expected) return false;
                expected *= sh[i];
            }
            return true;
        }
    }

    /*********************************************
     * xassignment_engine
     *********************************************/
    template <class E1, class E2, class Tag = void>
    class xassignment_engine;

    // Trivial contiguous assignment (both row-major, same shape)
    template <class E1, class E2>
    class xassignment_engine<E1, E2, trivial_assign>
    {
    public:
        static void run(E1& dst, const E2& src)
        {
            using T = typename E1::value_type;
            std::size_t count = dst.size();
            const T* src_ptr = src.data();
            T* dst_ptr = dst.data();

            if constexpr (is_simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec = count / simd_size;
                for (std::size_t i = 0; i < vec; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src_ptr + i * simd_size);
                    v.store_unaligned(dst_ptr + i * simd_size);
                }
                for (std::size_t i = vec * simd_size; i < count; ++i)
                    dst_ptr[i] = src_ptr[i];
            }
            else
            {
                std::copy(src_ptr, src_ptr + count, dst_ptr);
            }
        }
    };

    // Row-major assignment (broadcasting allowed)
    template <class E1, class E2>
    class xassignment_engine<E1, E2, row_major_assign>
    {
    public:
        static void run(E1& dst, const E2& src)
        {
            using T = typename E1::value_type;
            auto dst_shape = dst.shape();
            auto src_shape = src.shape();
            auto dst_strides = dst.strides();
            auto src_strides = src.strides();

            std::size_t ndim = dst_shape.size();
            if (ndim == 0) return;

            // Innermost dimension contiguous?
            if (dst_strides[ndim-1] == 1 && src_strides[ndim-1] == 1 &&
                dst_shape[ndim-1] == src_shape[ndim-1])
            {
                // Use SIMD for innermost dimension
                std::size_t inner_size = dst_shape[ndim-1];
                std::size_t outer_count = dst.size() / inner_size;

                T* dst_ptr = dst.data();
                const T* src_ptr = src.data();

                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    for (std::size_t o = 0; o < outer_count; ++o)
                    {
                        std::size_t offset_dst = 0, offset_src = 0;
                        auto idx = unravel_index(o * inner_size, dst_shape);
                        // Compute offsets based on multi-index
                        for (std::size_t d = 0; d < ndim - 1; ++d)
                        {
                            offset_dst += idx[d] * dst_strides[d];
                            offset_src += (idx[d] % src_shape[d]) * src_strides[d];
                        }
                        std::size_t j = 0;
                        for (; j + simd_size <= inner_size; j += simd_size)
                        {
                            simd_type v = simd_type::load_unaligned(src_ptr + offset_src + j);
                            v.store_unaligned(dst_ptr + offset_dst + j);
                        }
                        for (; j < inner_size; ++j)
                            dst_ptr[offset_dst + j] = src_ptr[offset_src + j];
                    }
                }
                else
                {
                    // scalar copy
                    std::vector<std::size_t> idx(ndim, 0);
                    for (std::size_t i = 0; i < dst.size(); ++i)
                    {
                        auto dst_idx = unravel_index(i, dst_shape);
                        std::size_t dst_off = 0, src_off = 0;
                        for (std::size_t d = 0; d < ndim; ++d)
                        {
                            dst_off += dst_idx[d] * dst_strides[d];
                            src_off += (dst_idx[d] % src_shape[d]) * src_strides[d];
                        }
                        dst_ptr[dst_off] = src_ptr[src_off];
                    }
                }
            }
            else
            {
                // Fully strided copy
                run_strided(dst, src);
            }
        }

    private:
        static void run_strided(E1& dst, const E2& src)
        {
            using T = typename E1::value_type;
            auto dst_shape = dst.shape();
            auto src_shape = src.shape();
            auto dst_strides = dst.strides();
            auto src_strides = src.strides();
            std::size_t ndim = dst_shape.size();

            T* dst_ptr = dst.data();
            const T* src_ptr = src.data();

            std::vector<std::size_t> idx(ndim, 0);
            for (std::size_t i = 0; i < dst.size(); ++i)
            {
                std::size_t dst_off = 0, src_off = 0;
                for (std::size_t d = 0; d < ndim; ++d)
                {
                    dst_off += idx[d] * dst_strides[d];
                    src_off += (idx[d] % src_shape[d]) * src_strides[d];
                }
                dst_ptr[dst_off] = src_ptr[src_off];

                // Increment index
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
                {
                    idx[static_cast<std::size_t>(d)]++;
                    if (idx[static_cast<std::size_t>(d)] < dst_shape[static_cast<std::size_t>(d)])
                        break;
                    idx[static_cast<std::size_t>(d)] = 0;
                }
            }
        }
    };

    // Column-major assignment (similar to row_major_assign but innermost dimension is first)
    template <class E1, class E2>
    class xassignment_engine<E1, E2, column_major_assign>
    {
    public:
        static void run(E1& dst, const E2& src)
        {
            // For column-major, contiguous dimension is 0
            using T = typename E1::value_type;
            auto dst_shape = dst.shape();
            auto src_shape = src.shape();
            auto dst_strides = dst.strides();
            auto src_strides = src.strides();
            std::size_t ndim = dst_shape.size();
            if (ndim == 0) return;

            T* dst_ptr = dst.data();
            const T* src_ptr = src.data();

            if (dst_strides[0] == 1 && src_strides[0] == 1 && dst_shape[0] == src_shape[0])
            {
                std::size_t inner_size = dst_shape[0];
                std::size_t outer_count = dst.size() / inner_size;
                if constexpr (is_simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    for (std::size_t o = 0; o < outer_count; ++o)
                    {
                        std::size_t off_dst = o * inner_size * dst_strides[1]; // simplified
                        std::size_t off_src = o * inner_size * src_strides[1];
                        std::size_t j = 0;
                        for (; j + simd_size <= inner_size; j += simd_size)
                        {
                            simd_type v = simd_type::load_unaligned(src_ptr + off_src + j);
                            v.store_unaligned(dst_ptr + off_dst + j);
                        }
                        for (; j < inner_size; ++j)
                            dst_ptr[off_dst + j] = src_ptr[off_src + j];
                    }
                }
                else
                {
                    std::copy(src_ptr, src_ptr + dst.size(), dst_ptr);
                }
            }
            else
            {
                // fallback to strided
                xassignment_engine<E1, E2, strided_assign>::run(dst, src);
            }
        }
    };

    // Fully strided (generic) assignment
    template <class E1, class E2>
    class xassignment_engine<E1, E2, strided_assign>
    {
    public:
        static void run(E1& dst, const E2& src)
        {
            using T = typename E1::value_type;
            auto dst_shape = dst.shape();
            auto src_shape = src.shape();
            auto dst_strides = dst.strides();
            auto src_strides = src.strides();
            std::size_t ndim = dst_shape.size();

            T* dst_ptr = dst.data();
            const T* src_ptr = src.data();

            std::vector<std::size_t> idx(ndim, 0);
            for (std::size_t i = 0; i < dst.size(); ++i)
            {
                std::size_t dst_off = 0, src_off = 0;
                for (std::size_t d = 0; d < ndim; ++d)
                {
                    dst_off += idx[d] * dst_strides[d];
                    src_off += (idx[d] % src_shape[d]) * src_strides[d];
                }
                dst_ptr[dst_off] = src_ptr[src_off];

                // Increment index (row-major order)
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
                {
                    idx[static_cast<std::size_t>(d)]++;
                    if (idx[static_cast<std::size_t>(d)] < dst_shape[static_cast<std::size_t>(d)])
                        break;
                    idx[static_cast<std::size_t>(d)] = 0;
                }
            }
        }
    };

    /*********************************************
     * Automatic tag deduction and dispatching
     *********************************************/
    namespace detail
    {
        template <class E1, class E2>
        struct assign_tag
        {
            using type = std::conditional_t<
                is_contiguous_row_major(std::declval<E1>()) && is_contiguous_row_major(std::declval<E2>()) &&
                std::declval<E1>().shape() == std::declval<E2>().shape(),
                trivial_assign,
                std::conditional_t<
                    is_contiguous_row_major(std::declval<E1>()) && is_contiguous_row_major(std::declval<E2>()),
                    row_major_assign,
                    std::conditional_t<
                        is_contiguous_column_major(std::declval<E1>()) && is_contiguous_column_major(std::declval<E2>()),
                        column_major_assign,
                        strided_assign
                    >
                >
            >;
        };
    }

    // Top-level assign function
    template <class E1, class E2>
    inline void assign(E1& dst, const E2& src)
    {
        using tag = typename detail::assign_tag<E1, E2>::type;
        xassignment_engine<E1, E2, tag>::run(dst, src);
    }

} // namespace xt

#endif // XTENSOR_XASSIGN_HPP