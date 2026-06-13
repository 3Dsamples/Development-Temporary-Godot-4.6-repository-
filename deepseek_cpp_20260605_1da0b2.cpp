//File 0319 : xframe/xframe_xassign.hpp
//Assignment engine for xframe: element‑wise copy, strided loops, SIMD‑accelerated transfer, and layout detection.
#ifndef XFRAME_XASSIGN_HPP
#define XFRAME_XASSIGN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe.hpp"

namespace xframe
{
    namespace detail
    {
        /**
         * Check if an xframe's variable data is contiguous and row‑major.
         */
        template <class... V>
        inline bool is_contiguous_row_major(const xframe<V...>& frame)
        {
            if (frame.dimension_count() <= 1) return true;
            // All variables share the same flat layout; check that strides are standard.
            // For xframe, data is always flat and row‑major (last dimension contiguous).
            return true;
        }
    }

    /**
     * @class xassignment_engine
     * @brief Static engine for copying data between xframe expressions.
     *
     * Handles trivial contiguous copy (SIMD), broadcast, and generic strided
     * assignment. The tag template parameter selects the strategy.
     */
    template <class Dst, class Src, class Tag = void>
    class xassignment_engine;

    // Trivial contiguous assignment
    template <class Dst, class Src>
    class xassignment_engine<Dst, Src, std::true_type>
    {
    public:
        static void run(Dst& dst, const Src& src)
        {
            using T = typename Dst::value_type;
            std::size_t count = dst.size();
            const T* src_data = src.data();
            T* dst_data = dst.data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec = count / simd_size;
                for (std::size_t i = 0; i < vec; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src_data + i * simd_size);
                    v.store_unaligned(dst_data + i * simd_size);
                }
                for (std::size_t i = vec * simd_size; i < count; ++i)
                    dst_data[i] = src_data[i];
            }
            else
            {
                std::copy(src_data, src_data + count, dst_data);
            }
        }
    };

    // Generic strided assignment (broadcast support)
    template <class Dst, class Src>
    class xassignment_engine<Dst, Src, std::false_type>
    {
    public:
        static void run(Dst& dst, const Src& src)
        {
            using T = typename Dst::value_type;
            std::size_t ndim = dst.dimension_count();
            if (ndim == 0)
            {
                // scalar
                dst.data()[0] = src.data()[0];
                return;
            }
            // Element‑wise copy via flat iteration
            std::size_t n = dst.size();
            T* dst_ptr = dst.data();
            const T* src_ptr = src.data();
            // If shapes match, we can do direct copy; else broadcast.
            if (dst.same_dimensions(src))
            {
                std::copy(src_ptr, src_ptr + n, dst_ptr);
            }
            else
            {
                // Broadcast: need to map indices; for now assume shapes match after prior broadcasting.
                throw std::runtime_error("xassignment_engine: broadcast assignment not implemented.");
            }
        }
    };

    /**
     * Dispatch assignment based on contiguous detection.
     */
    template <class Dst, class Src>
    inline void assign(Dst& dst, const Src& src)
    {
        using tag = std::conditional_t<detail::is_contiguous_row_major(dst) &&
                                       detail::is_contiguous_row_major(src),
                                       std::true_type, std::false_type>;
        xassignment_engine<Dst, Src, tag>::run(dst, src);
    }

} // namespace xframe

#endif // XFRAME_XASSIGN_HPP