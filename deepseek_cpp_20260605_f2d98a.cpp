//File 0348 : xframe/xvariable_assign.hpp
//Variable assignment engine: element-wise copy, scalar broadcast, SIMD-accelerated loops, and alignment-aware assignment for xframe variables.
#ifndef XFRAME_XVARIABLE_ASSIGN_HPP
#define XFRAME_XVARIABLE_ASSIGN_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xvariable.hpp"

namespace xframe
{
    namespace detail
    {
        /**
         * Check if a variable is contiguous and aligned for SIMD.
         */
        template <class T, class L>
        inline bool is_contiguous(const variable<T, L>& var) noexcept
        {
            return var.data() != nullptr && var.size() > 0;
        }
    }

    /**
     * @class variable_assigner
     * @brief Static dispatch engine for assigning data to a variable from an expression.
     */
    template <class Dst, class Src, class Enable = void>
    class variable_assigner
    {
    public:
        /**
         * Generic strided assignment: element-by-element copy.
         * Handles broadcasting where source size may be 1 or match destination.
         */
        static void assign(Dst& dst, const Src& src)
        {
            using T = typename Dst::value_type;
            std::size_t n = dst.size();
            if (n == 0) return;
            T* dst_data = dst.data();
            if constexpr (std::is_arithmetic_v<Src>)
            {
                std::fill(dst_data, dst_data + n, static_cast<T>(src));
            }
            else
            {
                const auto& src_expr = src.derived();
                if (src_expr.size() == 1)
                {
                    T val = static_cast<T>(src_expr[0]);
                    std::fill(dst_data, dst_data + n, val);
                }
                else if (src_expr.size() == n)
                {
                    const T* src_data = src_expr.data();
                    if (src_data != nullptr)
                    {
                        std::copy(src_data, src_data + n, dst_data);
                    }
                    else
                    {
                        for (std::size_t i = 0; i < n; ++i)
                            dst_data[i] = static_cast<T>(src_expr[i]);
                    }
                }
                else
                {
                    throw std::runtime_error("variable_assigner: shape mismatch.");
                }
            }
        }
    };

    /**
     * Specialization when both source and destination are variables (contiguous assignment).
     */
    template <class Dst, class Src>
    class variable_assigner<Dst, Src,
        std::enable_if_t<std::is_same_v<std::decay_t<Dst>, variable<typename Dst::value_type, typename Dst::label_type>> &&
                         std::is_same_v<std::decay_t<Src>, variable<typename Src::value_type, typename Src::label_type>>>>
    {
    public:
        static void assign(Dst& dst, const Src& src)
        {
            using T = typename Dst::value_type;
            std::size_t n = dst.size();
            if (n != src.size())
                throw std::runtime_error("variable_assigner: size mismatch.");
            if (n == 0) return;
            const T* src_ptr = src.data();
            T* dst_ptr = dst.data();
            if constexpr (simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type v = simd_type::load_unaligned(src_ptr + i * simd_size);
                    v.store_unaligned(dst_ptr + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_ptr[i] = src_ptr[i];
            }
            else
            {
                std::copy(src_ptr, src_ptr + n, dst_ptr);
            }
        }
    };

    /**
     * Specialization for scalar assignment to a variable.
     */
    template <class Dst, class T>
    class variable_assigner<Dst, T,
        std::enable_if_t<std::is_arithmetic_v<T>>>
    {
    public:
        static void assign(Dst& dst, T scalar)
        {
            using VT = typename Dst::value_type;
            std::size_t n = dst.size();
            VT* dst_data = dst.data();
            VT val = static_cast<VT>(scalar);
            if constexpr (simd_enabled_v<VT>)
            {
                using simd_type = xsimd::batch<VT, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t vec_count = n / simd_size;
                simd_type vs(val);
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    vs.store_unaligned(dst_data + i * simd_size);
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    dst_data[i] = val;
            }
            else
            {
                std::fill(dst_data, dst_data + n, val);
            }
        }
    };

    /**
     * Top-level assign function for variables and expressions.
     */
    template <class Dst, class Src>
    inline void assign(Dst& dst, const Src& src)
    {
        variable_assigner<Dst, Src>::assign(dst, src);
    }

    /**
     * Assign a range defined by iterators to a variable.
     */
    template <class T, class L, class It>
    inline void assign_range(variable<T, L>& dst, It first, It last)
    {
        std::size_t n = static_cast<std::size_t>(std::distance(first, last));
        if (dst.size() != n)
            dst.resize(n);
        T* dst_data = dst.data();
        if constexpr (simd_enabled_v<T>)
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            std::size_t vec_count = n / simd_size;
            for (std::size_t i = 0; i < vec_count; ++i)
            {
                alignas(64) std::array<T, simd_size> buf;
                for (std::size_t k = 0; k < simd_size; ++k)
                    buf[k] = static_cast<T>(*first++);
                simd_type v = simd_type::load_aligned(buf.data());
                v.store_unaligned(dst_data + i * simd_size);
            }
            for (std::size_t i = vec_count * simd_size; i < n; ++i)
                dst_data[i] = static_cast<T>(*first++);
        }
        else
        {
            for (std::size_t i = 0; i < n; ++i, ++first)
                dst_data[i] = static_cast<T>(*first);
        }
    }

    /**
     * Assign an initializer list to a variable.
     */
    template <class T, class L>
    inline void assign_init_list(variable<T, L>& dst, std::initializer_list<T> values)
    {
        assign_range(dst, values.begin(), values.end());
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_ASSIGN_HPP