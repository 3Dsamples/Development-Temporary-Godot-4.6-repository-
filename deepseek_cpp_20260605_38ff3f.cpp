//File 0010 : core/xeval.hpp
//Forces immediate evaluation of any expression into a temporary array using SIMD-accelerated copy and aligned memory allocation.
#ifndef XTENSOR_XEVAL_HPP
#define XTENSOR_XEVAL_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xreducer.hpp"
#include "xaccumulator.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    namespace detail
    {
        /**
         * Detect whether a type has a data() member (i.e., is a container).
         */
        template <class T, class = void>
        struct has_data : std::false_type {};

        template <class T>
        struct has_data<T, std::void_t<decltype(std::declval<const T&>().data())>> : std::true_type {};

        template <class T>
        inline constexpr bool has_data_v = has_data<T>::value;

        /**
         * Evaluate an expression into a newly allocated temporary array using SIMD copy.
         */
        template <class E, class T = typename std::decay_t<E>::value_type>
        inline auto evaluate_expression(E&& expr)
        {
            using expr_type = std::decay_t<E>;
            auto shape = expr.shape();
            using container_type = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
            container_type result(shape);

            // If expression has contiguous data, copy directly with SIMD
            if constexpr (has_data_v<expr_type> && std::is_same_v<decltype(expr.data()), const T*>)
            {
                const T* src = expr.data();
                T* dst = result.data();
                std::size_t count = expr.size();
                if (count > 0)
                {
                    // Use SIMD memory copy if size allows
                    if constexpr (is_simd_enabled_v<T>)
                    {
                        using simd_type = xsimd::batch<T, xsimd::default_arch>;
                        constexpr std::size_t simd_size = simd_type::size;
                        std::size_t vec_count = count / simd_size;
                        for (std::size_t i = 0; i < vec_count; ++i)
                        {
                            simd_type v = simd_type::load_unaligned(src + i * simd_size);
                            v.store_unaligned(dst + i * simd_size);
                        }
                        // copy remainder
                        for (std::size_t i = vec_count * simd_size; i < count; ++i)
                        {
                            dst[i] = src[i];
                        }
                    }
                    else
                    {
                        std::copy(src, src + count, dst);
                    }
                }
            }
            else
            {
                // Element-by-element evaluation via assign_temporary or loop
                result = expr;  // uses xcontainer_semantic assignment which does element copy
            }
            return result;
        }
    }

    /**
     * Evaluate an expression into an xarray with its actual type.
     */
    template <class E>
    inline auto eval(E&& expr)
    {
        return detail::evaluate_expression(std::forward<E>(expr));
    }

    /**
     * Evaluate an expression into an xarray with a specified value type.
     */
    template <class T, class E>
    inline auto eval_as(E&& expr)
    {
        using expr_type = std::decay_t<E>;
        using container_type = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        container_type result(expr.shape());
        // Manual assignment to force type conversion
        auto flat_result = result.data();
        std::size_t sz = result.size();
        // For simplicity, iterate sequentially; could be optimized with SIMD if T matches
        for (std::size_t i = 0; i < sz; ++i)
        {
            flat_result[i] = static_cast<T>(expr.data()[i]);
        }
        return result;
    }

    /**
     * Force evaluation of an expression and return as a fixed-rank xtensor.
     */
    template <std::size_t N, class E>
    inline auto eval_xtensor(E&& expr)
    {
        using expr_type = std::decay_t<E>;
        using T = typename expr_type::value_type;
        auto shape = expr.shape();
        std::array<std::size_t, N> fixed_shape;
        std::copy(shape.begin(), shape.begin() + std::min(N, shape.size()), fixed_shape.begin());
        using container_type = xtensor_container<xt::uvector<T>, N, DEFAULT_LAYOUT>;
        container_type result(fixed_shape);
        result = expr; // evaluation via expression assignment
        return result;
    }

}  // namespace xt

#endif  // XTENSOR_XEVAL_HPP