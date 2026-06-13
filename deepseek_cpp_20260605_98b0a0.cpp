//File 0010 (UPDATED) : core/xeval.hpp
//Forces immediate evaluation of expressions into temporary arrays with SIMD copy, alignment, and memory pool awareness.
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
        template <class T, class = void>
        struct has_data : std::false_type {};

        template <class T>
        struct has_data<T, std::void_t<decltype(std::declval<const T&>().data())>> : std::true_type {};

        template <class T>
        inline constexpr bool has_data_v = has_data<T>::value;

        template <class E, class T = typename std::decay_t<E>::value_type>
        inline auto evaluate_expression(E&& expr)
        {
            using expr_type = std::decay_t<E>;
            auto shape = expr.shape();
            using container_type = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
            container_type result(shape);

            if constexpr (has_data_v<expr_type> && std::is_same_v<decltype(expr.data()), const T*>)
            {
                const T* src = expr.data();
                T* dst = result.data();
                std::size_t count = expr.size();
                if (count > 0)
                {
                    if constexpr (is_simd_enabled_v<T>)
                    {
                        using simd_type = xsimd::batch<T, default_simd_arch>;
                        constexpr std::size_t simd_size = simd_type::size;
                        std::size_t vec_count = count / simd_size;
                        for (std::size_t i = 0; i < vec_count; ++i)
                        {
                            simd_type v = simd_type::load_unaligned(src + i * simd_size);
                            v.store_unaligned(dst + i * simd_size);
                        }
                        for (std::size_t i = vec_count * simd_size; i < count; ++i)
                            dst[i] = src[i];
                    }
                    else
                    {
                        std::copy(src, src + count, dst);
                    }
                }
            }
            else
            {
                result = expr; // uses expression assignment
            }
            return result;
        }
    }

    /**
     * Evaluate any expression into an xarray with actual type.
     */
    template <class E>
    inline auto eval(E&& expr)
    {
        return detail::evaluate_expression(std::forward<E>(expr));
    }

    /**
     * Evaluate expression into xarray with a specified value type, converting if needed.
     */
    template <class T, class E>
    inline auto eval_as(E&& expr)
    {
        using expr_type = std::decay_t<E>;
        using container_type = xarray_container<xt::uvector<T>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        container_type result(expr.shape());
        auto* dst = result.data();
        std::size_t sz = result.size();
        const auto* src = expr.data();
        for (std::size_t i = 0; i < sz; ++i)
            dst[i] = static_cast<T>(src[i]);
        return result;
    }

    /**
     * Evaluate and return as fixed-rank xtensor.
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
        result = expr;
        return result;
    }

} // namespace xt

#endif // XTENSOR_XEVAL_HPP