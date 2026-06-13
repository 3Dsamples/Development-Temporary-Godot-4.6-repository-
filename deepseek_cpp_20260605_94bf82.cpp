//File 0343 : xframe/xdynamic_variable_impl.hpp
//Implementation details for xdynamic_variable: variant visitors, SIMD-accelerated arithmetic, type-aware apply, and helper functions for runtime type dispatch.
#ifndef XFRAME_XDYNAMIC_VARIABLE_IMPL_HPP
#define XFRAME_XDYNAMIC_VARIABLE_IMPL_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xvariable.hpp"
#include "xdynamic_variable.hpp"

namespace xframe
{
    namespace detail
    {
        /**
         * @struct dynamic_variable_visitor
         * @brief Applies a binary functor to two dynamic variables that hold the same concrete type.
         *        If the types differ, an exception is thrown.
         */
        template <class F>
        struct dynamic_variable_visitor
        {
            F m_func;

            template <class T, class U>
            auto operator()(const variable<T, label_type>& a, const variable<U, label_type>& b) const
                -> variable<std::common_type_t<T, U>, label_type>
            {
                if constexpr (std::is_same_v<T, U>)
                {
                    using common = T;
                    variable<common, label_type> result(a.size(), a.name() + label_type("_op_") + b.name());
                    const common* ad = a.data();
                    const common* bd = b.data();
                    common* rd = result.data();
                    std::size_t n = a.size();
                    if constexpr (simd_enabled_v<common>)
                    {
                        using simd_type = xsimd::batch<common, default_simd_arch>;
                        constexpr std::size_t simd_size = simd_type::size;
                        std::size_t vec_count = n / simd_size;
                        for (std::size_t i = 0; i < vec_count; ++i)
                        {
                            simd_type va = simd_type::load_unaligned(ad + i * simd_size);
                            simd_type vb = simd_type::load_unaligned(bd + i * simd_size);
                            simd_type vr = m_func.simd_apply ? m_func.simd_apply(va, vb) : m_func(va, vb);
                            vr.store_unaligned(rd + i * simd_size);
                        }
                        for (std::size_t i = vec_count * simd_size; i < n; ++i)
                            rd[i] = m_func(ad[i], bd[i]);
                    }
                    else
                    {
                        for (std::size_t i = 0; i < n; ++i)
                            rd[i] = m_func(ad[i], bd[i]);
                    }
                    return result;
                }
                else
                {
                    throw std::runtime_error("dynamic_variable_visitor: type mismatch in variant operation.");
                }
            }
        };

        /**
         * @struct dynamic_variable_unary_visitor
         * @brief Applies a unary functor to a dynamic variable.
         */
        template <class F>
        struct dynamic_variable_unary_visitor
        {
            F m_func;

            template <class T>
            auto operator()(const variable<T, label_type>& a) const
            {
                variable<T, label_type> result(a.size(), a.name());
                const T* ad = a.data();
                T* rd = result.data();
                std::size_t n = a.size();
                if constexpr (simd_enabled_v<T>)
                {
                    using simd_type = xsimd::batch<T, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    std::size_t vec_count = n / simd_size;
                    for (std::size_t i = 0; i < vec_count; ++i)
                    {
                        simd_type va = simd_type::load_unaligned(ad + i * simd_size);
                        simd_type vr = m_func.simd_apply ? m_func.simd_apply(va) : m_func(va);
                        vr.store_unaligned(rd + i * simd_size);
                    }
                    for (std::size_t i = vec_count * simd_size; i < n; ++i)
                        rd[i] = m_func(ad[i]);
                }
                else
                {
                    for (std::size_t i = 0; i < n; ++i)
                        rd[i] = m_func(ad[i]);
                }
                return result;
            }
        };

        /**
         * @struct dynamic_variable_apply
         * @brief Applies a generic functor to the concrete variable inside a dynamic variable.
         */
        template <class F>
        struct dynamic_variable_apply
        {
            F m_func;

            template <class T>
            auto operator()(variable<T, label_type>& v) const
            {
                return m_func(v);
            }

            template <class T>
            auto operator()(const variable<T, label_type>& v) const
            {
                return m_func(v);
            }
        };

        /**
         * @struct simd_dispatch_binary
         * @brief Functor with simd_apply that delegates to scalar operator().
         */
        struct simd_plus
        {
            template <class T> T operator()(T a, T b) const { return a + b; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a + b; }
        };

        struct simd_minus
        {
            template <class T> T operator()(T a, T b) const { return a - b; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a - b; }
        };

        struct simd_multiplies
        {
            template <class T> T operator()(T a, T b) const { return a * b; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a * b; }
        };

        struct simd_divides
        {
            template <class T> T operator()(T a, T b) const { return a / b; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a, xsimd::batch<T, default_simd_arch> b) const { return a / b; }
        };

        struct simd_negate
        {
            template <class T> T operator()(T a) const { return -a; }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a) const { return -a; }
        };

        struct simd_abs
        {
            template <class T> T operator()(T a) const { return std::abs(a); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a) const { return xsimd::abs(a); }
        };

        struct simd_sqrt
        {
            template <class T> T operator()(T a) const { return std::sqrt(a); }
            template <class T> auto simd_apply(xsimd::batch<T, default_simd_arch> a) const { return xsimd::sqrt(a); }
        };

        /**
         * Helper to get the underlying concrete variable pointer from a dynamic variable.
         */
        template <class T, class... Ts>
        inline variable<T, label_type>* get_if(xdynamic_variable<Ts...>& dv) noexcept
        {
            return std::get_if<variable<T, label_type>>(&dv.variant());
        }

        template <class T, class... Ts>
        inline const variable<T, label_type>* get_if(const xdynamic_variable<Ts...>& dv) noexcept
        {
            return std::get_if<variable<T, label_type>>(&dv.variant());
        }

        /**
         * Helper to call a function on the concrete variable inside a dynamic variable.
         */
        template <class... Ts, class F>
        inline auto visit(xdynamic_variable<Ts...>& dv, F&& f)
        {
            return std::visit(dynamic_variable_apply<F>{std::forward<F>(f)}, dv.variant());
        }

        template <class... Ts, class F>
        inline auto visit(const xdynamic_variable<Ts...>& dv, F&& f)
        {
            return std::visit(dynamic_variable_apply<F>{std::forward<F>(f)}, dv.variant());
        }

        /**
         * Binary operation on two dynamic variables of same variant type.
         */
        template <class... Ts, class F>
        inline auto binary_visit(const xdynamic_variable<Ts...>& a,
                                 const xdynamic_variable<Ts...>& b,
                                 F&& f)
        {
            return std::visit(dynamic_variable_visitor<F>{std::forward<F>(f)}, a.variant(), b.variant());
        }

    } // namespace detail

    // Operator overloads for xdynamic_variable
    template <class... Ts>
    inline auto operator+(const xdynamic_variable<Ts...>& a, const xdynamic_variable<Ts...>& b)
    {
        return xdynamic_variable<Ts...>(detail::binary_visit(a, b, detail::simd_plus{}));
    }

    template <class... Ts>
    inline auto operator-(const xdynamic_variable<Ts...>& a, const xdynamic_variable<Ts...>& b)
    {
        return xdynamic_variable<Ts...>(detail::binary_visit(a, b, detail::simd_minus{}));
    }

    template <class... Ts>
    inline auto operator*(const xdynamic_variable<Ts...>& a, const xdynamic_variable<Ts...>& b)
    {
        return xdynamic_variable<Ts...>(detail::binary_visit(a, b, detail::simd_multiplies{}));
    }

    template <class... Ts>
    inline auto operator/(const xdynamic_variable<Ts...>& a, const xdynamic_variable<Ts...>& b)
    {
        return xdynamic_variable<Ts...>(detail::binary_visit(a, b, detail::simd_divides{}));
    }

    template <class... Ts>
    inline auto operator-(const xdynamic_variable<Ts...>& a)
    {
        return xdynamic_variable<Ts...>(detail::visit(a, [](const auto& v) {
            return detail::dynamic_variable_unary_visitor<detail::simd_negate>{detail::simd_negate{}}(v);
        }));
    }

    // Scalar multiplication / division
    template <class... Ts>
    inline auto operator*(const xdynamic_variable<Ts...>& a, double scalar)
    {
        return detail::visit(a, [scalar](auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            auto result = v;
            result *= static_cast<T>(scalar);
            return xdynamic_variable<Ts...>(std::move(result));
        });
    }

    template <class... Ts>
    inline auto operator*(double scalar, const xdynamic_variable<Ts...>& a)
    {
        return a * scalar;
    }

    template <class... Ts>
    inline auto operator/(const xdynamic_variable<Ts...>& a, double scalar)
    {
        return detail::visit(a, [scalar](auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            auto result = v;
            result /= static_cast<T>(scalar);
            return xdynamic_variable<Ts...>(std::move(result));
        });
    }

    // Element‑wise math functions for dynamic variables
    template <class... Ts>
    inline auto abs(const xdynamic_variable<Ts...>& a)
    {
        return xdynamic_variable<Ts...>(detail::visit(a, [](const auto& v) {
            return detail::dynamic_variable_unary_visitor<detail::simd_abs>{detail::simd_abs{}}(v);
        }));
    }

    template <class... Ts>
    inline auto sqrt(const xdynamic_variable<Ts...>& a)
    {
        return xdynamic_variable<Ts...>(detail::visit(a, [](const auto& v) {
            return detail::dynamic_variable_unary_visitor<detail::simd_sqrt>{detail::simd_sqrt{}}(v);
        }));
    }

} // namespace xframe

#endif // XFRAME_XDYNAMIC_VARIABLE_IMPL_HPP