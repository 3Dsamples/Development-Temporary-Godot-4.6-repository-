//File 0360 : xframe/xvariable_variant.hpp
//Variable variant: a type‑safe union of multiple variable types using std::variant, enabling runtime polymorphism with SIMD‑accelerated dispatch and expression integration.
#ifndef XFRAME_XVARIABLE_VARIANT_HPP
#define XFRAME_XVARIABLE_VARIANT_HPP

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
#include "xvariable_meta.hpp"

namespace xframe
{
    namespace detail
    {
        // Helper to check if a type is part of a variant's type list.
        template <class T, class Variant>
        struct is_in_variant : std::false_type {};

        template <class T, class... Ts>
        struct is_in_variant<T, std::variant<Ts...>>
            : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

        template <class T, class Variant>
        inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

        // Visitor that applies a unary functor to the concrete variable inside the variant.
        template <class F>
        struct variant_unary_visitor
        {
            F m_func;

            template <class T, class L>
            auto operator()(const variable<T, L>& v) const
            {
                return m_func(v);
            }
        };

        // Visitor that applies a binary functor to two concrete variables.
        template <class F>
        struct variant_binary_visitor
        {
            F m_func;

            template <class T, class U, class L>
            auto operator()(const variable<T, L>& a, const variable<U, L>& b) const
                -> std::enable_if_t<std::is_same_v<T, U>, variable<T, L>>
            {
                return m_func(a, b);
            }

            template <class T, class U, class L>
            auto operator()(const variable<T, L>& a, const variable<U, L>& b) const
                -> std::enable_if_t<!std::is_same_v<T, U>, variable<std::common_type_t<T, U>, L>>
            {
                throw std::runtime_error("variant_binary_visitor: type mismatch in variant operation.");
            }
        };
    }

    /**
     * @class xvariable_variant
     * @brief A type‑erased variable that can hold any variable type from a predefined list.
     *
     * Internally uses std::variant<variable<T1, L>, variable<T2, L>, ...>.
     * Provides a uniform interface for size, element access, arithmetic,
     * resizing, filling, and SIMD loads. The actual operation is dispatched
     * to the concrete variable via std::visit.
     */
    template <class L = label_type, class... Ts>
    class xvariable_variant : public expression<xvariable_variant<L, Ts...>>
    {
    public:
        using self_type = xvariable_variant<L, Ts...>;
        using variant_type = std::variant<variable<Ts, L>...>;
        using size_type = std::size_t;
        using value_type = double; // common type for access
        using label_type = L;

        static_assert(sizeof...(Ts) > 0, "xvariable_variant requires at least one variable type.");

        /**
         * Default constructor: holds a default‑constructed variable of the first type.
         */
        xvariable_variant()
            : m_var(variable<std::tuple_element_t<0, std::tuple<Ts...>>, L>())
        {
        }

        /**
         * Construct from a concrete variable of any of the allowed types.
         */
        template <class T, std::enable_if_t<detail::is_in_variant_v<variable<T, L>, variant_type>, int> = 0>
        explicit xvariable_variant(const variable<T, L>& var)
            : m_var(var)
        {
        }

        template <class T, std::enable_if_t<detail::is_in_variant_v<variable<T, L>, variant_type>, int> = 0>
        explicit xvariable_variant(variable<T, L>&& var) noexcept
            : m_var(std::move(var))
        {
        }

        xvariable_variant(const self_type&) = default;
        xvariable_variant& operator=(const self_type&) = default;
        xvariable_variant(self_type&&) = default;
        xvariable_variant& operator=(self_type&&) = default;

        /**
         * Size: delegates to the concrete variable.
         */
        size_type size() const
        {
            return std::visit([](const auto& v) { return v.size(); }, m_var);
        }

        bool empty() const
        {
            return std::visit([](const auto& v) { return v.empty(); }, m_var);
        }

        /**
         * Name of the variable.
         */
        label_type name() const
        {
            return std::visit([](const auto& v) -> label_type { return v.name(); }, m_var);
        }

        void set_name(const label_type& n)
        {
            std::visit([&](auto& v) { v.set_name(n); }, m_var);
        }

        /**
         * Element access: returns a common numeric value (double).
         */
        double operator[](size_type i) const
        {
            return std::visit([i](const auto& v) -> double { return static_cast<double>(v[i]); }, m_var);
        }

        /**
         * Fill with a constant value.
         */
        void fill(double val)
        {
            std::visit([val](auto& v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                v.fill(static_cast<T>(val));
            }, m_var);
        }

        /**
         * Resize.
         */
        void resize(size_type n)
        {
            std::visit([n](auto& v) { v.resize(n); }, m_var);
        }

        /**
         * SIMD load: delegates to concrete variable, converting to double batch.
         */
        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            return std::visit([i](const auto& v) {
                return v.template load_simd<Align, T>(i);
            }, m_var);
        }

        /**
         * Arithmetic operators: element‑wise with another variant or scalar.
         */
        self_type& operator+=(const self_type& rhs)
        {
            std::visit([&](auto& lhs) {
                std::visit([&](const auto& rhs_var) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, std::decay_t<decltype(rhs_var)>>)
                    {
                        lhs += rhs_var;
                    }
                    else
                    {
                        throw std::runtime_error("xvariable_variant::operator+=: type mismatch.");
                    }
                }, rhs.m_var);
            }, m_var);
            return *this;
        }

        self_type& operator-=(const self_type& rhs)
        {
            std::visit([&](auto& lhs) {
                std::visit([&](const auto& rhs_var) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, std::decay_t<decltype(rhs_var)>>)
                        lhs -= rhs_var;
                    else
                        throw std::runtime_error("xvariable_variant::operator-=: type mismatch.");
                }, rhs.m_var);
            }, m_var);
            return *this;
        }

        self_type& operator*=(double scalar)
        {
            std::visit([scalar](auto& v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                v *= static_cast<T>(scalar);
            }, m_var);
            return *this;
        }

        self_type& operator/=(double scalar)
        {
            std::visit([scalar](auto& v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                v /= static_cast<T>(scalar);
            }, m_var);
            return *this;
        }

        /**
         * Access the underlying variant.
         */
        variant_type& variant() noexcept { return m_var; }
        const variant_type& variant() const noexcept { return m_var; }

        /**
         * Check if the variant holds a specific variable type.
         */
        template <class T>
        bool holds() const noexcept
        {
            return std::holds_alternative<variable<T, L>>(m_var);
        }

        /**
         * Get a pointer to the concrete variable of type T, or nullptr.
         */
        template <class T>
        variable<T, L>* get_if() noexcept
        {
            return std::get_if<variable<T, L>>(&m_var);
        }

        template <class T>
        const variable<T, L>* get_if() const noexcept
        {
            return std::get_if<variable<T, L>>(&m_var);
        }

        /**
         * Apply a unary operation to the concrete variable, returning a new variant.
         */
        template <class F>
        auto apply(F&& f) const -> xvariable_variant<L, Ts...>
        {
            return xvariable_variant<L, Ts...>(
                std::visit(detail::variant_unary_visitor<std::decay_t<F>>{std::forward<F>(f)}, m_var));
        }

    private:
        variant_type m_var;
    };

    // Operator overloads for xvariable_variant
    template <class L, class... Ts>
    inline auto operator+(const xvariable_variant<L, Ts...>& a,
                          const xvariable_variant<L, Ts...>& b)
    {
        auto result = a;
        result += b;
        return result;
    }

    template <class L, class... Ts>
    inline auto operator-(const xvariable_variant<L, Ts...>& a,
                          const xvariable_variant<L, Ts...>& b)
    {
        auto result = a;
        result -= b;
        return result;
    }

    template <class L, class... Ts>
    inline auto operator*(const xvariable_variant<L, Ts...>& a, double scalar)
    {
        auto result = a;
        result *= scalar;
        return result;
    }

    template <class L, class... Ts>
    inline auto operator*(double scalar, const xvariable_variant<L, Ts...>& a)
    {
        return a * scalar;
    }

    template <class L, class... Ts>
    inline auto operator/(const xvariable_variant<L, Ts...>& a, double scalar)
    {
        auto result = a;
        result /= scalar;
        return result;
    }

    /**
     * Helper to create a variant variable from a concrete variable.
     */
    template <class L = label_type, class... Ts, class T>
    inline auto make_variant_variable(const variable<T, L>& var)
    {
        return xvariable_variant<L, Ts...>(var);
    }

} // namespace xframe

#endif // XFRAME_XVARIABLE_VARIANT_HPP