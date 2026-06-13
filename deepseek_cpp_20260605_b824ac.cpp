//File 0328 : xframe/xaxis_variant.hpp
//Axis variant: a type‑safe union of multiple axis scalar types, allowing runtime dispatch for different coordinate types with SIMD‑optimized access.
#ifndef XFRAME_XAXIS_VARIANT_HPP
#define XFRAME_XAXIS_VARIANT_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"
#include "xaxis_scalar.hpp"

namespace xframe
{
    /**
     * @class xaxis_variant
     * @brief A variant type that can hold any of the specified axis scalar types.
     *
     * It provides a uniform interface to access the underlying axis scalar
     * regardless of the concrete type. The variant visitor pattern is used
     * for element access and operations.
     */
    template <class... Ts>
    class xaxis_variant : public expression<xaxis_variant<Ts...>>
    {
    public:
        using self_type = xaxis_variant<Ts...>;
        using variant_type = std::variant<xaxis_scalar<Ts>...>;
        using size_type = std::size_t;
        using label_type = std::string;

        /**
         * Construct from any axis scalar type that is part of the variant.
         */
        template <class T, std::enable_if_t<(std::is_same_v<T, Ts> || ...), int> = 0>
        explicit xaxis_variant(T&& scalar)
            : m_value(std::forward<T>(scalar))
        {
        }

        xaxis_variant(const self_type&) = default;
        xaxis_variant& operator=(const self_type&) = default;
        xaxis_variant(self_type&&) = default;
        xaxis_variant& operator=(self_type&&) = default;

        std::size_t dimension_count() const noexcept
        {
            return std::visit([](const auto& s) { return s.dimension_count(); }, m_value);
        }

        std::size_t size() const noexcept
        {
            return std::visit([](const auto& s) { return s.size(); }, m_value);
        }

        /**
         * Returns the dimension descriptor (name, coordinate, etc.).
         */
        auto dimension(std::size_t i) const
        {
            return std::visit([i](const auto& s) -> decltype(s.dimension(i)) { return s.dimension(i); }, m_value);
        }

        /**
         * Element access: delegates to the underlying axis scalar.
         */
        template <class... Args>
        auto operator()(Args... args) const
        {
            return std::visit([&](const auto& s) -> decltype(s(args...)) { return s(args...); }, m_value);
        }

        template <class... Args>
        auto operator()(Args... args)
        {
            return std::visit([&](auto& s) -> decltype(s(args...)) { return s(args...); }, m_value);
        }

        auto operator[](size_type i) const
        {
            return std::visit([i](const auto& s) -> decltype(s[i]) { return s[i]; }, m_value);
        }

        auto operator[](size_type i)
        {
            return std::visit([i](auto& s) -> decltype(s[i]) { return s[i]; }, m_value);
        }

        template <class... Labels>
        auto locate(Labels... labels) const
        {
            return std::visit([&](const auto& s) -> decltype(s.locate(labels...)) { return s.locate(labels...); }, m_value);
        }

        template <class... Labels>
        auto locate(Labels... labels)
        {
            return std::visit([&](auto& s) -> decltype(s.locate(labels...)) { return s.locate(labels...); }, m_value);
        }

        auto data() noexcept
        {
            return std::visit([](auto& s) { return s.data(); }, m_value);
        }

        auto data() const noexcept
        {
            return std::visit([](const auto& s) { return s.data(); }, m_value);
        }

        /**
         * Get the underlying axis dimension regardless of type.
         */
        auto axis_dimension() const
        {
            return std::visit([](const auto& s) -> decltype(s.axis_dimension()) { return s.axis_dimension(); }, m_value);
        }

        /**
         * Arithmetic operators: apply element‑wise to the underlying scalar.
         */
        template <class T, std::enable_if_t<(std::is_same_v<T, Ts> || ...), int> = 0>
        self_type& operator+=(const xaxis_scalar<T>& rhs)
        {
            std::visit([&](auto& s) {
                if constexpr (std::is_same_v<std::decay_t<decltype(s)>, xaxis_scalar<T>>)
                    s += rhs;
                else
                    throw std::runtime_error("xaxis_variant: incompatible types for operator+=");
            }, m_value);
            return *this;
        }

        template <class T>
        self_type& operator*=(T scalar)
        {
            std::visit([scalar](auto& s) { s *= scalar; }, m_value);
            return *this;
        }

        template <class T>
        self_type& operator/=(T scalar)
        {
            std::visit([scalar](auto& s) { s /= scalar; }, m_value);
            return *this;
        }

        /**
         * SIMD load.
         */
        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            return std::visit([i](const auto& s) { return s.template load_simd<Align, T>(i); }, m_value);
        }

    private:
        variant_type m_value;
    };

    /**
     * Helper to create an axis variant from any axis scalar.
     */
    template <class T, class L = label_type>
    inline auto make_axis_variant(xaxis_scalar<T, L>&& scalar)
    {
        return xaxis_variant<T>(std::move(scalar));
    }

} // namespace xframe

#endif // XFRAME_XAXIS_VARIANT_HPP