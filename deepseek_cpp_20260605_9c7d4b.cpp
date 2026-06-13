//File 0342 : xframe/xdynamic_variable.hpp
//Dynamic variable: a type‑erased variable container using std::variant for multiple numeric types, supporting SIMD‑accelerated element access and arithmetic.
#ifndef XFRAME_XDYNAMIC_VARIABLE_HPP
#define XFRAME_XDYNAMIC_VARIABLE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
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

namespace xframe
{
    /**
     * @class xdynamic_variable
     * @brief A type‑erased variable that can hold any supported numeric type.
     *
     * Internally uses std::variant to store an xvariable of one of the
     * allowed types. Provides a uniform interface for size, element access,
     * arithmetic, and SIMD loads. This enables runtime polymorphism for
     * variables while retaining the performance of the concrete type.
     */
    template <class... Ts>
    class xdynamic_variable : public expression<xdynamic_variable<Ts...>>
    {
    public:
        using self_type = xdynamic_variable<Ts...>;
        using size_type = std::size_t;
        using value_type = double; // fallback common type
        using label_type = std::string;

        // Type‑erased storage: variant of xvariable<T> for each T in Ts...
        using variant_type = std::variant<variable<Ts, label_type>...>;

        xdynamic_variable() noexcept = default;

        /**
         * Construct from a concrete variable (must be one of the allowed types).
         */
        template <class T, std::enable_if_t<(std::is_same_v<T, Ts> || ...), int> = 0>
        explicit xdynamic_variable(variable<T, label_type> var) noexcept
            : m_var(std::move(var))
        {
        }

        xdynamic_variable(const self_type&) = default;
        xdynamic_variable& operator=(const self_type&) = default;
        xdynamic_variable(self_type&&) = default;
        xdynamic_variable& operator=(self_type&&) = default;

        /**
         * Size of the underlying variable.
         */
        size_type size() const noexcept
        {
            return std::visit([](const auto& v) { return v.size(); }, m_var);
        }

        /**
         * Element access (read‑only).
         * Returns a common type (e.g., double) for uniformity.
         */
        double operator[](size_type i) const
        {
            return std::visit([i](const auto& v) -> double { return static_cast<double>(v[i]); }, m_var);
        }

        /**
         * Element access (mutable) – requires concrete type knowledge;
         * throws if the variant does not hold the requested type.
         */
        template <class T>
        T& get(size_type i)
        {
            if (auto* p = std::get_if<variable<T, label_type>>(&m_var))
                return (*p)[i];
            throw std::runtime_error("xdynamic_variable::get: wrong type.");
        }

        /**
         * Name of the variable.
         */
        const label_type& name() const
        {
            return std::visit([](const auto& v) -> const label_type& { return v.name(); }, m_var);
        }

        /**
         * Set name.
         */
        void set_name(const label_type& n)
        {
            std::visit([&](auto& v) { v.set_name(n); }, m_var);
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
         * SIMD load (converts to double batch).
         */
        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            return std::visit([i](const auto& v) {
                return v.template load_simd<Align, T>(i);
            }, m_var);
        }

        /**
         * Arithmetic in‑place with another dynamic variable (same variant types).
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
                        throw std::runtime_error("xdynamic_variable::operator+=: type mismatch.");
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
                        throw std::runtime_error("xdynamic_variable::operator-=: type mismatch.");
                }, rhs.m_var);
            }, m_var);
            return *this;
        }

        self_type& operator*=(double scalar)
        {
            std::visit([scalar](auto& v) { v *= scalar; }, m_var);
            return *this;
        }

        self_type& operator/=(double scalar)
        {
            std::visit([scalar](auto& v) { v /= scalar; }, m_var);
            return *this;
        }

        /**
         * Access the underlying variant (advanced).
         */
        variant_type& variant() noexcept { return m_var; }
        const variant_type& variant() const noexcept { return m_var; }

    private:
        variant_type m_var;
    };

    /**
     * Helper to create a dynamic variable from a concrete variable.
     */
    template <class... Ts, class T>
    inline auto make_dynamic_variable(variable<T, label_type> var)
    {
        return xdynamic_variable<Ts...>(std::move(var));
    }

} // namespace xframe

#endif // XFRAME_XDYNAMIC_VARIABLE_HPP