//File 0310 : xframe/xframe_scalar.hpp
//Scalar constant expression for xframe: wraps a single value into the expression system, enabling broadcast operations with labeled arrays.
#ifndef XFRAME_SCALAR_HPP
#define XFRAME_SCALAR_HPP

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"

namespace xframe
{
    /**
     * @class xframe_scalar
     * @brief Expression representing a scalar constant broadcast across dimensions.
     *
     * When used in arithmetic with an xframe, the scalar is treated as having
     * the same dimensions as the other operand, but with a single repeated value.
     * This enables writing `frame * 2.0` or `3.0 + frame`.
     */
    template <class T>
    class xframe_scalar : public expression<xframe_scalar<T>>
    {
    public:
        using self_type = xframe_scalar<T>;
        using value_type = T;
        using size_type = std::size_t;

        /**
         * Construct from a scalar value.
         */
        explicit xframe_scalar(T val) noexcept : m_value(val) {}

        xframe_scalar(const self_type&) = default;
        xframe_scalar& operator=(const self_type&) = default;
        xframe_scalar(self_type&&) = default;
        xframe_scalar& operator=(self_type&&) = default;

        /**
         * A scalar has zero dimensions.
         */
        static constexpr std::size_t dimension_count() noexcept { return 0; }

        /**
         * Scalar size is 1.
         */
        size_type size() const noexcept { return 1; }

        /**
         * Element access (ignores indices).
         */
        template <class... Args>
        value_type operator()(Args...) const noexcept { return m_value; }

        template <class... Args>
        value_type locate(Args...) const noexcept { return m_value; }

        value_type operator[](size_type) const noexcept { return m_value; }

        const value_type* data() const noexcept { return &m_value; }

    private:
        T m_value;
    };

    /**
     * Helper to create a scalar expression from a value.
     */
    template <class T>
    inline auto make_scalar(T&& val)
    {
        return xframe_scalar<std::decay_t<T>>(std::forward<T>(val));
    }

    /**
     * Overloaded operators for scalar–expression combinations.
     */
    template <class E, class T, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator+(const expression<E>& e, T scalar)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return a + s; },
            e.derived());
    }

    template <class T, class E, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator+(T scalar, const expression<E>& e)
    {
        return e + scalar;
    }

    template <class E, class T, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator-(const expression<E>& e, T scalar)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return a - s; },
            e.derived());
    }

    template <class T, class E, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator-(T scalar, const expression<E>& e)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return s - a; },
            e.derived());
    }

    template <class E, class T, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator*(const expression<E>& e, T scalar)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return a * s; },
            e.derived());
    }

    template <class T, class E, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator*(T scalar, const expression<E>& e)
    {
        return e * scalar;
    }

    template <class E, class T, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator/(const expression<E>& e, T scalar)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return a / s; },
            e.derived());
    }

    template <class T, class E, std::enable_if_t<!is_expression_v<T>, int> = 0>
    inline auto operator/(T scalar, const expression<E>& e)
    {
        return make_xframe_function(
            [s = scalar](auto a) { return s / a; },
            e.derived());
    }

} // namespace xframe

#endif // XFRAME_SCALAR_HPP