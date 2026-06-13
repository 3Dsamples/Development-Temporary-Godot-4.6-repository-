//File 0106 : numdot/elementwise.h
//Expression template engine for element-wise operations, lazy evaluation, broadcasting, SIMD acceleration, and operator overloads.
#ifndef NUMDOT_ELEMENTWISE_H
#define NUMDOT_ELEMENTWISE_H

#include <type_traits>
#include <utility>
#include <tuple>
#include <cstddef>
#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"

namespace numdot
{
    // Base expression class using CRTP
    template <class D>
    class expression
    {
    public:
        using derived_type = D;
        derived_type& derived() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived() const noexcept { return *static_cast<const derived_type*>(this); }

        // Default implementations that derived classes should override
        auto size() const noexcept { return derived().size(); }
        auto shape() const noexcept { return derived().shape(); }
        auto strides() const noexcept { return derived().strides(); }
        auto data() const noexcept { return derived().data(); }
        auto data() noexcept { return derived().data(); }

        template <class... Args>
        decltype(auto) operator()(Args... args) { return derived()(args...); }
        template <class... Args>
        decltype(auto) operator()(Args... args) const { return derived()(args...); }

        decltype(auto) operator[](std::size_t i) { return derived()[i]; }
        decltype(auto) operator[](std::size_t i) const { return derived()[i]; }
    };

    // Function expression node: applies a functor to arguments
    template <class F, class... CT>
    class function : public expression<function<F, CT...>>
    {
    public:
        using self_type = function<F, CT...>;
        using value_type = decltype(std::declval<F>()(std::declval<typename CT::value_type>()...));
        using const_reference = const value_type&;
        using size_type = std::size_t;
        using shape_type = std::vector<size_type>;
        using strides_type = shape_type;
        static constexpr layout layout = default_layout;

        static constexpr std::size_t arity = sizeof...(CT);

        template <class Func, class... Args>
        function(Func&& f, Args&&... args) noexcept
            : m_f(std::forward<Func>(f)), m_args(std::forward<Args>(args)...)
        {
            m_shape = broadcast_args_shapes();
            m_strides = compute_strides(m_shape);
        }

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        value_type* data() noexcept { return nullptr; }
        const value_type* data() const noexcept { return nullptr; }

        template <class... Args>
        value_type operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<arity>{}, args...);
        }

        template <class... Args>
        value_type operator()(Args... args)
        {
            return access_impl(std::make_index_sequence<arity>{}, args...);
        }

        value_type operator[](std::size_t i) const
        {
            auto idx = unravel_index(i, m_shape);
            return element(idx.begin(), idx.end());
        }

        value_type operator[](std::size_t i)
        {
            return static_cast<const self_type*>(this)->operator[](i);
        }

        template <class It>
        value_type element(It first, It last) const
        {
            return access_by_index(first, last, std::make_index_sequence<arity>{});
        }

        // SIMD load
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            alignas(64) std::array<T, simd_size> buf;
            for (std::size_t k = 0; k < simd_size; ++k)
                buf[k] = operator()(i + k);
            return simd_type::load_aligned(buf.data());
        }

    private:
        F m_f;
        std::tuple<CT...> m_args;
        shape_type m_shape;
        strides_type m_strides;

        shape_type broadcast_args_shapes() const
        {
            return broadcast_shapes_impl(std::make_index_sequence<arity>{});
        }

        template <std::size_t... I>
        shape_type broadcast_args_shapes_impl(std::index_sequence<I...>) const
        {
            return broadcast_shapes(std::get<I>(m_args).shape()...);
        }

        template <std::size_t... I, class... Args>
        value_type access_impl(std::index_sequence<I...>, Args... args) const
        {
            return m_f(std::get<I>(m_args)(args...)...);
        }

        template <class It, std::size_t... I>
        value_type access_by_index(It first, It last, std::index_sequence<I...>) const
        {
            auto idx = std::vector<std::size_t>(first, last);
            return access_impl(std::index_sequence<I...>{}, idx[I]...);
        }
    };

    // Scalar constant expression
    template <class T>
    class scalar : public expression<scalar<T>>
    {
    public:
        using value_type = T;
        using const_reference = const T&;
        using size_type = std::size_t;
        using shape_type = std::array<size_type, 0>;
        using strides_type = shape_type;
        static constexpr layout layout = default_layout;

        scalar() noexcept : m_value(T{}) {}
        explicit scalar(T val) noexcept : m_value(val) {}

        size_type size() const noexcept { return 1; }
        shape_type shape() const noexcept { return {}; }
        strides_type strides() const noexcept { return {}; }
        const value_type* data() const noexcept { return &m_value; }

        template <class... Args>
        const_reference operator()(Args...) const noexcept { return m_value; }
        const_reference operator[](std::size_t) const noexcept { return m_value; }

    private:
        T m_value;
    };

    // Helper to wrap a scalar into an expression
    template <class T>
    inline auto make_scalar(T&& val)
    {
        return scalar<std::decay_t<T>>(std::forward<T>(val));
    }

    // Make a function expression
    template <class F, class... E>
    inline auto make_function(F&& f, E&&... e)
    {
        return function<std::decay_t<F>, std::decay_t<E>...>(std::forward<F>(f), std::forward<E>(e)...);
    }

    // Operator overloads for expressions
    template <class E1, class E2>
    inline auto operator+(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_function([](auto a, auto b) { return a + b; }, e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator-(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_function([](auto a, auto b) { return a - b; }, e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator*(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_function([](auto a, auto b) { return a * b; }, e1.derived(), e2.derived());
    }

    template <class E1, class E2>
    inline auto operator/(const expression<E1>& e1, const expression<E2>& e2)
    {
        return make_function([](auto a, auto b) { return a / b; }, e1.derived(), e2.derived());
    }

    template <class E>
    inline auto operator-(const expression<E>& e)
    {
        return make_function([](auto a) { return -a; }, e.derived());
    }

    // Scalar-expression mixed operators
    template <class E, class T, disable_if_expression_t<T, int> = 0>
    inline auto operator+(const expression<E>& e, T scalar)
    {
        return make_function([s=scalar](auto a) { return a + s; }, e.derived());
    }

    template <class T, class E, disable_if_expression_t<T, int> = 0>
    inline auto operator+(T scalar, const expression<E>& e)
    {
        return e + scalar;
    }

    template <class E, class T, disable_if_expression_t<T, int> = 0>
    inline auto operator-(const expression<E>& e, T scalar)
    {
        return make_function([s=scalar](auto a) { return a - s; }, e.derived());
    }

    template <class T, class E, disable_if_expression_t<T, int> = 0>
    inline auto operator-(T scalar, const expression<E>& e)
    {
        return make_function([s=scalar](auto a) { return s - a; }, e.derived());
    }

    template <class E, class T, disable_if_expression_t<T, int> = 0>
    inline auto operator*(const expression<E>& e, T scalar)
    {
        return make_function([s=scalar](auto a) { return a * s; }, e.derived());
    }

    template <class T, class E, disable_if_expression_t<T, int> = 0>
    inline auto operator*(T scalar, const expression<E>& e)
    {
        return e * scalar;
    }

    template <class E, class T, disable_if_expression_t<T, int> = 0>
    inline auto operator/(const expression<E>& e, T scalar)
    {
        return make_function([s=scalar](auto a) { return a / s; }, e.derived());
    }

    // Compound assignment operators (return new array)
    template <class E1, class E2>
    inline auto& operator+=(array<typename E1::value_type>& lhs, const expression<E2>& rhs)
    {
        lhs = lhs + rhs;
        return lhs;
    }

    template <class E1, class E2>
    inline auto& operator-=(array<typename E1::value_type>& lhs, const expression<E2>& rhs)
    {
        lhs = lhs - rhs;
        return lhs;
    }

    template <class E1, class E2>
    inline auto& operator*=(array<typename E1::value_type>& lhs, const expression<E2>& rhs)
    {
        lhs = lhs * rhs;
        return lhs;
    }

    template <class E1, class E2>
    inline auto& operator/=(array<typename E1::value_type>& lhs, const expression<E2>& rhs)
    {
        lhs = lhs / rhs;
        return lhs;
    }

} // namespace numdot

#endif // NUMDOT_ELEMENTWISE_H