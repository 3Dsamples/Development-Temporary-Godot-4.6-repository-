//File 0362 : xframe/xvector_variant.hpp
//Vector variant: type‑safe union of multiple vector‑like containers (std::vector, variable, etc.) with SIMD‑accelerated dispatch, broadcasting, and expression integration.
#ifndef XFRAME_XVECTOR_VARIANT_HPP
#define XFRAME_XVECTOR_VARIANT_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
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
    namespace detail
    {
        // SFINAE check if type appears in variant
        template <class T, class Variant>
        struct is_in_variant : std::false_type {};

        template <class T, class... Ts>
        struct is_in_variant<T, std::variant<Ts...>>
            : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

        template <class T, class Variant>
        inline constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;

        // Visitor that applies a unary operation to the contained vector/variable
        template <class F>
        struct vector_variant_unary_visitor
        {
            F m_func;

            template <class T>
            auto operator()(const T& container) const
            {
                return m_func(container);
            }
        };

        // Visitor that applies a binary operation to two containers of same type
        template <class F>
        struct vector_variant_binary_visitor
        {
            F m_func;

            template <class T, class U>
            auto operator()(const T& a, const U& b) const
                -> std::enable_if_t<std::is_same_v<T, U>, T>
            {
                return m_func(a, b);
            }

            template <class T, class U>
            auto operator()(const T&, const U&) const
                -> std::enable_if_t<!std::is_same_v<T, U>, T>
            {
                throw std::runtime_error("vector_variant_binary_visitor: type mismatch.");
            }
        };
    }

    /**
     * @class xvector_variant
     * @brief Type‑erased vector container that can hold any of the specified container types.
     *
     * This allows runtime polymorphism for 1D data structures: std::vector<T>,
     * variable<T, L>, etc. The interface provides size, element access, iteration,
     * SIMD loads, arithmetic operations, and conversion between the held types.
     * All operations are dispatched to the concrete container via std::visit.
     */
    template <class L = label_type, class... Containers>
    class xvector_variant : public expression<xvector_variant<L, Containers...>>
    {
    public:
        using self_type = xvector_variant<L, Containers...>;
        using variant_type = std::variant<Containers...>;
        using size_type = std::size_t;
        using value_type = double; // common numeric type for access
        using label_type = L;

        static_assert(sizeof...(Containers) > 0, "xvector_variant requires at least one container type.");

        /**
         * Default constructor: holds a default‑constructed instance of the first container type.
         */
        xvector_variant()
            : m_var(std::tuple_element_t<0, std::tuple<Containers...>>())
        {
        }

        /**
         * Construct from a concrete container.
         */
        template <class C, std::enable_if_t<detail::is_in_variant_v<std::decay_t<C>, variant_type>, int> = 0>
        explicit xvector_variant(C&& container)
            : m_var(std::forward<C>(container))
        {
        }

        xvector_variant(const self_type&) = default;
        xvector_variant& operator=(const self_type&) = default;
        xvector_variant(self_type&&) = default;
        xvector_variant& operator=(self_type&&) = default;

        /**
         * Size of the held container.
         */
        size_type size() const
        {
            return std::visit([](const auto& c) -> size_type { return c.size(); }, m_var);
        }

        bool empty() const
        {
            return std::visit([](const auto& c) { return c.empty(); }, m_var);
        }

        /**
         * Name: if the held type has a name() method, return it; else empty.
         */
        label_type name() const
        {
            return std::visit([](const auto& c) -> label_type {
                if constexpr (has_name_v<std::decay_t<decltype(c)>>)
                    return c.name();
                else
                    return label_type{};
            }, m_var);
        }

        /**
         * Element access by index (returns double).
         */
        double operator[](size_type i) const
        {
            return std::visit([i](const auto& c) -> double {
                return static_cast<double>(c[i]);
            }, m_var);
        }

        /**
         * Fill with a constant value.
         */
        void fill(double val)
        {
            std::visit([val](auto& c) {
                using T = typename std::decay_t<decltype(c)>::value_type;
                if constexpr (has_fill_v<std::decay_t<decltype(c)>>)
                    c.fill(static_cast<T>(val));
                else
                    std::fill(c.begin(), c.end(), static_cast<T>(val));
            }, m_var);
        }

        /**
         * Resize the container.
         */
        void resize(size_type n)
        {
            std::visit([n](auto& c) { c.resize(n); }, m_var);
        }

        /**
         * Reserve capacity.
         */
        void reserve(size_type n)
        {
            std::visit([n](auto& c) {
                if constexpr (has_reserve_v<std::decay_t<decltype(c)>>)
                    c.reserve(n);
            }, m_var);
        }

        /**
         * Get raw data pointer if available.
         */
        auto data() const
        {
            return std::visit([](const auto& c) -> const double* {
                if constexpr (has_data_v<std::decay_t<decltype(c)>>)
                    return reinterpret_cast<const double*>(c.data());
                else
                    return nullptr;
            }, m_var);
        }

        auto data()
        {
            return std::visit([](auto& c) -> double* {
                if constexpr (has_data_v<std::decay_t<decltype(c)>>)
                    return reinterpret_cast<double*>(c.data());
                else
                    return nullptr;
            }, m_var);
        }

        /**
         * SIMD load.
         */
        template <class Align, class T = double>
        auto load_simd(std::size_t i) const
        {
            return std::visit([i](const auto& c) {
                return c.template load_simd<Align, T>(i);
            }, m_var);
        }

        /**
         * Arithmetic operators with another variant (same variant types).
         */
        self_type& operator+=(const self_type& rhs)
        {
            std::visit([&](auto& lhs) {
                std::visit([&](const auto& rhs_c) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, std::decay_t<decltype(rhs_c)>>)
                    {
                        lhs += rhs_c;
                    }
                    else
                    {
                        throw std::runtime_error("xvector_variant::operator+=: type mismatch.");
                    }
                }, rhs.m_var);
            }, m_var);
            return *this;
        }

        self_type& operator-=(const self_type& rhs)
        {
            std::visit([&](auto& lhs) {
                std::visit([&](const auto& rhs_c) {
                    if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, std::decay_t<decltype(rhs_c)>>)
                        lhs -= rhs_c;
                    else
                        throw std::runtime_error("xvector_variant::operator-=: type mismatch.");
                }, rhs.m_var);
            }, m_var);
            return *this;
        }

        self_type& operator*=(double scalar)
        {
            std::visit([scalar](auto& c) {
                using T = typename std::decay_t<decltype(c)>::value_type;
                if constexpr (has_scale_assign_v<std::decay_t<decltype(c)>, T>)
                    c *= static_cast<T>(scalar);
                else
                    for (auto& v : c) v *= static_cast<T>(scalar);
            }, m_var);
            return *this;
        }

        self_type& operator/=(double scalar)
        {
            std::visit([scalar](auto& c) {
                using T = typename std::decay_t<decltype(c)>::value_type;
                if constexpr (has_scale_assign_v<std::decay_t<decltype(c)>, T>)
                    c /= static_cast<T>(scalar);
                else
                    for (auto& v : c) v /= static_cast<T>(scalar);
            }, m_var);
            return *this;
        }

        /**
         * Access the underlying variant.
         */
        variant_type& variant() noexcept { return m_var; }
        const variant_type& variant() const noexcept { return m_var; }

        /**
         * Check if the variant holds a specific container type.
         */
        template <class C>
        bool holds() const noexcept
        {
            return std::holds_alternative<C>(m_var);
        }

        /**
         * Get a pointer to the concrete container, or nullptr.
         */
        template <class C>
        C* get_if() noexcept
        {
            return std::get_if<C>(&m_var);
        }

        template <class C>
        const C* get_if() const noexcept
        {
            return std::get_if<C>(&m_var);
        }

        /**
         * Apply a unary operation, returning a new variant (the result type must be one of the variant types).
         */
        template <class F>
        auto apply(F&& f) const
        {
            return std::visit([&](const auto& c) -> variant_type {
                return f(c);
            }, m_var);
        }

    private:
        variant_type m_var;

        // Type trait helpers
        template <class C, class = void>
        struct has_name : std::false_type {};
        template <class C>
        struct has_name<C, std::void_t<decltype(std::declval<const C&>().name())>> : std::true_type {};
        template <class C>
        static inline constexpr bool has_name_v = has_name<C>::value;

        template <class C, class = void>
        struct has_data : std::false_type {};
        template <class C>
        struct has_data<C, std::void_t<decltype(std::declval<C&>().data())>> : std::true_type {};
        template <class C>
        static inline constexpr bool has_data_v = has_data<C>::value;

        template <class C, class = void>
        struct has_fill : std::false_type {};
        template <class C>
        struct has_fill<C, std::void_t<decltype(std::declval<C&>().fill(std::declval<typename C::value_type>()))>> : std::true_type {};
        template <class C>
        static inline constexpr bool has_fill_v = has_fill<C>::value;

        template <class C, class = void>
        struct has_reserve : std::false_type {};
        template <class C>
        struct has_reserve<C, std::void_t<decltype(std::declval<C&>().reserve(std::size_t{}))>> : std::true_type {};
        template <class C>
        static inline constexpr bool has_reserve_v = has_reserve<C>::value;

        template <class C, class T, class = void>
        struct has_scale_assign : std::false_type {};
        template <class C, class T>
        struct has_scale_assign<C, T, std::void_t<decltype(std::declval<C&>() *= std::declval<T>())>> : std::true_type {};
        template <class C, class T>
        static inline constexpr bool has_scale_assign_v = has_scale_assign<C, T>::value;
    };

    // Operator overloads
    template <class L, class... Cs>
    inline auto operator+(const xvector_variant<L, Cs...>& a,
                          const xvector_variant<L, Cs...>& b)
    {
        auto result = a;
        result += b;
        return result;
    }

    template <class L, class... Cs>
    inline auto operator-(const xvector_variant<L, Cs...>& a,
                          const xvector_variant<L, Cs...>& b)
    {
        auto result = a;
        result -= b;
        return result;
    }

    template <class L, class... Cs>
    inline auto operator*(const xvector_variant<L, Cs...>& a, double scalar)
    {
        auto result = a;
        result *= scalar;
        return result;
    }

    template <class L, class... Cs>
    inline auto operator*(double scalar, const xvector_variant<L, Cs...>& a)
    {
        return a * scalar;
    }

    template <class L, class... Cs>
    inline auto operator/(const xvector_variant<L, Cs...>& a, double scalar)
    {
        auto result = a;
        result /= scalar;
        return result;
    }

    /**
     * Helper to create a vector variant from a concrete container.
     */
    template <class L = label_type, class... Cs, class C>
    inline auto make_vector_variant(C&& container)
    {
        return xvector_variant<L, Cs...>(std::forward<C>(container));
    }

} // namespace xframe

#endif // XFRAME_XVECTOR_VARIANT_HPP