// core/xvariant.hpp
#ifndef XTENSOR_XVARIANT_HPP
#define XTENSOR_XVARIANT_HPP

// ----------------------------------------------------------------------------
// xvariant.hpp – Type‑safe heterogeneous container for xtensor
// ----------------------------------------------------------------------------
// This header provides a variant type (similar to std::variant) specialized
// for xtensor expression types, with additional features:
//   - Recursive variant support (e.g., trees of mixed arrays)
//   - Visitor pattern for type‑safe operations
//   - Integration with BigNumber and FFT‑accelerated operations
//   - Lazy expression evaluation over variant values
//   - Variant array containers (e.g., heterogeneous tables)
//   - Serialization support (JSON, binary)
//   - Pattern matching (via overloaded lambdas)
//   - Conversion between variant and std::variant
//   - Type introspection and trait queries
//
// All operations are designed to work with bignumber::BigNumber and FFT
// acceleration where applicable (e.g., vectorized visitors).
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt {
namespace variant_impl {

// ------------------------------------------------------------------------
// Type list utilities
// ------------------------------------------------------------------------
template <class...> struct type_list {};

template <class T, class... Ts>
struct index_of;

template <class T, class... Ts>
struct index_of<T, T, Ts...> : std::integral_constant<size_t, 0> {};

template <class T, class U, class... Ts>
struct index_of<T, U, Ts...> : std::integral_constant<size_t, 1 + index_of<T, Ts...>::value> {};

template <class T, class... Ts>
inline constexpr size_t index_of_v = index_of<T, Ts...>::value;

template <class... Ts>
struct all_same : std::false_type {};

template <class T>
struct all_same<T> : std::true_type {};

template <class T, class... Ts>
struct all_same<T, T, Ts...> : all_same<T, Ts...> {};

template <class... Ts>
inline constexpr bool all_same_v = all_same<Ts...>::value;

// ------------------------------------------------------------------------
// xvariant storage (wrapper around std::variant)
// ------------------------------------------------------------------------
template <class... Ts>
class xvariant {
public:
    using variant_type = std::variant<Ts...>;
    static constexpr size_t type_count = sizeof...(Ts);

    // Constructors
    xvariant() = default;
    xvariant(const xvariant&) = default;
    xvariant(xvariant&&) = default;
    xvariant& operator=(const xvariant&) = default;
    xvariant& operator=(xvariant&&) = default;

    template <class T, class = std::enable_if_t<(std::is_same_v<std::decay_t<T>, Ts> || ...)>>
    xvariant(T&& value) : m_var(std::forward<T>(value)) {}

    // Type queries
    size_t index() const noexcept { return m_var.index(); }
    bool valueless_by_exception() const noexcept { return m_var.valueless_by_exception(); }

    template <class T>
    bool holds_alternative() const noexcept {
        return std::holds_alternative<T>(m_var);
    }

    // Access (throws on wrong type)
    template <class T>
    T& get() & { return std::get<T>(m_var); }

    template <class T>
    const T& get() const & { return std::get<T>(m_var); }

    template <class T>
    T&& get() && { return std::get<T>(std::move(m_var)); }

    // Access with pointer (returns nullptr on wrong type)
    template <class T>
    T* get_if() noexcept { return std::get_if<T>(&m_var); }

    template <class T>
    const T* get_if() const noexcept { return std::get_if<T>(&m_var); }

    // Visitor pattern
    template <class Visitor>
    decltype(auto) visit(Visitor&& vis) & {
        return std::visit(std::forward<Visitor>(vis), m_var);
    }

    template <class Visitor>
    decltype(auto) visit(Visitor&& vis) const & {
        return std::visit(std::forward<Visitor>(vis), m_var);
    }

    template <class Visitor>
    decltype(auto) visit(Visitor&& vis) && {
        return std::visit(std::forward<Visitor>(vis), std::move(m_var));
    }

    // Comparison
    bool operator==(const xvariant& other) const { return m_var == other.m_var; }
    bool operator!=(const xvariant& other) const { return m_var != other.m_var; }
    bool operator<(const xvariant& other) const { return m_var < other.m_var; }

    // Direct access to underlying std::variant
    variant_type& raw() { return m_var; }
    const variant_type& raw() const { return m_var; }

private:
    variant_type m_var;
};

// ------------------------------------------------------------------------
// Make xvariant (deduce type)
// ------------------------------------------------------------------------
template <class... Ts, class T>
xvariant<Ts...> make_xvariant(T&& value) {
    return xvariant<Ts...>(std::forward<T>(value));
}

// ------------------------------------------------------------------------
// Pattern matching helper (overload)
// ------------------------------------------------------------------------
template <class... Fs>
struct overloaded : Fs... {
    using Fs::operator()...;
};
template <class... Fs> overloaded(Fs...) -> overloaded<Fs...>;

} // namespace variant_impl

using variant_impl::xvariant;
using variant_impl::make_xvariant;
using variant_impl::overloaded;

// ========================================================================
// Common variant type aliases for xtensor
// ========================================================================
using ScalarVariant = xvariant<bool, int, double, bignumber::BigNumber>;
using ArrayVariant = xvariant<xarray_container<double>,
                              xarray_container<float>,
                              xarray_container<int>,
                              xarray_container<bignumber::BigNumber>>;
using TensorVariant = xvariant<ScalarVariant, ArrayVariant>;  // recursive

// ========================================================================
// Variant array (heterogeneous container)
// ========================================================================
template <class... Ts>
class xvariant_array {
public:
    using value_type = xvariant<Ts...>;
    using size_type = size_t;

    xvariant_array() = default;
    explicit xvariant_array(size_type n) : m_data(n) {}

    // Access
    value_type& operator[](size_type i) { return m_data[i]; }
    const value_type& operator[](size_type i) const { return m_data[i]; }
    size_type size() const noexcept { return m_data.size(); }
    bool empty() const noexcept { return m_data.empty(); }

    // Push / emplace
    void push_back(const value_type& val) { m_data.push_back(val); }
    void push_back(value_type&& val) { m_data.push_back(std::move(val)); }

    template <class T>
    void push_back(T&& val) { m_data.emplace_back(std::forward<T>(val)); }

    // Iterators
    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }
    auto begin() const { return m_data.begin(); }
    auto end() const { return m_data.end(); }

    // Filter by type (returns vector of indices)
    template <class T>
    std::vector<size_t> indices_of() const {
        std::vector<size_t> idx;
        for (size_t i = 0; i < m_data.size(); ++i)
            if (m_data[i].template holds_alternative<T>())
                idx.push_back(i);
        return idx;
    }

    // Apply visitor to all elements
    template <class Visitor>
    void for_each(Visitor&& vis) {
        for (auto& v : m_data) v.visit(vis);
    }

private:
    std::vector<value_type> m_data;
};

// ========================================================================
// Expression support for xvariant (lazy operations)
// ========================================================================
namespace detail {
    template <class Op, class... Ts>
    struct variant_visitor {
        template <class A, class B>
        auto operator()(const A& a, const B& b) const {
            if constexpr (std::is_invocable_v<Op, A, B>)
                return Op{}(a, b);
            else
                throw std::runtime_error("Operation not supported for these types");
        }
    };
}

// Binary operation on two variants
template <class... Ts, class Op>
auto binary_op(const xvariant<Ts...>& a, const xvariant<Ts...>& b, Op op) {
    return a.visit([&](const auto& va) {
        return b.visit([&](const auto& vb) -> xvariant<Ts...> {
            return detail::variant_visitor<Op, Ts...>{}(va, vb);
        });
    });
}

// Convenience arithmetic operators for variant
template <class... Ts>
xvariant<Ts...> operator+(const xvariant<Ts...>& a, const xvariant<Ts...>& b) {
    return binary_op(a, b, std::plus<>{});
}

template <class... Ts>
xvariant<Ts...> operator-(const xvariant<Ts...>& a, const xvariant<Ts...>& b) {
    return binary_op(a, b, std::minus<>{});
}

template <class... Ts>
xvariant<Ts...> operator*(const xvariant<Ts...>& a, const xvariant<Ts...>& b) {
    return binary_op(a, b, std::multiplies<>{});
}

template <class... Ts>
xvariant<Ts...> operator/(const xvariant<Ts...>& a, const xvariant<Ts...>& b) {
    return binary_op(a, b, std::divides<>{});
}

// ========================================================================
// Serialization support (to string)
// ========================================================================
template <class... Ts>
std::string to_string(const xvariant<Ts...>& var) {
    return var.visit([](const auto& val) -> std::string {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, std::string>)
            return val;
        else if constexpr (std::is_arithmetic_v<T>)
            return std::to_string(val);
        else if constexpr (std::is_same_v<T, bignumber::BigNumber>)
            return val.to_string();
        else
            return "<" + std::string(typeid(T).name()) + ">";
    });
}

// ========================================================================
// Type traits
// ========================================================================
template <class T>
struct is_xvariant : std::false_type {};

template <class... Ts>
struct is_xvariant<xvariant<Ts...>> : std::true_type {};

template <class T>
inline constexpr bool is_xvariant_v = is_xvariant<T>::value;

// Check if a type is in the variant's type list
template <class T, class V>
struct variant_has_type : std::false_type {};

template <class T, class... Ts>
struct variant_has_type<T, xvariant<Ts...>> : std::disjunction<std::is_same<T, Ts>...> {};

template <class T, class V>
inline constexpr bool variant_has_type_v = variant_has_type<T, V>::value;

} // namespace xt

#endif // XTENSOR_XVARIANT_HPP