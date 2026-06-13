//File 0301 : xframe/xframe_forward.hpp
//Forward declarations for all xframe components: variable, coordinate, dimension, xframe array, views, and expression tags.
#ifndef XFRAME_FORWARD_HPP
#define XFRAME_FORWARD_HPP

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#include "xframe_config.hpp"

namespace xframe
{
    // Expression base tag
    struct expression_tag {};

    // Forward declarations
    template <class D>
    class expression;

    template <class T, class L>
    class variable;

    template <class T>
    class coordinate;

    template <class L>
    class dimension;

    template <class... V>
    class xframe;

    template <class CT, class... S>
    class xframe_view;

    template <class CT>
    class xframe_offset_view;

    template <class F, class... CT>
    class xframe_function;

    template <class T>
    class xframe_scalar;

    // Key types
    using label_type = std::string;
    template <class T>
    using label_list = std::vector<T>;

    // Traits
    template <class T>
    struct is_variable : std::false_type {};
    template <class T, class L>
    struct is_variable<variable<T, L>> : std::true_type {};
    template <class T>
    inline constexpr bool is_variable_v = is_variable<T>::value;

    template <class T>
    struct is_coordinate : std::false_type {};
    template <class T>
    struct is_coordinate<coordinate<T>> : std::true_type {};
    template <class T>
    inline constexpr bool is_coordinate_v = is_coordinate<T>::value;

    template <class T>
    struct is_dimension : std::false_type {};
    template <class L>
    struct is_dimension<dimension<L>> : std::true_type {};
    template <class T>
    inline constexpr bool is_dimension_v = is_dimension<T>::value;

    // Expression detection
    template <class T, class = void>
    struct is_expression : std::false_type {};
    template <class T>
    struct is_expression<T, std::void_t<decltype(std::declval<const T&>().derived())>>
        : std::is_base_of<expression_tag, T> {};
    template <class T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

    // Operators
    template <class E1, class E2>
    auto operator+(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2>
    auto operator-(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2>
    auto operator*(const expression<E1>&, const expression<E2>&);
    template <class E1, class E2>
    auto operator/(const expression<E1>&, const expression<E2>&);
}

#endif