//File 0103 : numdot/types.h
//Type traits, value type deduction, scalar detection, expression properties, and compile‑time utilities for NumDot types.
#ifndef NUMDOT_TYPES_H
#define NUMDOT_TYPES_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include "config.h"
#include "forward.h"

namespace numdot
{
    // Primary type traits for any expression
    template <class T>
    struct type_traits
    {
        using value_type = typename T::value_type;
        using reference = typename T::reference;
        using const_reference = typename T::const_reference;
        using pointer = typename T::pointer;
        using const_pointer = typename T::const_pointer;
        using size_type = typename T::size_type;
        using difference_type = typename T::difference_type;
        using shape_type = typename T::shape_type;
        using strides_type = typename T::strides_type;
        using layout = typename T::layout;
    };

    // Specialization for fundamental types (scalars)
    template <class T>
    struct scalar_traits
    {
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = std::array<size_type, 0>;
        using strides_type = std::array<size_type, 0>;
        static constexpr layout layout_value = default_layout;
    };

    // Detect if a type is an expression
    template <class T, class = void>
    struct is_expression : std::false_type {};

    template <class T>
    struct is_expression<T, std::void_t<decltype(std::declval<const T&>().derived())>>
        : std::is_base_of<expression_tag, T> {};

    template <class T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

    // Detect if a type is a scalar (not an expression)
    template <class T>
    struct is_scalar : std::negation<is_expression<T>> {};

    template <class T>
    inline constexpr bool is_scalar_v = is_scalar<T>::value;

    // Value type deduction
    template <class T, class = void>
    struct value_type_of
    {
        using type = T;
    };

    template <class T>
    struct value_type_of<T, std::enable_if_t<is_expression_v<T>>>
    {
        using type = typename T::value_type;
    };

    template <class T>
    using value_type_of_t = typename value_type_of<T>::type;

    // Common value type of multiple expressions/scalars
    template <class... Ts>
    using common_value_type_t = std::common_type_t<value_type_of_t<Ts>...>;

    // Shape type deduction
    template <class T, class = void>
    struct shape_type_of
    {
        using type = typename T::shape_type;
    };

    template <class T>
    struct shape_type_of<T, std::enable_if_t<is_scalar_v<T>>>
    {
        using type = std::array<std::size_t, 0>;
    };

    template <class T>
    using shape_type_of_t = typename shape_type_of<T>::type;

    // Strides type deduction
    template <class T, class = void>
    struct strides_type_of
    {
        using type = typename T::strides_type;
    };

    template <class T>
    struct strides_type_of<T, std::enable_if_t<is_scalar_v<T>>>
    {
        using type = std::array<std::size_t, 0>;
    };

    template <class T>
    using strides_type_of_t = typename strides_type_of<T>::type;

    // Size type deduction
    template <class T, class = void>
    struct size_type_of
    {
        using type = typename T::size_type;
    };

    template <class T>
    struct size_type_of<T, std::enable_if_t<is_scalar_v<T>>>
    {
        using type = std::size_t;
    };

    template <class T>
    using size_type_of_t = typename size_type_of<T>::type;

    // Layout deduction
    template <class T, class = void>
    struct layout_of
    {
        static constexpr layout value = T::layout;
    };

    template <class T>
    struct layout_of<T, std::enable_if_t<is_scalar_v<T>>>
    {
        static constexpr layout value = default_layout;
    };

    template <class T>
    inline constexpr layout layout_of_v = layout_of<T>::value;

    // Check if type has contiguous data interface
    template <class T, class = void>
    struct has_data_interface : std::false_type {};

    template <class T>
    struct has_data_interface<T, std::void_t<decltype(std::declval<const T&>().data())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_data_interface_v = has_data_interface<T>::value;

    // Check if type has shape interface
    template <class T, class = void>
    struct has_shape_interface : std::false_type {};

    template <class T>
    struct has_shape_interface<T, std::void_t<decltype(std::declval<const T&>().shape())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_shape_interface_v = has_shape_interface<T>::value;

    // Check if type has strides interface
    template <class T, class = void>
    struct has_strides_interface : std::false_type {};

    template <class T>
    struct has_strides_interface<T, std::void_t<decltype(std::declval<const T&>().strides())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_strides_interface_v = has_strides_interface<T>::value;

    // Check if type supports SIMD loading
    template <class T, class = void>
    struct has_simd_interface : std::false_type {};

    template <class T>
    struct has_simd_interface<T, std::void_t<decltype(std::declval<const T&>().template load_simd<aligned_mode::unaligned>(std::size_t(0)))>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_simd_interface_v = has_simd_interface<T>::value;

    // Rank (number of dimensions) deduction
    template <class T, class = void>
    struct rank_of
    {
        static constexpr std::size_t value = 0;
    };

    template <class T>
    struct rank_of<T, std::void_t<decltype(std::declval<const T&>().dimension())>>
    {
        static constexpr std::size_t value = std::decay_t<T>::dimension();
    };

    template <class T>
    struct rank_of<T, std::void_t<decltype(std::declval<const T&>().shape())>>
    {
        static constexpr std::size_t value = std::tuple_size_v<shape_type_of_t<T>>;
    };

    template <class T>
    inline constexpr std::size_t rank_of_v = rank_of<T>::value;

    // Check if an expression is a temporary (rvalue) or an lvalue reference
    template <class T>
    struct is_temporary : std::false_type {};

    template <class T>
    struct is_temporary<T&&> : std::true_type {};

    template <class T>
    inline constexpr bool is_temporary_v = is_temporary<T>::value;

    // Type for temporary array
    template <class T>
    using temporary_type = array<value_type_of_t<T>>;

    // Enable if expression
    template <class T, class R = void>
    using enable_if_expression_t = std::enable_if_t<is_expression_v<T>, R>;

    // Disable if expression (for scalar overloads)
    template <class T, class R = void>
    using disable_if_expression_t = std::enable_if_t<!is_expression_v<T>, R>;

    // Numeric constants for a type
    template <class T>
    struct constants
    {
        static constexpr T pi = T(3.14159265358979323846);
        static constexpr T e  = T(2.71828182845904523536);
        static constexpr T sqrt2 = T(1.41421356237309504880);
        static constexpr T log2e = T(1.44269504088896340736);
        static constexpr T log10e = T(0.43429448190325182765);
        static constexpr T ln2 = T(0.69314718055994530942);
    };

    // Allow arithmetic types to be used as expressions? No, just traits.

} // namespace numdot

#endif // NUMDOT_TYPES_H