//File 0358 : xframe/xvariable_meta.hpp
//Compile‑time metadata for variables: type traits, value type detection, size deduction, layout queries, and SFINAE helpers for xframe variable types.
#ifndef XFRAME_XVARIABLE_META_HPP
#define XFRAME_XVARIABLE_META_HPP

#include <cstddef>
#include <type_traits>
#include <string>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xvariable.hpp"

namespace xframe {
namespace meta {

    // ========== is_variable trait ==========
    template <class T, class = void>
    struct is_variable : std::false_type {};

    template <class T, class L>
    struct is_variable<variable<T, L>> : std::true_type {};

    template <class T>
    inline constexpr bool is_variable_v = is_variable<T>::value;

    // ========== value_type_of ==========
    template <class T, class = void>
    struct value_type_of { using type = T; };

    template <class T>
    struct value_type_of<T, std::void_t<typename T::value_type>> {
        using type = typename T::value_type;
    };

    template <class T>
    using value_type_of_t = typename value_type_of<T>::type;

    // ========== label_type_of ==========
    template <class T, class = void>
    struct label_type_of { using type = std::string; };

    template <class T>
    struct label_type_of<T, std::void_t<typename T::label_type>> {
        using type = typename T::label_type;
    };

    template <class T>
    using label_type_of_t = typename label_type_of<T>::type;

    // ========== size_type_of ==========
    template <class T, class = void>
    struct size_type_of { using type = std::size_t; };

    template <class T>
    struct size_type_of<T, std::void_t<typename T::size_type>> {
        using type = typename T::size_type;
    };

    template <class T>
    using size_type_of_t = typename size_type_of<T>::type;

    // ========== has_data ==========
    template <class T, class = void>
    struct has_data : std::false_type {};

    template <class T>
    struct has_data<T, std::void_t<decltype(std::declval<T&>().data())>> : std::true_type {};

    template <class T>
    inline constexpr bool has_data_v = has_data<T>::value;

    // ========== has_size ==========
    template <class T, class = void>
    struct has_size : std::false_type {};

    template <class T>
    struct has_size<T, std::void_t<decltype(std::declval<const T&>().size())>> : std::true_type {};

    template <class T>
    inline constexpr bool has_size_v = has_size<T>::value;

    // ========== is_numeric_variable ==========
    template <class T>
    struct is_numeric_variable : std::false_type {};

    template <class T, class L>
    struct is_numeric_variable<variable<T, L>>
        : std::bool_constant<std::is_arithmetic_v<T>> {};

    template <class T>
    inline constexpr bool is_numeric_variable_v = is_numeric_variable<T>::value;

    // ========== common_numeric_type ==========
    template <class... Vars>
    using common_numeric_type_t = std::common_type_t<value_type_of_t<Vars>...>;

    // ========== rank_of ==========
    template <class T, class = void>
    struct rank_of { static constexpr std::size_t value = 0; };

    template <class T>
    struct rank_of<T, std::void_t<decltype(std::declval<const T&>().dimension_count())>> {
        static constexpr std::size_t value = std::decay_t<T>::dimension_count();
    };

    template <class T>
    inline constexpr std::size_t rank_of_v = rank_of<T>::value;

    // ========== is_expression ==========
    template <class T, class = void>
    struct is_expression : std::false_type {};

    template <class T>
    struct is_expression<T, std::void_t<decltype(std::declval<const T&>().derived())>>
        : std::is_base_of<expression_tag, T> {};

    template <class T>
    inline constexpr bool is_expression_v = is_expression<T>::value;

    // ========== disable_if_variable (for SFINAE) ==========
    template <class T, class R = void>
    using disable_if_variable_t = std::enable_if_t<!is_variable_v<T>, R>;

    template <class T, class R = void>
    using enable_if_variable_t = std::enable_if_t<is_variable_v<T>, R>;

} // namespace meta
} // namespace xframe

#endif // XFRAME_XVARIABLE_META_HPP