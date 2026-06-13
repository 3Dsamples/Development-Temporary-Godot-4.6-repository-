//File 0045 : core/xexpression_traits.hpp
//Expression traits for compile‑time type resolution, shape deduction, SIMD interface detection, and temporary type generation.
#ifndef XTENSOR_XEXPRESSION_TRAITS_HPP
#define XTENSOR_XEXPRESSION_TRAITS_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xexpression.hpp"

namespace xt
{
    /*************************************
     * is_xexpression
     *************************************/
    template <class T, class = void>
    struct is_xexpression : std::false_type {};

    template <class T>
    struct is_xexpression<T, std::void_t<decltype(std::declval<const T&>().derived_cast())>>
        : std::is_base_of<xtensor_expression_tag, T> {};

    template <class T>
    inline constexpr bool is_xexpression_v = is_xexpression<T>::value;

    /*************************************
     * xexpression_value_type
     *************************************/
    template <class E, class = void>
    struct xexpression_value_type
    {
        using type = void;
    };

    template <class E>
    struct xexpression_value_type<E, std::void_t<typename E::value_type>>
    {
        using type = typename E::value_type;
    };

    template <class E>
    using xexpression_value_type_t = typename xexpression_value_type<E>::type;

    /*************************************
     * xexpression_shape_type
     *************************************/
    template <class E, class = void>
    struct xexpression_shape_type
    {
        using type = typename E::shape_type;
    };

    template <class E>
    struct xexpression_shape_type<E, std::void_t<typename E::shape_type>>
    {
        using type = typename E::shape_type;
    };

    template <class E>
    using xexpression_shape_type_t = typename xexpression_shape_type<E>::type;

    /*************************************
     * xexpression_size_type
     *************************************/
    template <class E, class = void>
    struct xexpression_size_type
    {
        using type = typename E::size_type;
    };

    template <class E>
    using xexpression_size_type_t = typename xexpression_size_type<E>::type;

    /*************************************
     * xexpression_layout
     *************************************/
    template <class E, class = void>
    struct xexpression_layout
    {
        static constexpr layout_type value = E::layout;
    };

    template <class E>
    inline constexpr layout_type xexpression_layout_v = xexpression_layout<E>::value;

    /*************************************
     * has_simd_interface
     *************************************/
    template <class E, class = void>
    struct has_simd_interface : std::false_type {};

    template <class E>
    struct has_simd_interface<E, std::void_t<decltype(std::declval<const E&>().template load_simd<aligned_mode::unaligned>(std::size_t(0)))>>
        : std::true_type {};

    template <class E>
    inline constexpr bool has_simd_interface_v = has_simd_interface<E>::value;

    /*************************************
     * temporary_type_from_shape
     *************************************/
    template <class E, class S>
    using temporary_type_from_shape_t = xarray_container<
        uvector<xexpression_value_type_t<E>>,
        xexpression_layout_v<E>,
        S>;

    /*************************************
     * common_value_type
     *************************************/
    template <class... E>
    using common_value_type = std::common_type_t<xexpression_value_type_t<E>...>;

    /*************************************
     * is_convertible_expression
     *************************************/
    template <class From, class To>
    struct is_convertible_expression : std::false_type {};

    template <class From, class To>
    struct is_convertible_expression<From, To,
        std::enable_if_t<is_xexpression_v<From> && is_xexpression_v<To>>>
    {
        static constexpr bool value = std::is_convertible_v<
            xexpression_value_type_t<From>,
            xexpression_value_type_t<To>>;
    };

    template <class From, class To>
    inline constexpr bool is_convertible_expression_v = is_convertible_expression<From, To>::value;

    /*************************************
     * disable_xexpression (for SFINAE)
     *************************************/
    template <class T>
    using disable_xexpression = std::enable_if_t<!is_xexpression_v<std::decay_t<T>>>;

    /*************************************
     * expression_rank
     *************************************/
    template <class E, class = void>
    struct expression_rank
    {
        static constexpr std::size_t value = 0;
    };

    template <class E>
    struct expression_rank<E, std::void_t<decltype(std::declval<const E&>().dimension())>>
    {
        static constexpr std::size_t value = std::decay_t<E>::dimension();
    };

    template <class E>
    inline constexpr std::size_t expression_rank_v = expression_rank<E>::value;

    /*************************************
     * expression_data_pointer
     *************************************/
    template <class E, class = void>
    struct expression_data_pointer
    {
        using type = void;
    };

    template <class E>
    struct expression_data_pointer<E, std::void_t<decltype(std::declval<E&>().data())>>
    {
        using type = decltype(std::declval<E&>().data());
    };

    template <class E>
    using expression_data_pointer_t = typename expression_data_pointer<E>::type;

    /*************************************
     * has_strides
     *************************************/
    template <class E, class = void>
    struct has_strides : std::false_type {};

    template <class E>
    struct has_strides<E, std::void_t<decltype(std::declval<const E&>().strides())>>
        : std::true_type {};

    template <class E>
    inline constexpr bool has_strides_v = has_strides<E>::value;

    /*************************************
     * has_backstrides
     *************************************/
    template <class E, class = void>
    struct has_backstrides : std::false_type {};

    template <class E>
    struct has_backstrides<E, std::void_t<decltype(std::declval<const E&>().backstrides())>>
        : std::true_type {};

    template <class E>
    inline constexpr bool has_backstrides_v = has_backstrides<E>::value;

} // namespace xt

#endif // XTENSOR_XEXPRESSION_TRAITS_HPP