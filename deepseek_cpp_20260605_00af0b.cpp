//File 0038 : core/xoperation.hpp
//Expression operation dispatch: make_xfunction, xfunction_type_t, scalar promotion, and function object deduction.
#ifndef XTENSOR_XOPERATION_HPP
#define XTENSOR_XOPERATION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xfunction.hpp"
#include "xscalar.hpp"

namespace xt
{
    /*******************************
     * detail::make_xfunction helper
     *******************************/
    namespace detail
    {
        // Primary template: construct xfunction from functor and arguments.
        template <class F, class... CT>
        inline auto make_xfunction(F&& f, CT&&... e) noexcept
        {
            using function_type = xfunction<F, CT...>;
            return function_type(std::forward<F>(f), std::forward<CT>(e)...);
        }

        // Overload for when the first argument is a functor that should be deduced,
        // but we already have the perfect forwarding version.
    }

    /*******************************
     * Scalar operation promotion
     *******************************/
    namespace detail
    {
        // Promote a scalar value to an xscalar expression.
        template <class T>
        inline auto promote_scalar(T&& val)
        {
            return xscalar<std::decay_t<T>>(std::forward<T>(val));
        }

        // When both operands are expressions, no promotion needed.
        template <class E1, class E2,
                  std::enable_if_t<is_xexpression<std::decay_t<E1>>::value &&
                                   is_xexpression<std::decay_t<E2>>::value, int> = 0>
        inline auto promote_scalar(E1&& e1, E2&& e2)
        {
            return std::make_pair(std::forward<E1>(e1), std::forward<E2>(e2));
        }

        // Promote one scalar operand to xscalar.
        template <class E, class T,
                  std::enable_if_t<is_xexpression<std::decay_t<E>>::value &&
                                   !is_xexpression<std::decay_t<T>>::value, int> = 0>
        inline auto promote_scalar(E&& e, T&& val)
        {
            return std::make_pair(std::forward<E>(e), promote_scalar(std::forward<T>(val)));
        }

        template <class T, class E,
                  std::enable_if_t<!is_xexpression<std::decay_t<T>>::value &&
                                   is_xexpression<std::decay_t<E>>::value, int> = 0>
        inline auto promote_scalar(T&& val, E&& e)
        {
            return std::make_pair(promote_scalar(std::forward<T>(val)), std::forward<E>(e));
        }
    }

    /*******************************
     * is_xexpression trait
     *******************************/
    template <class T, class = void>
    struct is_xexpression : std::false_type {};

    template <class T>
    struct is_xexpression<T, std::void_t<decltype(std::declval<T>().derived_cast())>>
        : std::is_base_of<xtensor_expression_tag, T> {};

    template <class T>
    constexpr bool is_xexpression_v = is_xexpression<T>::value;

} // namespace xt

#endif // XTENSOR_XOPERATION_HPP