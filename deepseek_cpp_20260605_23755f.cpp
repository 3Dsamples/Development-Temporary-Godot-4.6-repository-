//File 0017 : core/xtensor_forward.hpp
//Forward declarations and type traits for all core components, enabling decoupled include dependencies.
#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"

namespace xt
{
    // Tag for expression types
    struct xtensor_expression_tag {};

    // Base expression class template
    template <class D>
    class xexpression;

    // Forward declaration of xcontainer_semantic
    template <class D>
    class xcontainer_semantic;

    // Forward declaration of xview_semantic
    template <class D>
    class xview_semantic;

    // Forward declaration of xstrided_container
    template <class D>
    class xstrided_container;

    // Forward declaration of xfunction
    template <class F, class... CT>
    class xfunction;

    // Forward declaration of xscalar
    template <class T>
    class xscalar;

    // Forward declaration of xarray_container
    template <class EC, layout_type L = DEFAULT_LAYOUT, class SC = std::vector<typename EC::size_type>, class Tag = xtensor_expression_tag>
    class xarray_container;

    // Forward declaration of xtensor_container
    template <class EC, std::size_t N, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_container;

    // Forward declaration of xview (fixed rank)
    template <class CT, std::size_t N, class... S>
    class xview;

    // Forward declaration of xstrided_view (dynamic rank)
    template <class CT, class S = std::vector<std::size_t>, layout_type L = DEFAULT_LAYOUT, class FST = void>
    class xstrided_view;

    // Forward declaration of xreducer
    template <class F, class E, class X, class ES = void>
    class xreducer;

    // Forward declaration of xaccumulator
    template <class F, class E>
    class xaccumulator;

    // Forward declaration of xstepper
    template <class C>
    class xstepper;

    // Primary template for xcontainer_inner_types (specialized for each container)
    template <class T>
    struct xcontainer_inner_types;

    // Helper to detect if a type is an xexpression
    template <class T>
    using disable_xexpression = std::enable_if_t<!std::is_base_of<xtensor_expression_tag, T>::value>;

    // Traits for layout
    template <layout_type L>
    struct layout_trait {
        static constexpr layout_type value = L;
    };

    // Default strategy reducers placeholder
    struct sequential_strategy {};
    struct parallel_strategy {};
    using DEFAULT_STRATEGY_REDUCERS = sequential_strategy;

    // Forward declaration of numeric_constants
    template <class T>
    struct numeric_constants;

    // Forward declare detail helpers used in expression types
    namespace detail
    {
        template <class F, class... CT>
        using xfunction_type_t = xfunction<F, CT...>;

        // Plus, minus, etc. need to be forward declared
        struct plus;
        struct minus;
        struct multiplies;
        struct divides;
        struct negate;
        struct less;
        struct less_equal;
        struct greater;
        struct greater_equal;
        struct equal_to;
        struct not_equal_to;
    }

    // Some common shapes
    template <class T>
    using xshape = std::vector<T>;

    // Forward declaration of math functions? Not needed.
}

#endif // XTENSOR_FORWARD_HPP