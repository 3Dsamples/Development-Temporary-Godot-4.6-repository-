//File 0017 (UPDATED) : core/xtensor_forward.hpp
//Forward declarations and type traits for all core components, including new semantic types, adaptors, and expression helpers.
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
    struct xtensor_expression_tag {};

    template <class D>
    class xexpression;

    // semantic base classes
    template <class D>
    class xsemantic_base;

    template <class D>
    class xcontainer_semantic;

    template <class D>
    class xview_semantic;

    template <class D>
    class xsharable_expression;

    template <class D, class T>
    class scalar_computed_assign;

    template <class D>
    class xstrided_container;

    template <class F, class... CT>
    class xfunction;

    template <class T>
    class xscalar;

    template <class EC, layout_type L = DEFAULT_LAYOUT, class SC = std::vector<typename EC::size_type>, class Tag = xtensor_expression_tag>
    class xarray_container;

    template <class EC, std::size_t N, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xtensor_container;

    // adaptor for external storage
    template <class EC, layout_type L = DEFAULT_LAYOUT, class Tag = xtensor_expression_tag>
    class xarray_adaptor;

    template <class CT, std::size_t N, class... S>
    class xview;

    template <class CT, class S = std::vector<std::size_t>, layout_type L = DEFAULT_LAYOUT, class FST = void>
    class xstrided_view;

    template <class F, class E, class X, class ES = void>
    class xreducer;

    template <class F, class E>
    class xaccumulator;

    template <class C>
    class xstepper;

    // expression traits
    template <class T>
    struct xcontainer_inner_types;

    template <class T>
    using disable_xexpression = std::enable_if_t<!std::is_base_of<xtensor_expression_tag, T>::value>;

    template <layout_type L>
    struct layout_trait { static constexpr layout_type value = L; };

    struct sequential_strategy {};
    struct parallel_strategy {};
    using DEFAULT_STRATEGY_REDUCERS = sequential_strategy;

    template <class T>
    struct numeric_constants;

    namespace detail
    {
        template <class F, class... CT>
        using xfunction_type_t = xfunction<F, CT...>;

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

    template <class T>
    using xshape = std::vector<T>;
}

#endif // XTENSOR_FORWARD_HPP