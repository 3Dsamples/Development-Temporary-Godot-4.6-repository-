// core/xtensor_forward.hpp
#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

// ----------------------------------------------------------------------------
// xtensor_forward.hpp – Forward declarations for xtensor + BigNumber + FFT
// ----------------------------------------------------------------------------
// Provides forward declarations of all container, view, expression, and
// algorithm types. Includes traits for BigNumber detection and FFT dispatch.
// Designed for high‑performance scientific simulation, real‑time animation,
// and multi‑scale physics (micro to galactic).
// ----------------------------------------------------------------------------

#include <cstddef>
#include <vector>
#include <type_traits>
#include "xtensor_config.hpp"
#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    // ------------------------------------------------------------------------
    // Layout enumeration
    // ------------------------------------------------------------------------
    enum class layout_type
    {
        row_major,
        column_major,
        dynamic
    };
    static constexpr layout_type DEFAULT_LAYOUT = config::default_layout;

    // ------------------------------------------------------------------------
    // Type aliases
    // ------------------------------------------------------------------------
    using index_type   = config::index_type;
    using size_type    = config::size_type;
    using shape_type   = std::vector<size_type>;
    using strides_type = std::vector<size_type>;
    using value_type   = config::value_type;

    // ------------------------------------------------------------------------
    // CRTP base classes
    // ------------------------------------------------------------------------
    template <class D> class xexpression;
    template <class D> class xcontainer_expression;
    template <class D> class xview_expression;

    // ------------------------------------------------------------------------
    // Containers
    // ------------------------------------------------------------------------
    template <class T, layout_type L = DEFAULT_LAYOUT, class A = std::allocator<T>>
    class xarray_container;
    template <class T, std::size_t N, layout_type L = DEFAULT_LAYOUT, class A = std::allocator<T>>
    class xtensor_container;
    template <class EC, layout_type L = DEFAULT_LAYOUT>
    class xarray_adaptor;
    template <class EC, std::size_t N, layout_type L = DEFAULT_LAYOUT>
    class xtensor_adaptor;

    // ------------------------------------------------------------------------
    // Views
    // ------------------------------------------------------------------------
    template <class CT, class... S> class xview;
    template <class CT, class S, layout_type L = DEFAULT_LAYOUT, class FST = typename CT::strides_type>
    class xstrided_view;
    template <class CT> class xdynamic_view;

    // ------------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------------
    template <class F, class... E> class xfunction;
    template <class F, class... E> class xscalar;
    template <class E> class xbroadcast;

    // ------------------------------------------------------------------------
    // Reducers / accumulators
    // ------------------------------------------------------------------------
    template <class E, class X, class Reducer> class xreducer;
    template <class E> class xaccumulator;

    // ------------------------------------------------------------------------
    // Sparse formats
    // ------------------------------------------------------------------------
    template <class T> class xcoo_scheme;
    template <class T> class xcsr_scheme;

    // ------------------------------------------------------------------------
    // FFT module
    // ------------------------------------------------------------------------
    namespace fft
    {
        class fft_plan;
        template <class E> auto fft(const xexpression<E>& e);
        template <class E> auto ifft(const xexpression<E>& e);
        template <class E> auto rfft(const xexpression<E>& e);
        template <class E> auto irfft(const xexpression<E>& e);
        template <class E> auto fft2(const xexpression<E>& e);
        template <class E> auto ifft2(const xexpression<E>& e);
        template <class E> auto fftn(const xexpression<E>& e, const std::vector<size_type>& axes = {});
        template <class E> auto ifftn(const xexpression<E>& e, const std::vector<size_type>& axes = {});
        template <class E1, class E2> auto convolve(const xexpression<E1>& a, const xexpression<E2>& b);
        template <class E1, class E2> auto correlate(const xexpression<E1>& a, const xexpression<E2>& b);
    }

    // ------------------------------------------------------------------------
    // Linear algebra
    // ------------------------------------------------------------------------
    namespace linalg
    {
        template <class E> auto inv(const xexpression<E>& a);
        template <class E> auto det(const xexpression<E>& a);
        template <class E> auto solve(const xexpression<E>& a, const xexpression<E>& b);
        template <class E> auto cholesky(const xexpression<E>& a);
        template <class E> auto qr(const xexpression<E>& a);
        template <class E> auto svd(const xexpression<E>& a);
        template <class E> auto eig(const xexpression<E>& a);
    }

    // ------------------------------------------------------------------------
    // Statistics / Metrics
    // ------------------------------------------------------------------------
    template <class E> auto mean(const xexpression<E>& e);
    template <class E> auto variance(const xexpression<E>& e);
    template <class E> auto stddev(const xexpression<E>& e);
    template <class E1, class E2> auto mse(const xexpression<E1>& y_true, const xexpression<E2>& y_pred);

    // ------------------------------------------------------------------------
    // Image and mesh
    // ------------------------------------------------------------------------
    template <class T, size_t C> class ximage;
    template <class T> class xmesh;

    // ------------------------------------------------------------------------
    // Optimization / Integration
    // ------------------------------------------------------------------------
    namespace optimize { template <class F> auto minimize(F&& f, const shape_type& x0); }
    namespace integrate { template <class F> auto integrate(F&& f, value_type a, value_type b); }

    // ------------------------------------------------------------------------
    // Type traits
    // ------------------------------------------------------------------------
    template <class T> struct is_xexpression : std::false_type {};
    template <class D> struct is_xexpression<xexpression<D>> : std::true_type {};
    template <class T> inline constexpr bool is_xexpression_v = is_xexpression<T>::value;

    template <class T> struct is_bignumber_expression : std::false_type {};
    template <class E> struct is_bignumber_expression<E>
        : std::conjunction<is_xexpression<E>, std::is_same<typename E::value_type, bignumber::BigNumber>> {};
    template <class T> inline constexpr bool is_bignumber_expression_v = is_bignumber_expression<T>::value;

    template <class E1, class E2>
    struct use_fft_multiplication : std::conjunction<
        is_bignumber_expression<E1>, is_bignumber_expression<E2>,
        std::bool_constant<config::use_fft_multiply>> {};
    template <class E1, class E2>
    inline constexpr bool use_fft_multiplication_v = use_fft_multiplication<E1, E2>::value;

    // ------------------------------------------------------------------------
    // Factory helpers
    // ------------------------------------------------------------------------
    template <class F, class... E> auto make_xfunction(F&& f, E&&... e);
    template <class T> xarray_container<T> empty(const shape_type& shape);
    template <class T> xarray_container<T> zeros(const shape_type& shape);
    template <class T> xarray_container<T> ones(const shape_type& shape);

} // namespace xt

#endif // XTENSOR_FORWARD_HPP