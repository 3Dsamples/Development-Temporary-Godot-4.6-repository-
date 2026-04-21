// core/xreducer.hpp
#ifndef XTENSOR_XREDUCER_HPP
#define XTENSOR_XREDUCER_HPP

// ----------------------------------------------------------------------------
// xreducer.hpp – Reduction operations on xtensor expressions
// ----------------------------------------------------------------------------
// This header defines the xreducer class and reduction functions (sum, prod,
// mean, etc.) for xtensor expressions. It supports:
//   - Global reductions (to scalar)
//   - Axis‑wise reductions (reducing specific dimensions)
//   - Custom reducer functors
//   - FFT‑accelerated product for BigNumber arrays
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <iterator>
#include <stdexcept>
#include <limits>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xarray.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace detail
    {
        // --------------------------------------------------------------------
        // Helper to remove a dimension from a shape
        // --------------------------------------------------------------------
        shape_type remove_dimension(const shape_type& shape, size_type axis);
        // --------------------------------------------------------------------
        // Compute strides for a reduced array
        // --------------------------------------------------------------------
        strides_type reduced_strides(const shape_type& reduced_shape,
                                    const strides_type& original_strides,
                                    const std::vector<size_type>& reduced_axes);
        // --------------------------------------------------------------------
        // Check if an axis is in a set of axes
        // --------------------------------------------------------------------
        bool is_axis_reduced(size_type axis, const std::vector<size_type>& axes);
        // --------------------------------------------------------------------
        // Convert flat index to multi‑dimensional index
        // --------------------------------------------------------------------
        std::vector<size_type> unravel_index(size_type flat_index, const shape_type& shape);
        // --------------------------------------------------------------------
        // Convert multi‑dimensional index to flat offset using strides
        // --------------------------------------------------------------------
        size_type ravel_index(const std::vector<size_type>& index, const strides_type& strides);
        // --------------------------------------------------------------------
        // Compute size from shape
        // --------------------------------------------------------------------
        size_type compute_size(const shape_type& shape) noexcept;

        // ====================================================================
        // Reducer functors
        // ====================================================================
        template <class T> struct reducer_sum;
        template <class T> struct reducer_prod;
        template <class T> struct reducer_mean;
        template <class T> struct reducer_min;
        template <class T> struct reducer_max;
        template <class T> struct reducer_stddev;
    }

    // ========================================================================
    // xreducer – Lazy reduction expression
    // ========================================================================
    template <class E, class X, class Reducer>
    class xreducer : public xexpression<xreducer<E, X, Reducer>>
    {
    public:
        using self_type = xreducer<E, X, Reducer>;
        using value_type = typename Reducer::value_type;
        using size_type = xt::size_type;
        using shape_type = xt::shape_type;
        using strides_type = xt::strides_type;
        using expression_type = E;
        using axes_type = X;
        using reducer_type = Reducer;
        static constexpr layout_type layout = layout_type::dynamic;

        template <class Ex, class Ax>
        xreducer(Ex&& e, Ax&& axes, Reducer&& reducer = Reducer());

        const shape_type& shape() const noexcept;
        size_type size() const;
        size_type dimension() const noexcept;
        bool empty() const noexcept;
        layout_type layout() const noexcept;

        const expression_type& expression() const noexcept;
        const axes_type& axes() const noexcept;
        const reducer_type& reducer() const noexcept;

        value_type operator()() const;
        template <class... Args> value_type operator()(Args... args) const;
        template <class S> value_type element(const S& indices) const;

        operator xarray_container<value_type>() const;
        xarray_container<value_type> evaluate() const;

        template <class C> void assign_to(xcontainer_expression<C>& dst) const;

    private:
        E m_expression;
        X m_axes;
        Reducer m_reducer;
        shape_type m_shape;

        shape_type make_reduced_shape() const;
        value_type reduce_global() const;
        xarray_container<value_type> reduce_axes() const;
        void reduce_axes_recursive(xarray_container<value_type>& result,
                                   std::vector<size_type>& reduced_indices,
                                   size_type& result_flat, size_type dim) const;
        value_type reduce_at(const std::vector<size_type>& reduced_indices) const;
        void reduce_axes_inner(value_type& result, bool& first, size_type& count,
                               size_type base_offset, std::vector<size_type>& axis_indices,
                               size_type axis_idx) const;
        template <class S> value_type evaluate_at(const S& indices) const;
    };

    // ========================================================================
    // Factory functions for reductions (global)
    // ========================================================================
    template <class E> auto sum(const xexpression<E>& e);
    template <class E> auto prod(const xexpression<E>& e);
    template <class E> auto mean(const xexpression<E>& e);
    template <class E> auto min(const xexpression<E>& e);
    template <class E> auto max(const xexpression<E>& e);

    // Axis‑wise reductions (variadic and container versions)
    template <class E, class... Axes> auto sum(const xexpression<E>& e, Axes... axes);
    template <class E, class C> auto sum(const xexpression<E>& e, const C& axes);
    template <class E, class... Axes> auto prod(const xexpression<E>& e, Axes... axes);
    template <class E, class C> auto prod(const xexpression<E>& e, const C& axes);
    template <class E, class... Axes> auto mean(const xexpression<E>& e, Axes... axes);
    template <class E, class C> auto mean(const xexpression<E>& e, const C& axes);
    template <class E, class... Axes> auto min(const xexpression<E>& e, Axes... axes);
    template <class E, class C> auto min(const xexpression<E>& e, const C& axes);
    template <class E, class... Axes> auto max(const xexpression<E>& e, Axes... axes);
    template <class E, class C> auto max(const xexpression<E>& e, const C& axes);

    // ------------------------------------------------------------------------
    // Variance and standard deviation (global)
    // ------------------------------------------------------------------------
    template <class E> auto variance(const xexpression<E>& e);
    template <class E> auto stddev(const xexpression<E>& e);
    template <class E, class... Axes> auto variance(const xexpression<E>& e, Axes... axes);
    template <class E, class... Axes> auto stddev(const xexpression<E>& e, Axes... axes);

    // ------------------------------------------------------------------------
    // All / any reductions (for boolean expressions)
    // ------------------------------------------------------------------------
    template <class E> bool all(const xexpression<E>& e);
    template <class E> bool any(const xexpression<E>& e);
    template <class E, class... Axes> auto all(const xexpression<E>& e, Axes... axes);
    template <class E, class... Axes> auto any(const xexpression<E>& e, Axes... axes);

    // ========================================================================
    // Inner types specialization for xreducer
    // ========================================================================
    template <class E, class X, class Reducer>
    struct xcontainer_inner_types<xreducer<E, X, Reducer>>
    {
        using temporary_type = xarray_container<typename Reducer::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with TODO and //comment in each)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace detail
    {
        inline shape_type remove_dimension(const shape_type& shape, size_type axis)
        { /* TODO: implement */ (void)axis; return shape; } //comment

        inline strides_type reduced_strides(const shape_type& reduced_shape,
                                            const strides_type& original_strides,
                                            const std::vector<size_type>& reduced_axes)
        { /* TODO: implement */ (void)reduced_shape; (void)reduced_axes; return original_strides; } //comment

        inline bool is_axis_reduced(size_type axis, const std::vector<size_type>& axes)
        { /* TODO: implement */ (void)axis; (void)axes; return false; } //comment

        inline std::vector<size_type> unravel_index(size_type flat_index, const shape_type& shape)
        { /* TODO: implement */ (void)flat_index; (void)shape; return {}; } //comment

        inline size_type ravel_index(const std::vector<size_type>& index, const strides_type& strides)
        { /* TODO: implement */ (void)index; (void)strides; return 0; } //comment

        inline size_type compute_size(const shape_type& shape) noexcept
        { size_type s = 1; for (auto d : shape) s *= d; return s; } //comment

        template <class T>
        struct reducer_sum
        {
            using value_type = T;
            T init() const noexcept { return T(0); } //comment
            T operator()(const T& a, const T& b) const { return a + b; } //comment
            T finalize(const T& value, size_type) const { return value; } //comment
        };

        template <class T>
        struct reducer_prod
        {
            using value_type = T;
            T init() const noexcept { return T(1); } //comment
            T operator()(const T& a, const T& b) const
            { /* TODO: FFT dispatch */ return a * b; } //comment
            T finalize(const T& value, size_type) const { return value; } //comment
        };

        template <class T>
        struct reducer_mean
        {
            using value_type = T;
            T init() const noexcept { return T(0); } //comment
            T operator()(const T& a, const T& b) const { return a + b; } //comment
            T finalize(const T& sum, size_type count) const
            { return count ? sum / T(count) : T(0); } //comment
        };

        template <class T>
        struct reducer_min
        {
            using value_type = T;
            T init() const noexcept { return std::numeric_limits<T>::max(); } //comment
            T operator()(const T& a, const T& b) const { return a < b ? a : b; } //comment
            T finalize(const T& value, size_type) const { return value; } //comment
        };

        template <class T>
        struct reducer_max
        {
            using value_type = T;
            T init() const noexcept { return std::numeric_limits<T>::lowest(); } //comment
            T operator()(const T& a, const T& b) const { return a > b ? a : b; } //comment
            T finalize(const T& value, size_type) const { return value; } //comment
        };

        template <class T>
        struct reducer_stddev
        {
            using value_type = T;
            struct state { T sum; T sum_sq; size_type count; };
            state init() const noexcept { return state{T(0), T(0), 0}; } //comment
            state operator()(const state& s, const T& val) const
            { return state{s.sum + val, s.sum_sq + val * val, s.count + 1}; } //comment
            state merge(const state& s1, const state& s2) const
            { return state{s1.sum + s2.sum, s1.sum_sq + s2.sum_sq, s1.count + s2.count}; } //comment
            T finalize(const state& s, size_type) const
            { if (s.count <= 1) return T(0); T m = s.sum / T(s.count); return std::sqrt((s.sum_sq - T(2)*m*s.sum + m*m*T(s.count)) / T(s.count - 1)); } //comment
        };
    }

    // xreducer member functions
    template <class E, class X, class Reducer>
    template <class Ex, class Ax>
    xreducer<E, X, Reducer>::xreducer(Ex&& e, Ax&& axes, Reducer&& reducer)
        : m_expression(std::forward<Ex>(e)), m_axes(std::forward<Ax>(axes)),
          m_reducer(std::move(reducer)), m_shape(make_reduced_shape())
    { /* TODO: implement */ } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::shape() const noexcept -> const shape_type&
    { return m_shape; } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::size() const -> size_type
    { return detail::compute_size(m_shape); } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::dimension() const noexcept -> size_type
    { return m_shape.size(); } //comment

    template <class E, class X, class Reducer>
    inline bool xreducer<E, X, Reducer>::empty() const noexcept
    { return m_expression.empty(); } //comment

    template <class E, class X, class Reducer>
    inline layout_type xreducer<E, X, Reducer>::layout() const noexcept
    { return layout_type::dynamic; } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::expression() const noexcept -> const expression_type&
    { return m_expression; } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::axes() const noexcept -> const axes_type&
    { return m_axes; } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::reducer() const noexcept -> const reducer_type&
    { return m_reducer; } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::operator()() const -> value_type
    { return reduce_global(); } //comment

    template <class E, class X, class Reducer>
    template <class... Args>
    inline auto xreducer<E, X, Reducer>::operator()(Args... args) const -> value_type
    { std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...}; return evaluate_at(indices); } //comment

    template <class E, class X, class Reducer>
    template <class S>
    inline auto xreducer<E, X, Reducer>::element(const S& indices) const -> value_type
    { return evaluate_at(indices); } //comment

    template <class E, class X, class Reducer>
    inline xreducer<E, X, Reducer>::operator xarray_container<value_type>() const
    { return evaluate(); } //comment

    template <class E, class X, class Reducer>
    inline auto xreducer<E, X, Reducer>::evaluate() const -> xarray_container<value_type>
    { return m_axes.empty() ? xarray_container<value_type>({}, reduce_global()) : reduce_axes(); } //comment

    template <class E, class X, class Reducer>
    template <class C>
    inline void xreducer<E, X, Reducer>::assign_to(xcontainer_expression<C>& dst) const
    { auto result = evaluate(); dst.derived_cast().resize(result.shape()); std::copy(result.begin(), result.end(), dst.derived_cast().begin()); } //comment

    // Private helpers
    template <class E, class X, class Reducer>
    auto xreducer<E, X, Reducer>::make_reduced_shape() const -> shape_type
    { /* TODO: implement */ return {}; } //comment

    template <class E, class X, class Reducer>
    auto xreducer<E, X, Reducer>::reduce_global() const -> value_type
    { /* TODO: implement */ return m_reducer.init(); } //comment

    template <class E, class X, class Reducer>
    auto xreducer<E, X, Reducer>::reduce_axes() const -> xarray_container<value_type>
    { /* TODO: implement */ return xarray_container<value_type>(m_shape); } //comment

    template <class E, class X, class Reducer>
    void xreducer<E, X, Reducer>::reduce_axes_recursive(xarray_container<value_type>& result,
                                                        std::vector<size_type>& reduced_indices,
                                                        size_type& result_flat, size_type dim) const
    { /* TODO: implement */ } //comment

    template <class E, class X, class Reducer>
    auto xreducer<E, X, Reducer>::reduce_at(const std::vector<size_type>& reduced_indices) const -> value_type
    { /* TODO: implement */ return value_type(); } //comment

    template <class E, class X, class Reducer>
    void xreducer<E, X, Reducer>::reduce_axes_inner(value_type& result, bool& first, size_type& count,
                                                    size_type base_offset, std::vector<size_type>& axis_indices,
                                                    size_type axis_idx) const
    { /* TODO: implement */ } //comment

    template <class E, class X, class Reducer>
    template <class S>
    auto xreducer<E, X, Reducer>::evaluate_at(const S& indices) const -> value_type
    { /* TODO: implement */ return value_type(); } //comment

    // Factory functions (global)
    template <class E> auto sum(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<value_type>>(e.derived_cast(), std::vector<size_type>{})(); } //comment
    template <class E> auto prod(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(e.derived_cast(), std::vector<size_type>{})(); } //comment
    template <class E> auto mean(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(e.derived_cast(), std::vector<size_type>{})(); } //comment
    template <class E> auto min(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(e.derived_cast(), std::vector<size_type>{})(); } //comment
    template <class E> auto max(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(e.derived_cast(), std::vector<size_type>{})(); } //comment

    // Axis‑wise (stubs)
    template <class E, class... Axes> auto sum(const xexpression<E>& e, Axes... axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...}; return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class C> auto sum(const xexpression<E>& e, const C& axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec(axes.begin(), axes.end()); return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class... Axes> auto prod(const xexpression<E>& e, Axes... axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...}; return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class C> auto prod(const xexpression<E>& e, const C& axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec(axes.begin(), axes.end()); return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class... Axes> auto mean(const xexpression<E>& e, Axes... axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...}; return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class C> auto mean(const xexpression<E>& e, const C& axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec(axes.begin(), axes.end()); return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class... Axes> auto min(const xexpression<E>& e, Axes... axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...}; return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class C> auto min(const xexpression<E>& e, const C& axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec(axes.begin(), axes.end()); return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class... Axes> auto max(const xexpression<E>& e, Axes... axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...}; return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment
    template <class E, class C> auto max(const xexpression<E>& e, const C& axes)
    { using value_type = typename E::value_type; std::vector<size_type> axes_vec(axes.begin(), axes.end()); return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(e.derived_cast(), std::move(axes_vec)); } //comment

    template <class E> auto variance(const xexpression<E>& e)
    { using value_type = typename E::value_type; auto reducer = detail::reducer_stddev<value_type>(); auto state = reducer.init(); for (size_t i=0; i<e.size(); ++i) state = reducer(state, e.flat(i)); return reducer.finalize(state, e.size()); } //comment
    template <class E> auto stddev(const xexpression<E>& e)
    { return std::sqrt(variance(e)); } //comment
    template <class E, class... Axes> auto variance(const xexpression<E>& e, Axes... axes)
    { /* TODO: implement */ return e; } //comment
    template <class E, class... Axes> auto stddev(const xexpression<E>& e, Axes... axes)
    { /* TODO: implement */ return e; } //comment

    template <class E> bool all(const xexpression<E>& e)
    { for (size_t i=0; i<e.size(); ++i) if (!e.flat(i)) return false; return true; } //comment
    template <class E> bool any(const xexpression<E>& e)
    { for (size_t i=0; i<e.size(); ++i) if (e.flat(i)) return true; return false; } //comment
    template <class E, class... Axes> auto all(const xexpression<E>& e, Axes... axes)
    { /* TODO: implement */ return false; } //comment
    template <class E, class... Axes> auto any(const xexpression<E>& e, Axes... axes)
    { /* TODO: implement */ return false; } //comment

} // namespace xt

#endif // XTENSOR_XREDUCER_HPPe_type;
        return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(
            e.derived_cast(), std::vector<size_type>{}
        )();
    }

    template <class E>
    inline auto mean(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(
            e.derived_cast(), std::vector<size_type>{}
        )();
    }

    template <class E>
    inline auto min(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(
            e.derived_cast(), std::vector<size_type>{}
        )();
    }

    template <class E>
    inline auto max(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(
            e.derived_cast(), std::vector<size_type>{}
        )();
    }

    // Axis‑wise reductions (accept variadic axes or container)
    template <class E, class... Axes>
    inline auto sum(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class C>
    inline auto sum(const xexpression<E>& e, const C& axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec(axes.begin(), axes.end());
        return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class... Axes>
    inline auto prod(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class C>
    inline auto prod(const xexpression<E>& e, const C& axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec(axes.begin(), axes.end());
        return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class... Axes>
    inline auto mean(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class C>
    inline auto mean(const xexpression<E>& e, const C& axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec(axes.begin(), axes.end());
        return xreducer<const E&, std::vector<size_type>, detail::reducer_mean<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class... Axes>
    inline auto min(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class C>
    inline auto min(const xexpression<E>& e, const C& axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec(axes.begin(), axes.end());
        return xreducer<const E&, std::vector<size_type>, detail::reducer_min<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class... Axes>
    inline auto max(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    template <class E, class C>
    inline auto max(const xexpression<E>& e, const C& axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec(axes.begin(), axes.end());
        return xreducer<const E&, std::vector<size_type>, detail::reducer_max<value_type>>(
            e.derived_cast(), std::move(axes_vec)
        );
    }

    // ------------------------------------------------------------------------
    // Variance and standard deviation (global)
    // ------------------------------------------------------------------------
    template <class E>
    inline auto variance(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto reducer = detail::reducer_stddev<value_type>();
        auto state = reducer.init();
        const auto& expr = e.derived_cast();
        for (size_type i = 0; i < expr.size(); ++i)
            state = reducer(state, expr.flat(i));
        return reducer.finalize(state, expr.size());
    }

    template <class E>
    inline auto stddev(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        auto var = variance(e);
        if constexpr (std::is_same_v<value_type, bignumber::BigNumber>)
        {
            return bignumber::sqrt(var);
        }
        else
        {
            return std::sqrt(var);
        }
    }

    // Axis‑wise variance and stddev (to be implemented fully)
    template <class E, class... Axes>
    inline auto variance(const xexpression<E>& e, Axes... axes)
    {
        using value_type = typename E::value_type;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        auto reducer = detail::reducer_stddev<value_type>();
        return xreducer<const E&, std::vector<size_type>, detail::reducer_stddev<value_type>>(
            e.derived_cast(), std::move(axes_vec), std::move(reducer)
        );
    }

    template <class E, class... Axes>
    inline auto stddev(const xexpression<E>& e, Axes... axes)
    {
        auto var = variance(e, axes...);
        if constexpr (std::is_same_v<typename decltype(var)::value_type, bignumber::BigNumber>)
        {
            return var.transform([](const bignumber::BigNumber& x) { return bignumber::sqrt(x); });
        }
        else
        {
            return var.transform([](auto x) { return std::sqrt(x); });
        }
    }

    // ------------------------------------------------------------------------
    // All / any reductions (for boolean expressions)
    // ------------------------------------------------------------------------
    template <class E>
    inline bool all(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        for (size_type i = 0; i < expr.size(); ++i)
        {
            if (!expr.flat(i))
                return false;
        }
        return true;
    }

    template <class E>
    inline bool any(const xexpression<E>& e)
    {
        const auto& expr = e.derived_cast();
        for (size_type i = 0; i < expr.size(); ++i)
        {
            if (expr.flat(i))
                return true;
        }
        return false;
    }

    template <class E, class... Axes>
    inline auto all(const xexpression<E>& e, Axes... axes)
    {
        using value_type = bool;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        auto reducer = detail::reducer_prod<bool>(); // logical AND via product
        return xreducer<const E&, std::vector<size_type>, detail::reducer_prod<bool>>(
            e.derived_cast(), std::move(axes_vec), std::move(reducer)
        );
    }

    template <class E, class... Axes>
    inline auto any(const xexpression<E>& e, Axes... axes)
    {
        using value_type = bool;
        std::vector<size_type> axes_vec = {static_cast<size_type>(axes)...};
        auto reducer = detail::reducer_sum<bool>(); // logical OR via sum
        return xreducer<const E&, std::vector<size_type>, detail::reducer_sum<bool>>(
            e.derived_cast(), std::move(axes_vec), std::move(reducer)
        ).transform([](int x) { return x > 0; });
    }

    // ========================================================================
    // Inner types specialization for xreducer
    // ========================================================================
    template <class E, class X, class Reducer>
    struct xcontainer_inner_types<xreducer<E, X, Reducer>>
    {
        using temporary_type = xarray_container<typename Reducer::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

#endif // XTENSOR_XREDUCER_HPP