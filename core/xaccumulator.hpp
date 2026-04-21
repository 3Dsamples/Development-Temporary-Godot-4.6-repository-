// core/xaccumulator.hpp
#ifndef XTENSOR_XACCUMULATOR_HPP
#define XTENSOR_XACCUMULATOR_HPP

// ----------------------------------------------------------------------------
// xaccumulator.hpp – Cumulative operations on xtensor expressions
// ----------------------------------------------------------------------------
// This header defines the xaccumulator class and cumulative functions
// (cumsum, cumprod, cummin, cummax) for xtensor expressions. It supports:
//   - Global cumulative operations (flattened order)
//   - Axis‑wise cumulative operations
//   - Lazy evaluation via expression templates
//   - FFT‑accelerated cumulative product for BigNumber arrays
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>
#include <stdexcept>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xarray.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace detail
    {
        // --------------------------------------------------------------------
        // Compute size from shape
        // --------------------------------------------------------------------
        inline size_type compute_size(const shape_type& shape) noexcept;
        // --------------------------------------------------------------------
        // Check if an axis is valid
        // --------------------------------------------------------------------
        inline void validate_axis(size_type axis, size_type dimension);
        // --------------------------------------------------------------------
        // Accumulator functors
        // --------------------------------------------------------------------
        template <class T> struct accumulator_sum;
        template <class T> struct accumulator_prod;
        template <class T> struct accumulator_min;
        template <class T> struct accumulator_max;
    }

    // ========================================================================
    // xaccumulator – Lazy cumulative expression
    // ========================================================================
    template <class E, class Accumulator>
    class xaccumulator : public xexpression<xaccumulator<E, Accumulator>>
    {
    public:
        using self_type = xaccumulator<E, Accumulator>;
        using value_type = typename Accumulator::value_type;
        using size_type = xt::size_type;
        using shape_type = xt::shape_type;
        using strides_type = xt::strides_type;

        using expression_type = E;
        using accumulator_type = Accumulator;

        static constexpr layout_type layout = E::layout;

        // Constructor for global accumulation (no axis)
        template <class Ex>
        xaccumulator(Ex&& e, Accumulator&& acc = Accumulator());

        // Constructor for axis‑wise accumulation
        template <class Ex>
        xaccumulator(Ex&& e, size_type axis, Accumulator&& acc = Accumulator());

        const shape_type& shape() const noexcept;
        size_type size() const;
        size_type dimension() const noexcept;
        bool empty() const noexcept;
        layout_type layout() const noexcept;

        const expression_type& expression() const noexcept;
        const accumulator_type& accumulator() const noexcept;
        const std::optional<size_type>& axis() const noexcept;

        value_type flat(size_type i) const;
        template <class... Args> value_type operator()(Args... args) const;
        template <class S> value_type element(const S& indices) const;

        operator xarray_container<value_type>() const;
        xarray_container<value_type> evaluate() const;

        template <class C> void assign_to(xcontainer_expression<C>& dst) const;

    private:
        E m_expression;
        std::optional<size_type> m_axis;
        Accumulator m_accumulator;
        shape_type m_shape;

        value_type flat_global(size_type i) const;
        void evaluate_global(xarray_container<value_type>& result) const;
        value_type flat_axis(size_type flat_index, size_type axis) const;
        template <class S> value_type element_axis(const S& indices, size_type axis) const;
        void evaluate_axis(xarray_container<value_type>& result, size_type axis) const;
    };

    // ========================================================================
    // Factory functions for accumulators
    // ========================================================================
    template <class E> auto cumsum(const xexpression<E>& e);
    template <class E> auto cumsum(const xexpression<E>& e, size_type axis);
    template <class E> auto cumprod(const xexpression<E>& e);
    template <class E> auto cumprod(const xexpression<E>& e, size_type axis);
    template <class E> auto cummin(const xexpression<E>& e);
    template <class E> auto cummin(const xexpression<E>& e, size_type axis);
    template <class E> auto cummax(const xexpression<E>& e);
    template <class E> auto cummax(const xexpression<E>& e, size_type axis);

    // ------------------------------------------------------------------------
    // diff – difference between consecutive elements
    // ------------------------------------------------------------------------
    template <class E> auto diff(const xexpression<E>& e, size_type n = 1, size_type axis = 0);

    // ------------------------------------------------------------------------
    // trapz – trapezoidal numerical integration
    // ------------------------------------------------------------------------
    template <class E> auto trapz(const xexpression<E>& y, const xexpression<E>& x, size_type axis = 0);
    template <class E> auto trapz(const xexpression<E>& y, value_type dx = value_type(1), size_type axis = 0);

    // ========================================================================
    // Inner types specialization for xaccumulator
    // ========================================================================
    template <class E, class Accumulator>
    struct xcontainer_inner_types<xaccumulator<E, Accumulator>>
    {
        using temporary_type = xarray_container<typename Accumulator::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (with //comment in each)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace detail
    {
        inline size_type compute_size(const shape_type& shape) noexcept
        { size_type s = 1; for (auto d : shape) s *= d; return s; } //comment

        inline void validate_axis(size_type axis, size_type dimension)
        { if (axis >= dimension) XTENSOR_THROW(std::out_of_range, "Axis out of range"); } //comment

        template <class T>
        struct accumulator_sum
        {
            using value_type = T;
            T init() const noexcept { return T(0); } //comment
            T operator()(const T& acc, const T& val) const { return acc + val; } //comment
        };

        template <class T>
        struct accumulator_prod
        {
            using value_type = T;
            T init() const noexcept { return T(1); } //comment
            T operator()(const T& acc, const T& val) const
            { /* TODO: FFT dispatch */ return acc * val; } //comment
        };

        template <class T>
        struct accumulator_min
        {
            using value_type = T;
            T init() const noexcept { return std::numeric_limits<T>::max(); } //comment
            T operator()(const T& acc, const T& val) const { return acc < val ? acc : val; } //comment
        };

        template <class T>
        struct accumulator_max
        {
            using value_type = T;
            T init() const noexcept { return std::numeric_limits<T>::lowest(); } //comment
            T operator()(const T& acc, const T& val) const { return acc > val ? acc : val; } //comment
        };
    }

    // xaccumulator member functions
    template <class E, class Accumulator>
    template <class Ex>
    xaccumulator<E, Accumulator>::xaccumulator(Ex&& e, Accumulator&& acc)
        : m_expression(std::forward<Ex>(e)), m_axis(std::nullopt),
          m_accumulator(std::move(acc)), m_shape(m_expression.shape())
    { /* TODO: implement */ } //comment

    template <class E, class Accumulator>
    template <class Ex>
    xaccumulator<E, Accumulator>::xaccumulator(Ex&& e, size_type axis, Accumulator&& acc)
        : m_expression(std::forward<Ex>(e)), m_axis(axis),
          m_accumulator(std::move(acc)), m_shape(m_expression.shape())
    { detail::validate_axis(axis, m_expression.dimension()); } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::shape() const noexcept -> const shape_type&
    { return m_shape; } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::size() const -> size_type
    { return detail::compute_size(m_shape); } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::dimension() const noexcept -> size_type
    { return m_shape.size(); } //comment

    template <class E, class Accumulator>
    inline bool xaccumulator<E, Accumulator>::empty() const noexcept
    { return m_expression.empty(); } //comment

    template <class E, class Accumulator>
    inline layout_type xaccumulator<E, Accumulator>::layout() const noexcept
    { return m_expression.layout(); } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::expression() const noexcept -> const expression_type&
    { return m_expression; } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::accumulator() const noexcept -> const accumulator_type&
    { return m_accumulator; } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::axis() const noexcept -> const std::optional<size_type>&
    { return m_axis; } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::flat(size_type i) const -> value_type
    { return m_axis.has_value() ? flat_axis(i, *m_axis) : flat_global(i); } //comment

    template <class E, class Accumulator>
    template <class... Args>
    inline auto xaccumulator<E, Accumulator>::operator()(Args... args) const -> value_type
    { std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...}; return element(indices); } //comment

    template <class E, class Accumulator>
    template <class S>
    inline auto xaccumulator<E, Accumulator>::element(const S& indices) const -> value_type
    { return m_axis.has_value() ? element_axis(indices, *m_axis) : flat_global(detail::ravel_index(indices, m_shape)); } //comment

    template <class E, class Accumulator>
    inline xaccumulator<E, Accumulator>::operator xarray_container<value_type>() const
    { return evaluate(); } //comment

    template <class E, class Accumulator>
    inline auto xaccumulator<E, Accumulator>::evaluate() const -> xarray_container<value_type>
    { xarray_container<value_type> result(m_shape); m_axis.has_value() ? evaluate_axis(result, *m_axis) : evaluate_global(result); return result; } //comment

    template <class E, class Accumulator>
    template <class C>
    inline void xaccumulator<E, Accumulator>::assign_to(xcontainer_expression<C>& dst) const
    { auto result = evaluate(); dst.derived_cast().resize(result.shape()); std::copy(result.begin(), result.end(), dst.derived_cast().begin()); } //comment

    // Private helpers
    template <class E, class Accumulator>
    auto xaccumulator<E, Accumulator>::flat_global(size_type i) const -> value_type
    { value_type result = m_accumulator.init(); for (size_type j = 0; j <= i; ++j) result = m_accumulator(result, m_expression.flat(j)); return result; } //comment

    template <class E, class Accumulator>
    void xaccumulator<E, Accumulator>::evaluate_global(xarray_container<value_type>& result) const
    { if (m_expression.empty()) return; value_type running = m_expression.flat(0); result.flat(0) = running; for (size_type i = 1; i < m_expression.size(); ++i) { running = m_accumulator(running, m_expression.flat(i)); result.flat(i) = running; } } //comment

    template <class E, class Accumulator>
    auto xaccumulator<E, Accumulator>::flat_axis(size_type flat_index, size_type axis) const -> value_type
    { std::vector<size_type> index = detail::unravel_index(flat_index, m_shape); return element_axis(index, axis); } //comment

    template <class E, class Accumulator>
    template <class S>
    auto xaccumulator<E, Accumulator>::element_axis(const S& indices, size_type axis) const -> value_type
    { /* TODO: implement */ return value_type(); } //comment

    template <class E, class Accumulator>
    void xaccumulator<E, Accumulator>::evaluate_axis(xarray_container<value_type>& result, size_type axis) const
    { /* TODO: implement */ } //comment

    // Factory functions
    template <class E> auto cumsum(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_sum<value_type>>(e.derived_cast()); } //comment
    template <class E> auto cumsum(const xexpression<E>& e, size_type axis)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_sum<value_type>>(e.derived_cast(), axis); } //comment
    template <class E> auto cumprod(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_prod<value_type>>(e.derived_cast()); } //comment
    template <class E> auto cumprod(const xexpression<E>& e, size_type axis)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_prod<value_type>>(e.derived_cast(), axis); } //comment
    template <class E> auto cummin(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_min<value_type>>(e.derived_cast()); } //comment
    template <class E> auto cummin(const xexpression<E>& e, size_type axis)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_min<value_type>>(e.derived_cast(), axis); } //comment
    template <class E> auto cummax(const xexpression<E>& e)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_max<value_type>>(e.derived_cast()); } //comment
    template <class E> auto cummax(const xexpression<E>& e, size_type axis)
    { using value_type = typename E::value_type; return xaccumulator<const E&, detail::accumulator_max<value_type>>(e.derived_cast(), axis); } //comment

    template <class E> auto diff(const xexpression<E>& e, size_type n, size_type axis)
    { /* TODO: implement */ return e; } //comment
    template <class E> auto trapz(const xexpression<E>& y, const xexpression<E>& x, size_type axis)
    { /* TODO: implement */ return y; } //comment
    template <class E> auto trapz(const xexpression<E>& y, value_type dx, size_type axis)
    { /* TODO: implement */ return y; } //comment

} // namespace xt

#endif // XTENSOR_XACCUMULATOR_HPP  size_type coord = remaining % expr_shape[d];
                    remaining /= expr_shape[d];
                    prefix_offset += coord * expr_strides[d];
                }

                for (size_type inner = 0; inner < inner_size; ++inner)
                {
                    // Compute suffix offset for this inner slice
                    size_type suffix_offset = 0;
                    size_type rem = inner;
                    for (size_type d = axis + 1; d < expr_shape.size(); ++d)
                    {
                        size_type coord = rem % expr_shape[d];
                        rem /= expr_shape[d];
                        suffix_offset += coord * expr_strides[d];
                    }

                    // Perform cumulative operation along the axis
                    value_type running = m_accumulator.init();
                    bool first = true;

                    for (size_type i = 0; i < axis_size; ++i)
                    {
                        size_type offset = prefix_offset + i * axis_stride + suffix_offset;
                        const value_type& val = m_expression.flat(offset);

                        if (first)
                        {
                            running = val;
                            first = false;
                        }
                        else
                        {
                            running = m_accumulator(running, val);
                        }

                        // Store result in the corresponding position
                        size_type result_offset = prefix_offset + i * axis_stride + suffix_offset;
                        result.flat(result_offset) = running;
                    }
                }
            }
        }
    };

    // ========================================================================
    // Factory functions for accumulators
    // ========================================================================

    // Global cumulative sum (flattened order)
    template <class E>
    inline auto cumsum(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_sum<value_type>>(
            e.derived_cast()
        );
    }

    // Axis‑wise cumulative sum
    template <class E>
    inline auto cumsum(const xexpression<E>& e, size_type axis)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_sum<value_type>>(
            e.derived_cast(), axis
        );
    }

    // Global cumulative product
    template <class E>
    inline auto cumprod(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_prod<value_type>>(
            e.derived_cast()
        );
    }

    // Axis‑wise cumulative product
    template <class E>
    inline auto cumprod(const xexpression<E>& e, size_type axis)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_prod<value_type>>(
            e.derived_cast(), axis
        );
    }

    // Global cumulative minimum
    template <class E>
    inline auto cummin(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_min<value_type>>(
            e.derived_cast()
        );
    }

    // Axis‑wise cumulative minimum
    template <class E>
    inline auto cummin(const xexpression<E>& e, size_type axis)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_min<value_type>>(
            e.derived_cast(), axis
        );
    }

    // Global cumulative maximum
    template <class E>
    inline auto cummax(const xexpression<E>& e)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_max<value_type>>(
            e.derived_cast()
        );
    }

    // Axis‑wise cumulative maximum
    template <class E>
    inline auto cummax(const xexpression<E>& e, size_type axis)
    {
        using value_type = typename E::value_type;
        return xaccumulator<const E&, detail::accumulator_max<value_type>>(
            e.derived_cast(), axis
        );
    }

    // ------------------------------------------------------------------------
    // diff – difference between consecutive elements
    // ------------------------------------------------------------------------
    template <class E>
    inline auto diff(const xexpression<E>& e, size_type n = 1, size_type axis = 0)
    {
        using value_type = typename E::value_type;
        const auto& expr = e.derived_cast();
        const auto& shp = expr.shape();
        size_type dim = shp.size();

        if (axis >= dim)
            XTENSOR_THROW(std::out_of_range, "diff: axis out of range");

        if (n == 0)
            return xarray_container<value_type>(expr);

        size_type new_axis_size = shp[axis] > n ? shp[axis] - n : 0;
        shape_type new_shape = shp;
        new_shape[axis] = new_axis_size;

        xarray_container<value_type> result(new_shape);

        // Iterate over all slices orthogonal to axis
        size_type outer_size = 1;
        for (size_type d = 0; d < axis; ++d)
            outer_size *= shp[d];

        size_type inner_size = 1;
        for (size_type d = axis + 1; d < dim; ++d)
            inner_size *= shp[d];

        size_type axis_stride = expr.strides()[axis];
        size_type axis_size = shp[axis];

        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0;
            size_type remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % shp[d];
                remaining /= shp[d];
                prefix_offset += coord * expr.strides()[d];
            }

            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0;
                size_type rem = inner;
                for (size_type d = axis + 1; d < dim; ++d)
                {
                    size_type coord = rem % shp[d];
                    rem /= shp[d];
                    suffix_offset += coord * expr.strides()[d];
                }

                size_type base_offset = prefix_offset + suffix_offset;

                for (size_type i = 0; i < new_axis_size; ++i)
                {
                    size_type offset1 = base_offset + i * axis_stride;
                    size_type offset2 = base_offset + (i + n) * axis_stride;
                    result.flat(base_offset + i * axis_stride) =
                        expr.flat(offset2) - expr.flat(offset1);
                }
            }
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // trapz – trapezoidal numerical integration
    // ------------------------------------------------------------------------
    template <class E>
    inline auto trapz(const xexpression<E>& y, const xexpression<E>& x, size_type axis = 0)
    {
        using value_type = typename E::value_type;
        const auto& y_expr = y.derived_cast();
        const auto& x_expr = x.derived_cast();

        const auto& y_shape = y_expr.shape();
        const auto& x_shape = x_expr.shape();

        if (x_shape.size() != 1)
            XTENSOR_THROW(std::invalid_argument, "trapz: x must be 1‑dimensional");

        if (axis >= y_shape.size())
            XTENSOR_THROW(std::out_of_range, "trapz: axis out of range");

        if (x_shape[0] != y_shape[axis])
            XTENSOR_THROW(std::invalid_argument, "trapz: length of x must match size of y along axis");

        size_type n = y_shape[axis];
        if (n < 2)
            XTENSOR_THROW(std::invalid_argument, "trapz: need at least 2 points along integration axis");

        // Build result shape by removing the integration axis
        shape_type result_shape;
        for (size_type d = 0; d < y_shape.size(); ++d)
        {
            if (d != axis)
                result_shape.push_back(y_shape[d]);
        }

        xarray_container<value_type> result(result_shape, value_type(0));

        // Precompute dx values (half the differences)
        std::vector<value_type> dx(n - 1);
        for (size_type i = 0; i < n - 1; ++i)
        {
            dx[i] = (x_expr.flat(i + 1) - x_expr.flat(i)) / value_type(2);
        }

        // Integration loop
        size_type outer_size = 1;
        for (size_type d = 0; d < axis; ++d)
            outer_size *= y_shape[d];

        size_type inner_size = 1;
        for (size_type d = axis + 1; d < y_shape.size(); ++d)
            inner_size *= y_shape[d];

        size_type axis_stride = y_expr.strides()[axis];

        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0;
            size_type remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % y_shape[d];
                remaining /= y_shape[d];
                prefix_offset += coord * y_expr.strides()[d];
            }

            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0;
                size_type rem = inner;
                for (size_type d = axis + 1; d < y_shape.size(); ++d)
                {
                    size_type coord = rem % y_shape[d];
                    rem /= y_shape[d];
                    suffix_offset += coord * y_expr.strides()[d];
                }

                size_type base_offset = prefix_offset + suffix_offset;
                size_type result_offset = prefix_offset + suffix_offset; // same indexing without axis

                value_type sum = value_type(0);
                for (size_type i = 0; i < n - 1; ++i)
                {
                    size_type offset1 = base_offset + i * axis_stride;
                    size_type offset2 = base_offset + (i + 1) * axis_stride;
                    sum = sum + (y_expr.flat(offset1) + y_expr.flat(offset2)) * dx[i];
                }
                result.flat(result_offset) = sum;
            }
        }

        return result;
    }

    template <class E>
    inline auto trapz(const xexpression<E>& y, value_type dx = value_type(1), size_type axis = 0)
    {
        using value_type = typename E::value_type;
        const auto& y_expr = y.derived_cast();
        const auto& y_shape = y_expr.shape();

        if (axis >= y_shape.size())
            XTENSOR_THROW(std::out_of_range, "trapz: axis out of range");

        size_type n = y_shape[axis];
        if (n < 2)
            XTENSOR_THROW(std::invalid_argument, "trapz: need at least 2 points along integration axis");

        // Build result shape
        shape_type result_shape;
        for (size_type d = 0; d < y_shape.size(); ++d)
        {
            if (d != axis)
                result_shape.push_back(y_shape[d]);
        }

        xarray_container<value_type> result(result_shape, value_type(0));

        value_type half_dx = dx / value_type(2);
        value_type first_last_factor = half_dx;
        value_type interior_factor = dx;

        size_type outer_size = 1;
        for (size_type d = 0; d < axis; ++d)
            outer_size *= y_shape[d];

        size_type inner_size = 1;
        for (size_type d = axis + 1; d < y_shape.size(); ++d)
            inner_size *= y_shape[d];

        size_type axis_stride = y_expr.strides()[axis];

        for (size_type outer = 0; outer < outer_size; ++outer)
        {
            size_type prefix_offset = 0;
            size_type remaining = outer;
            for (size_type d = 0; d < axis; ++d)
            {
                size_type coord = remaining % y_shape[d];
                remaining /= y_shape[d];
                prefix_offset += coord * y_expr.strides()[d];
            }

            for (size_type inner = 0; inner < inner_size; ++inner)
            {
                size_type suffix_offset = 0;
                size_type rem = inner;
                for (size_type d = axis + 1; d < y_shape.size(); ++d)
                {
                    size_type coord = rem % y_shape[d];
                    rem /= y_shape[d];
                    suffix_offset += coord * y_expr.strides()[d];
                }

                size_type base_offset = prefix_offset + suffix_offset;

                value_type sum = (y_expr.flat(base_offset) + y_expr.flat(base_offset + (n - 1) * axis_stride)) * half_dx;
                for (size_type i = 1; i < n - 1; ++i)
                {
                    sum = sum + y_expr.flat(base_offset + i * axis_stride) * interior_factor;
                }
                result.flat(base_offset) = sum;
            }
        }

        return result;
    }

    // ========================================================================
    // Inner types specialization for xaccumulator
    // ========================================================================
    template <class E, class Accumulator>
    struct xcontainer_inner_types<xaccumulator<E, Accumulator>>
    {
        using temporary_type = xarray_container<typename Accumulator::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

#endif // XTENSOR_XACCUMULATOR_HPP
