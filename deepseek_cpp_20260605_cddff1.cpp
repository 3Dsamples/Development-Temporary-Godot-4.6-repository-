//File 0073 : iterators/xaxis_slice_iterator.hpp
//Iterator over all 1D slices along a specified axis, providing begin/end axis iterators for each slice, supporting nested loops and SIMD-aligned traversal.
#ifndef XTENSOR_XAXIS_SLICE_ITERATOR_HPP
#define XTENSOR_XAXIS_SLICE_ITERATOR_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xstrides.hpp"
#include "xaxis_iterator.hpp"

namespace xt
{
    /**
     * @class xaxis_slice_iterator
     * @brief Forward iterator over all 1D slices along a given axis.
     *
     * Each dereference returns a pair of axis iterators (begin, end) that span
     * the current slice. Incrementing the slice iterator advances to the next
     * slice in row-major order of the remaining dimensions.
     */
    template <class E>
    class xaxis_slice_iterator
    {
    public:
        using self_type = xaxis_slice_iterator<E>;
        using expression_type = E;
        using value_type = std::pair<xaxis_iterator<E>, xaxis_iterator<E>>;
        using reference = value_type;
        using const_reference = std::pair<xaxis_const_iterator<E>, xaxis_const_iterator<E>>;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        using iterator_category = std::forward_iterator_tag;

        /**
         * Construct a slice iterator.
         * @param expr Pointer to the expression.
         * @param axis The axis along which slices are taken.
         * @param slice_index Linear index of the current slice (0 = first slice).
         */
        xaxis_slice_iterator(expression_type* expr, size_type axis, size_type slice_index = 0) noexcept
            : p_expression(expr), m_axis(axis), m_slice_index(slice_index)
        {
            compute_offsets();
        }

        /**
         * Advance to the next slice.
         */
        self_type& operator++()
        {
            ++m_slice_index;
            compute_offsets();
            return *this;
        }

        self_type operator++(int)
        {
            self_type tmp(*this);
            ++(*this);
            return tmp;
        }

        /**
         * Return the begin/end axis iterators for the current slice.
         */
        reference operator*() const
        {
            return std::make_pair(
                xaxis_iterator<E>(p_expression, m_axis, m_slice_start_offset),
                xaxis_iterator<E>(p_expression, m_axis,
                                 m_slice_start_offset + p_expression->shape()[m_axis] * p_expression->strides()[m_axis]));
        }

        const_reference operator*() const
        {
            // Const version (if expression is const)
            return std::make_pair(
                xaxis_const_iterator<E>(static_cast<const E*>(p_expression), m_axis, m_slice_start_offset),
                xaxis_const_iterator<E>(static_cast<const E*>(p_expression), m_axis,
                                        m_slice_start_offset + p_expression->shape()[m_axis] * p_expression->strides()[m_axis]));
        }

        bool operator==(const self_type& rhs) const
        {
            return p_expression == rhs.p_expression && m_axis == rhs.m_axis && m_slice_index == rhs.m_slice_index;
        }

        bool operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }

        size_type slice_index() const noexcept { return m_slice_index; }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_slice_index;
        size_type m_slice_start_offset;

        /**
         * Compute the linear offset for the start of the current slice.
         * The slice index is interpreted as a flat index over all dimensions
         * except the slice axis, in row-major order.
         */
        void compute_offsets()
        {
            auto& shape = p_expression->shape();
            auto& strides = p_expression->strides();
            std::size_t ndim = shape.size();

            // Decompose slice_index into indices for all dimensions except m_axis.
            std::size_t remaining = m_slice_index;
            m_slice_start_offset = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis) continue;
                std::size_t dim_size = shape[d];
                std::size_t idx = remaining % dim_size;
                remaining /= dim_size;
                m_slice_start_offset += idx * strides[d];
            }
        }
    };

    /**
     * @class xaxis_slice_const_iterator
     * @brief Const version of xaxis_slice_iterator.
     */
    template <class E>
    class xaxis_slice_const_iterator
    {
    public:
        using self_type = xaxis_slice_const_iterator<E>;
        using expression_type = const E;
        using value_type = std::pair<xaxis_const_iterator<E>, xaxis_const_iterator<E>>;
        using reference = value_type;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        using iterator_category = std::forward_iterator_tag;

        xaxis_slice_const_iterator(expression_type* expr, size_type axis, size_type slice_index = 0) noexcept
            : p_expression(expr), m_axis(axis), m_slice_index(slice_index)
        {
            compute_offsets();
        }

        self_type& operator++()
        {
            ++m_slice_index;
            compute_offsets();
            return *this;
        }

        self_type operator++(int)
        {
            self_type tmp(*this);
            ++(*this);
            return tmp;
        }

        reference operator*() const
        {
            return std::make_pair(
                xaxis_const_iterator<E>(p_expression, m_axis, m_slice_start_offset),
                xaxis_const_iterator<E>(p_expression, m_axis,
                                        m_slice_start_offset + p_expression->shape()[m_axis] * p_expression->strides()[m_axis]));
        }

        bool operator==(const self_type& rhs) const
        {
            return p_expression == rhs.p_expression && m_axis == rhs.m_axis && m_slice_index == rhs.m_slice_index;
        }

        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

        size_type slice_index() const noexcept { return m_slice_index; }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_slice_index;
        size_type m_slice_start_offset;

        void compute_offsets()
        {
            auto& shape = p_expression->shape();
            auto& strides = p_expression->strides();
            std::size_t ndim = shape.size();
            std::size_t remaining = m_slice_index;
            m_slice_start_offset = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis) continue;
                std::size_t dim_size = shape[d];
                std::size_t idx = remaining % dim_size;
                remaining /= dim_size;
                m_slice_start_offset += idx * strides[d];
            }
        }
    };

    /**
     * Free functions to create slice iterators.
     */
    template <class E>
    inline auto axis_slice_begin(E& expr, std::size_t axis)
    {
        return xaxis_slice_iterator<E>(&expr, axis, 0);
    }

    template <class E>
    inline auto axis_slice_end(E& expr, std::size_t axis)
    {
        auto& shape = expr.shape();
        std::size_t total_slices = 1;
        for (std::size_t d = 0; d < shape.size(); ++d)
            if (d != axis) total_slices *= shape[d];
        return xaxis_slice_iterator<E>(&expr, axis, total_slices);
    }

    template <class E>
    inline auto axis_slice_cbegin(const E& expr, std::size_t axis)
    {
        return xaxis_slice_const_iterator<E>(&expr, axis, 0);
    }

    template <class E>
    inline auto axis_slice_cend(const E& expr, std::size_t axis)
    {
        auto& shape = expr.shape();
        std::size_t total_slices = 1;
        for (std::size_t d = 0; d < shape.size(); ++d)
            if (d != axis) total_slices *= shape[d];
        return xaxis_slice_const_iterator<E>(&expr, axis, total_slices);
    }

} // namespace xt

#endif // XTENSOR_XAXIS_SLICE_ITERATOR_HPP