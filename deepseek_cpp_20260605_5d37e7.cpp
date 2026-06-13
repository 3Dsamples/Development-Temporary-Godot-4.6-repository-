//File 0072 : iterators/xaxis_iterator.hpp
//Axis-aligned iteration over a single dimension: xaxis_iterator provides linear traversal along a given axis with step semantics.
#ifndef XTENSOR_XAXIS_ITERATOR_HPP
#define XTENSOR_XAXIS_ITERATOR_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xstrides.hpp"

namespace xt
{
    /**
     * @class xaxis_iterator
     * @brief Random-access iterator stepping along a specified axis.
     *
     * Given a multi-dimensional expression, this iterator visits elements
     * sequentially along one axis (dimension), while keeping the coordinates
     * along all other dimensions fixed. This is useful for applying operations
     * along rows, columns, or any chosen dimension.
     */
    template <class E>
    class xaxis_iterator
    {
    public:
        using self_type = xaxis_iterator<E>;
        using expression_type = E;
        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        /**
         * Construct an axis iterator at a given linear offset for a given axis.
         * @param expr Pointer to the expression.
         * @param axis The axis along which to iterate.
         * @param linear_index Linear offset of the first element along the axis.
         */
        xaxis_iterator(expression_type* expr, size_type axis, size_type linear_index) noexcept
            : p_expression(expr), m_axis(axis), m_linear_index(linear_index)
        {
        }

        /**
         * Step forward along the axis.
         */
        self_type& operator++()
        {
            m_linear_index += p_expression->strides()[m_axis];
            return *this;
        }

        /**
         * Step backward along the axis.
         */
        self_type& operator--()
        {
            m_linear_index -= p_expression->strides()[m_axis];
            return *this;
        }

        /**
         * Step by n elements along the axis.
         */
        self_type& operator+=(difference_type n)
        {
            m_linear_index += static_cast<size_type>(n) * p_expression->strides()[m_axis];
            return *this;
        }

        self_type& operator-=(difference_type n)
        {
            m_linear_index -= static_cast<size_type>(n) * p_expression->strides()[m_axis];
            return *this;
        }

        /**
         * Distance between two iterators (in number of steps along the axis).
         */
        difference_type operator-(const self_type& rhs) const
        {
            return static_cast<difference_type>(
                (m_linear_index - rhs.m_linear_index) / p_expression->strides()[m_axis]);
        }

        /**
         * Dereference: returns element at current position.
         */
        reference operator*() const
        {
            return (*p_expression)[m_linear_index];
        }

        pointer operator->() const
        {
            return &(operator*());
        }

        reference operator[](difference_type n) const
        {
            return *(*this + n);
        }

        bool operator==(const self_type& rhs) const
        {
            return p_expression == rhs.p_expression && m_linear_index == rhs.m_linear_index;
        }

        bool operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }

        bool operator<(const self_type& rhs) const
        {
            return m_linear_index < rhs.m_linear_index;
        }

        bool operator<=(const self_type& rhs) const
        {
            return m_linear_index <= rhs.m_linear_index;
        }

        bool operator>(const self_type& rhs) const
        {
            return m_linear_index > rhs.m_linear_index;
        }

        bool operator>=(const self_type& rhs) const
        {
            return m_linear_index >= rhs.m_linear_index;
        }

        /**
         * Reset the iterator to the first element of this axis slice.
         */
        void reset()
        {
            m_linear_index = m_linear_index % p_expression->strides()[m_axis];
        }

        size_type linear_index() const noexcept { return m_linear_index; }
        size_type axis() const noexcept { return m_axis; }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_linear_index;
    };

    /**
     * @class xaxis_const_iterator
     * @brief Const version of xaxis_iterator.
     */
    template <class E>
    class xaxis_const_iterator
    {
    public:
        using self_type = xaxis_const_iterator<E>;
        using expression_type = const E;
        using value_type = typename E::value_type;
        using const_reference = typename E::const_reference;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        xaxis_const_iterator(expression_type* expr, size_type axis, size_type linear_index) noexcept
            : p_expression(expr), m_axis(axis), m_linear_index(linear_index)
        {
        }

        self_type& operator++()
        {
            m_linear_index += p_expression->strides()[m_axis];
            return *this;
        }

        self_type& operator--()
        {
            m_linear_index -= p_expression->strides()[m_axis];
            return *this;
        }

        self_type& operator+=(difference_type n)
        {
            m_linear_index += static_cast<size_type>(n) * p_expression->strides()[m_axis];
            return *this;
        }

        self_type& operator-=(difference_type n)
        {
            m_linear_index -= static_cast<size_type>(n) * p_expression->strides()[m_axis];
            return *this;
        }

        difference_type operator-(const self_type& rhs) const
        {
            return static_cast<difference_type>(
                (m_linear_index - rhs.m_linear_index) / p_expression->strides()[m_axis]);
        }

        const_reference operator*() const
        {
            return (*p_expression)[m_linear_index];
        }

        const_pointer operator->() const
        {
            return &(operator*());
        }

        const_reference operator[](difference_type n) const
        {
            return *(*this + n);
        }

        bool operator==(const self_type& rhs) const
        {
            return p_expression == rhs.p_expression && m_linear_index == rhs.m_linear_index;
        }

        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }
        bool operator<(const self_type& rhs) const { return m_linear_index < rhs.m_linear_index; }
        bool operator<=(const self_type& rhs) const { return m_linear_index <= rhs.m_linear_index; }
        bool operator>(const self_type& rhs) const { return m_linear_index > rhs.m_linear_index; }
        bool operator>=(const self_type& rhs) const { return m_linear_index >= rhs.m_linear_index; }

        void reset()
        {
            m_linear_index = m_linear_index % p_expression->strides()[m_axis];
        }

        size_type linear_index() const noexcept { return m_linear_index; }
        size_type axis() const noexcept { return m_axis; }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_linear_index;
    };

    /**
     * Free functions to create axis iterators.
     */
    template <class E>
    inline auto axis_begin(E& expr, std::size_t axis, std::size_t slice_index = 0)
    {
        // Compute the linear offset for the start of the given slice along the axis.
        auto& shape = expr.shape();
        auto& strides = expr.strides();
        std::size_t ndim = shape.size();
        std::size_t offset = 0;
        std::size_t remaining = slice_index;
        for (std::size_t d = 0; d < ndim; ++d)
        {
            if (d == axis) continue;
            std::size_t dim_size = shape[d];
            std::size_t idx = remaining % dim_size;
            remaining /= dim_size;
            offset += idx * strides[d];
        }
        return xaxis_iterator<E>(&expr, axis, offset);
    }

    template <class E>
    inline auto axis_end(E& expr, std::size_t axis, std::size_t slice_index = 0)
    {
        auto it = axis_begin(expr, axis, slice_index);
        it += expr.shape()[axis];
        return it;
    }

    template <class E>
    inline auto axis_cbegin(const E& expr, std::size_t axis, std::size_t slice_index = 0)
    {
        auto& shape = expr.shape();
        auto& strides = expr.strides();
        std::size_t ndim = shape.size();
        std::size_t offset = 0;
        std::size_t remaining = slice_index;
        for (std::size_t d = 0; d < ndim; ++d)
        {
            if (d == axis) continue;
            std::size_t dim_size = shape[d];
            std::size_t idx = remaining % dim_size;
            remaining /= dim_size;
            offset += idx * strides[d];
        }
        return xaxis_const_iterator<E>(&expr, axis, offset);
    }

    template <class E>
    inline auto axis_cend(const E& expr, std::size_t axis, std::size_t slice_index = 0)
    {
        auto it = axis_cbegin(expr, axis, slice_index);
        it += expr.shape()[axis];
        return it;
    }

} // namespace xt

#endif // XTENSOR_XAXIS_ITERATOR_HPP