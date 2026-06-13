//File 0046 : core/xiterable.hpp
//Iteration support: CRTP mixin for expression iterators, random-access xiterator, axis iterators, and stepper-based traversal.
#ifndef XTENSOR_XITERABLE_HPP
#define XTENSOR_XITERABLE_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xstrides.hpp"

namespace xt
{
    /*********************************************
     * xiterable – CRTP mixin adding iteration
     *********************************************/
    template <class D>
    class xiterable : public xexpression<D>
    {
    public:
        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;

        // Default iterator types (raw pointers for contiguous storage)
        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        // Linear iterators
        iterator begin() noexcept { return derived_cast().data(); }
        iterator end() noexcept { return derived_cast().data() + derived_cast().size(); }
        const_iterator begin() const noexcept { return derived_cast().data(); }
        const_iterator end() const noexcept { return derived_cast().data() + derived_cast().size(); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Reverse iterators
        reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
        reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
        const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
        const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
        const_reverse_iterator crbegin() const noexcept { return rbegin(); }
        const_reverse_iterator crend() const noexcept { return rend(); }

    protected:
        xiterable() = default;
        ~xiterable() = default;
        xiterable(const xiterable&) = default;
        xiterable& operator=(const xiterable&) = default;
        xiterable(xiterable&&) = default;
        xiterable& operator=(xiterable&&) = default;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }
    };

    /*********************************************
     * xiterator – random access iterator for expressions
     *********************************************/
    template <class E>
    class xiterator : public xtl::xrandom_access_iterator_base<
                          xiterator<E>,
                          typename E::value_type,
                          typename E::difference_type,
                          typename E::pointer,
                          typename E::reference>
    {
    public:
        using self_type = xiterator<E>;
        using expression_type = E;
        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using pointer = typename E::pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        using shape_type = typename E::shape_type;

        xiterator() noexcept : p_expression(nullptr), m_linear_index(0) {}

        xiterator(expression_type* expr, size_type linear_index) noexcept
            : p_expression(expr), m_linear_index(linear_index)
        {
        }

        self_type& operator++()
        {
            ++m_linear_index;
            return *this;
        }

        self_type& operator--()
        {
            --m_linear_index;
            return *this;
        }

        self_type& operator+=(difference_type n)
        {
            m_linear_index += static_cast<size_type>(n);
            return *this;
        }

        self_type& operator-=(difference_type n)
        {
            m_linear_index -= static_cast<size_type>(n);
            return *this;
        }

        difference_type operator-(const self_type& rhs) const
        {
            return static_cast<difference_type>(m_linear_index - rhs.m_linear_index);
        }

        reference operator*() const
        {
            auto idx = unravel_index(m_linear_index, p_expression->shape());
            return p_expression->element(idx.begin(), idx.end());
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

    private:
        expression_type* p_expression;
        size_type m_linear_index;
    };

    /*********************************************
     * xconst_iterator – const random access iterator
     *********************************************/
    template <class E>
    class xconst_iterator : public xtl::xrandom_access_iterator_base<
                                xconst_iterator<E>,
                                typename E::value_type,
                                typename E::difference_type,
                                typename E::const_pointer,
                                typename E::const_reference>
    {
    public:
        using self_type = xconst_iterator<E>;
        using expression_type = const E;
        using value_type = typename E::value_type;
        using const_reference = typename E::const_reference;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        xconst_iterator() noexcept : p_expression(nullptr), m_linear_index(0) {}

        xconst_iterator(expression_type* expr, size_type linear_index) noexcept
            : p_expression(expr), m_linear_index(linear_index)
        {
        }

        self_type& operator++() { ++m_linear_index; return *this; }
        self_type& operator--() { --m_linear_index; return *this; }
        self_type& operator+=(difference_type n) { m_linear_index += static_cast<size_type>(n); return *this; }
        self_type& operator-=(difference_type n) { m_linear_index -= static_cast<size_type>(n); return *this; }
        difference_type operator-(const self_type& rhs) const { return static_cast<difference_type>(m_linear_index - rhs.m_linear_index); }

        const_reference operator*() const
        {
            auto idx = unravel_index(m_linear_index, p_expression->shape());
            return p_expression->element(idx.begin(), idx.end());
        }

        const_pointer operator->() const { return &(operator*()); }
        const_reference operator[](difference_type n) const { return *(*this + n); }

        bool operator==(const self_type& rhs) const { return p_expression == rhs.p_expression && m_linear_index == rhs.m_linear_index; }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }
        bool operator<(const self_type& rhs) const { return m_linear_index < rhs.m_linear_index; }
        bool operator<=(const self_type& rhs) const { return m_linear_index <= rhs.m_linear_index; }
        bool operator>(const self_type& rhs) const { return m_linear_index > rhs.m_linear_index; }
        bool operator>=(const self_type& rhs) const { return m_linear_index >= rhs.m_linear_index; }

    private:
        expression_type* p_expression;
        size_type m_linear_index;
    };

    /*********************************************
     * xstepper – multi-dimensional stepping iterator
     *********************************************/
    template <class E>
    class xstepper
    {
    public:
        using expression_type = E;
        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using size_type = typename E::size_type;
        using strides_type = typename E::strides_type;

        xstepper(expression_type* expr, size_type linear_index) noexcept
            : p_expression(expr), m_linear_index(linear_index)
        {
        }

        void step(size_type dim, size_type n = 1)
        {
            m_linear_index += n * p_expression->strides()[dim];
        }

        void step_back(size_type dim, size_type n = 1)
        {
            m_linear_index -= n * p_expression->strides()[dim];
        }

        void reset(size_type dim)
        {
            auto& strides = p_expression->strides();
            auto& shape = p_expression->shape();
            size_type offset = m_linear_index % strides[dim];
            m_linear_index -= offset;
        }

        void to_begin() { m_linear_index = 0; }
        void to_end() { m_linear_index = p_expression->size(); }

        reference operator*()
        {
            return (*p_expression)[m_linear_index];
        }

        const_reference operator*() const
        {
            return (*p_expression)[m_linear_index];
        }

        bool operator==(const xstepper& other) const
        {
            return p_expression == other.p_expression && m_linear_index == other.m_linear_index;
        }

        bool operator!=(const xstepper& other) const
        {
            return !(*this == other);
        }

    private:
        expression_type* p_expression;
        size_type m_linear_index;
    };

    /*********************************************
     * Axis iterators (1D slices along an axis)
     *********************************************/
    template <class E>
    class xaxis_iterator
    {
    public:
        using expression_type = E;
        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using size_type = typename E::size_type;

        xaxis_iterator(expression_type* expr, size_type axis, size_type linear_index) noexcept
            : p_expression(expr), m_axis(axis), m_linear_index(linear_index)
        {
        }

        void step(size_type n = 1)
        {
            m_linear_index += n * p_expression->strides()[m_axis];
        }

        reference operator*()
        {
            return (*p_expression)[m_linear_index];
        }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_linear_index;
    };

    /*********************************************
     * xaxis_slice_iterator – iterate over all 1D slices
     *********************************************/
    template <class E>
    class xaxis_slice_iterator
    {
    public:
        using self_type = xaxis_slice_iterator<E>;
        using expression_type = E;
        using value_type = typename E::value_type;
        using size_type = typename E::size_type;
        using shape_type = typename E::shape_type;

        xaxis_slice_iterator(expression_type* expr, size_type axis, size_type slice_index)
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

        bool operator==(const self_type& rhs) const
        {
            return p_expression == rhs.p_expression && m_slice_index == rhs.m_slice_index;
        }

        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

        // Return the beginning of the current slice
        xaxis_iterator<E> begin() const
        {
            return xaxis_iterator<E>(p_expression, m_axis, m_start_offset);
        }

        xaxis_iterator<E> end() const
        {
            return xaxis_iterator<E>(p_expression, m_axis, m_start_offset + p_expression->shape()[m_axis] * p_expression->strides()[m_axis]);
        }

    private:
        expression_type* p_expression;
        size_type m_axis;
        size_type m_slice_index;
        size_type m_start_offset;

        void compute_offsets()
        {
            auto shape = p_expression->shape();
            auto strides = p_expression->strides();
            std::size_t ndim = shape.size();
            // Map slice_index to the indices of the other dimensions
            std::size_t remaining = m_slice_index;
            m_start_offset = 0;
            for (std::size_t d = 0; d < ndim; ++d)
            {
                if (d == m_axis) continue;
                std::size_t dim_size = shape[d];
                std::size_t idx = remaining % dim_size;
                remaining /= dim_size;
                m_start_offset += idx * strides[d];
            }
        }
    };

} // namespace xt

#endif // XTENSOR_XITERABLE_HPP