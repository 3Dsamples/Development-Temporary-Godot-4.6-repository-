// include/xtu/core/xscalar.hpp
// xtensor-unified - Scalar wrapper for lazy expression system
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_CORE_XSCALAR_HPP
#define XTU_CORE_XSCALAR_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/core/xexpression.hpp"

XTU_NAMESPACE_BEGIN

// #############################################################################
// xscalar - Expression representing a scalar value (0-dimensional)
// #############################################################################
template <class T>
class xscalar : public xexpression<xscalar<T>> {
public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    using shape_type = std::vector<size_type>;
    using strides_type = std::vector<size_type>;
    using expression_tag = xarray_expression_tag;

private:
    value_type m_value;

public:
    // Constructors
    xscalar() = default;
    
    explicit xscalar(const value_type& value) : m_value(value) {}
    explicit xscalar(value_type&& value) : m_value(std::move(value)) {}
    
    // Copy / move
    xscalar(const xscalar&) = default;
    xscalar(xscalar&&) noexcept = default;
    xscalar& operator=(const xscalar&) = default;
    xscalar& operator=(xscalar&&) noexcept = default;
    
    // Assignment from value
    xscalar& operator=(const value_type& value) {
        m_value = value;
        return *this;
    }
    
    xscalar& operator=(value_type&& value) {
        m_value = std::move(value);
        return *this;
    }
    
    // Dimension and shape access (scalar has dimension 0)
    size_type dimension() const noexcept {
        return 0;
    }

    const shape_type& shape() const noexcept {
        static shape_type empty_shape;
        return empty_shape;
    }

    const strides_type& strides() const noexcept {
        static strides_type empty_strides;
        return empty_strides;
    }

    layout_type layout() const noexcept {
        return layout_type::any;
    }

    size_type size() const noexcept {
        return 1;
    }

    // Data access
    pointer data() noexcept {
        return &m_value;
    }

    const_pointer data() const noexcept {
        return &m_value;
    }

    // Element access (scalar can be accessed with any number of indices; all must be 0)
    reference operator()() {
        return m_value;
    }

    const_reference operator()() const {
        return m_value;
    }

    template <class... Idx>
    reference operator()(Idx... idxs) {
        // For scalar, all indices must be 0
        static_assert(sizeof...(Idx) > 0, "Use operator()() for scalar access");
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        for (size_type i : indices) {
            XTU_ASSERT_MSG(i == 0, "Scalar index out of bounds (must be 0)");
        }
        return m_value;
    }

    template <class... Idx>
    const_reference operator()(Idx... idxs) const {
        static_assert(sizeof...(Idx) > 0, "Use operator()() for scalar access");
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        for (size_type i : indices) {
            XTU_ASSERT_MSG(i == 0, "Scalar index out of bounds (must be 0)");
        }
        return m_value;
    }

    reference flat(size_type i) {
        XTU_ASSERT_MSG(i == 0, "Scalar flat index out of bounds");
        return m_value;
    }

    const_reference flat(size_type i) const {
        XTU_ASSERT_MSG(i == 0, "Scalar flat index out of bounds");
        return m_value;
    }

    reference operator[](size_type i) {
        return flat(i);
    }

    const_reference operator[](size_type i) const {
        return flat(i);
    }

    // Iterator support (scalar has exactly one element)
    class iterator {
    private:
        xscalar* m_scalar;
        bool m_end;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xscalar::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = typename xscalar::reference;

        iterator(xscalar* s, bool end = false) : m_scalar(s), m_end(end) {}

        reference operator*() const {
            XTU_ASSERT_MSG(!m_end, "Cannot dereference end iterator");
            return m_scalar->m_value;
        }

        iterator& operator++() {
            XTU_ASSERT_MSG(!m_end, "Cannot increment end iterator");
            m_end = true;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator& operator--() {
            XTU_ASSERT_MSG(m_end, "Cannot decrement begin iterator");
            m_end = false;
            return *this;
        }

        iterator operator--(int) {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        iterator& operator+=(difference_type n) {
            if (n > 0) {
                XTU_ASSERT_MSG(!m_end && n == 1, "Scalar iterator can only move by 0 or 1");
                m_end = true;
            } else if (n < 0) {
                XTU_ASSERT_MSG(m_end && n == -1, "Scalar iterator can only move by 0 or -1");
                m_end = false;
            }
            return *this;
        }

        iterator& operator-=(difference_type n) {
            return (*this += -n);
        }

        iterator operator+(difference_type n) const {
            iterator tmp = *this;
            tmp += n;
            return tmp;
        }

        iterator operator-(difference_type n) const {
            iterator tmp = *this;
            tmp -= n;
            return tmp;
        }

        difference_type operator-(const iterator& other) const {
            if (m_end == other.m_end) return 0;
            if (m_end) return 1;
            return -1;
        }

        bool operator==(const iterator& other) const {
            return m_scalar == other.m_scalar && m_end == other.m_end;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

        bool operator<(const iterator& other) const {
            return !m_end && other.m_end;
        }

        bool operator<=(const iterator& other) const {
            return !m_end || other.m_end;
        }

        bool operator>(const iterator& other) const {
            return m_end && !other.m_end;
        }

        bool operator>=(const iterator& other) const {
            return m_end || !other.m_end;
        }

        reference operator[](difference_type n) const {
            XTU_ASSERT_MSG(n == 0, "Scalar index out of bounds");
            return m_scalar->m_value;
        }
    };

    class const_iterator {
    private:
        const xscalar* m_scalar;
        bool m_end;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xscalar::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = typename xscalar::const_reference;

        const_iterator(const xscalar* s, bool end = false) : m_scalar(s), m_end(end) {}

        const_reference operator*() const {
            XTU_ASSERT_MSG(!m_end, "Cannot dereference end iterator");
            return m_scalar->m_value;
        }

        const_iterator& operator++() {
            XTU_ASSERT_MSG(!m_end, "Cannot increment end iterator");
            m_end = true;
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        const_iterator& operator--() {
            XTU_ASSERT_MSG(m_end, "Cannot decrement begin iterator");
            m_end = false;
            return *this;
        }

        const_iterator operator--(int) {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        const_iterator& operator+=(difference_type n) {
            if (n > 0) {
                XTU_ASSERT_MSG(!m_end && n == 1, "Scalar iterator can only move by 0 or 1");
                m_end = true;
            } else if (n < 0) {
                XTU_ASSERT_MSG(m_end && n == -1, "Scalar iterator can only move by 0 or -1");
                m_end = false;
            }
            return *this;
        }

        const_iterator& operator-=(difference_type n) {
            return (*this += -n);
        }

        const_iterator operator+(difference_type n) const {
            const_iterator tmp = *this;
            tmp += n;
            return tmp;
        }

        const_iterator operator-(difference_type n) const {
            const_iterator tmp = *this;
            tmp -= n;
            return tmp;
        }

        difference_type operator-(const const_iterator& other) const {
            if (m_end == other.m_end) return 0;
            if (m_end) return 1;
            return -1;
        }

        bool operator==(const const_iterator& other) const {
            return m_scalar == other.m_scalar && m_end == other.m_end;
        }

        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

        bool operator<(const const_iterator& other) const {
            return !m_end && other.m_end;
        }

        bool operator<=(const const_iterator& other) const {
            return !m_end || other.m_end;
        }

        bool operator>(const const_iterator& other) const {
            return m_end && !other.m_end;
        }

        bool operator>=(const const_iterator& other) const {
            return m_end || !other.m_end;
        }

        const_reference operator[](difference_type n) const {
            XTU_ASSERT_MSG(n == 0, "Scalar index out of bounds");
            return m_scalar->m_value;
        }
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(this, false); }
    iterator end() { return iterator(this, true); }
    const_iterator begin() const { return const_iterator(this, false); }
    const_iterator end() const { return const_iterator(this, true); }
    const_iterator cbegin() const { return const_iterator(this, false); }
    const_iterator cend() const { return const_iterator(this, true); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    // Implicit conversion to value_type
    operator value_type() const { return m_value; }
};

// #############################################################################
// Type trait to detect if an expression is a scalar
// #############################################################################
namespace detail {
    template <class E>
    struct is_scalar_expression : std::false_type {};

    template <class T>
    struct is_scalar_expression<xscalar<T>> : std::true_type {};

    template <class E>
    static constexpr bool is_scalar_expression_v = is_scalar_expression<E>::value;
}

// #############################################################################
// Helper function to create scalar expression
// #############################################################################
template <class T>
auto scalar(T&& value) {
    return xscalar<std::decay_t<T>>(std::forward<T>(value));
}

// #############################################################################
// Additional operator overloads for scalar and expressions
// #############################################################################
template <class T, class E, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator+(const T& scalar, const xexpression<E>& e) {
    return xscalar<T>(scalar) + e.derived_cast();
}

template <class T, class E, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator-(const T& scalar, const xexpression<E>& e) {
    return xscalar<T>(scalar) - e.derived_cast();
}

template <class T, class E, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator*(const T& scalar, const xexpression<E>& e) {
    return xscalar<T>(scalar) * e.derived_cast();
}

template <class T, class E, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator/(const T& scalar, const xexpression<E>& e) {
    return xscalar<T>(scalar) / e.derived_cast();
}

// Right-hand side versions (already in xfunction.hpp, but ensure completeness)
template <class E, class T, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator+(const xexpression<E>& e, const T& scalar) {
    return e.derived_cast() + xscalar<T>(scalar);
}

template <class E, class T, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator-(const xexpression<E>& e, const T& scalar) {
    return e.derived_cast() - xscalar<T>(scalar);
}

template <class E, class T, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator*(const xexpression<E>& e, const T& scalar) {
    return e.derived_cast() * xscalar<T>(scalar);
}

template <class E, class T, std::enable_if_t<!std::is_base_of<xexpression<E>, T>::value, int> = 0>
auto operator/(const xexpression<E>& e, const T& scalar) {
    return e.derived_cast() / xscalar<T>(scalar);
}

XTU_NAMESPACE_END

#endif // XTU_CORE_XSCALAR_HPP