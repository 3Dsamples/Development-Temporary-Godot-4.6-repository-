// include/xtu/core/xiterable.hpp
// xtensor-unified - Iterable base classes for expression traversal
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_CORE_XITERABLE_HPP
#define XTU_CORE_XITERABLE_HPP

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/core/xexpression.hpp"

XTU_NAMESPACE_BEGIN

// #############################################################################
// xiterable - CRTP base for expressions that support iteration
// #############################################################################
template <class D>
class xiterable : public xexpression<D> {
public:
    using derived_type = D;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    
    // Iterator types (to be specialized by derived classes)
    using iterator = typename derived_type::iterator;
    using const_iterator = typename derived_type::const_iterator;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // #########################################################################
    // Iteration interface
    // #########################################################################
    iterator begin() noexcept {
        return derived_cast().begin();
    }

    iterator end() noexcept {
        return derived_cast().end();
    }

    const_iterator begin() const noexcept {
        return derived_cast().begin();
    }

    const_iterator end() const noexcept {
        return derived_cast().end();
    }

    const_iterator cbegin() const noexcept {
        return derived_cast().cbegin();
    }

    const_iterator cend() const noexcept {
        return derived_cast().cend();
    }

    reverse_iterator rbegin() noexcept {
        return reverse_iterator(end());
    }

    reverse_iterator rend() noexcept {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(cend());
    }

    const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(cbegin());
    }

    // #########################################################################
    // Access to underlying derived type
    // #########################################################################
    derived_type& derived_cast() & noexcept {
        return *static_cast<derived_type*>(this);
    }

    const derived_type& derived_cast() const & noexcept {
        return *static_cast<const derived_type*>(this);
    }

    derived_type derived_cast() && noexcept {
        return std::move(*static_cast<derived_type*>(this));
    }

protected:
    xiterable() = default;
    ~xiterable() = default;
};

// #############################################################################
// xconst_iterable - For expressions that only provide const iteration
// #############################################################################
template <class D>
class xconst_iterable : public xexpression<D> {
public:
    using derived_type = D;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    
    using const_iterator = typename derived_type::const_iterator;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    const_iterator begin() const noexcept {
        return derived_cast().begin();
    }

    const_iterator end() const noexcept {
        return derived_cast().end();
    }

    const_iterator cbegin() const noexcept {
        return derived_cast().cbegin();
    }

    const_iterator cend() const noexcept {
        return derived_cast().cend();
    }

    const_reverse_iterator rbegin() const noexcept {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator rend() const noexcept {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crbegin() const noexcept {
        return const_reverse_iterator(cend());
    }

    const_reverse_iterator crend() const noexcept {
        return const_reverse_iterator(cbegin());
    }

    derived_type& derived_cast() & noexcept {
        return *static_cast<derived_type*>(this);
    }

    const derived_type& derived_cast() const & noexcept {
        return *static_cast<const derived_type*>(this);
    }

    derived_type derived_cast() && noexcept {
        return std::move(*static_cast<derived_type*>(this));
    }

protected:
    xconst_iterable() = default;
    ~xconst_iterable() = default;
};

// #############################################################################
// Stepper iterators for strided traversal (used by xview, xstrided_view)
// #############################################################################
template <class CT>
class xstepper {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::reference;
    using pointer = typename CT::pointer;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using size_type = xtu::size_type;

private:
    CT* m_container;
    std::vector<size_type> m_index;
    size_type m_linear_index;

public:
    xstepper(CT* cont, size_type linear_idx)
        : m_container(cont), m_linear_index(linear_idx) {
        if (m_container) {
            m_index.resize(m_container->dimension());
            unravel_index(linear_idx);
        }
    }

    xstepper(CT* cont, const std::vector<size_type>& idx)
        : m_container(cont), m_index(idx) {
        if (m_container) {
            m_linear_index = ravel_index();
        }
    }

    reference operator*() const {
        return (*m_container)(m_index);
    }

    pointer operator->() const {
        return &(operator*());
    }

    xstepper& operator++() {
        ++m_linear_index;
        increment_index();
        return *this;
    }

    xstepper operator++(int) {
        xstepper tmp = *this;
        ++(*this);
        return tmp;
    }

    xstepper& operator--() {
        --m_linear_index;
        decrement_index();
        return *this;
    }

    xstepper operator--(int) {
        xstepper tmp = *this;
        --(*this);
        return tmp;
    }

    xstepper& operator+=(difference_type n) {
        m_linear_index += static_cast<size_type>(n);
        unravel_index(m_linear_index);
        return *this;
    }

    xstepper& operator-=(difference_type n) {
        m_linear_index -= static_cast<size_type>(n);
        unravel_index(m_linear_index);
        return *this;
    }

    xstepper operator+(difference_type n) const {
        xstepper tmp = *this;
        tmp += n;
        return tmp;
    }

    xstepper operator-(difference_type n) const {
        xstepper tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const xstepper& other) const {
        return static_cast<difference_type>(m_linear_index - other.m_linear_index);
    }

    bool operator==(const xstepper& other) const {
        return m_linear_index == other.m_linear_index;
    }

    bool operator!=(const xstepper& other) const {
        return !(*this == other);
    }

    bool operator<(const xstepper& other) const {
        return m_linear_index < other.m_linear_index;
    }

    bool operator<=(const xstepper& other) const {
        return m_linear_index <= other.m_linear_index;
    }

    bool operator>(const xstepper& other) const {
        return m_linear_index > other.m_linear_index;
    }

    bool operator>=(const xstepper& other) const {
        return m_linear_index >= other.m_linear_index;
    }

    reference operator[](difference_type n) const {
        return *(*this + n);
    }

    size_type linear_index() const noexcept {
        return m_linear_index;
    }

    const std::vector<size_type>& index() const noexcept {
        return m_index;
    }

private:
    void unravel_index(size_type linear) {
        const auto& shp = m_container->shape();
        size_type temp = linear;
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            m_index[static_cast<size_t>(i)] = temp % shp[static_cast<size_t>(i)];
            temp /= shp[static_cast<size_t>(i)];
        }
    }

    size_type ravel_index() const {
        const auto& strd = m_container->strides();
        size_type result = 0;
        for (size_t i = 0; i < m_index.size(); ++i) {
            result += m_index[i] * strd[i];
        }
        return result;
    }

    void increment_index() {
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            ++m_index[static_cast<size_t>(i)];
            if (m_index[static_cast<size_t>(i)] < m_container->shape()[static_cast<size_t>(i)]) {
                break;
            }
            m_index[static_cast<size_t>(i)] = 0;
        }
    }

    void decrement_index() {
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            if (m_index[static_cast<size_t>(i)] > 0) {
                --m_index[static_cast<size_t>(i)];
                break;
            }
            m_index[static_cast<size_t>(i)] = m_container->shape()[static_cast<size_t>(i)] - 1;
        }
    }
};

// #############################################################################
// Linear iterator for contiguous storage (used by xarray_container)
// #############################################################################
template <class CT>
class xlinear_iterator {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::reference;
    using pointer = typename CT::pointer;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using size_type = xtu::size_type;

private:
    CT* m_container;
    size_type m_index;

public:
    xlinear_iterator(CT* cont, size_type idx) : m_container(cont), m_index(idx) {}

    reference operator*() const {
        return (*m_container)[m_index];
    }

    pointer operator->() const {
        return &(operator*());
    }

    xlinear_iterator& operator++() {
        ++m_index;
        return *this;
    }

    xlinear_iterator operator++(int) {
        xlinear_iterator tmp = *this;
        ++m_index;
        return tmp;
    }

    xlinear_iterator& operator--() {
        --m_index;
        return *this;
    }

    xlinear_iterator operator--(int) {
        xlinear_iterator tmp = *this;
        --m_index;
        return tmp;
    }

    xlinear_iterator& operator+=(difference_type n) {
        m_index += static_cast<size_type>(n);
        return *this;
    }

    xlinear_iterator& operator-=(difference_type n) {
        m_index -= static_cast<size_type>(n);
        return *this;
    }

    xlinear_iterator operator+(difference_type n) const {
        xlinear_iterator tmp = *this;
        tmp += n;
        return tmp;
    }

    xlinear_iterator operator-(difference_type n) const {
        xlinear_iterator tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const xlinear_iterator& other) const {
        return static_cast<difference_type>(m_index - other.m_index);
    }

    bool operator==(const xlinear_iterator& other) const {
        return m_index == other.m_index;
    }

    bool operator!=(const xlinear_iterator& other) const {
        return !(*this == other);
    }

    bool operator<(const xlinear_iterator& other) const {
        return m_index < other.m_index;
    }

    bool operator<=(const xlinear_iterator& other) const {
        return m_index <= other.m_index;
    }

    bool operator>(const xlinear_iterator& other) const {
        return m_index > other.m_index;
    }

    bool operator>=(const xlinear_iterator& other) const {
        return m_index >= other.m_index;
    }

    reference operator[](difference_type n) const {
        return (*m_container)[static_cast<size_type>(m_index + n)];
    }

    size_type index() const noexcept {
        return m_index;
    }
};

// #############################################################################
// Const versions of iterators
// #############################################################################
template <class CT>
class xconst_stepper {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::const_reference;
    using pointer = typename CT::const_pointer;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using size_type = xtu::size_type;

private:
    const CT* m_container;
    std::vector<size_type> m_index;
    size_type m_linear_index;

public:
    xconst_stepper(const CT* cont, size_type linear_idx)
        : m_container(cont), m_linear_index(linear_idx) {
        if (m_container) {
            m_index.resize(m_container->dimension());
            unravel_index(linear_idx);
        }
    }

    xconst_stepper(const CT* cont, const std::vector<size_type>& idx)
        : m_container(cont), m_index(idx) {
        if (m_container) {
            m_linear_index = ravel_index();
        }
    }

    reference operator*() const {
        return (*m_container)(m_index);
    }

    pointer operator->() const {
        return &(operator*());
    }

    xconst_stepper& operator++() {
        ++m_linear_index;
        increment_index();
        return *this;
    }

    xconst_stepper operator++(int) {
        xconst_stepper tmp = *this;
        ++(*this);
        return tmp;
    }

    xconst_stepper& operator--() {
        --m_linear_index;
        decrement_index();
        return *this;
    }

    xconst_stepper operator--(int) {
        xconst_stepper tmp = *this;
        --(*this);
        return tmp;
    }

    xconst_stepper& operator+=(difference_type n) {
        m_linear_index += static_cast<size_type>(n);
        unravel_index(m_linear_index);
        return *this;
    }

    xconst_stepper& operator-=(difference_type n) {
        m_linear_index -= static_cast<size_type>(n);
        unravel_index(m_linear_index);
        return *this;
    }

    xconst_stepper operator+(difference_type n) const {
        xconst_stepper tmp = *this;
        tmp += n;
        return tmp;
    }

    xconst_stepper operator-(difference_type n) const {
        xconst_stepper tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const xconst_stepper& other) const {
        return static_cast<difference_type>(m_linear_index - other.m_linear_index);
    }

    bool operator==(const xconst_stepper& other) const {
        return m_linear_index == other.m_linear_index;
    }

    bool operator!=(const xconst_stepper& other) const {
        return !(*this == other);
    }

    bool operator<(const xconst_stepper& other) const {
        return m_linear_index < other.m_linear_index;
    }

    bool operator<=(const xconst_stepper& other) const {
        return m_linear_index <= other.m_linear_index;
    }

    bool operator>(const xconst_stepper& other) const {
        return m_linear_index > other.m_linear_index;
    }

    bool operator>=(const xconst_stepper& other) const {
        return m_linear_index >= other.m_linear_index;
    }

    reference operator[](difference_type n) const {
        return *(*this + n);
    }

    size_type linear_index() const noexcept {
        return m_linear_index;
    }

    const std::vector<size_type>& index() const noexcept {
        return m_index;
    }

private:
    void unravel_index(size_type linear) {
        const auto& shp = m_container->shape();
        size_type temp = linear;
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            m_index[static_cast<size_t>(i)] = temp % shp[static_cast<size_t>(i)];
            temp /= shp[static_cast<size_t>(i)];
        }
    }

    size_type ravel_index() const {
        const auto& strd = m_container->strides();
        size_type result = 0;
        for (size_t i = 0; i < m_index.size(); ++i) {
            result += m_index[i] * strd[i];
        }
        return result;
    }

    void increment_index() {
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            ++m_index[static_cast<size_t>(i)];
            if (m_index[static_cast<size_t>(i)] < m_container->shape()[static_cast<size_t>(i)]) {
                break;
            }
            m_index[static_cast<size_t>(i)] = 0;
        }
    }

    void decrement_index() {
        for (int i = static_cast<int>(m_index.size()) - 1; i >= 0; --i) {
            if (m_index[static_cast<size_t>(i)] > 0) {
                --m_index[static_cast<size_t>(i)];
                break;
            }
            m_index[static_cast<size_t>(i)] = m_container->shape()[static_cast<size_t>(i)] - 1;
        }
    }
};

template <class CT>
class xconst_linear_iterator {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::const_reference;
    using pointer = typename CT::const_pointer;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    using size_type = xtu::size_type;

private:
    const CT* m_container;
    size_type m_index;

public:
    xconst_linear_iterator(const CT* cont, size_type idx) : m_container(cont), m_index(idx) {}

    reference operator*() const {
        return (*m_container)[m_index];
    }

    pointer operator->() const {
        return &(operator*());
    }

    xconst_linear_iterator& operator++() {
        ++m_index;
        return *this;
    }

    xconst_linear_iterator operator++(int) {
        xconst_linear_iterator tmp = *this;
        ++m_index;
        return tmp;
    }

    xconst_linear_iterator& operator--() {
        --m_index;
        return *this;
    }

    xconst_linear_iterator operator--(int) {
        xconst_linear_iterator tmp = *this;
        --m_index;
        return tmp;
    }

    xconst_linear_iterator& operator+=(difference_type n) {
        m_index += static_cast<size_type>(n);
        return *this;
    }

    xconst_linear_iterator& operator-=(difference_type n) {
        m_index -= static_cast<size_type>(n);
        return *this;
    }

    xconst_linear_iterator operator+(difference_type n) const {
        xconst_linear_iterator tmp = *this;
        tmp += n;
        return tmp;
    }

    xconst_linear_iterator operator-(difference_type n) const {
        xconst_linear_iterator tmp = *this;
        tmp -= n;
        return tmp;
    }

    difference_type operator-(const xconst_linear_iterator& other) const {
        return static_cast<difference_type>(m_index - other.m_index);
    }

    bool operator==(const xconst_linear_iterator& other) const {
        return m_index == other.m_index;
    }

    bool operator!=(const xconst_linear_iterator& other) const {
        return !(*this == other);
    }

    bool operator<(const xconst_linear_iterator& other) const {
        return m_index < other.m_index;
    }

    bool operator<=(const xconst_linear_iterator& other) const {
        return m_index <= other.m_index;
    }

    bool operator>(const xconst_linear_iterator& other) const {
        return m_index > other.m_index;
    }

    bool operator>=(const xconst_linear_iterator& other) const {
        return m_index >= other.m_index;
    }

    reference operator[](difference_type n) const {
        return (*m_container)[static_cast<size_type>(m_index + n)];
    }

    size_type index() const noexcept {
        return m_index;
    }
};

XTU_NAMESPACE_END

#endif // XTU_CORE_XITERABLE_HPP