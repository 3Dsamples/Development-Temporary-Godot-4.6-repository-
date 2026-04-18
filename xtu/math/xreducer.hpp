// include/xtu/math/xreducer.hpp
// xtensor-unified - Lazy reduction expressions (sum, mean, min, max, etc.)
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_MATH_XREDUCER_HPP
#define XTU_MATH_XREDUCER_HPP

#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/core/xexpression.hpp"
#include "xtu/views/xview.hpp"

XTU_NAMESPACE_BEGIN
namespace math {

// #############################################################################
// Reduction functors
// #############################################################################
namespace functor {
    template <class T>
    struct sum {
        using result_type = T;
        XTU_ALWAYS_INLINE
        T operator()(const T& a, const T& b) const { return a + b; }
        static constexpr T identity() { return T(0); }
    };

    template <class T>
    struct prod {
        using result_type = T;
        XTU_ALWAYS_INLINE
        T operator()(const T& a, const T& b) const { return a * b; }
        static constexpr T identity() { return T(1); }
    };

    template <class T>
    struct max {
        using result_type = T;
        XTU_ALWAYS_INLINE
        T operator()(const T& a, const T& b) const { return (a > b) ? a : b; }
        static T identity() { return std::numeric_limits<T>::lowest(); }
    };

    template <class T>
    struct min {
        using result_type = T;
        XTU_ALWAYS_INLINE
        T operator()(const T& a, const T& b) const { return (a < b) ? a : b; }
        static T identity() { return std::numeric_limits<T>::max(); }
    };

    template <class T>
    struct mean {
        using result_type = T;
        // Not a binary op; handled separately
    };

    template <class T>
    struct stddev {
        using result_type = T;
        // Not a binary op; handled separately
    };

    template <class T>
    struct variance {
        using result_type = T;
        // Not a binary op; handled separately
    };

    template <class T>
    struct all {
        using result_type = bool;
        XTU_ALWAYS_INLINE
        bool operator()(bool a, bool b) const { return a && b; }
        static constexpr bool identity() { return true; }
    };

    template <class T>
    struct any {
        using result_type = bool;
        XTU_ALWAYS_INLINE
        bool operator()(bool a, bool b) const { return a || b; }
        static constexpr bool identity() { return false; }
    };
}

// #############################################################################
// xreducer - Lazy reduction expression
// #############################################################################
template <class F, class E, class X>
class xreducer : public xexpression<xreducer<F, E, X>> {
public:
    using functor_type = F;
    using value_type = typename F::result_type;
    using reference = value_type;
    using const_reference = value_type;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    using shape_type = std::vector<size_type>;
    using strides_type = std::vector<size_type>;
    using expression_tag = typename E::expression_tag;

private:
    E m_expression;
    functor_type m_functor;
    std::vector<size_type> m_axes;
    shape_type m_shape;
    strides_type m_strides;
    size_type m_size;

public:
    // Constructor: reduce over specified axes
    xreducer(E&& expr, std::vector<size_type> axes)
        : m_expression(std::forward<E>(expr))
        , m_axes(std::move(axes))
        , m_functor() {
        // Validate axes
        for (size_type ax : m_axes) {
            XTU_ASSERT_MSG(ax < m_expression.dimension(), "Axis out of range");
        }
        // Sort and remove duplicates
        std::sort(m_axes.begin(), m_axes.end());
        m_axes.erase(std::unique(m_axes.begin(), m_axes.end()), m_axes.end());
        // Compute reduced shape
        const auto& src_shape = m_expression.shape();
        for (size_type d = 0; d < m_expression.dimension(); ++d) {
            if (std::find(m_axes.begin(), m_axes.end(), d) == m_axes.end()) {
                m_shape.push_back(src_shape[d]);
            }
        }
        // Compute strides (contiguous, row-major)
        m_strides.resize(m_shape.size());
        if (!m_shape.empty()) {
            m_strides.back() = 1;
            for (int i = static_cast<int>(m_shape.size()) - 2; i >= 0; --i) {
                m_strides[static_cast<size_t>(i)] = m_strides[static_cast<size_t>(i) + 1] * m_shape[static_cast<size_t>(i) + 1];
            }
        }
        // Compute total size
        m_size = 1;
        for (auto s : m_shape) m_size *= s;
    }

    // Dimension and shape access
    size_type dimension() const noexcept {
        return m_shape.size();
    }

    const shape_type& shape() const noexcept {
        return m_shape;
    }

    const strides_type& strides() const noexcept {
        return m_strides;
    }

    layout_type layout() const noexcept {
        return layout_type::row_major;
    }

    size_type size() const noexcept {
        return m_size;
    }

    // Access to underlying expression
    const E& expression() const noexcept {
        return m_expression;
    }

    // Reduction evaluation (eager, but returns value)
    value_type operator[](size_type i) const {
        return flat(i);
    }

    value_type flat(size_type i) const {
        // Map linear index to multi-dimensional index in reduced space
        std::vector<size_type> reduced_idx(dimension());
        size_type temp = i;
        for (int d = static_cast<int>(dimension()) - 1; d >= 0; --d) {
            reduced_idx[static_cast<size_t>(d)] = temp % m_shape[static_cast<size_t>(d)];
            temp /= m_shape[static_cast<size_t>(d)];
        }
        // Map reduced index to source index range and perform reduction
        return reduce_over_axes(reduced_idx);
    }

    template <class... Idx>
    value_type operator()(Idx... idxs) const {
        std::vector<size_type> reduced_idx{static_cast<size_type>(idxs)...};
        XTU_ASSERT_MSG(reduced_idx.size() == dimension(), "Number of indices must match reduced dimension");
        return reduce_over_axes(reduced_idx);
    }

    // Iterator support (read-only)
    class const_iterator {
    private:
        const xreducer* m_reducer;
        size_type m_index;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xreducer::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;

        const_iterator(const xreducer* r, size_type idx) : m_reducer(r), m_index(idx) {}

        value_type operator*() const { return (*m_reducer)[m_index]; }
        
        const_iterator& operator++() { ++m_index; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++m_index; return tmp; }
        const_iterator& operator--() { --m_index; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --m_index; return tmp; }
        const_iterator& operator+=(difference_type n) { m_index += n; return *this; }
        const_iterator& operator-=(difference_type n) { m_index -= n; return *this; }
        
        const_iterator operator+(difference_type n) const { return const_iterator(m_reducer, m_index + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(m_reducer, m_index - n); }
        difference_type operator-(const const_iterator& other) const { return m_index - other.m_index; }
        
        bool operator==(const const_iterator& other) const { return m_index == other.m_index; }
        bool operator!=(const const_iterator& other) const { return m_index != other.m_index; }
        bool operator<(const const_iterator& other) const { return m_index < other.m_index; }
        bool operator<=(const const_iterator& other) const { return m_index <= other.m_index; }
        bool operator>(const const_iterator& other) const { return m_index > other.m_index; }
        bool operator>=(const const_iterator& other) const { return m_index >= other.m_index; }
        
        value_type operator[](difference_type n) const { return (*m_reducer)[m_index + n]; }
    };

    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, m_size); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

private:
    value_type reduce_over_axes(const std::vector<size_type>& reduced_idx) const {
        // Build mapping from reduced index to full index range over reduction axes
        const auto& src_shape = m_expression.shape();
        std::vector<size_type> full_idx(m_expression.dimension());
        size_type reduced_pos = 0;
        for (size_type d = 0; d < m_expression.dimension(); ++d) {
            if (std::find(m_axes.begin(), m_axes.end(), d) == m_axes.end()) {
                full_idx[d] = reduced_idx[reduced_pos++];
            } else {
                full_idx[d] = 0;
            }
        }
        // Recursively iterate over reduction axes and accumulate
        return accumulate(full_idx, 0);
    }

    value_type accumulate(std::vector<size_type>& idx, size_type axis_pos) const {
        if (axis_pos == m_axes.size()) {
            // Compute linear index and get value
            size_type linear = 0;
            size_type stride = 1;
            const auto& src_strides = m_expression.strides();
            const auto& src_shape = m_expression.shape();
            for (int d = static_cast<int>(m_expression.dimension()) - 1; d >= 0; --d) {
                linear += idx[static_cast<size_t>(d)] * src_strides[static_cast<size_t>(d)];
            }
            return m_expression.flat(linear);
        }
        size_type ax = m_axes[axis_pos];
        value_type result = functor_type::identity();
        for (size_type i = 0; i < m_expression.shape()[ax]; ++i) {
            idx[ax] = i;
            value_type val = accumulate(idx, axis_pos + 1);
            result = m_functor(result, val);
        }
        return result;
    }
};

// #############################################################################
// Special reducer for mean (requires count)
// #############################################################################
template <class E>
class xmean_reducer : public xexpression<xmean_reducer<E>> {
public:
    using value_type = typename E::value_type;
    using size_type = xtu::size_type;
    using shape_type = std::vector<size_type>;
    using strides_type = std::vector<size_type>;

private:
    E m_expression;
    std::vector<size_type> m_axes;
    shape_type m_shape;
    strides_type m_strides;
    size_type m_size;
    size_type m_reduced_size;

public:
    xmean_reducer(E&& expr, std::vector<size_type> axes)
        : m_expression(std::forward<E>(expr))
        , m_axes(std::move(axes)) {
        // Validate axes
        for (size_type ax : m_axes) {
            XTU_ASSERT_MSG(ax < m_expression.dimension(), "Axis out of range");
        }
        std::sort(m_axes.begin(), m_axes.end());
        m_axes.erase(std::unique(m_axes.begin(), m_axes.end()), m_axes.end());
        // Compute reduced shape
        const auto& src_shape = m_expression.shape();
        m_reduced_size = 1;
        for (size_type ax : m_axes) m_reduced_size *= src_shape[ax];
        for (size_type d = 0; d < m_expression.dimension(); ++d) {
            if (std::find(m_axes.begin(), m_axes.end(), d) == m_axes.end()) {
                m_shape.push_back(src_shape[d]);
            }
        }
        // Compute strides
        m_strides.resize(m_shape.size());
        if (!m_shape.empty()) {
            m_strides.back() = 1;
            for (int i = static_cast<int>(m_shape.size()) - 2; i >= 0; --i) {
                m_strides[static_cast<size_t>(i)] = m_strides[static_cast<size_t>(i) + 1] * m_shape[static_cast<size_t>(i) + 1];
            }
        }
        m_size = 1;
        for (auto s : m_shape) m_size *= s;
    }

    size_type dimension() const noexcept { return m_shape.size(); }
    const shape_type& shape() const noexcept { return m_shape; }
    const strides_type& strides() const noexcept { return m_strides; }
    layout_type layout() const noexcept { return layout_type::row_major; }
    size_type size() const noexcept { return m_size; }

    value_type operator[](size_type i) const {
        std::vector<size_type> reduced_idx(dimension());
        size_type temp = i;
        for (int d = static_cast<int>(dimension()) - 1; d >= 0; --d) {
            reduced_idx[static_cast<size_t>(d)] = temp % m_shape[static_cast<size_t>(d)];
            temp /= m_shape[static_cast<size_t>(d)];
        }
        value_type sum = value_type(0);
        const auto& src_shape = m_expression.shape();
        std::vector<size_type> full_idx(m_expression.dimension());
        size_type reduced_pos = 0;
        for (size_type d = 0; d < m_expression.dimension(); ++d) {
            if (std::find(m_axes.begin(), m_axes.end(), d) == m_axes.end()) {
                full_idx[d] = reduced_idx[reduced_pos++];
            } else {
                full_idx[d] = 0;
            }
        }
        sum = accumulate_sum(full_idx, 0);
        return sum / static_cast<value_type>(m_reduced_size);
    }

    template <class... Idx>
    value_type operator()(Idx... idxs) const {
        std::vector<size_type> reduced_idx{static_cast<size_type>(idxs)...};
        return (*this)[compute_linear(reduced_idx)];
    }

    const E& expression() const noexcept { return m_expression; }

    class const_iterator {
    private:
        const xmean_reducer* m_reducer;
        size_type m_index;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xmean_reducer::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;

        const_iterator(const xmean_reducer* r, size_type idx) : m_reducer(r), m_index(idx) {}
        value_type operator*() const { return (*m_reducer)[m_index]; }
        const_iterator& operator++() { ++m_index; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++m_index; return tmp; }
        bool operator==(const const_iterator& other) const { return m_index == other.m_index; }
        bool operator!=(const const_iterator& other) const { return m_index != other.m_index; }
        difference_type operator-(const const_iterator& other) const { return m_index - other.m_index; }
    };

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, m_size); }

private:
    value_type accumulate_sum(std::vector<size_type>& idx, size_type axis_pos) const {
        if (axis_pos == m_axes.size()) {
            size_type linear = 0;
            const auto& strides = m_expression.strides();
            for (size_type d = 0; d < m_expression.dimension(); ++d) {
                linear += idx[d] * strides[d];
            }
            return m_expression.flat(linear);
        }
        size_type ax = m_axes[axis_pos];
        value_type sum = 0;
        for (size_type i = 0; i < m_expression.shape()[ax]; ++i) {
            idx[ax] = i;
            sum += accumulate_sum(idx, axis_pos + 1);
        }
        return sum;
    }

    size_type compute_linear(const std::vector<size_type>& idx) const {
        size_type linear = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            linear += idx[i] * m_strides[i];
        }
        return linear;
    }
};

// #############################################################################
// Free functions for reductions
// #############################################################################
template <class E>
auto sum(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        // Reduce all axes -> scalar
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::sum<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::sum<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

template <class E>
auto prod(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::prod<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::prod<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

template <class E>
auto mean(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xmean_reducer<xclosure_t<const E&>>(e.derived_cast(), std::move(all_axes));
    }
    return xmean_reducer<xclosure_t<const E&>>(e.derived_cast(), axes);
}

template <class E>
auto max(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::max<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::max<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

template <class E>
auto min(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::min<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::min<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

template <class E>
auto all(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::all<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::all<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

template <class E>
auto any(const xexpression<E>& e, const std::vector<size_type>& axes = {}) {
    if (axes.empty()) {
        std::vector<size_type> all_axes(e.derived_cast().dimension());
        std::iota(all_axes.begin(), all_axes.end(), 0);
        return xreducer<functor::any<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
            e.derived_cast(), std::move(all_axes)
        );
    }
    return xreducer<functor::any<typename E::value_type>, xclosure_t<const E&>, std::vector<size_type>>(
        e.derived_cast(), axes
    );
}

// Convenience overloads for reducing over a single axis (integer)
template <class E>
auto sum(const xexpression<E>& e, size_type axis) {
    return sum(e, std::vector<size_type>{axis});
}

template <class E>
auto mean(const xexpression<E>& e, size_type axis) {
    return mean(e, std::vector<size_type>{axis});
}

template <class E>
auto max(const xexpression<E>& e, size_type axis) {
    return max(e, std::vector<size_type>{axis});
}

template <class E>
auto min(const xexpression<E>& e, size_type axis) {
    return min(e, std::vector<size_type>{axis});
}

} // namespace math
XTU_NAMESPACE_END

#endif // XTU_MATH_XREDUCER_HPP