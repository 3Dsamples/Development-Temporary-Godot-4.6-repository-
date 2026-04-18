// include/xtu/views/xview.hpp
// xtensor-unified - Strided view for efficient slicing without data copy
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_VIEWS_XVIEW_HPP
#define XTU_VIEWS_XVIEW_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/core/xexpression.hpp"

XTU_NAMESPACE_BEGIN

// #############################################################################
// Slice types and utilities
// #############################################################################

/// Represents a single index (integer) in a slice specification
class xslice_index {
public:
    using size_type = xtu::size_type;
    
private:
    size_type m_index;
    
public:
    explicit xslice_index(size_type idx) : m_index(idx) {}
    size_type index() const noexcept { return m_index; }
};

/// Represents a range (start, stop, step) in a slice specification
class xrange {
public:
    using size_type = xtu::size_type;
    
private:
    size_type m_start;
    size_type m_stop;
    size_type m_step;
    
public:
    xrange(size_type start, size_type stop, size_type step = 1)
        : m_start(start), m_stop(stop), m_step(step) {
        XTU_ASSERT_MSG(step > 0, "Step must be positive");
    }
    
    size_type start() const noexcept { return m_start; }
    size_type stop() const noexcept { return m_stop; }
    size_type step() const noexcept { return m_step; }
    size_type size() const noexcept {
        if (m_stop <= m_start) return 0;
        return (m_stop - m_start + m_step - 1) / m_step;
    }
};

/// Represents "all elements" in a slice specification
class xall {};

/// Represents "new axis" insertion in a slice specification
class xnewaxis {};

// Predefined slice objects
inline xall all() { return xall{}; }
inline xnewaxis newaxis() { return xnewaxis{}; }
inline xrange range(size_t start, size_t stop, size_t step = 1) { return xrange(start, stop, step); }
inline xslice_index index(size_t idx) { return xslice_index(idx); }

// #############################################################################
// Helper to compute view shape and strides from slice specification
// #############################################################################
namespace detail {
    enum class slice_kind { index, range, all, newaxis };
    
    template <class T> struct slice_traits { static const slice_kind kind = slice_kind::index; };
    template <> struct slice_traits<xrange> { static const slice_kind kind = slice_kind::range; };
    template <> struct slice_traits<xall> { static const slice_kind kind = slice_kind::all; };
    template <> struct slice_traits<xnewaxis> { static const slice_kind kind = slice_kind::newaxis; };
    template <> struct slice_traits<xslice_index> { static const slice_kind kind = slice_kind::index; };
    
    template <class... S>
    struct slice_processor {
        using size_type = xtu::size_type;
        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        
        static void compute(const shape_type& src_shape, const strides_type& src_strides,
                            shape_type& view_shape, strides_type& view_strides,
                            const std::tuple<S...>& slices) {
            view_shape.clear();
            view_strides.clear();
            size_t src_dim = 0;
            compute_impl(src_shape, src_strides, view_shape, view_strides, slices, src_dim, std::index_sequence_for<S...>{});
        }
        
    private:
        template <size_t... I>
        static void compute_impl(const shape_type& src_shape, const strides_type& src_strides,
                                 shape_type& view_shape, strides_type& view_strides,
                                 const std::tuple<S...>& slices, size_t& src_dim,
                                 std::index_sequence<I...>) {
            (process_slice<I>(src_shape, src_strides, view_shape, view_strides, std::get<I>(slices), src_dim), ...);
        }
        
        template <size_t I, class Slice>
        static void process_slice(const shape_type& src_shape, const strides_type& src_strides,
                                  shape_type& view_shape, strides_type& view_strides,
                                  const Slice& slice, size_t& src_dim) {
            if constexpr (std::is_same_v<Slice, xall>) {
                XTU_ASSERT_MSG(src_dim < src_shape.size(), "Too many slices for source dimension");
                view_shape.push_back(src_shape[src_dim]);
                view_strides.push_back(src_strides[src_dim]);
                ++src_dim;
            } else if constexpr (std::is_same_v<Slice, xrange>) {
                XTU_ASSERT_MSG(src_dim < src_shape.size(), "Too many slices for source dimension");
                XTU_ASSERT_MSG(slice.stop() <= src_shape[src_dim], "Range stop exceeds dimension size");
                view_shape.push_back(slice.size());
                view_strides.push_back(src_strides[src_dim] * slice.step());
                ++src_dim;
            } else if constexpr (std::is_same_v<Slice, xslice_index>) {
                XTU_ASSERT_MSG(src_dim < src_shape.size(), "Too many slices for source dimension");
                XTU_ASSERT_MSG(slice.index() < src_shape[src_dim], "Index out of bounds");
                // Index slice removes the dimension (no shape added)
                ++src_dim;
            } else if constexpr (std::is_same_v<Slice, xnewaxis>) {
                // New axis adds a dimension of size 1 with stride 0
                view_shape.push_back(1);
                view_strides.push_back(0);
                // src_dim does not increment
            }
        }
    };
    
    // Compute linear offset for indexed slices
    template <class... S>
    size_t compute_index_offset(const std::vector<size_t>& src_strides, const std::tuple<S...>& slices) {
        size_t offset = 0;
        size_t src_dim = 0;
        compute_offset_impl(src_strides, slices, offset, src_dim, std::index_sequence_for<S...>{});
        return offset;
    }
    
    template <size_t... I>
    static void compute_offset_impl(const std::vector<size_t>& src_strides,
                                    const std::tuple<S...>& slices,
                                    size_t& offset, size_t& src_dim,
                                    std::index_sequence<I...>) {
        (accumulate_offset<I>(src_strides, std::get<I>(slices), offset, src_dim), ...);
    }
    
    template <size_t I, class Slice>
    static void accumulate_offset(const std::vector<size_t>& src_strides, const Slice& slice,
                                  size_t& offset, size_t& src_dim) {
        if constexpr (std::is_same_v<Slice, xslice_index>) {
            offset += slice.index() * src_strides[src_dim];
            ++src_dim;
        } else if constexpr (std::is_same_v<Slice, xrange>) {
            offset += slice.start() * src_strides[src_dim];
            ++src_dim;
        } else if constexpr (std::is_same_v<Slice, xall>) {
            ++src_dim;
        }
        // xnewaxis does not affect offset or src_dim
    }
}

// #############################################################################
// xview - Strided view of an underlying expression
// #############################################################################
template <class CT, class... S>
class xview : public xexpression<xview<CT, S...>> {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::reference;
    using const_reference = typename CT::const_reference;
    using pointer = typename CT::pointer;
    using const_pointer = typename CT::const_pointer;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    using shape_type = std::vector<size_type>;
    using strides_type = std::vector<size_type>;
    using expression_tag = typename CT::expression_tag;
    
    using slice_tuple = std::tuple<S...>;

private:
    CT m_expression;
    slice_tuple m_slices;
    shape_type m_shape;
    strides_type m_strides;
    size_type m_size;
    size_type m_offset;

public:
    // Constructor
    xview(CT&& expr, S&&... slices)
        : m_expression(std::forward<CT>(expr))
        , m_slices(std::forward<S>(slices)...) {
        // Compute view shape and strides
        const auto& src_shape = m_expression.shape();
        const auto& src_strides = m_expression.strides();
        std::vector<size_type> src_shape_vec(src_shape.begin(), src_shape.end());
        std::vector<size_type> src_strides_vec(src_strides.begin(), src_strides.end());
        
        detail::slice_processor<S...>::compute(src_shape_vec, src_strides_vec, m_shape, m_strides, m_slices);
        m_offset = detail::compute_index_offset(src_strides_vec, m_slices);
        
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
        return layout_type::dynamic;
    }

    size_type size() const noexcept {
        return m_size;
    }

    // Access to underlying expression
    const CT& expression() const noexcept {
        return m_expression;
    }

    // Element access via flat index
    reference flat(size_type i) {
        size_type src_index = m_offset;
        size_type temp = i;
        for (size_t d = m_shape.size(); d > 0; --d) {
            size_t dim = d - 1;
            size_type coord = temp % m_shape[dim];
            temp /= m_shape[dim];
            src_index += coord * m_strides[dim];
        }
        return m_expression.flat(src_index);
    }

    const_reference flat(size_type i) const {
        size_type src_index = m_offset;
        size_type temp = i;
        for (size_t d = m_shape.size(); d > 0; --d) {
            size_t dim = d - 1;
            size_type coord = temp % m_shape[dim];
            temp /= m_shape[dim];
            src_index += coord * m_strides[dim];
        }
        return m_expression.flat(src_index);
    }

    reference operator[](size_type i) {
        return flat(i);
    }

    const_reference operator[](size_type i) const {
        return flat(i);
    }

    // Multi-dimensional access (variadic)
    template <class... Idx>
    reference operator()(Idx... idxs) {
        static_assert(sizeof...(Idx) == sizeof...(S) - count_index_slices() + count_newaxis_slices(),
                      "Number of indices must match view dimension");
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        size_type src_index = m_offset;
        size_t view_idx = 0;
        size_t src_dim = 0;
        accumulate_indices(indices, view_idx, src_dim, src_index, std::index_sequence_for<S...>{});
        return m_expression.flat(src_index);
    }

    template <class... Idx>
    const_reference operator()(Idx... idxs) const {
        static_assert(sizeof...(Idx) == dimension(),
                      "Number of indices must match view dimension");
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        size_type src_index = m_offset;
        size_t view_idx = 0;
        size_t src_dim = 0;
        accumulate_indices(indices, view_idx, src_dim, src_index, std::index_sequence_for<S...>{});
        return m_expression.flat(src_index);
    }

    // Assignment from expression (broadcasting allowed)
    template <class E>
    xview& operator=(const xexpression<E>& e) {
        const E& expr = e.derived_cast();
        // Check broadcast compatibility
        std::vector<size_type> broadcast_shape;
        bool compatible = detail::broadcast_shapes(m_shape, expr.shape(), broadcast_shape);
        if (!compatible || broadcast_shape != m_shape) {
            XTU_THROW(std::runtime_error, "Cannot assign: shape mismatch");
        }
        // Assign element-wise
        for (size_type i = 0; i < m_size; ++i) {
            flat(i) = expr.flat(i);
        }
        return *this;
    }

    // Iterator support
    class iterator {
    private:
        xview* m_view;
        size_type m_index;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xview::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = typename xview::reference;

        iterator(xview* v, size_type idx) : m_view(v), m_index(idx) {}

        reference operator*() const { return (*m_view)[m_index]; }
        
        iterator& operator++() { ++m_index; return *this; }
        iterator operator++(int) { iterator tmp = *this; ++m_index; return tmp; }
        iterator& operator--() { --m_index; return *this; }
        iterator operator--(int) { iterator tmp = *this; --m_index; return tmp; }
        iterator& operator+=(difference_type n) { m_index += n; return *this; }
        iterator& operator-=(difference_type n) { m_index -= n; return *this; }
        
        iterator operator+(difference_type n) const { return iterator(m_view, m_index + n); }
        iterator operator-(difference_type n) const { return iterator(m_view, m_index - n); }
        difference_type operator-(const iterator& other) const { return m_index - other.m_index; }
        
        bool operator==(const iterator& other) const { return m_index == other.m_index; }
        bool operator!=(const iterator& other) const { return m_index != other.m_index; }
        bool operator<(const iterator& other) const { return m_index < other.m_index; }
        bool operator<=(const iterator& other) const { return m_index <= other.m_index; }
        bool operator>(const iterator& other) const { return m_index > other.m_index; }
        bool operator>=(const iterator& other) const { return m_index >= other.m_index; }
        
        reference operator[](difference_type n) const { return (*m_view)[m_index + n]; }
    };

    class const_iterator {
    private:
        const xview* m_view;
        size_type m_index;
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename xview::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = typename xview::const_reference;

        const_iterator(const xview* v, size_type idx) : m_view(v), m_index(idx) {}

        const_reference operator*() const { return (*m_view)[m_index]; }
        
        const_iterator& operator++() { ++m_index; return *this; }
        const_iterator operator++(int) { const_iterator tmp = *this; ++m_index; return tmp; }
        const_iterator& operator--() { --m_index; return *this; }
        const_iterator operator--(int) { const_iterator tmp = *this; --m_index; return tmp; }
        const_iterator& operator+=(difference_type n) { m_index += n; return *this; }
        const_iterator& operator-=(difference_type n) { m_index -= n; return *this; }
        
        const_iterator operator+(difference_type n) const { return const_iterator(m_view, m_index + n); }
        const_iterator operator-(difference_type n) const { return const_iterator(m_view, m_index - n); }
        difference_type operator-(const const_iterator& other) const { return m_index - other.m_index; }
        
        bool operator==(const const_iterator& other) const { return m_index == other.m_index; }
        bool operator!=(const const_iterator& other) const { return m_index != other.m_index; }
        bool operator<(const const_iterator& other) const { return m_index < other.m_index; }
        bool operator<=(const const_iterator& other) const { return m_index <= other.m_index; }
        bool operator>(const const_iterator& other) const { return m_index > other.m_index; }
        bool operator>=(const const_iterator& other) const { return m_index >= other.m_index; }
        
        const_reference operator[](difference_type n) const { return (*m_view)[m_index + n]; }
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, m_size); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, m_size); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, m_size); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

private:
    static constexpr size_t count_index_slices() {
        return (0 + ... + (std::is_same_v<S, xslice_index> ? 1 : 0));
    }
    
    static constexpr size_t count_newaxis_slices() {
        return (0 + ... + (std::is_same_v<S, xnewaxis> ? 1 : 0));
    }

    template <size_t... I>
    void accumulate_indices(const std::array<size_type, sizeof...(S) - count_index_slices() + count_newaxis_slices()>& indices,
                            size_t& view_idx, size_t& src_dim, size_type& src_index,
                            std::index_sequence<I...>) const {
        (process_view_index<I>(indices, view_idx, src_dim, src_index, std::get<I>(m_slices)), ...);
    }

    template <size_t I, class Slice>
    void process_view_index(const std::array<size_type, sizeof...(S) - count_index_slices() + count_newaxis_slices()>& indices,
                            size_t& view_idx, size_t& src_dim, size_type& src_index,
                            const Slice& slice) const {
        if constexpr (std::is_same_v<Slice, xall>) {
            src_index += indices[view_idx++] * m_expression.strides()[src_dim++];
        } else if constexpr (std::is_same_v<Slice, xrange>) {
            src_index += (slice.start() + indices[view_idx++] * slice.step()) * m_expression.strides()[src_dim++];
        } else if constexpr (std::is_same_v<Slice, xnewaxis>) {
            // newaxis dimension is always 1, index must be 0
            XTU_ASSERT_MSG(indices[view_idx++] == 0, "Index for newaxis must be 0");
            // no src_dim increment, no offset change
        } else if constexpr (std::is_same_v<Slice, xslice_index>) {
            // index slice already accounted for in m_offset; src_dim increments
            ++src_dim;
        }
    }
};

// #############################################################################
// Free functions to create views
// #############################################################################
template <class E, class... S>
auto view(xexpression<E>& e, S&&... slices) {
    return xview<xclosure_t<E&>, S...>(e.derived_cast(), std::forward<S>(slices)...);
}

template <class E, class... S>
auto view(const xexpression<E>& e, S&&... slices) {
    return xview<xclosure_t<const E&>, S...>(e.derived_cast(), std::forward<S>(slices)...);
}

// #############################################################################
// Operator() overload for slicing on containers
// #############################################################################
#define XTU_DEFINE_SLICING_OP(Container) \
template <class... S> \
auto operator()(const Container& c, S... slices) -> decltype(view(c, slices...)) { \
    return view(c, slices...); \
} \
template <class... S> \
auto operator()(Container& c, S... slices) -> decltype(view(c, slices...)) { \
    return view(c, slices...); \
}

XTU_NAMESPACE_END

#endif // XTU_VIEWS_XVIEW_HPP