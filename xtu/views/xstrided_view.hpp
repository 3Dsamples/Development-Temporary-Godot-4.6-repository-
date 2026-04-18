// include/xtu/views/xstrided_view.hpp
// xtensor-unified - Strided view with arbitrary strides and shape
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_VIEWS_XSTRIDED_VIEW_HPP
#define XTU_VIEWS_XSTRIDED_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/core/xexpression.hpp"
#include "xtu/core/xiterable.hpp"

XTU_NAMESPACE_BEGIN

namespace detail {
    // Compute linear offset from strides and indices
    template <class StridesIter, class IndicesIter>
    size_t compute_offset(StridesIter strides_begin, StridesIter strides_end,
                          IndicesIter indices_begin, IndicesIter indices_end) {
        size_t offset = 0;
        auto s_it = strides_begin;
        auto i_it = indices_begin;
        for (; s_it != strides_end && i_it != indices_end; ++s_it, ++i_it) {
            offset += (*i_it) * (*s_it);
        }
        return offset;
    }

    // Validate that new shape and strides are consistent
    inline bool validate_strides(const std::vector<size_t>& shape,
                                 const std::vector<size_t>& strides) {
        if (shape.size() != strides.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == 0) return false;
            // Strides can be arbitrary (including 0 for broadcasting)
        }
        return true;
    }

    // Compute default strides from shape (row-major)
    inline std::vector<size_t> default_strides(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        if (shape.empty()) return strides;
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
        }
        return strides;
    }
}

// #############################################################################
// xstrided_view - View with explicit shape and strides
// #############################################################################
template <class CT, class S, layout_type L = layout_type::dynamic>
class xstrided_view : public xiterable<xstrided_view<CT, S, L>> {
public:
    using value_type = typename CT::value_type;
    using reference = typename CT::reference;
    using const_reference = typename CT::const_reference;
    using pointer = typename CT::pointer;
    using const_pointer = typename CT::const_pointer;
    using size_type = xtu::size_type;
    using difference_type = std::ptrdiff_t;
    using shape_type = S;
    using strides_type = S;
    using expression_tag = typename CT::expression_tag;

    using iterator = xstepper<xstrided_view>;
    using const_iterator = xconst_stepper<xstrided_view>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

private:
    CT m_expression;
    shape_type m_shape;
    strides_type m_strides;
    size_type m_size;
    size_type m_offset;

public:
    // Constructor with explicit shape and strides
    xstrided_view(CT&& expr, const shape_type& shape, const strides_type& strides, size_type offset = 0)
        : m_expression(std::forward<CT>(expr))
        , m_shape(shape)
        , m_strides(strides)
        , m_offset(offset) {
        XTU_ASSERT_MSG(detail::validate_strides(m_shape, m_strides), "Invalid shape or strides");
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), size_type(1), std::multiplies<size_type>());
        // Ensure offset is within bounds
        size_t max_offset = 0;
        for (size_t i = 0; i < m_shape.size(); ++i) {
            max_offset += (m_shape[i] - 1) * m_strides[i];
        }
        XTU_ASSERT_MSG(offset + max_offset <= m_expression.size(), "Strided view exceeds underlying storage");
    }

    // Constructor with shape only (row-major strides)
    xstrided_view(CT&& expr, const shape_type& shape, size_type offset = 0)
        : m_expression(std::forward<CT>(expr))
        , m_shape(shape)
        , m_strides(detail::default_strides(shape))
        , m_offset(offset) {
        m_size = std::accumulate(m_shape.begin(), m_shape.end(), size_type(1), std::multiplies<size_type>());
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
        return L;
    }

    size_type size() const noexcept {
        return m_size;
    }

    // Access to underlying expression
    const CT& expression() const noexcept {
        return m_expression;
    }

    CT& expression() noexcept {
        return m_expression;
    }

    // Element access
    template <class... Idx>
    reference operator()(Idx... idxs) {
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        XTU_ASSERT_MSG(indices.size() == dimension(), "Number of indices must match dimension");
        size_type linear = m_offset;
        for (size_t d = 0; d < dimension(); ++d) {
            XTU_ASSERT_MSG(indices[d] < m_shape[d], "Index out of bounds");
            linear += indices[d] * m_strides[d];
        }
        return m_expression.flat(linear);
    }

    template <class... Idx>
    const_reference operator()(Idx... idxs) const {
        std::array<size_type, sizeof...(Idx)> indices = {static_cast<size_type>(idxs)...};
        XTU_ASSERT_MSG(indices.size() == dimension(), "Number of indices must match dimension");
        size_type linear = m_offset;
        for (size_t d = 0; d < dimension(); ++d) {
            XTU_ASSERT_MSG(indices[d] < m_shape[d], "Index out of bounds");
            linear += indices[d] * m_strides[d];
        }
        return m_expression.flat(linear);
    }

    reference operator[](size_type i) {
        return flat(i);
    }

    const_reference operator[](size_type i) const {
        return flat(i);
    }

    reference flat(size_type i) {
        XTU_ASSERT_MSG(i < m_size, "Flat index out of bounds");
        // Convert linear index to multi-index using shape
        std::vector<size_type> idx(dimension());
        size_type temp = i;
        for (int d = static_cast<int>(dimension()) - 1; d >= 0; --d) {
            idx[static_cast<size_t>(d)] = temp % m_shape[static_cast<size_t>(d)];
            temp /= m_shape[static_cast<size_t>(d)];
        }
        size_type linear = m_offset;
        for (size_t d = 0; d < dimension(); ++d) {
            linear += idx[d] * m_strides[d];
        }
        return m_expression.flat(linear);
    }

    const_reference flat(size_type i) const {
        XTU_ASSERT_MSG(i < m_size, "Flat index out of bounds");
        std::vector<size_type> idx(dimension());
        size_type temp = i;
        for (int d = static_cast<int>(dimension()) - 1; d >= 0; --d) {
            idx[static_cast<size_t>(d)] = temp % m_shape[static_cast<size_t>(d)];
            temp /= m_shape[static_cast<size_t>(d)];
        }
        size_type linear = m_offset;
        for (size_t d = 0; d < dimension(); ++d) {
            linear += idx[d] * m_strides[d];
        }
        return m_expression.flat(linear);
    }

    // Assignment
    template <class E>
    xstrided_view& operator=(const xexpression<E>& e) {
        const E& expr = e.derived_cast();
        // Check broadcast compatibility
        std::vector<size_type> bcast_shape;
        bool compatible = detail::broadcast_shapes(
            std::vector<size_type>(m_shape.begin(), m_shape.end()),
            std::vector<size_type>(expr.shape().begin(), expr.shape().end()),
            bcast_shape);
        if (!compatible || bcast_shape != std::vector<size_type>(m_shape.begin(), m_shape.end())) {
            XTU_THROW(std::runtime_error, "Cannot assign: shape mismatch");
        }
        // Assign element-wise
        for (size_type i = 0; i < m_size; ++i) {
            flat(i) = expr.flat(i);
        }
        return *this;
    }

    // Iterator support
    iterator begin() noexcept {
        return iterator(this, 0);
    }

    iterator end() noexcept {
        return iterator(this, m_size);
    }

    const_iterator begin() const noexcept {
        return const_iterator(this, 0);
    }

    const_iterator end() const noexcept {
        return const_iterator(this, m_size);
    }

    const_iterator cbegin() const noexcept {
        return const_iterator(this, 0);
    }

    const_iterator cend() const noexcept {
        return const_iterator(this, m_size);
    }
};

// #############################################################################
// Free functions to create strided views
// #############################################################################
template <class E, class S>
auto strided_view(xexpression<E>& e, const S& shape, const S& strides, size_t offset = 0) {
    return xstrided_view<xclosure_t<E&>, S>(e.derived_cast(), shape, strides, offset);
}

template <class E, class S>
auto strided_view(const xexpression<E>& e, const S& shape, const S& strides, size_t offset = 0) {
    return xstrided_view<xclosure_t<const E&>, S>(e.derived_cast(), shape, strides, offset);
}

// #############################################################################
// Transpose via strided view
// #############################################################################
template <class E>
auto transpose_view(const xexpression<E>& e, const std::vector<size_t>& perm = {}) {
    const auto& expr = e.derived_cast();
    size_t ndim = expr.dimension();
    std::vector<size_t> axes;
    if (perm.empty()) {
        axes.resize(ndim);
        for (size_t i = 0; i < ndim; ++i) axes[i] = ndim - 1 - i;
    } else {
        XTU_ASSERT_MSG(perm.size() == ndim, "Permutation size must match dimension");
        axes = perm;
    }
    
    std::vector<size_t> new_shape(ndim);
    std::vector<size_t> new_strides(ndim);
    const auto& old_shape = expr.shape();
    const auto& old_strides = expr.strides();
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = old_shape[axes[i]];
        new_strides[i] = old_strides[axes[i]];
    }
    return strided_view(e, new_shape, new_strides);
}

// #############################################################################
// Diagonal view (extract diagonal from 2D matrix)
// #############################################################################
template <class E>
auto diagonal_view(const xexpression<E>& e, int offset = 0) {
    const auto& expr = e.derived_cast();
    XTU_ASSERT_MSG(expr.dimension() == 2, "Diagonal view requires 2D matrix");
    size_t nrows = expr.shape()[0];
    size_t ncols = expr.shape()[1];
    const auto& strides = expr.strides();
    
    size_t diag_len;
    size_t start_offset;
    if (offset >= 0) {
        diag_len = std::min(nrows, ncols - static_cast<size_t>(offset));
        start_offset = static_cast<size_t>(offset) * strides[1];
    } else {
        diag_len = std::min(nrows - static_cast<size_t>(-offset), ncols);
        start_offset = static_cast<size_t>(-offset) * strides[0];
    }
    
    std::vector<size_t> new_shape = {diag_len};
    std::vector<size_t> new_strides = {strides[0] + strides[1]};
    return strided_view(e, new_shape, new_strides, start_offset);
}

// #############################################################################
// Slice view (similar to xview but with stride support)
// #############################################################################
template <class E, class... S>
auto slice_view(const xexpression<E>& e, S... slices) {
    const auto& expr = e.derived_cast();
    size_t ndim = expr.dimension();
    const auto& old_shape = expr.shape();
    const auto& old_strides = expr.strides();
    
    std::vector<size_t> new_shape;
    std::vector<size_t> new_strides;
    size_t offset = 0;
    size_t src_dim = 0;
    
    auto process_slice = [&](auto&& slice) {
        using SliceType = std::decay_t<decltype(slice)>;
        if constexpr (std::is_same_v<SliceType, xall>) {
            new_shape.push_back(old_shape[src_dim]);
            new_strides.push_back(old_strides[src_dim]);
            ++src_dim;
        } else if constexpr (std::is_same_v<SliceType, xrange>) {
            XTU_ASSERT_MSG(slice.stop() <= old_shape[src_dim], "Range exceeds dimension");
            new_shape.push_back(slice.size());
            new_strides.push_back(old_strides[src_dim] * slice.step());
            offset += slice.start() * old_strides[src_dim];
            ++src_dim;
        } else if constexpr (std::is_same_v<SliceType, xslice_index>) {
            XTU_ASSERT_MSG(slice.index() < old_shape[src_dim], "Index out of bounds");
            offset += slice.index() * old_strides[src_dim];
            ++src_dim;
        } else if constexpr (std::is_same_v<SliceType, xnewaxis>) {
            new_shape.push_back(1);
            new_strides.push_back(0);
        } else {
            static_assert(sizeof(SliceType) == 0, "Unsupported slice type");
        }
    };
    
    (process_slice(slices), ...);
    return strided_view(e, new_shape, new_strides, offset);
}

// #############################################################################
// Broadcast view (repeat data without copying)
// #############################################################################
template <class E>
auto broadcast_view(const xexpression<E>& e, const std::vector<size_t>& target_shape) {
    const auto& expr = e.derived_cast();
    const auto& src_shape = expr.shape();
    const auto& src_strides = expr.strides();
    size_t ndim_src = src_shape.size();
    size_t ndim_target = target_shape.size();
    
    XTU_ASSERT_MSG(ndim_target >= ndim_src, "Target shape must have at least as many dimensions as source");
    
    std::vector<size_t> new_strides(ndim_target, 0);
    for (size_t i = 0; i < ndim_target; ++i) {
        size_t src_i = (i < ndim_target - ndim_src) ? static_cast<size_t>(-1) : i - (ndim_target - ndim_src);
        if (src_i != static_cast<size_t>(-1)) {
            if (src_shape[src_i] == 1) {
                new_strides[i] = 0;
            } else {
                XTU_ASSERT_MSG(src_shape[src_i] == target_shape[i], "Shape not broadcastable");
                new_strides[i] = src_strides[src_i];
            }
        } else {
            XTU_ASSERT_MSG(target_shape[i] == 1 || target_shape[i] == src_shape[0] ? true : false,
                           "Leading dimensions must be 1 or match");
            new_strides[i] = 0;
        }
    }
    return strided_view(e, target_shape, new_strides);
}

// #############################################################################
// Ravel view (flatten without copy if contiguous)
// #############################################################################
template <class E>
auto ravel_view(const xexpression<E>& e) {
    const auto& expr = e.derived_cast();
    // Check if already contiguous
    const auto& strides = expr.strides();
    bool is_contiguous = true;
    size_t expected_stride = 1;
    for (int i = static_cast<int>(strides.size()) - 1; i >= 0; --i) {
        if (strides[static_cast<size_t>(i)] != expected_stride) {
            is_contiguous = false;
            break;
        }
        expected_stride *= expr.shape()[static_cast<size_t>(i)];
    }
    if (is_contiguous) {
        std::vector<size_t> new_shape = {expr.size()};
        std::vector<size_t> new_strides = {1};
        return strided_view(e, new_shape, new_strides);
    } else {
        // Cannot create a 1D strided view for non-contiguous; return copy via ravel()
        XTU_THROW(std::runtime_error, "ravel_view requires contiguous storage; use ravel() for a copy");
    }
}

XTU_NAMESPACE_END

#endif // XTU_VIEWS_XSTRIDED_VIEW_HPP