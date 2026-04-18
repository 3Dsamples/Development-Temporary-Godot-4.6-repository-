// include/xtu/manipulation/xmanipulation.hpp
// xtensor-unified - Array manipulation routines (reshape, concatenate, stack, split, etc.)
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_MANIPULATION_XMANIPULATION_HPP
#define XTU_MANIPULATION_XMANIPULATION_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/views/xview.hpp"

XTU_NAMESPACE_BEGIN
namespace manipulation {

// #############################################################################
// Reshape
// #############################################################################
template <class E>
auto reshape(const xexpression<E>& e, const std::vector<size_t>& new_shape) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    XTU_ASSERT_MSG(new_size == expr.size(), "Reshape cannot change total number of elements");
    
    // Create a copy with new shape (contiguous layout)
    xarray_container<value_type> result(new_shape);
    std::copy(expr.begin(), expr.end(), result.begin());
    return result;
}

template <class E, class... S>
auto reshape(const xexpression<E>& e, S... new_shape) {
    return reshape(e, std::vector<size_t>{static_cast<size_t>(new_shape)...});
}

// #############################################################################
// Ravel (flatten to 1D)
// #############################################################################
template <class E>
auto ravel(const xexpression<E>& e) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    xarray_container<value_type> result({expr.size()});
    std::copy(expr.begin(), expr.end(), result.begin());
    return result;
}

// #############################################################################
// Transpose (reverse axes)
// #############################################################################
template <class E>
auto transpose(const xexpression<E>& e, const std::vector<size_t>& axes = {}) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    
    std::vector<size_t> perm;
    if (axes.empty()) {
        // Reverse axes
        perm.resize(ndim);
        for (size_t i = 0; i < ndim; ++i) perm[i] = ndim - 1 - i;
    } else {
        XTU_ASSERT_MSG(axes.size() == ndim, "Number of axes must match dimension");
        perm = axes;
        // Validate permutation
        std::vector<bool> seen(ndim, false);
        for (size_t ax : perm) {
            XTU_ASSERT_MSG(ax < ndim, "Axis index out of range");
            XTU_ASSERT_MSG(!seen[ax], "Duplicate axis in transpose");
            seen[ax] = true;
        }
    }
    
    // Compute new shape
    std::vector<size_t> new_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = expr.shape()[perm[i]];
    }
    xarray_container<value_type> result(new_shape);
    
    // Perform transpose (naive element-wise copy)
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            dst_idx[perm[dim]] = i;
            copy_rec(dim + 1);
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Concatenate along an existing axis
// #############################################################################
template <class E1, class E2, class... Es>
auto concatenate(const xexpression<E1>& e1, const xexpression<E2>& e2, size_t axis = 0) {
    const auto& a1 = e1.derived_cast();
    const auto& a2 = e2.derived_cast();
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    
    size_t ndim = a1.dimension();
    XTU_ASSERT_MSG(a2.dimension() == ndim, "All arrays must have same number of dimensions");
    XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
    
    // Check shapes except along axis
    for (size_t d = 0; d < ndim; ++d) {
        if (d != axis) {
            XTU_ASSERT_MSG(a1.shape()[d] == a2.shape()[d], "Shapes must match except along concatenation axis");
        }
    }
    
    // Compute new shape
    std::vector<size_t> new_shape(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        new_shape[d] = (d == axis) ? a1.shape()[d] + a2.shape()[d] : a1.shape()[d];
    }
    xarray_container<value_type> result(new_shape);
    
    // Copy first array
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t, const auto&, size_t)> copy_rec = [&](size_t dim, const auto& src, size_t offset) {
        if (dim == ndim) {
            result(dst_idx) = src(src_idx);
            return;
        }
        for (size_t i = 0; i < src.shape()[dim]; ++i) {
            src_idx[dim] = i;
            dst_idx[dim] = (dim == axis) ? i + offset : i;
            copy_rec(dim + 1, src, offset);
        }
    };
    copy_rec(0, a1, 0);
    copy_rec(0, a2, a1.shape()[axis]);
    return result;
}

// Variadic concatenate
template <class E1, class E2, class... Es>
auto concatenate(const xexpression<E1>& e1, const xexpression<E2>& e2, const xexpression<Es>&... es) {
    auto first = concatenate(e1, e2);
    return concatenate(first, es...);
}

// #############################################################################
// Stack along a new axis
// #############################################################################
template <class E1, class E2, class... Es>
auto stack(const xexpression<E1>& e1, const xexpression<E2>& e2, size_t axis = 0) {
    const auto& a1 = e1.derived_cast();
    const auto& a2 = e2.derived_cast();
    using value_type = typename std::common_type<typename E1::value_type, typename E2::value_type>::type;
    
    size_t ndim = a1.dimension();
    XTU_ASSERT_MSG(a2.dimension() == ndim, "All arrays must have same number of dimensions");
    XTU_ASSERT_MSG(axis <= ndim, "Axis out of range (max is number of dimensions)");
    XTU_ASSERT_MSG(a1.shape() == a2.shape(), "All arrays must have identical shapes");
    
    // New shape has an extra dimension at axis
    std::vector<size_t> new_shape(ndim + 1);
    for (size_t d = 0; d < axis; ++d) new_shape[d] = a1.shape()[d];
    new_shape[axis] = 2;  // two arrays
    for (size_t d = axis; d < ndim; ++d) new_shape[d + 1] = a1.shape()[d];
    
    xarray_container<value_type> result(new_shape);
    
    // Copy arrays into new axis position
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim + 1);
    std::function<void(size_t, const auto&, size_t)> copy_rec = [&](size_t dim, const auto& src, size_t stack_pos) {
        if (dim == ndim) {
            result(dst_idx) = src(src_idx);
            return;
        }
        size_t dst_dim = (dim < axis) ? dim : dim + 1;
        for (size_t i = 0; i < src.shape()[dim]; ++i) {
            src_idx[dim] = i;
            dst_idx[dst_dim] = i;
            copy_rec(dim + 1, src, stack_pos);
        }
    };
    // First array
    dst_idx[axis] = 0;
    copy_rec(0, a1, 0);
    // Second array
    dst_idx[axis] = 1;
    copy_rec(0, a2, 1);
    return result;
}

template <class E1, class E2, class... Es>
auto stack(const xexpression<E1>& e1, const xexpression<E2>& e2, const xexpression<Es>&... es) {
    auto first = stack(e1, e2);
    // For more than 2, we need to stack repeatedly (not efficient but correct)
    // Better to collect all and create result directly, but for simplicity we'll implement recursively.
    // Full implementation would require a different approach; we'll note that.
    XTU_THROW(std::runtime_error, "Stack with more than 2 arrays not yet implemented; use concatenate along new axis instead.");
}

// #############################################################################
// Split array into sub-arrays along an axis
// #############################################################################
template <class E>
std::vector<xarray_container<typename E::value_type>> 
split(const xexpression<E>& e, size_t sections, size_t axis = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
    size_t axis_size = expr.shape()[axis];
    XTU_ASSERT_MSG(axis_size % sections == 0, "Axis size must be divisible by number of sections");
    size_t section_size = axis_size / sections;
    
    std::vector<xarray_container<value_type>> result;
    for (size_t s = 0; s < sections; ++s) {
        std::vector<size_t> new_shape = expr.shape();
        new_shape[axis] = section_size;
        xarray_container<value_type> part(new_shape);
        
        std::vector<size_t> src_idx(ndim);
        std::vector<size_t> dst_idx(ndim);
        std::function<void(size_t)> copy_rec = [&](size_t dim) {
            if (dim == ndim) {
                part(dst_idx) = expr(src_idx);
                return;
            }
            for (size_t i = 0; i < new_shape[dim]; ++i) {
                src_idx[dim] = (dim == axis) ? i + s * section_size : i;
                dst_idx[dim] = i;
                copy_rec(dim + 1);
            }
        };
        copy_rec(0);
        result.push_back(std::move(part));
    }
    return result;
}

// #############################################################################
// Tile (repeat array)
// #############################################################################
template <class E>
auto tile(const xexpression<E>& e, const std::vector<size_t>& reps) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(reps.size() == ndim, "Number of repetitions must match dimension");
    
    std::vector<size_t> new_shape(ndim);
    for (size_t d = 0; d < ndim; ++d) new_shape[d] = expr.shape()[d] * reps[d];
    xarray_container<value_type> result(new_shape);
    
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t r = 0; r < reps[dim]; ++r) {
            for (size_t i = 0; i < expr.shape()[dim]; ++i) {
                src_idx[dim] = i;
                dst_idx[dim] = r * expr.shape()[dim] + i;
                copy_rec(dim + 1);
            }
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Repeat elements
// #############################################################################
template <class E>
auto repeat(const xexpression<E>& e, size_t repeats, size_t axis) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
    
    std::vector<size_t> new_shape = expr.shape();
    new_shape[axis] *= repeats;
    xarray_container<value_type> result(new_shape);
    
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            if (dim == axis) {
                for (size_t r = 0; r < repeats; ++r) {
                    dst_idx[dim] = i * repeats + r;
                    copy_rec(dim + 1);
                }
            } else {
                dst_idx[dim] = i;
                copy_rec(dim + 1);
            }
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Pad array with constant values
// #############################################################################
template <class E>
auto pad(const xexpression<E>& e, const std::vector<std::pair<size_t, size_t>>& pad_width,
         typename E::value_type constant_value = 0) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(pad_width.size() == ndim, "Pad width must match number of dimensions");
    
    std::vector<size_t> new_shape(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        new_shape[d] = expr.shape()[d] + pad_width[d].first + pad_width[d].second;
    }
    xarray_container<value_type> result(new_shape, constant_value);
    
    // Copy original data into center
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            dst_idx[dim] = i + pad_width[dim].first;
            copy_rec(dim + 1);
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Flip array along specified axis
// #############################################################################
template <class E>
auto flip(const xexpression<E>& e, size_t axis) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
    
    xarray_container<value_type> result(expr.shape());
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            dst_idx[dim] = (dim == axis) ? expr.shape()[dim] - 1 - i : i;
            copy_rec(dim + 1);
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Roll array elements along axis
// #############################################################################
template <class E>
auto roll(const xexpression<E>& e, int shift, size_t axis) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis < ndim, "Axis out of range");
    size_t axis_size = expr.shape()[axis];
    // Normalize shift
    int norm_shift = shift % static_cast<int>(axis_size);
    if (norm_shift < 0) norm_shift += axis_size;
    
    xarray_container<value_type> result(expr.shape());
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            if (dim == axis) {
                dst_idx[dim] = (i + norm_shift) % axis_size;
            } else {
                dst_idx[dim] = i;
            }
            copy_rec(dim + 1);
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Squeeze (remove dimensions of size 1)
// #############################################################################
template <class E>
auto squeeze(const xexpression<E>& e) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    
    // Determine which axes to keep
    std::vector<size_t> keep_axes;
    for (size_t d = 0; d < ndim; ++d) {
        if (expr.shape()[d] != 1) keep_axes.push_back(d);
    }
    // If all axes are size 1, keep one dimension of size 1
    if (keep_axes.empty()) keep_axes.push_back(0);
    
    std::vector<size_t> new_shape;
    for (size_t ax : keep_axes) new_shape.push_back(expr.shape()[ax]);
    xarray_container<value_type> result(new_shape);
    
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(keep_axes.size());
    std::function<void(size_t, size_t)> copy_rec = [&](size_t dim, size_t out_dim) {
        if (dim == ndim) {
            result(dst_idx) = expr(src_idx);
            return;
        }
        if (std::find(keep_axes.begin(), keep_axes.end(), dim) != keep_axes.end()) {
            for (size_t i = 0; i < expr.shape()[dim]; ++i) {
                src_idx[dim] = i;
                dst_idx[out_dim] = i;
                copy_rec(dim + 1, out_dim + 1);
            }
        } else {
            src_idx[dim] = 0;
            copy_rec(dim + 1, out_dim);
        }
    };
    copy_rec(0, 0);
    return result;
}

// #############################################################################
// Expand dimensions (add new axis of size 1)
// #############################################################################
template <class E>
auto expand_dims(const xexpression<E>& e, size_t axis) {
    const auto& expr = e.derived_cast();
    using value_type = typename E::value_type;
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis <= ndim, "Axis out of range");
    
    std::vector<size_t> new_shape(ndim + 1);
    for (size_t d = 0; d < axis; ++d) new_shape[d] = expr.shape()[d];
    new_shape[axis] = 1;
    for (size_t d = axis; d < ndim; ++d) new_shape[d + 1] = expr.shape()[d];
    xarray_container<value_type> result(new_shape);
    
    std::vector<size_t> src_idx(ndim);
    std::vector<size_t> dst_idx(ndim + 1);
    std::function<void(size_t)> copy_rec = [&](size_t dim) {
        if (dim == ndim) {
            dst_idx[axis] = 0;
            result(dst_idx) = expr(src_idx);
            return;
        }
        for (size_t i = 0; i < expr.shape()[dim]; ++i) {
            src_idx[dim] = i;
            size_t dst_dim = (dim < axis) ? dim : dim + 1;
            dst_idx[dst_dim] = i;
            copy_rec(dim + 1);
        }
    };
    copy_rec(0);
    return result;
}

// #############################################################################
// Swap axes
// #############################################################################
template <class E>
auto swapaxes(const xexpression<E>& e, size_t axis1, size_t axis2) {
    const auto& expr = e.derived_cast();
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(axis1 < ndim && axis2 < ndim, "Axis out of range");
    if (axis1 == axis2) return expr; // No change
    
    std::vector<size_t> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[axis1], perm[axis2]);
    return transpose(e, perm);
}

// #############################################################################
// Move axis to a new position
// #############################################################################
template <class E>
auto moveaxis(const xexpression<E>& e, size_t source, size_t destination) {
    const auto& expr = e.derived_cast();
    size_t ndim = expr.dimension();
    XTU_ASSERT_MSG(source < ndim && destination < ndim, "Axis out of range");
    if (source == destination) return expr;
    
    std::vector<size_t> perm;
    for (size_t d = 0; d < ndim; ++d) {
        if (d == source) continue;
        perm.push_back(d);
    }
    perm.insert(perm.begin() + destination, source);
    return transpose(e, perm);
}

} // namespace manipulation

// Bring manipulation functions into main namespace for convenience
using manipulation::reshape;
using manipulation::ravel;
using manipulation::transpose;
using manipulation::concatenate;
using manipulation::stack;
using manipulation::split;
using manipulation::tile;
using manipulation::repeat;
using manipulation::pad;
using manipulation::flip;
using manipulation::roll;
using manipulation::squeeze;
using manipulation::expand_dims;
using manipulation::swapaxes;
using manipulation::moveaxis;

XTU_NAMESPACE_END

#endif // XTU_MANIPULATION_XMANIPULATION_HPP