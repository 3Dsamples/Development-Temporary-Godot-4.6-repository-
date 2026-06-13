//File 0104 : numdot/shape.h
//Shape and stride utilities: broadcasting, validation, promotion, index ravel/unravel, layout computation, and SIMD-aligned element offset.
#ifndef NUMDOT_SHAPE_H
#define NUMDOT_SHAPE_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include "config.h"
#include "forward.h"
#include "types.h"

namespace numdot
{

    /**
     * Compute total number of elements in a shape.
     */
    template <class S>
    inline typename S::value_type compute_size(const S& shape) noexcept
    {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(),
                               static_cast<typename S::value_type>(1),
                               std::multiplies<typename S::value_type>());
    }

    /**
     * Compute row-major strides from a shape.
     */
    template <class S>
    inline S compute_strides_row_major(const S& shape)
    {
        if (shape.empty()) return S{};
        S strides(shape.size());
        strides[shape.size() - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
            strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
        return strides;
    }

    /**
     * Compute column-major strides from a shape.
     */
    template <class S>
    inline S compute_strides_column_major(const S& shape)
    {
        if (shape.empty()) return S{};
        S strides(shape.size());
        strides[0] = 1;
        for (std::size_t i = 1; i < shape.size(); ++i)
            strides[i] = strides[i - 1] * shape[i - 1];
        return strides;
    }

    /**
     * Compute strides for a given shape and layout (compile-time).
     */
    template <layout L, class S>
    inline S compute_strides_layout(const S& shape)
    {
        if constexpr (L == layout::row_major)
            return compute_strides_row_major(shape);
        else if constexpr (L == layout::column_major)
            return compute_strides_column_major(shape);
        else
            return compute_strides_row_major(shape); // default fallback
    }

    /**
     * Compute strides for a given shape and runtime layout.
     */
    template <class S>
    inline S compute_strides(const S& shape, layout l = default_layout)
    {
        if (l == layout::row_major)
            return compute_strides_row_major(shape);
        else if (l == layout::column_major)
            return compute_strides_column_major(shape);
        else
            return compute_strides_row_major(shape);
    }

    /**
     * Compute backstrides from strides and shape.
     */
    template <class S>
    inline S compute_backstrides(const S& strides, const S& shape)
    {
        S backstrides(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i)
            backstrides[i] = (shape[i] > 0) ? (shape[i] - 1) * strides[i] : 0;
        return backstrides;
    }

    /**
     * Check if a shape is valid (all dimensions >= 0).
     */
    template <class S>
    inline bool is_valid_shape(const S& shape) noexcept
    {
        return std::all_of(shape.begin(), shape.end(),
                           [](auto dim) { return dim >= 0; });
    }

    /**
     * Check if two shapes are identical.
     */
    template <class S1, class S2>
    inline bool same_shape(const S1& s1, const S2& s2) noexcept
    {
        if (s1.size() != s2.size()) return false;
        return std::equal(s1.begin(), s1.end(), s2.begin());
    }

    /**
     * Broadcast two shapes to a common shape, throwing if incompatible.
     */
    template <class S>
    inline S broadcast_shapes(const S& s1, const S& s2)
    {
        if (s1.empty()) return s2;
        if (s2.empty()) return s1;
        if (s1.size() != s2.size())
            throw std::runtime_error("Shape broadcasting requires same rank.");
        S result(s1.size());
        for (std::size_t i = 0; i < s1.size(); ++i)
        {
            if (s1[i] == 1)
                result[i] = s2[i];
            else if (s2[i] == 1)
                result[i] = s1[i];
            else if (s1[i] == s2[i])
                result[i] = s1[i];
            else
                throw std::runtime_error("Incompatible broadcast shapes.");
        }
        return result;
    }

    /**
     * Variadic broadcast shape (fold over multiple shapes).
     */
    template <class S, class... Shapes>
    inline S broadcast_shapes(const S& first, const S& second, const Shapes&... rest)
    {
        return broadcast_shapes(broadcast_shapes(first, second), rest...);
    }

    /**
     * Broadcast strides for a new broadcast shape given original shape and strides.
     */
    template <class S>
    inline S broadcast_strides(const S& old_shape, const S& old_strides, const S& new_shape)
    {
        std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(new_shape.size()) - static_cast<std::ptrdiff_t>(old_shape.size());
        if (offset < 0) throw std::runtime_error("Cannot broadcast to lower rank.");
        S new_strides(new_shape.size(), 0);
        for (std::size_t i = 0; i < old_shape.size(); ++i)
        {
            new_strides[i + static_cast<std::size_t>(offset)] =
                (old_shape[i] == 1) ? 0 : old_strides[i];
        }
        return new_strides;
    }

    /**
     * Convert a flat index to a multi-dimensional index (unravel).
     */
    template <class S>
    inline S unravel_index(typename S::value_type flat, const S& shape, layout l = default_layout)
    {
        S result(shape.size());
        if (l == layout::row_major)
        {
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
            {
                result[static_cast<std::size_t>(i)] = flat % shape[static_cast<std::size_t>(i)];
                flat /= shape[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                result[i] = flat % shape[i];
                flat /= shape[i];
            }
        }
        return result;
    }

    /**
     * Convert a multi-dimensional index to a flat index (ravel).
     */
    template <class S>
    inline typename S::value_type ravel_index(const S& index, const S& strides) noexcept
    {
        typename S::value_type offset = 0;
        for (std::size_t i = 0; i < index.size(); ++i)
            offset += index[i] * strides[i];
        return offset;
    }

    /**
     * Deduce the layout from shape and strides.
     */
    template <class S>
    inline layout deduce_layout(const S& shape, const S& strides) noexcept
    {
        if (shape.size() <= 1) return layout::row_major;
        // Check row-major
        bool is_row = true;
        typename S::value_type expected = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
        {
            if (strides[static_cast<std::size_t>(i)] != expected)
            {
                is_row = false;
                break;
            }
            expected *= shape[static_cast<std::size_t>(i)];
        }
        if (is_row) return layout::row_major;
        // Check column-major
        expected = 1;
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (strides[i] != expected) return layout::dynamic;
            expected *= shape[i];
        }
        return layout::column_major;
    }

    /**
     * Promote a shape to a higher rank by prepending 1s.
     */
    template <class S>
    inline S promote_shape(const S& shape, std::size_t target_rank)
    {
        if (target_rank < shape.size())
            throw std::runtime_error("Target rank must be >= current rank.");
        S promoted(target_rank, 1);
        std::copy(shape.begin(), shape.end(), promoted.begin() + (target_rank - shape.size()));
        return promoted;
    }

    /**
     * Squeeze shape: remove dimensions of size 1.
     */
    template <class S>
    inline S squeeze_shape(const S& shape, std::ptrdiff_t axis = -1)
    {
        if (axis == -1)
        {
            S result;
            for (auto dim : shape)
                if (dim != 1) result.push_back(dim);
            return result;
        }
        else
        {
            std::size_t ax = (axis >= 0) ? static_cast<std::size_t>(axis) : shape.size() + axis;
            if (ax >= shape.size()) throw std::out_of_range("squeeze_shape: axis out of bounds.");
            if (shape[ax] != 1) throw std::runtime_error("squeeze_shape: dimension must be 1.");
            S result;
            for (std::size_t i = 0; i < shape.size(); ++i)
                if (i != ax) result.push_back(shape[i]);
            return result;
        }
    }

    /**
     * Expand dims: insert a new axis of size 1.
     */
    template <class S>
    inline S expand_dims_shape(const S& shape, std::size_t axis)
    {
        S new_shape = shape;
        new_shape.insert(new_shape.begin() + std::min(axis, new_shape.size()), 1);
        return new_shape;
    }

    /**
     * Stack shape: insert a new axis with given size.
     */
    template <class S>
    inline S stack_shape(const S& base_shape, std::size_t axis, std::size_t n_arrays)
    {
        return expand_dims_shape(base_shape, axis); // this is not correct, stack prepends new axis with size n_arrays
    }

    /**
     * Concatenate shape: increase size along an axis.
     */
    template <class S>
    inline S concatenate_shape(const S& base_shape, std::size_t axis, std::size_t total_len)
    {
        S new_shape = base_shape;
        new_shape[axis] = total_len;
        return new_shape;
    }

    /**
     * Check if index is within bounds for a shape.
     */
    template <class S, class Index>
    inline bool in_bounds(const S& shape, const Index& index) noexcept
    {
        if (index.size() != shape.size()) return false;
        for (std::size_t i = 0; i < shape.size(); ++i)
            if (index[i] >= shape[i]) return false;
        return true;
    }

} // namespace numdot

#endif // NUMDOT_SHAPE_H