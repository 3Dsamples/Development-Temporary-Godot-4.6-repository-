//File 0114 : numdot/strides.h
//Stride computation utilities, layout handling, broadcasting stride mapping, and SIMD-aligned memory access helpers.
#ifndef NUMDOT_STRIDES_H
#define NUMDOT_STRIDES_H

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "config.h"
#include "forward.h"
#include "types.h"
#include "shape.h"

namespace numdot
{
    /**
     * Compute row-major strides for a given shape.
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
     * Compute column-major strides for a given shape.
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
     * Compute strides based on compile-time layout.
     */
    template <layout L, class S>
    inline S compute_strides_layout(const S& shape)
    {
        if constexpr (L == layout::row_major)
            return compute_strides_row_major(shape);
        else if constexpr (L == layout::column_major)
            return compute_strides_column_major(shape);
        else
            return compute_strides_row_major(shape);
    }

    /**
     * Compute strides for a given shape at runtime.
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
     * Compute backstrides (max offset for negative indexing).
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
     * Broadcast strides from old shape/strides to a new broadcast shape.
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
     * Adapt strides when reshaping an array (keeps same total size, contiguous layout).
     */
    template <class S>
    inline void adapt_strides(const S& new_shape, S& strides, layout l = default_layout)
    {
        strides = compute_strides(new_shape, l);
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
     * Check if strides describe a contiguous layout.
     */
    template <class S>
    inline bool is_contiguous(const S& shape, const S& strides, layout l = default_layout) noexcept
    {
        if (shape.empty()) return true;
        auto expected = compute_strides(shape, l);
        return std::equal(strides.begin(), strides.end(), expected.begin());
    }

    /**
     * Align an offset to a given boundary (SIMD friendly).
     */
    inline std::size_t align_offset(std::size_t offset, std::size_t alignment = simd_alignment) noexcept
    {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Return the minimal stride that is a multiple of SIMD alignment for a given type.
     */
    template <class T>
    constexpr std::size_t simd_stride() noexcept
    {
        return simd_alignment / sizeof(T);
    }

    /**
     * Compute the total number of elements accessible from a shape and strides (unused, but available).
     */
    template <class S>
    inline typename S::value_type compute_data_size(const S& shape, const S& strides) noexcept
    {
        if (shape.empty()) return 0;
        typename S::value_type max_offset = 0;
        for (std::size_t i = 0; i < shape.size(); ++i)
            max_offset += (shape[i] - 1) * strides[i];
        return max_offset + 1;
    }

    /**
     * Convert a multi-dimensional index to a flat offset using strides.
     */
    template <class S>
    inline typename S::value_type index_to_offset(const S& index, const S& strides) noexcept
    {
        typename S::value_type offset = 0;
        for (std::size_t i = 0; i < index.size(); ++i)
            offset += index[i] * strides[i];
        return offset;
    }

    /**
     * Convert a flat offset to a multi-dimensional index given shape and strides (unravel).
     */
    template <class S>
    inline S offset_to_index(typename S::value_type offset, const S& shape, const S& strides,
                             layout l = default_layout)
    {
        S index(shape.size());
        if (l == layout::row_major)
        {
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
            {
                index[static_cast<std::size_t>(i)] = offset / strides[static_cast<std::size_t>(i)];
                offset %= strides[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                index[i] = offset / strides[i];
                offset %= strides[i];
            }
        }
        return index;
    }

} // namespace numdot

#endif // NUMDOT_STRIDES_H