//File 0054 : core/xshape.hpp
//Shape utilities for validation, promotion, broadcasting, compile-time inference, and manipulation of multidimensional array shapes with C++17 constexpr support.
#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    namespace detail
    {
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
         * Compute the total number of elements from a shape.
         */
        template <class S>
        inline auto compute_size(const S& shape)
        {
            using size_type = typename S::value_type;
            if (shape.empty()) return size_type(0);
            return std::accumulate(shape.begin(), shape.end(), size_type(1),
                                   std::multiplies<size_type>());
        }

        /**
         * Promote two shapes to their common broadcast shape.
         */
        template <class S>
        inline S broadcast_shape_impl(const S& s1, const S& s2)
        {
            if (s1.empty()) return s2;
            if (s2.empty()) return s1;
            if (s1.size() != s2.size())
                throw std::runtime_error("Incompatible shape dimensions for broadcasting.");

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
                    throw std::runtime_error("Incompatible shapes for broadcasting.");
            }
            return result;
        }

        /**
         * Variadic broadcast shape.
         */
        template <class S>
        inline S broadcast_shape_variadic(const S& first) { return first; }

        template <class S, class... Shapes>
        inline S broadcast_shape_variadic(const S& first, const S& second, const Shapes&... rest)
        {
            return broadcast_shape_variadic(broadcast_shape_impl(first, second), rest...);
        }
    }

    /**
     * Promote multiple shapes to their common broadcast shape.
     */
    template <class S1, class S2, class... S>
    inline S1 broadcast_shape(const S1& s1, const S2& s2, const S&... rest)
    {
        return detail::broadcast_shape_variadic(s1, s2, rest...);
    }

    /**
     * Check if a shape is a scalar (all dimensions 1).
     */
    template <class S>
    inline bool is_scalar_shape(const S& shape) noexcept
    {
        return shape.empty() || std::all_of(shape.begin(), shape.end(),
                                            [](auto dim) { return dim == 1; });
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
     * Compute the shape resulting from stacking arrays along a new axis.
     */
    template <class S>
    inline S stack_shape(const S& base_shape, std::size_t axis, std::size_t n_arrays)
    {
        S new_shape = base_shape;
        new_shape.insert(new_shape.begin() + std::min(axis, new_shape.size()), n_arrays);
        return new_shape;
    }

    /**
     * Compute the shape resulting from concatenating arrays along an existing axis.
     */
    template <class S, class It>
    inline S concatenate_shape(const S& base_shape, std::size_t axis, It sizes_begin, It sizes_end)
    {
        S new_shape = base_shape;
        new_shape[axis] = std::accumulate(sizes_begin, sizes_end,
                                          typename S::value_type(0));
        return new_shape;
    }

    /**
     * Squeeze shape: remove dimensions of size 1.
     */
    template <class S>
    inline S squeeze_shape(const S& shape, std::ptrdiff_t axis = -1)
    {
        S result;
        if (axis == -1)
        {
            for (auto dim : shape)
                if (dim != 1) result.push_back(dim);
        }
        else
        {
            std::size_t ax = static_cast<std::size_t>(axis);
            if (ax >= shape.size()) throw std::out_of_range("squeeze_shape: axis out of bounds.");
            if (shape[ax] != 1) throw std::runtime_error("squeeze_shape: dimension must be 1.");
            for (std::size_t i = 0; i < shape.size(); ++i)
                if (i != ax) result.push_back(shape[i]);
        }
        return result;
    }

    /**
     * Expand dims shape: add a dimension of size 1 at the given axis.
     */
    template <class S>
    inline S expand_dims_shape(const S& shape, std::size_t axis)
    {
        S new_shape = shape;
        new_shape.insert(new_shape.begin() + std::min(axis, new_shape.size()), 1);
        return new_shape;
    }

    /**
     * Repeat shape: multiply the size of a given axis by repeats.
     */
    template <class S>
    inline S repeat_shape(const S& shape, std::size_t repeats, std::ptrdiff_t axis)
    {
        S new_shape = shape;
        std::size_t ax = (axis >= 0) ? static_cast<std::size_t>(axis) : shape.size() + axis;
        if (ax >= shape.size()) throw std::out_of_range("repeat_shape: axis out of bounds.");
        new_shape[ax] *= repeats;
        return new_shape;
    }

    /**
     * Tile shape: multiply each dimension by the corresponding repetition factor.
     */
    template <class S>
    inline S tile_shape(const S& shape, const S& reps)
    {
        S new_shape(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            std::size_t rep = (i < reps.size()) ? reps[i] : 1;
            new_shape[i] = shape[i] * rep;
        }
        return new_shape;
    }

    /**
     * Flatten shape: collapse all dimensions into one.
     */
    template <class S>
    inline auto flatten_shape(const S& shape)
    {
        using size_type = typename S::value_type;
        std::vector<size_type> result(1);
        result[0] = detail::compute_size(shape);
        return result;
    }

    /**
     * Reshape validation: check that new shape has the same total number of elements.
     */
    template <class S1, class S2>
    inline bool is_valid_reshape(const S1& old_shape, const S2& new_shape)
    {
        return detail::compute_size(old_shape) == detail::compute_size(new_shape);
    }

    /**
     * Check that a multi-dimensional index is within bounds for a given shape.
     */
    template <class S, class Index>
    inline bool in_bounds(const S& shape, const Index& index) noexcept
    {
        if (index.size() != shape.size()) return false;
        for (std::size_t i = 0; i < shape.size(); ++i)
            if (index[i] >= shape[i]) return false;
        return true;
    }

    /**
     * Convert a shape from one container type to another.
     */
    template <class TargetContainer, class SourceContainer>
    inline TargetContainer convert_shape(const SourceContainer& source)
    {
        return TargetContainer(source.begin(), source.end());
    }

    /**
     * Compile-time shape operations (for fixed-dimension arrays).
     */
    template <class S, std::size_t NewSize>
    struct reshape_shape_fixed;

    template <std::size_t... Dims, std::size_t NewSize>
    struct reshape_shape_fixed<std::array<std::size_t, sizeof...(Dims)>, NewSize>
    {
        static_assert(sizeof...(Dims) > 0, "Cannot reshape empty shape.");
        // Compile-time reshape not fully supported; use runtime.
    };

    /**
     * Prepend dimensions of size 1 to match a given rank.
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
     * Remove leading dimensions of size 1 (reduce rank if possible).
     */
    template <class S>
    inline S demote_shape(const S& shape)
    {
        auto first_not_one = std::find_if(shape.begin(), shape.end(),
                                          [](auto dim) { return dim != 1; });
        return S(first_not_one, shape.end());
    }

    /**
     * Get the stride vector for a contiguous layout given a shape.
     */
    template <class S>
    inline S contiguous_strides(const S& shape, layout_type l = DEFAULT_LAYOUT)
    {
        S strides(shape.size());
        if (l == layout_type::row_major)
        {
            if (shape.empty()) return strides;
            strides[shape.size() - 1] = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2; i >= 0; --i)
            {
                strides[static_cast<std::size_t>(i)] = strides[static_cast<std::size_t>(i) + 1] *
                                                       shape[static_cast<std::size_t>(i) + 1];
            }
        }
        else
        {
            if (shape.empty()) return strides;
            strides[0] = 1;
            for (std::size_t i = 1; i < shape.size(); ++i)
            {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
        return strides;
    }

    /**
     * Compute the offset of a flat index in a multi-dimensional array given strides.
     */
    template <class S, class Index>
    inline auto ravel_index(const Index& multi_index, const S& strides)
    {
        using size_type = typename S::value_type;
        size_type offset = 0;
        for (std::size_t i = 0; i < multi_index.size(); ++i)
            offset += multi_index[i] * strides[i];
        return offset;
    }

    /**
     * Convert a flat index to a multi-dimensional index given a shape.
     */
    template <class S>
    inline S unravel_index(typename S::value_type flat_index, const S& shape,
                           layout_type l = DEFAULT_LAYOUT)
    {
        S result(shape.size());
        if (l == layout_type::row_major)
        {
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i)
            {
                result[static_cast<std::size_t>(i)] = flat_index % shape[static_cast<std::size_t>(i)];
                flat_index /= shape[static_cast<std::size_t>(i)];
            }
        }
        else
        {
            for (std::size_t i = 0; i < shape.size(); ++i)
            {
                result[i] = flat_index % shape[i];
                flat_index /= shape[i];
            }
        }
        return result;
    }

    /**
     * Converts an initializer_list of initializer_lists to a shape (for nested initializers).
     */
    template <class T>
    inline std::vector<std::size_t> shape_from_initializer_list(
        const std::initializer_list<std::initializer_list<T>>& init)
    {
        std::vector<std::size_t> result(2);
        result[0] = init.size();
        result[1] = init.begin()->size();
        return result;
    }

} // namespace xt

#endif // XTENSOR_XSHAPE_HPP