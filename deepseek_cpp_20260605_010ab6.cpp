//File 0011 (UPDATED) : core/xmanipulation.hpp
//Array manipulation: transpose, flip, roll, concatenate, stack, split, squeeze, expand_dims, repeat, tile with SIMD memory movement and full axis handling.
#ifndef XTENSOR_XMANIPULATION_HPP
#define XTENSOR_XMANIPULATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xstrided_view.hpp"
#include "xreducer.hpp"
#include "xaccumulator.hpp"
#include "xeval.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /****************************************
     * Transpose
     ****************************************/
    namespace detail
    {
        template <class shape_type, class strides_type>
        inline strides_type transpose_strides(const shape_type& shape, const strides_type& strides,
                                              const std::vector<std::size_t>& permutation)
        {
            strides_type new_strides(shape.size());
            for (std::size_t i = 0; i < permutation.size(); ++i)
                new_strides[i] = strides[permutation[i]];
            return new_strides;
        }
    }

    /**
     * Returns a view with axes permuted according to the given permutation.
     * If permutation is empty, reverses axes.
     */
    template <class E>
    inline auto transpose(E&& e, const std::vector<std::size_t>& permutation = {})
    {
        using expr_type = std::decay_t<E>;
        auto old_shape = e.shape();
        std::size_t ndim = old_shape.size();

        if (permutation.empty())
        {
            std::vector<std::size_t> rev(ndim);
            for (std::size_t i = 0; i < ndim; ++i) rev[i] = ndim - 1 - i;
            return transpose(std::forward<E>(e), rev);
        }

        if (permutation.size() != ndim)
            throw std::runtime_error("Transpose permutation size must match dimensionality.");

        // Validate permutation
        std::vector<bool> seen(ndim, false);
        for (auto p : permutation)
        {
            if (p >= ndim) throw std::runtime_error("Invalid permutation index.");
            if (seen[p]) throw std::runtime_error("Duplicate in permutation.");
            seen[p] = true;
        }

        // New shape
        std::vector<std::size_t> new_shape(ndim);
        for (std::size_t i = 0; i < ndim; ++i)
            new_shape[i] = old_shape[permutation[i]];

        auto old_strides = compute_strides(old_shape, DEFAULT_LAYOUT);
        auto new_strides = detail::transpose_strides(old_shape, old_strides, permutation);

        return strided_view(std::forward<E>(e), new_shape, new_strides, 0, DEFAULT_LAYOUT);
    }

    /****************************************
     * Swap axes
     ****************************************/
    template <class E>
    inline auto swapaxes(E&& e, std::size_t axis1, std::size_t axis2)
    {
        auto ndim = e.dimension();
        if (axis1 >= ndim || axis2 >= ndim)
            throw std::runtime_error("swapaxes: axis out of bounds.");
        std::vector<std::size_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[axis1], perm[axis2]);
        return transpose(std::forward<E>(e), perm);
    }

    /****************************************
     * Flip (reverse along axis)
     ****************************************/
    template <class E>
    inline auto flip(const E& e, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (axis >= ndim)
            throw std::runtime_error("Flip axis out of bounds.");

        auto result = xt::eval(e);
        auto& res_data = result.storage();

        // Calculate strides
        std::size_t outer_stride = 1;
        for (std::size_t i = axis + 1; i < ndim; ++i) outer_stride *= shape[i];
        std::size_t block_size = outer_stride;
        std::size_t axis_len = shape[axis];
        std::size_t stride_axis = outer_stride;
        std::size_t outer_loop = result.size() / (axis_len * outer_stride);

        if constexpr (is_simd_enabled_v<value_type>)
        {
            using simd_type = xsimd::batch<value_type, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                for (std::size_t j = 0; j < axis_len / 2; ++j)
                {
                    auto idx1 = base + j * stride_axis;
                    auto idx2 = base + (axis_len - 1 - j) * stride_axis;
                    for (std::size_t b = 0; b < block_size; ++b)
                    {
                        std::size_t simd_b = b / simd_size * simd_size;
                        if (b % simd_size == 0 && b + simd_size <= block_size)
                        {
                            simd_type v1 = simd_type::load_unaligned(&res_data[idx1 + b]);
                            simd_type v2 = simd_type::load_unaligned(&res_data[idx2 + b]);
                            v2.store_unaligned(&res_data[idx1 + b]);
                            v1.store_unaligned(&res_data[idx2 + b]);
                            b += simd_size - 1;
                        }
                        else
                        {
                            std::swap(res_data[idx1 + b], res_data[idx2 + b]);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                for (std::size_t j = 0; j < axis_len / 2; ++j)
                {
                    auto idx1 = base + j * stride_axis;
                    auto idx2 = base + (axis_len - 1 - j) * stride_axis;
                    for (std::size_t b = 0; b < block_size; ++b)
                        std::swap(res_data[idx1 + b], res_data[idx2 + b]);
                }
            }
        }
        return result;
    }

    /****************************************
     * Roll (shift along axis)
     ****************************************/
    template <class E>
    inline auto roll(const E& e, std::ptrdiff_t shift, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (axis >= ndim)
            throw std::runtime_error("Roll axis out of bounds.");

        std::size_t axis_len = shape[axis];
        if (axis_len == 0) return xt::eval(e);
        shift = ((shift % static_cast<std::ptrdiff_t>(axis_len)) + static_cast<std::ptrdiff_t>(axis_len)) % static_cast<std::ptrdiff_t>(axis_len);
        if (shift == 0) return xt::eval(e);

        auto result = xt::eval(e);
        auto& res_data = result.storage();

        std::size_t outer_stride = 1;
        for (std::size_t i = axis + 1; i < ndim; ++i) outer_stride *= shape[i];
        std::size_t block_size = outer_stride;
        std::size_t stride_axis = outer_stride;
        std::size_t outer_loop = result.size() / (axis_len * outer_stride);

        std::vector<value_type> temp(block_size);
        if constexpr (is_simd_enabled_v<value_type>)
        {
            using simd_type = xsimd::batch<value_type, default_simd_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                // Save the first 'shift' blocks
                for (std::size_t s = 0; s < static_cast<std::size_t>(shift); ++s)
                {
                    auto src = base + s * stride_axis;
                    std::copy(&res_data[src], &res_data[src + block_size], temp.begin() + s * block_size);
                }
                // Shift remaining blocks left
                for (std::size_t j = 0; j < axis_len - static_cast<std::size_t>(shift); ++j)
                {
                    auto src = base + (j + static_cast<std::size_t>(shift)) * stride_axis;
                    auto dst = base + j * stride_axis;
                    for (std::size_t b = 0; b < block_size; b += simd_size)
                    {
                        if (b + simd_size <= block_size)
                        {
                            simd_type v = simd_type::load_unaligned(&res_data[src + b]);
                            v.store_unaligned(&res_data[dst + b]);
                        }
                        else
                        {
                            std::copy(&res_data[src + b], &res_data[src + block_size], &res_data[dst + b]);
                            break;
                        }
                    }
                }
                // Place saved blocks at end
                for (std::size_t j = axis_len - static_cast<std::size_t>(shift); j < axis_len; ++j)
                {
                    auto dst = base + j * stride_axis;
                    auto t_src = (j - (axis_len - static_cast<std::size_t>(shift))) * block_size;
                    for (std::size_t b = 0; b < block_size; b += simd_size)
                    {
                        if (b + simd_size <= block_size)
                        {
                            simd_type v = simd_type::load_unaligned(temp.data() + t_src + b);
                            v.store_unaligned(&res_data[dst + b]);
                        }
                        else
                        {
                            std::copy(temp.begin() + t_src + b, temp.begin() + t_src + block_size, &res_data[dst + b]);
                            break;
                        }
                    }
                }
            }
        }
        else
        {
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                // Save first shift blocks
                for (std::size_t s = 0; s < static_cast<std::size_t>(shift); ++s)
                    std::copy(&res_data[base + s * stride_axis],
                              &res_data[base + s * stride_axis + block_size],
                              temp.begin() + s * block_size);
                // Shift remaining left
                for (std::size_t j = 0; j < axis_len - static_cast<std::size_t>(shift); ++j)
                    std::copy(&res_data[base + (j + static_cast<std::size_t>(shift)) * stride_axis],
                              &res_data[base + (j + static_cast<std::size_t>(shift)) * stride_axis + block_size],
                              &res_data[base + j * stride_axis]);
                // Copy saved to end
                for (std::size_t j = axis_len - static_cast<std::size_t>(shift); j < axis_len; ++j)
                    std::copy(temp.begin() + (j - (axis_len - static_cast<std::size_t>(shift))) * block_size,
                              temp.begin() + (j - (axis_len - static_cast<std::size_t>(shift)) + 1) * block_size,
                              &res_data[base + j * stride_axis]);
            }
        }
        return result;
    }

    /****************************************
     * Concatenate
     ****************************************/
    template <class... Es>
    inline auto concatenate(std::size_t axis, Es&&... args)
    {
        auto arrays = std::make_tuple(xt::eval(std::forward<Es>(args))...);
        auto first_shape = std::get<0>(arrays).shape();
        std::size_t ndim = first_shape.size();
        if (axis >= ndim)
            throw std::runtime_error("Concatenate axis out of bounds.");

        using value_type = typename std::tuple_element<0, decltype(arrays)>::type::value_type;
        using container = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>;

        std::size_t total_axis_len = 0;
        std::apply([&](auto&&... arr) { ((total_axis_len += arr.shape()[axis]), ...); }, arrays);

        auto new_shape = first_shape;
        new_shape[axis] = total_axis_len;
        container result(new_shape);

        std::size_t axis_offset = 0;
        std::apply([&](auto&&... arr) {
            auto copy_one = [&](auto& a) {
                auto a_shape = a.shape();
                std::size_t a_axis_len = a_shape[axis];
                std::size_t block_size = 1;
                for (std::size_t i = axis + 1; i < a_shape.size(); ++i) block_size *= a_shape[i];
                std::size_t outer_stride = 1;
                for (std::size_t i = 0; i < axis; ++i) outer_stride *= a_shape[i];

                const value_type* src = a.data();
                value_type* dst = result.data();
                std::size_t total_blocks = a.size() / (a_axis_len * block_size);

                for (std::size_t i = 0; i < total_blocks; ++i)
                {
                    std::size_t src_start = i * a_axis_len * block_size;
                    std::size_t dst_start = i * total_axis_len * block_size + axis_offset * block_size;
                    std::copy(src + src_start, src + src_start + a_axis_len * block_size, dst + dst_start);
                }
                axis_offset += a_axis_len;
            };
            (copy_one(arr), ...);
        }, arrays);
        return result;
    }

    /****************************************
     * Stack (join along new axis)
     ****************************************/
    template <class... Es>
    inline auto stack(std::size_t axis, Es&&... args)
    {
        auto arrays = std::make_tuple(xt::eval(std::forward<Es>(args))...);
        auto first_shape = std::get<0>(arrays).shape();
        using value_type = typename std::tuple_element<0, decltype(arrays)>::type::value_type;
        std::size_t n_arrays = sizeof...(Es);
        std::size_t ndim = first_shape.size();

        // Validate all shapes equal
        std::apply([&](auto&&... arr) {
            ([&](auto& a) {
                if (a.shape() != first_shape)
                    throw std::runtime_error("stack: all arrays must have same shape.");
            }(arr), ...);
        }, arrays);

        std::vector<std::size_t> new_shape = first_shape;
        new_shape.insert(new_shape.begin() + std::min(axis, ndim), n_arrays);
        using container = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        container result(new_shape);

        std::size_t block_size_before = 1;
        for (std::size_t i = 0; i < axis; ++i) block_size_before *= first_shape[i];
        std::size_t block_size_after = 1;
        for (std::size_t i = axis; i < ndim; ++i) block_size_after *= first_shape[i];
        std::size_t total_elements = first_shape.empty() ? 1 :
            std::accumulate(first_shape.begin(), first_shape.end(), std::size_t(1), std::multiplies<std::size_t>());

        std::size_t array_idx = 0;
        std::apply([&](auto&&... arr) {
            auto copy_one = [&](auto& a) {
                const value_type* src = a.data();
                value_type* dst = result.data();
                for (std::size_t i = 0; i < total_elements; ++i)
                {
                    std::size_t outer = i / block_size_after;
                    std::size_t inner = i % block_size_after;
                    std::size_t dst_idx = outer * (n_arrays * block_size_after) + array_idx * block_size_after + inner;
                    dst[dst_idx] = src[i];
                }
                ++array_idx;
            };
            (copy_one(arr), ...);
        }, arrays);
        return result;
    }

    template <class... Es> inline auto hstack(Es&&... args) { return concatenate(1, std::forward<Es>(args)...); }
    template <class... Es> inline auto vstack(Es&&... args) { return concatenate(0, std::forward<Es>(args)...); }
    template <class... Es> inline auto dstack(Es&&... args) { return concatenate(2, std::forward<Es>(args)...); }

    /****************************************
     * Split
     ****************************************/
    template <class E>
    inline auto split(const E& e, std::size_t sections, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (axis >= ndim) throw std::runtime_error("Split axis out of bounds.");
        std::size_t axis_len = shape[axis];
        if (axis_len % sections != 0)
            throw std::runtime_error("Split sections must evenly divide axis length.");
        std::size_t sub_len = axis_len / sections;
        std::vector<xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>> result;
        result.reserve(sections);
        auto full_data = e.data();
        std::size_t block_size = 1;
        for (std::size_t i = axis + 1; i < ndim; ++i) block_size *= shape[i];
        std::size_t outer_stride = 1;
        for (std::size_t i = 0; i < axis; ++i) outer_stride *= shape[i];
        for (std::size_t s = 0; s < sections; ++s)
        {
            auto sub_shape = shape;
            sub_shape[axis] = sub_len;
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> sub(sub_shape);
            auto* dst = sub.data();
            for (std::size_t o = 0; o < outer_stride; ++o)
            {
                std::size_t src_start = o * axis_len * block_size + s * sub_len * block_size;
                std::size_t dst_start = o * sub_len * block_size;
                std::copy(full_data + src_start, full_data + src_start + sub_len * block_size, dst + dst_start);
            }
            result.push_back(std::move(sub));
        }
        return result;
    }

    template <class E> inline auto hsplit(const E& e, std::size_t sections) { return split(e, sections, 1); }
    template <class E> inline auto vsplit(const E& e, std::size_t sections) { return split(e, sections, 0); }
    template <class E> inline auto dsplit(const E& e, std::size_t sections) { return split(e, sections, 2); }

    /****************************************
     * Squeeze (remove dimensions of size 1)
     ****************************************/
    template <class E>
    inline auto squeeze(E&& e, std::ptrdiff_t axis = -1)
    {
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        std::vector<std::size_t> new_shape;
        if (axis == -1)
        {
            for (auto s : shape)
                if (s != 1) new_shape.push_back(s);
        }
        else
        {
            std::size_t ax = static_cast<std::size_t>(axis);
            if (ax >= ndim) throw std::runtime_error("squeeze: axis out of bounds.");
            if (shape[ax] != 1) throw std::runtime_error("squeeze: dimension at given axis must be 1.");
            for (std::size_t i = 0; i < ndim; ++i)
                if (i != ax) new_shape.push_back(shape[i]);
        }
        auto old_strides = compute_strides(shape);
        auto new_strides = std::vector<std::size_t>();
        // Recompute strides from shape? Use strided_view
        return strided_view(std::forward<E>(e), new_shape, compute_strides(new_shape), 0, DEFAULT_LAYOUT);
    }

    /****************************************
     * Expand dims (add size-1 dimension at axis)
     ****************************************/
    template <class E>
    inline auto expand_dims(E&& e, std::size_t axis)
    {
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (axis > ndim) throw std::runtime_error("expand_dims: axis out of bounds.");
        std::vector<std::size_t> new_shape = shape;
        new_shape.insert(new_shape.begin() + axis, 1);
        auto new_strides = compute_strides(new_shape);
        return strided_view(std::forward<E>(e), new_shape, new_strides, 0, DEFAULT_LAYOUT);
    }

    /****************************************
     * Repeat (repeat elements of an array)
     ****************************************/
    template <class E>
    inline auto repeat(const E& e, std::size_t repeats, std::ptrdiff_t axis = -1)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto arr = xt::eval(e);
        auto shape = arr.shape();
        std::size_t ndim = shape.size();

        if (axis == -1)
        {
            // Flatten, repeat, reshape
            std::vector<std::size_t> new_shape = {arr.size() * repeats};
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
            auto* dst = result.data();
            const auto* src = arr.data();
            for (std::size_t i = 0; i < arr.size(); ++i)
                for (std::size_t r = 0; r < repeats; ++r)
                    *dst++ = src[i];
            return result;
        }
        else
        {
            std::size_t ax = static_cast<std::size_t>(axis);
            if (ax >= ndim) throw std::runtime_error("repeat: axis out of bounds.");
            auto new_shape = shape;
            new_shape[ax] *= repeats;
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
            auto* dst = result.data();
            const auto* src = arr.data();
            // Iterate and copy repeating along axis
            std::vector<std::size_t> idx(ndim, 0);
            std::function<void(std::size_t)> copy_rec = [&](std::size_t dim) {
                if (dim == ndim) return;
                // not fully generic, but works for contiguous data by exploiting loops
            };
            // Simplified: just do repeated copy block-wise
            std::size_t outer = 1;
            for (std::size_t i = 0; i < ax; ++i) outer *= shape[i];
            std::size_t inner = 1;
            for (std::size_t i = ax + 1; i < ndim; ++i) inner *= shape[i];
            std::size_t axis_len = shape[ax];
            for (std::size_t o = 0; o < outer; ++o)
            {
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    for (std::size_t r = 0; r < repeats; ++r)
                    {
                        std::size_t src_start = o * axis_len * inner + i * inner;
                        std::size_t dst_start = o * axis_len * repeats * inner + (i * repeats + r) * inner;
                        std::copy(src + src_start, src + src_start + inner, dst + dst_start);
                    }
                }
            }
            return result;
        }
    }

    /****************************************
     * Tile (repeat array to construct a tiled one)
     ****************************************/
    template <class E>
    inline auto tile(const E& e, const std::vector<std::size_t>& reps)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        std::size_t ndim = shape.size();
        if (reps.size() > ndim)
            throw std::runtime_error("tile: reps length cannot exceed array dimensions.");

        std::vector<std::size_t> new_shape(ndim);
        for (std::size_t i = 0; i < ndim; ++i)
        {
            std::size_t rep = (i < reps.size()) ? reps[i] : 1;
            new_shape[i] = shape[i] * rep;
        }

        xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
        auto* dst = result.data();
        const auto* src = e.data();

        // Nested loops to copy tiles
        std::vector<std::size_t> dst_idx(ndim, 0);
        std::function<void(std::size_t)> tile_rec = [&](std::size_t dim) {
            if (dim == ndim)
            {
                // Compute source index
                std::size_t src_linear = 0;
                std::size_t stride = 1;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
                {
                    src_linear += (dst_idx[d] % shape[d]) * stride;
                    stride *= shape[d];
                }
                // Compute dst linear index
                std::size_t dst_linear = 0;
                stride = 1;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d)
                {
                    dst_linear += dst_idx[d] * stride;
                    stride *= new_shape[d];
                }
                dst[dst_linear] = src[src_linear];
                return;
            }
            for (std::size_t i = 0; i < new_shape[dim]; ++i)
            {
                dst_idx[dim] = i;
                tile_rec(dim + 1);
            }
        };
        tile_rec(0);
        return result;
    }

} // namespace xt

#endif // XTENSOR_XMANIPULATION_HPP