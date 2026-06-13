//File 0011 : core/xmanipulation.hpp
//Array manipulation: transpose, flip, roll, concatenate, stack, split with SIMD-accelerated memory movement and views.
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
        /**
         * Compute permuted strides for transpose.
         */
        template <class shape_type, class strides_type>
        inline strides_type transpose_strides(const shape_type& shape, const strides_type& strides,
                                              const std::vector<std::size_t>& permutation)
        {
            strides_type new_strides(shape.size());
            for (std::size_t i = 0; i < permutation.size(); ++i)
            {
                new_strides[i] = strides[permutation[i]];
            }
            return new_strides;
        }
    }

    /**
     * Returns a view with axes permuted according to the given permutation.
     */
    template <class E>
    inline auto transpose(E&& e, const std::vector<std::size_t>& permutation = {})
    {
        using expr_type = std::decay_t<E>;
        auto old_shape = e.shape();
        if (permutation.empty())
        {
            // default reverse axes
            std::vector<std::size_t> rev(old_shape.size());
            std::iota(rev.rbegin(), rev.rend(), 0);
            return transpose(std::forward<E>(e), rev);
        }
        if (permutation.size() != old_shape.size())
            throw std::runtime_error("Transpose permutation size does not match dimensionality.");
        // new shape
        std::vector<std::size_t> new_shape(old_shape.size());
        for (std::size_t i = 0; i < permutation.size(); ++i)
            new_shape[i] = old_shape[permutation[i]];
        auto old_strides = xt::compute_strides<DEFAULT_LAYOUT>(old_shape); // assuming default layout
        auto new_strides = detail::transpose_strides(old_shape, old_strides, permutation);
        // Create strided view with new shape and new strides
        return strided_view(std::forward<E>(e), new_shape, new_strides, 0, DEFAULT_LAYOUT);
    }

    /****************************************
     * Flip (reverse along axis)
     ****************************************/
    /**
     * Flip array along given axis, returning a new array with reversed data.
     */
    template <class E>
    inline auto flip(const E& e, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        if (axis >= shape.size())
            throw std::runtime_error("Flip axis out of bounds.");
        auto result = xt::eval(e); // full copy
        auto& res_data = result.storage();
        std::size_t outer_stride = 1;
        for (std::size_t i = axis + 1; i < shape.size(); ++i)
            outer_stride *= shape[i];
        std::size_t block_size = outer_stride;
        std::size_t axis_len = shape[axis];
        std::size_t stride_axis = outer_stride;
        std::size_t outer_loop = result.size() / (axis_len * outer_stride);
        // Use SIMD for copying flipped blocks if value_type supports SIMD
        if constexpr (is_simd_enabled_v<value_type>)
        {
            using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                for (std::size_t j = 0; j < axis_len / 2; ++j)
                {
                    auto idx1 = base + j * stride_axis;
                    auto idx2 = base + (axis_len - 1 - j) * stride_axis;
                    // swap blocks of block_size elements using SIMD
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
    /**
     * Roll array elements along a given axis, wrapping around.
     */
    template <class E>
    inline auto roll(const E& e, std::ptrdiff_t shift, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        if (axis >= shape.size())
            throw std::runtime_error("Roll axis out of bounds.");
        std::size_t axis_len = shape[axis];
        if (axis_len == 0) return xt::eval(e);
        shift = ((shift % static_cast<std::ptrdiff_t>(axis_len)) + axis_len) % axis_len; // positive shift
        if (shift == 0) return xt::eval(e);
        auto result = xt::eval(e);
        auto& res_data = result.storage();
        std::size_t outer_stride = 1;
        for (std::size_t i = axis + 1; i < shape.size(); ++i)
            outer_stride *= shape[i];
        std::size_t block_size = outer_stride;
        std::size_t stride_axis = outer_stride;
        std::size_t outer_loop = result.size() / (axis_len * outer_stride);
        std::vector<value_type> temp(block_size);
        // Use SIMD for copying blocks when shifting
        if constexpr (is_simd_enabled_v<value_type>)
        {
            using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
            constexpr std::size_t simd_size = simd_type::size;
            for (std::size_t i = 0; i < outer_loop; ++i)
            {
                auto base = i * axis_len * stride_axis;
                // save first block
                std::copy(&res_data[base], &res_data[base + block_size], temp.begin());
                // shift blocks left by shift
                for (std::size_t j = 0; j < axis_len - shift; ++j)
                {
                    auto src = base + (j + shift) * stride_axis;
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
                // place saved block at the end
                for (std::size_t j = axis_len - shift; j < axis_len; ++j)
                {
                    auto dst = base + j * stride_axis;
                    for (std::size_t b = 0; b < block_size; b += simd_size)
                    {
                        if (b + simd_size <= block_size)
                        {
                            simd_type v = simd_type::load_unaligned(temp.data() + b);
                            v.store_unaligned(&res_data[dst + b]);
                        }
                        else
                        {
                            std::copy(temp.begin() + b, temp.end(), &res_data[dst + b]);
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
                std::copy(&res_data[base], &res_data[base + block_size], temp.begin());
                for (std::size_t j = 0; j < axis_len - shift; ++j)
                    std::copy(&res_data[base + (j + shift) * stride_axis],
                              &res_data[base + (j + shift) * stride_axis + block_size],
                              &res_data[base + j * stride_axis]);
                for (std::size_t j = axis_len - shift; j < axis_len; ++j)
                    std::copy(temp.begin(), temp.end(), &res_data[base + j * stride_axis]);
            }
        }
        return result;
    }

    /****************************************
     * Concatenate
     ****************************************/
    /**
     * Concatenate multiple arrays along an existing axis.
     */
    template <class... Es>
    inline auto concatenate(std::size_t axis, Es&&... args)
    {
        auto arrays = std::make_tuple(xt::eval(std::forward<Es>(args))...);
        auto first_shape = std::get<0>(arrays).shape();
        if (axis >= first_shape.size())
            throw std::runtime_error("Concatenate axis out of bounds.");
        using value_type = typename std::tuple_element<0, decltype(arrays)>::type::value_type;
        using container = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        std::size_t total_axis_len = 0;
        std::apply([&](auto&&... arr) {
            ((total_axis_len += arr.shape()[axis]), ...);
        }, arrays);
        auto new_shape = first_shape;
        new_shape[axis] = total_axis_len;
        container result(new_shape);
        std::size_t axis_offset = 0;
        std::apply([&](auto&&... arr) {
            auto copy_one = [&](auto& a) {
                auto a_shape = a.shape();
                std::size_t a_axis_len = a_shape[axis];
                std::size_t block_size = 1;
                for (std::size_t i = axis + 1; i < a_shape.size(); ++i)
                    block_size *= a_shape[i];
                std::size_t outer_stride_before = 1;
                for (std::size_t i = 0; i < axis; ++i)
                    outer_stride_before *= a_shape[i];
                // Copy all blocks to result using SIMD
                const value_type* src = a.data();
                value_type* dst = result.data() + axis_offset * block_size;
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
    /**
     * Stack arrays along a new axis.
     */
    template <class... Es>
    inline auto stack(std::size_t axis, Es&&... args)
    {
        auto arrays = std::make_tuple(xt::eval(std::forward<Es>(args))...);
        auto first_shape = std::get<0>(arrays).shape();
        using value_type = typename std::tuple_element<0, decltype(arrays)>::type::value_type;
        std::size_t n_arrays = sizeof...(Es);
        std::vector<std::size_t> new_shape = first_shape;
        new_shape.insert(new_shape.begin() + axis, n_arrays);
        using container = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>;
        container result(new_shape);
        std::size_t block_size_before = 1;
        for (std::size_t i = 0; i < axis; ++i) block_size_before *= first_shape[i];
        std::size_t block_size_after = 1;
        for (std::size_t i = axis; i < first_shape.size(); ++i) block_size_after *= first_shape[i];
        std::size_t total_elements = first_shape.empty() ? 1 : std::accumulate(first_shape.begin(), first_shape.end(), std::size_t(1), std::multiplies<std::size_t>());
        std::size_t array_idx = 0;
        std::apply([&](auto&&... arr) {
            auto copy_one = [&](auto& a) {
                const value_type* src = a.data();
                value_type* dst = result.data();
                // Place this array's data at appropriate positions
                for (std::size_t i = 0; i < total_elements; ++i)
                {
                    // Map linear index i in original array to new index with extra axis = array_idx
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

    /**
     * Convenience: hstack
     */
    template <class... Es>
    inline auto hstack(Es&&... args)
    {
        return concatenate(1, std::forward<Es>(args)...);
    }

    /**
     * Convenience: vstack
     */
    template <class... Es>
    inline auto vstack(Es&&... args)
    {
        return concatenate(0, std::forward<Es>(args)...);
    }

    /**
     * Convenience: dstack (depth stack, axis=2)
     */
    template <class... Es>
    inline auto dstack(Es&&... args)
    {
        return concatenate(2, std::forward<Es>(args)...);
    }

    /****************************************
     * Split (divide array along axis)
     ****************************************/
    /**
     * Split an array into multiple sub-arrays along an axis.
     */
    template <class E>
    inline auto split(const E& e, std::size_t sections, std::size_t axis)
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto shape = e.shape();
        if (axis >= shape.size())
            throw std::runtime_error("Split axis out of bounds.");
        std::size_t axis_len = shape[axis];
        if (axis_len % sections != 0)
            throw std::runtime_error("Split sections must evenly divide axis length.");
        std::size_t sub_len = axis_len / sections;
        std::vector<xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>>> result;
        result.reserve(sections);
        auto full_data = e.data();
        std::size_t block_size = 1;
        for (std::size_t i = axis + 1; i < shape.size(); ++i)
            block_size *= shape[i];
        std::size_t outer_stride = 1;
        for (std::size_t i = 0; i < axis; ++i)
            outer_stride *= shape[i];
        for (std::size_t s = 0; s < sections; ++s)
        {
            auto sub_shape = shape;
            sub_shape[axis] = sub_len;
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> sub(sub_shape);
            auto* dst = sub.data();
            // copy all blocks corresponding to this section
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

    /**
     * Convenience: hsplit (split along axis=1)
     */
    template <class E>
    inline auto hsplit(const E& e, std::size_t sections)
    {
        return split(e, sections, 1);
    }

    /**
     * Convenience: vsplit (split along axis=0)
     */
    template <class E>
    inline auto vsplit(const E& e, std::size_t sections)
    {
        return split(e, sections, 0);
    }

    /**
     * Convenience: dsplit (split along axis=2)
     */
    template <class E>
    inline auto dsplit(const E& e, std::size_t sections)
    {
        return split(e, sections, 2);
    }

}  // namespace xt

#endif  // XTENSOR_XMANIPULATION_HPP