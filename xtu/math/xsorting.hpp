// math/xsorting.hpp

#ifndef XTENSOR_XSORTING_HPP
#define XTENSOR_XSORTING_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <cmath>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        // --------------------------------------------------------------------
        // Sorting algorithms for tensors
        // --------------------------------------------------------------------
        
        namespace detail
        {
            // Helper to get the sorting axis
            inline std::size_t get_sort_axis(std::ptrdiff_t axis, std::size_t ndim)
            {
                return normalize_axis(axis, ndim);
            }
            
            // Copy elements along a specified axis into a 1D buffer for sorting
            template <class E>
            void extract_axis_slice(const xexpression<E>& expr,
                                    std::size_t axis,
                                    std::size_t slice_index,
                                    std::vector<typename E::value_type>& buffer)
            {
                const auto& e = expr.derived_cast();
                std::size_t ndim = e.dimension();
                std::size_t axis_len = e.shape()[axis];
                buffer.resize(axis_len);
                
                // Compute base index for the slice
                std::vector<std::size_t> coords(ndim, 0);
                std::size_t temp = slice_index;
                for (std::size_t d = 0; d < ndim; ++d)
                {
                    if (d == axis) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < ndim; ++k)
                        if (k != axis) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                
                // Fill buffer
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    coords[axis] = i;
                    buffer[i] = e.element(coords);
                }
            }
            
            // Write sorted buffer back to the expression along the axis
            template <class E>
            void write_axis_slice(xexpression<E>& expr,
                                  std::size_t axis,
                                  std::size_t slice_index,
                                  const std::vector<typename E::value_type>& buffer)
            {
                auto& e = expr.derived_cast();
                std::size_t ndim = e.dimension();
                std::size_t axis_len = e.shape()[axis];
                
                std::vector<std::size_t> coords(ndim, 0);
                std::size_t temp = slice_index;
                for (std::size_t d = 0; d < ndim; ++d)
                {
                    if (d == axis) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < ndim; ++k)
                        if (k != axis) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    coords[axis] = i;
                    e.element(coords) = buffer[i];
                }
            }
            
            // Get number of slices along an axis
            template <class E>
            std::size_t get_num_slices(const xexpression<E>& expr, std::size_t axis)
            {
                const auto& e = expr.derived_cast();
                std::size_t total = e.size();
                std::size_t axis_len = e.shape()[axis];
                return total / axis_len;
            }
            
        } // namespace detail
        
        // --------------------------------------------------------------------
        // sort - in-place sort along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline void sort(xexpression<E>& expr, std::size_t axis = 0)
        {
            auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::sort(buffer.begin(), buffer.end());
                detail::write_axis_slice(e, ax, slice, buffer);
            }
        }
        
        template <class E, class Compare>
        inline void sort(xexpression<E>& expr, std::size_t axis, Compare comp)
        {
            auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::sort(buffer.begin(), buffer.end(), comp);
                detail::write_axis_slice(e, ax, slice, buffer);
            }
        }
        
        // --------------------------------------------------------------------
        // argsort - return indices that would sort the array along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto argsort(const xexpression<E>& expr, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            using shape_type = typename std::decay_t<decltype(e)>::shape_type;
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            shape_type result_shape = e.shape();
            
            // Create result array of indices (same shape as input)
            xarray_container<size_type> result(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            std::vector<std::pair<value_type, size_type>> indexed_buffer(axis_len);
            std::vector<value_type> buffer;
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                
                // Create pairs of (value, original_index)
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indexed_buffer[i] = {buffer[i], i};
                }
                
                // Sort by value
                std::sort(indexed_buffer.begin(), indexed_buffer.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                
                // Write indices back
                std::vector<size_type> indices(axis_len);
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indices[i] = indexed_buffer[i].second;
                }
                
                // Write to result along the axis
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < e.dimension(); ++k)
                        if (k != ax) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    coords[ax] = i;
                    result.element(coords) = indices[i];
                }
            }
            
            return result;
        }
        
        template <class E, class Compare>
        inline auto argsort(const xexpression<E>& expr, std::size_t axis, Compare comp)
        {
            const auto& e = expr.derived_cast();
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            using shape_type = typename std::decay_t<decltype(e)>::shape_type;
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            shape_type result_shape = e.shape();
            xarray_container<size_type> result(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            std::vector<std::pair<value_type, size_type>> indexed_buffer(axis_len);
            std::vector<value_type> buffer;
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indexed_buffer[i] = {buffer[i], i};
                }
                
                std::sort(indexed_buffer.begin(), indexed_buffer.end(),
                          [&comp](const auto& a, const auto& b) { return comp(a.first, b.first); });
                
                std::vector<size_type> indices(axis_len);
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indices[i] = indexed_buffer[i].second;
                }
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < e.dimension(); ++k)
                        if (k != ax) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    coords[ax] = i;
                    result.element(coords) = indices[i];
                }
            }
            
            return result;
        }
        
        // --------------------------------------------------------------------
        // partition - in-place partial sort along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline void partition(xexpression<E>& expr, std::size_t kth, std::size_t axis = 0)
        {
            auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            if (kth >= axis_len)
            {
                XTENSOR_THROW(std::out_of_range, "partition: kth out of range");
            }
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::nth_element(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(kth), buffer.end());
                detail::write_axis_slice(e, ax, slice, buffer);
            }
        }
        
        template <class E, class Compare>
        inline void partition(xexpression<E>& expr, std::size_t kth, std::size_t axis, Compare comp)
        {
            auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            if (kth >= axis_len)
            {
                XTENSOR_THROW(std::out_of_range, "partition: kth out of range");
            }
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::nth_element(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(kth), buffer.end(), comp);
                detail::write_axis_slice(e, ax, slice, buffer);
            }
        }
        
        // --------------------------------------------------------------------
        // argpartition - indices that would partially sort
        // --------------------------------------------------------------------
        template <class E>
        inline auto argpartition(const xexpression<E>& expr, std::size_t kth, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            using shape_type = typename std::decay_t<decltype(e)>::shape_type;
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t axis_len = e.shape()[ax];
            
            if (kth >= axis_len)
            {
                XTENSOR_THROW(std::out_of_range, "argpartition: kth out of range");
            }
            
            shape_type result_shape = e.shape();
            xarray_container<size_type> result(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            
            std::vector<std::pair<value_type, size_type>> indexed_buffer(axis_len);
            std::vector<value_type> buffer;
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indexed_buffer[i] = {buffer[i], i};
                }
                
                std::nth_element(indexed_buffer.begin(),
                                 indexed_buffer.begin() + static_cast<std::ptrdiff_t>(kth),
                                 indexed_buffer.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
                
                std::vector<size_type> indices(axis_len);
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indices[i] = indexed_buffer[i].second;
                }
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < e.dimension(); ++k)
                        if (k != ax) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    coords[ax] = i;
                    result.element(coords) = indices[i];
                }
            }
            
            return result;
        }
        
        // --------------------------------------------------------------------
        // unique - find unique elements (flattened)
        // --------------------------------------------------------------------
        template <class E>
        inline auto unique(const xexpression<E>& expr)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            std::vector<value_type> flattened(e.size());
            std::copy(e.begin(), e.end(), flattened.begin());
            
            std::sort(flattened.begin(), flattened.end());
            auto last = std::unique(flattened.begin(), flattened.end());
            flattened.erase(last, flattened.end());
            
            return flattened;
        }
        
        template <class E>
        inline auto unique(const xexpression<E>& expr, bool return_index, bool return_inverse, bool return_counts)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            
            std::vector<value_type> flattened(e.size());
            std::copy(e.begin(), e.end(), flattened.begin());
            
            // Create indexed pairs for stable unique
            std::vector<std::pair<value_type, size_type>> indexed(flattened.size());
            for (size_type i = 0; i < flattened.size(); ++i)
            {
                indexed[i] = {flattened[i], i};
            }
            
            std::sort(indexed.begin(), indexed.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            
            std::vector<value_type> uniq_vals;
            std::vector<size_type> uniq_indices;
            std::vector<size_type> inverse(flattened.size());
            std::vector<size_type> counts;
            
            if (indexed.empty())
            {
                if (return_index && return_inverse && return_counts)
                    return std::make_tuple(uniq_vals, uniq_indices, inverse, counts);
                else if (return_index && return_inverse)
                    return std::make_tuple(uniq_vals, uniq_indices, inverse);
                else if (return_index && return_counts)
                    return std::make_tuple(uniq_vals, uniq_indices, counts);
                else if (return_inverse && return_counts)
                    return std::make_tuple(uniq_vals, inverse, counts);
                else if (return_index)
                    return std::make_tuple(uniq_vals, uniq_indices);
                else if (return_inverse)
                    return std::make_tuple(uniq_vals, inverse);
                else if (return_counts)
                    return std::make_tuple(uniq_vals, counts);
                else
                    return uniq_vals;
            }
            
            uniq_vals.push_back(indexed[0].first);
            if (return_index) uniq_indices.push_back(indexed[0].second);
            size_type current_label = 0;
            inverse[indexed[0].second] = current_label;
            size_type current_count = 1;
            
            for (size_type i = 1; i < indexed.size(); ++i)
            {
                if (indexed[i].first != indexed[i-1].first)
                {
                    if (return_counts) counts.push_back(current_count);
                    current_count = 1;
                    ++current_label;
                    uniq_vals.push_back(indexed[i].first);
                    if (return_index) uniq_indices.push_back(indexed[i].second);
                }
                else
                {
                    ++current_count;
                }
                inverse[indexed[i].second] = current_label;
            }
            if (return_counts) counts.push_back(current_count);
            
            // Return appropriate tuple based on requested outputs
            if (return_index && return_inverse && return_counts)
                return std::make_tuple(uniq_vals, uniq_indices, inverse, counts);
            else if (return_index && return_inverse)
                return std::make_tuple(uniq_vals, uniq_indices, inverse);
            else if (return_index && return_counts)
                return std::make_tuple(uniq_vals, uniq_indices, counts);
            else if (return_inverse && return_counts)
                return std::make_tuple(uniq_vals, inverse, counts);
            else if (return_index)
                return std::make_tuple(uniq_vals, uniq_indices);
            else if (return_inverse)
                return std::make_tuple(uniq_vals, inverse);
            else if (return_counts)
                return std::make_tuple(uniq_vals, counts);
            else
                return uniq_vals;
        }
        
        // --------------------------------------------------------------------
        // top_k - find k largest elements along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto top_k(const xexpression<E>& expr, std::size_t k, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t axis_len = e.shape()[ax];
            
            if (k > axis_len)
            {
                XTENSOR_THROW(std::out_of_range, "top_k: k larger than axis size");
            }
            
            auto result_shape = e.shape();
            result_shape[ax] = k;
            xarray_container<value_type> values(result_shape);
            xarray_container<size_type> indices(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::vector<value_type> buffer;
            std::vector<std::pair<value_type, size_type>> indexed(axis_len);
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indexed[i] = {buffer[i], i};
                }
                
                std::partial_sort(indexed.begin(),
                                  indexed.begin() + static_cast<std::ptrdiff_t>(k),
                                  indexed.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t kk = d + 1; kk < e.dimension(); ++kk)
                        if (kk != ax) stride_after *= e.shape()[kk];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                for (std::size_t i = 0; i < k; ++i)
                {
                    coords[ax] = i;
                    values.element(coords) = indexed[i].first;
                    indices.element(coords) = indexed[i].second;
                }
            }
            
            return std::make_pair(values, indices);
        }
        
        // --------------------------------------------------------------------
        // bottom_k - find k smallest elements along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto bottom_k(const xexpression<E>& expr, std::size_t k, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t axis_len = e.shape()[ax];
            
            if (k > axis_len)
            {
                XTENSOR_THROW(std::out_of_range, "bottom_k: k larger than axis size");
            }
            
            auto result_shape = e.shape();
            result_shape[ax] = k;
            xarray_container<value_type> values(result_shape);
            xarray_container<size_type> indices(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::vector<value_type> buffer;
            std::vector<std::pair<value_type, size_type>> indexed(axis_len);
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                for (std::size_t i = 0; i < axis_len; ++i)
                {
                    indexed[i] = {buffer[i], i};
                }
                
                std::partial_sort(indexed.begin(),
                                  indexed.begin() + static_cast<std::ptrdiff_t>(k),
                                  indexed.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t kk = d + 1; kk < e.dimension(); ++kk)
                        if (kk != ax) stride_after *= e.shape()[kk];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                for (std::size_t i = 0; i < k; ++i)
                {
                    coords[ax] = i;
                    values.element(coords) = indexed[i].first;
                    indices.element(coords) = indexed[i].second;
                }
            }
            
            return std::make_pair(values, indices);
        }
        
        // --------------------------------------------------------------------
        // median - median along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto median(const xexpression<E>& expr, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            auto result_shape = e.shape();
            result_shape[ax] = 1;
            
            xarray_container<value_type> result(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            std::vector<value_type> buffer;
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::sort(buffer.begin(), buffer.end());
                
                value_type med;
                if (axis_len % 2 == 1)
                {
                    med = buffer[axis_len / 2];
                }
                else
                {
                    med = (buffer[axis_len / 2 - 1] + buffer[axis_len / 2]) / static_cast<value_type>(2);
                }
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < e.dimension(); ++k)
                        if (k != ax) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                coords[ax] = 0;
                result.element(coords) = med;
            }
            
            return result;
        }
        
        // --------------------------------------------------------------------
        // quantile - q-th quantile along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline auto quantile(const xexpression<E>& expr, double q, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            
            if (q < 0.0 || q > 1.0)
            {
                XTENSOR_THROW(std::invalid_argument, "quantile: q must be in [0,1]");
            }
            
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            auto result_shape = e.shape();
            result_shape[ax] = 1;
            
            xarray_container<value_type> result(result_shape);
            
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            std::vector<value_type> buffer;
            
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                std::sort(buffer.begin(), buffer.end());
                
                double pos = q * static_cast<double>(axis_len - 1);
                std::size_t idx_low = static_cast<std::size_t>(std::floor(pos));
                std::size_t idx_high = static_cast<std::size_t>(std::ceil(pos));
                double frac = pos - std::floor(pos);
                
                value_type q_val;
                if (idx_low == idx_high)
                {
                    q_val = buffer[idx_low];
                }
                else
                {
                    q_val = buffer[idx_low] * (1.0 - frac) + buffer[idx_high] * frac;
                }
                
                std::vector<std::size_t> coords(e.dimension(), 0);
                std::size_t temp = slice;
                for (std::size_t d = 0; d < e.dimension(); ++d)
                {
                    if (d == ax) continue;
                    std::size_t stride_after = 1;
                    for (std::size_t k = d + 1; k < e.dimension(); ++k)
                        if (k != ax) stride_after *= e.shape()[k];
                    coords[d] = temp / stride_after;
                    temp %= stride_after;
                }
                coords[ax] = 0;
                result.element(coords) = q_val;
            }
            
            return result;
        }
        
        // --------------------------------------------------------------------
        // percentile - p-th percentile along an axis (equivalent to quantile * 100)
        // --------------------------------------------------------------------
        template <class E>
        inline auto percentile(const xexpression<E>& expr, double p, std::size_t axis = 0)
        {
            if (p < 0.0 || p > 100.0)
            {
                XTENSOR_THROW(std::invalid_argument, "percentile: p must be in [0,100]");
            }
            return quantile(expr, p / 100.0, axis);
        }
        
        // --------------------------------------------------------------------
        // sort flattened
        // --------------------------------------------------------------------
        template <class E>
        inline auto sort_flattened(const xexpression<E>& expr)
        {
            auto result = eval(expr);
            std::sort(result.begin(), result.end());
            return result;
        }
        
        template <class E, class Compare>
        inline auto sort_flattened(const xexpression<E>& expr, Compare comp)
        {
            auto result = eval(expr);
            std::sort(result.begin(), result.end(), comp);
            return result;
        }
        
        // --------------------------------------------------------------------
        // is_sorted - check if array is sorted along an axis
        // --------------------------------------------------------------------
        template <class E>
        inline bool is_sorted(const xexpression<E>& expr, std::size_t axis = 0)
        {
            const auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                if (!std::is_sorted(buffer.begin(), buffer.end()))
                {
                    return false;
                }
            }
            return true;
        }
        
        template <class E, class Compare>
        inline bool is_sorted(const xexpression<E>& expr, std::size_t axis, Compare comp)
        {
            const auto& e = expr.derived_cast();
            std::size_t ax = detail::get_sort_axis(static_cast<std::ptrdiff_t>(axis), e.dimension());
            std::size_t num_slices = detail::get_num_slices(e, ax);
            std::size_t axis_len = e.shape()[ax];
            
            std::vector<typename E::value_type> buffer;
            for (std::size_t slice = 0; slice < num_slices; ++slice)
            {
                detail::extract_axis_slice(e, ax, slice, buffer);
                if (!std::is_sorted(buffer.begin(), buffer.end(), comp))
                {
                    return false;
                }
            }
            return true;
        }
        
        // --------------------------------------------------------------------
        // searchsorted - find indices where elements should be inserted
        // --------------------------------------------------------------------
        template <class E, class V>
        inline auto searchsorted(const xexpression<E>& sorted_expr, const V& values, bool side_left = true)
        {
            const auto& e = sorted_expr.derived_cast();
            using value_type = typename std::decay_t<decltype(e)>::value_type;
            using size_type = typename std::decay_t<decltype(e)>::size_type;
            
            // Ensure the input is sorted
            if (!is_sorted(e, 0))
            {
                XTENSOR_THROW(std::runtime_error, "searchsorted: input array must be sorted");
            }
            
            std::vector<value_type> flat_sorted(e.size());
            std::copy(e.begin(), e.end(), flat_sorted.begin());
            
            // Handle scalar or array values
            if constexpr (std::is_arithmetic<V>::value)
            {
                auto it = side_left 
                    ? std::lower_bound(flat_sorted.begin(), flat_sorted.end(), values)
                    : std::upper_bound(flat_sorted.begin(), flat_sorted.end(), values);
                return static_cast<size_type>(std::distance(flat_sorted.begin(), it));
            }
            else
            {
                const auto& v_expr = values.derived_cast();
                std::vector<size_type> result(v_expr.size());
                for (std::size_t i = 0; i < v_expr.size(); ++i)
                {
                    value_type val = v_expr.flat(i);
                    auto it = side_left
                        ? std::lower_bound(flat_sorted.begin(), flat_sorted.end(), val)
                        : std::upper_bound(flat_sorted.begin(), flat_sorted.end(), val);
                    result[i] = static_cast<size_type>(std::distance(flat_sorted.begin(), it));
                }
                xarray_container<size_type> res(v_expr.shape());
                std::copy(result.begin(), result.end(), res.begin());
                return res;
            }
        }
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XSORTING_HPP

// math/xsorting.hpp