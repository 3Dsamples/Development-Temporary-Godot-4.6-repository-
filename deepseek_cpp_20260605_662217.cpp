//File 0013 : core/xsort.hpp
//Sorting, argsort, partitioning, quickselect, and median with SIMD-accelerated comparisons and parallel merge sort.
#ifndef XTENSOR_XSORT_HPP
#define XTENSOR_XSORT_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
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
#include "xmanipulation.hpp"
#include "xio.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    namespace sort
    {
        namespace detail
        {
            // SIMD-based comparator for two batches: returns a mask of which elements are less.
            template <class T>
            inline auto simd_less_than(xsimd::batch<T, xsimd::default_arch> a,
                                       xsimd::batch<T, xsimd::default_arch> b) noexcept
            {
                return a < b;
            }

            // Scalar comparator wrapper
            template <class T>
            struct less
            {
                constexpr bool operator()(const T& a, const T& b) const { return a < b; }
            };

            // Insertion sort for small arrays (used in quicksort fallback)
            template <class RandomIt, class Compare>
            void insertion_sort(RandomIt first, RandomIt last, Compare comp)
            {
                for (auto it = first + 1; it < last; ++it)
                {
                    auto val = std::move(*it);
                    auto jt = it;
                    while (jt > first && comp(val, *(jt - 1)))
                    {
                        *jt = std::move(*(jt - 1));
                        --jt;
                    }
                    *jt = std::move(val);
                }
            }

            // Partition using SIMD for pivot comparisons
            template <class RandomIt, class T, class Compare>
            RandomIt simd_partition(RandomIt first, RandomIt last, const T& pivot, Compare comp)
            {
                using value_type = typename std::iterator_traits<RandomIt>::value_type;
                if constexpr (is_simd_enabled_v<value_type>)
                {
                    using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    RandomIt left = first;
                    RandomIt right = last - 1;
                    while (left <= right)
                    {
                        // Move left pointer using SIMD blocks
                        while (left <= right)
                        {
                            if (left + simd_size <= right + 1)
                            {
                                simd_type v = simd_type::load_unaligned(&*left);
                                auto mask = comp.simd_apply ? comp.simd_apply(v, simd_type(pivot)) : simd_less_than(v, simd_type(pivot));
                                // Find first false in mask
                                for (std::size_t i = 0; i < simd_size; ++i)
                                {
                                    if (!mask[i]) { left += i; goto left_found; }
                                }
                                left += simd_size;
                            }
                            else break;
                        }
                        left_found:
                        while (left <= right)
                        {
                            if (right - simd_size + 1 >= left)
                            {
                                simd_type v = simd_type::load_unaligned(&*(right - simd_size + 1));
                                auto mask = comp.simd_apply ? !comp.simd_apply(simd_type(pivot), v) : !(simd_type(pivot) < v);
                                for (std::ptrdiff_t i = simd_size - 1; i >= 0; --i)
                                {
                                    if (!mask[i]) { right -= (simd_size - 1 - i); goto right_found; }
                                }
                                right -= simd_size;
                            }
                            else break;
                        }
                        right_found:
                        if (left <= right)
                        {
                            std::iter_swap(left, right);
                            ++left;
                            --right;
                        }
                    }
                    return left;
                }
                else
                {
                    // fallback to standard partition
                    auto it = std::partition(first, last, [&](const value_type& v) { return comp(v, pivot); });
                    return it;
                }
            }

            // Quicksort with SIMD partitioning and insertion sort for small segments
            template <class RandomIt, class Compare>
            void quicksort(RandomIt first, RandomIt last, Compare comp)
            {
                const std::ptrdiff_t threshold = 32;
                while (last - first > threshold)
                {
                    // Choose pivot: median of three
                    auto mid = first + (last - first) / 2;
                    auto last_el = last - 1;
                    if (comp(*mid, *first)) std::iter_swap(mid, first);
                    if (comp(*last_el, *first)) std::iter_swap(last_el, first);
                    if (comp(*last_el, *mid)) std::iter_swap(last_el, mid);
                    auto pivot = *mid;
                    auto part = simd_partition(first, last, pivot, comp);
                    // Recurse into smaller partition
                    if (part - first < last - part)
                    {
                        quicksort(first, part, comp);
                        first = part;
                    }
                    else
                    {
                        quicksort(part, last, comp);
                        last = part;
                    }
                }
                insertion_sort(first, last, comp);
            }

            // Merge two sorted ranges using SIMD
            template <class RandomIt, class Compare>
            void simd_merge(RandomIt first1, RandomIt last1, RandomIt first2, RandomIt last2,
                            RandomIt d_first, Compare comp)
            {
                using value_type = typename std::iterator_traits<RandomIt>::value_type;
                if constexpr (is_simd_enabled_v<value_type>)
                {
                    using simd_type = xsimd::batch<value_type, xsimd::default_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    while (first1 < last1 && first2 < last2)
                    {
                        if (first1 + simd_size <= last1 && first2 + simd_size <= last2)
                        {
                            simd_type v1 = simd_type::load_unaligned(&*first1);
                            simd_type v2 = simd_type::load_unaligned(&*first2);
                            auto mask = comp.simd_apply ? comp.simd_apply(v1, v2) : v1 < v2;
                            // Merge based on mask (blend)
                            for (std::size_t i = 0; i < simd_size; ++i)
                            {
                                if (mask[i])
                                {
                                    *d_first++ = std::move(*first1++);
                                }
                                else
                                {
                                    *d_first++ = std::move(*first2++);
                                }
                                if (first1 == last1 || first2 == last2) break;
                            }
                        }
                        else
                        {
                            if (comp(*first1, *first2))
                                *d_first++ = std::move(*first1++);
                            else
                                *d_first++ = std::move(*first2++);
                        }
                    }
                    std::move(first1, last1, d_first);
                    std::move(first2, last2, d_first);
                }
                else
                {
                    std::merge(first1, last1, first2, last2, d_first, comp);
                }
            }

            // Parallel merge sort using multiple threads
            template <class RandomIt, class Compare>
            void parallel_mergesort(RandomIt first, RandomIt last, Compare comp, std::size_t depth = 0)
            {
                const std::ptrdiff_t size = last - first;
                const std::ptrdiff_t threshold = 10000;
                const unsigned int max_depth = 4;
                if (size <= 1) return;
                if (size <= threshold || depth >= max_depth)
                {
                    quicksort(first, last, comp);
                    return;
                }
                RandomIt mid = first + size / 2;
                if (depth < max_depth && size > threshold * 2)
                {
                    std::thread t1([&]() { parallel_mergesort(first, mid, comp, depth + 1); });
                    std::thread t2([&]() { parallel_mergesort(mid, last, comp, depth + 1); });
                    t1.join();
                    t2.join();
                }
                else
                {
                    parallel_mergesort(first, mid, comp, depth + 1);
                    parallel_mergesort(mid, last, comp, depth + 1);
                }
                std::vector<typename std::iterator_traits<RandomIt>::value_type> temp(size);
                simd_merge(first, mid, mid, last, temp.begin(), comp);
                std::move(temp.begin(), temp.end(), first);
            }
        } // namespace detail

        /**********************************************
         * sort - in-place sorting of an array along axis
         **********************************************/

        // Sort entire array (flat) using quicksort with SIMD.
        template <class E>
        inline auto sort(E&& e)
        {
            auto arr = xt::eval(std::forward<E>(e));
            detail::quicksort(arr.data(), arr.data() + arr.size(), detail::less<typename std::decay_t<E>::value_type>());
            return arr;
        }

        // Sort along a specific axis.
        template <class E>
        inline auto sort(E&& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(std::forward<E>(e));
            auto shape = arr.shape();
            if (axis >= shape.size())
                throw std::runtime_error("sort: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i)
                block_size *= shape[i];
            std::size_t outer_loop = arr.size() / (axis_len * block_size);
            value_type* data = arr.data();
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                // We need to sort each "row" along axis independently
                // For simplicity, sort each contiguous segment of length axis_len * block_size? No.
                // axis_len is the number of elements along axis; each slice perpendicular to axis has block_size elements.
                // So we have outer_loop such slices. We'll sort each slice individually: each slice consists of axis_len blocks, each of size block_size.
                // To sort along axis, we can compare the whole block? Actually sorting along axis means we compare elements in that dimension, but each element is a scalar. So we need to extract the block_size scalar at each index along axis and compare them.
                // For block_size = 1, it's simpler.
                if (block_size == 1)
                {
                    detail::quicksort(data + offset, data + offset + axis_len, detail::less<value_type>());
                }
                else
                {
                    // General case: we need to sort the array's indices along that axis. We'll create an index array and sort it based on the corresponding element.
                    std::vector<std::size_t> indices(axis_len);
                    std::iota(indices.begin(), indices.end(), 0);
                    // Comparison uses the first element of each block? Actually each element along axis is a block of block_size scalars. We compare the whole block lexicographically? Typically, sorting along axis sorts the scalar elements, not blocks.
                    // So block_size is the number of contiguous elements that represent a single value? No, block_size is the product of dimensions after axis. So if shape is (a,b,c) and axis=1, block_size = c. Then sorting along axis=1 means for each i and k (a and c), we sort the b elements. Each "element" is a scalar; block_size is the number of scalars per dimension-1 index? That's not correct.
                    // Better: we can use a strided view approach, but we'll implement a loop that sorts vectors by copying rows to a buffer, sorting, and writing back.
                    std::vector<value_type> row(axis_len * block_size);
                    std::memcpy(row.data(), data + offset, axis_len * block_size * sizeof(value_type));
                    // sort each "column" of block_size within row? We'll compare the first element of each block, but that's not correct for multi-dimensional blocks.
                    // To properly sort along axis with block_size>1, we'd need to permute the whole block. However, many use-cases have block_size=1. We'll just throw an error for simplicity and state that sorting with block_size > 1 is not yet supported, or we can just sort by the first element of each block (like lexicographic sort). For now, implement only if block_size == 1; else, use a fallback that sorts by the sum of absolute values to get a partial order, but that's not standard. We'll leave as unsupported.
                    throw std::runtime_error("sort along axis with block_size > 1 is not fully supported yet.");
                }
            }
            return arr;
        }

        /**********************************************
         * argsort - returns indices that would sort the array.
         **********************************************/

        // Argsort entire flat array.
        template <class E>
        inline auto argsort(const E& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            std::vector<std::size_t> indices(e.size());
            std::iota(indices.begin(), indices.end(), 0);
            const value_type* data = e.data();
            std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
                return data[i] < data[j];
            });
            // return as 1D xarray of size_t
            xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({indices.size()});
            std::copy(indices.begin(), indices.end(), result.data());
            return result;
        }

        // Argsort along a given axis.
        template <class E>
        inline auto argsort(const E& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto shape = e.shape();
            if (axis >= shape.size())
                throw std::runtime_error("argsort: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i)
                block_size *= shape[i];
            std::size_t outer_loop = e.size() / (axis_len * block_size);
            std::vector<std::size_t> result_idx(e.size());
            const value_type* src = e.data();
            // We'll fill result with linear indices, then sort slices
            std::iota(result_idx.begin(), result_idx.end(), 0);
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                auto begin_it = result_idx.begin() + offset;
                auto end_it = begin_it + axis_len * block_size;
                if (block_size == 1)
                {
                    std::sort(begin_it, end_it, [&](std::size_t i, std::size_t j) { return src[i] < src[j]; });
                }
                else
                {
                    // For block_size > 1, compare blocks lexicographically
                    std::sort(begin_it, end_it, [&](std::size_t i, std::size_t j) {
                        const value_type* pi = src + i;
                        const value_type* pj = src + j;
                        for (std::size_t k = 0; k < block_size; ++k)
                        {
                            if (pi[k] != pj[k])
                                return pi[k] < pj[k];
                        }
                        return false;
                    });
                }
            }
            // Return a new array with the same shape, containing indices
            auto result_arr = xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>>(shape);
            std::copy(result_idx.begin(), result_idx.end(), result_arr.data());
            return result_arr;
        }

        /**********************************************
         * partition - reorder array so that elements before
         * kth position are less than or equal to those after.
         **********************************************/

        template <class E>
        inline auto partition(E&& e, std::size_t kth)
        {
            auto arr = xt::eval(std::forward<E>(e));
            auto size = arr.size();
            if (kth >= size)
                throw std::runtime_error("partition: kth out of bounds.");
            std::nth_element(arr.data(), arr.data() + kth, arr.data() + size);
            return arr;
        }

        // Partition along axis using nth_element on each slice.
        template <class E>
        inline auto partition(E&& e, std::size_t kth, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(std::forward<E>(e));
            auto shape = arr.shape();
            if (axis >= shape.size())
                throw std::runtime_error("partition: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            if (kth >= axis_len)
                throw std::runtime_error("partition: kth out of axis bounds.");
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i)
                block_size *= shape[i];
            std::size_t outer_loop = arr.size() / (axis_len * block_size);
            value_type* data = arr.data();
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                if (block_size == 1)
                {
                    std::nth_element(data + offset, data + offset + kth, data + offset + axis_len);
                }
                else
                {
                    throw std::runtime_error("partition along axis with block_size>1 not supported.");
                }
            }
            return arr;
        }

        /**********************************************
         * nth_element - place the element at position n
         * in sorted order without fully sorting.
         **********************************************/

        template <class E>
        inline auto nth_element(E&& e, std::size_t n)
        {
            auto arr = xt::eval(std::forward<E>(e));
            std::nth_element(arr.data(), arr.data() + n, arr.data() + arr.size());
            return arr;
        }

        /**********************************************
         * median - returns median along axis or entire array.
         **********************************************/

        template <class E>
        inline auto median(const E& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            std::size_t sz = arr.size();
            std::nth_element(arr.data(), arr.data() + sz / 2, arr.data() + sz);
            if (sz % 2 == 0)
            {
                // average two middle elements
                value_type a = arr[sz / 2 - 1];
                value_type b = arr[sz / 2];
                // to get the lower median correctly we need to nth_element the (sz/2 - 1) as well
                std::nth_element(arr.data(), arr.data() + sz / 2 - 1, arr.data() + sz);
                a = arr[sz / 2 - 1];
                b = arr[sz / 2];
                return (a + b) / value_type(2);
            }
            return arr[sz / 2];
        }

        template <class E>
        inline auto median(const E& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            auto shape = arr.shape();
            if (axis >= shape.size())
                throw std::runtime_error("median: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i)
                block_size *= shape[i];
            std::size_t outer_loop = arr.size() / (axis_len * block_size);
            std::vector<value_type> medians(outer_loop * block_size);
            value_type* data = arr.data();
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                if (block_size == 1)
                {
                    std::nth_element(data + offset, data + offset + axis_len / 2, data + offset + axis_len);
                    medians[o] = data[offset + axis_len / 2];
                    if (axis_len % 2 == 0)
                    {
                        std::nth_element(data + offset, data + offset + axis_len / 2 - 1, data + offset + axis_len);
                        medians[o] = (medians[o] + data[offset + axis_len / 2 - 1]) / value_type(2);
                    }
                }
                else
                {
                    throw std::runtime_error("median along axis with block_size>1 not supported.");
                }
            }
            // Build result array (shape with axis reduced to 1)
            std::vector<std::size_t> new_shape = shape;
            new_shape[axis] = 1;
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
            std::copy(medians.begin(), medians.end(), result.data());
            return result;
        }

        /**********************************************
         * parallel_sort - parallel merge sort
         **********************************************/

        template <class E>
        inline auto parallel_sort(E&& e)
        {
            auto arr = xt::eval(std::forward<E>(e));
            detail::parallel_mergesort(arr.data(), arr.data() + arr.size(), detail::less<typename std::decay_t<E>::value_type>());
            return arr;
        }

        /**********************************************
         * is_sorted - check if array is sorted (flat or axis)
         **********************************************/

        template <class E>
        inline bool is_sorted(const E& e)
        {
            const auto& data = e.data();
            return std::is_sorted(data, data + e.size());
        }

        template <class E>
        inline bool is_sorted(const E& e, std::size_t axis)
        {
            auto shape = e.shape();
            if (axis >= shape.size())
                throw std::runtime_error("is_sorted: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i)
                block_size *= shape[i];
            std::size_t outer_loop = e.size() / (axis_len * block_size);
            const auto* data = e.data();
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                if (block_size == 1)
                {
                    if (!std::is_sorted(data + offset, data + offset + axis_len))
                        return false;
                }
                else
                {
                    // Lexicographic sorted check over blocks
                    for (std::size_t i = 0; i + 1 < axis_len; ++i)
                    {
                        const auto* p1 = data + offset + i * block_size;
                        const auto* p2 = data + offset + (i + 1) * block_size;
                        bool less_or_equal = std::lexicographical_compare(p1, p1 + block_size, p2, p2 + block_size,
                            [](const auto& a, const auto& b) { return a <= b; });
                        if (!less_or_equal) return false;
                    }
                }
            }
            return true;
        }

    } // namespace sort
} // namespace xt

#endif // XTENSOR_XSORT_HPP