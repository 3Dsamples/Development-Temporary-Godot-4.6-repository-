//File 0013 (UPDATED) : core/xsort.hpp
//Sorting, argsort, partition, quickselect, median, stable sort, and parallel merge sort with SIMD-accelerated comparisons and 64-bit alignment.
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
            inline auto simd_less_than(xsimd::batch<T, default_simd_arch> a,
                                       xsimd::batch<T, default_simd_arch> b) noexcept
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
                    using simd_type = xsimd::batch<value_type, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    RandomIt left = first;
                    RandomIt right = last - 1;
                    while (left <= right)
                    {
                        while (left <= right)
                        {
                            if (left + simd_size <= right + 1)
                            {
                                simd_type v = simd_type::load_unaligned(&*left);
                                auto mask = comp.simd_apply ? comp.simd_apply(v, simd_type(pivot)) : simd_less_than(v, simd_type(pivot));
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
                    auto mid = first + (last - first) / 2;
                    auto last_el = last - 1;
                    if (comp(*mid, *first)) std::iter_swap(mid, first);
                    if (comp(*last_el, *first)) std::iter_swap(last_el, first);
                    if (comp(*last_el, *mid)) std::iter_swap(last_el, mid);
                    auto pivot = *mid;
                    auto part = simd_partition(first, last, pivot, comp);
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
                    using simd_type = xsimd::batch<value_type, default_simd_arch>;
                    constexpr std::size_t simd_size = simd_type::size;
                    while (first1 < last1 && first2 < last2)
                    {
                        if (first1 + simd_size <= last1 && first2 + simd_size <= last2)
                        {
                            simd_type v1 = simd_type::load_unaligned(&*first1);
                            simd_type v2 = simd_type::load_unaligned(&*first2);
                            auto mask = comp.simd_apply ? comp.simd_apply(v1, v2) : v1 < v2;
                            for (std::size_t i = 0; i < simd_size; ++i)
                            {
                                if (mask[i])
                                    *d_first++ = std::move(*first1++);
                                else
                                    *d_first++ = std::move(*first2++);
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
        /**
         * Sort entire array (flat) using quicksort with SIMD.
         */
        template <class E>
        inline auto sort(E&& e)
        {
            auto arr = xt::eval(std::forward<E>(e));
            detail::quicksort(arr.data(), arr.data() + arr.size(), detail::less<typename std::decay_t<E>::value_type>());
            return arr;
        }

        /**
         * Sort along a specific axis.
         */
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
                if (block_size == 1)
                {
                    detail::quicksort(data + offset, data + offset + axis_len, detail::less<value_type>());
                }
                else
                {
                    // Sort blocks by comparing first element (lexicographic sort)
                    std::vector<std::size_t> indices(axis_len);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
                        const value_type* pi = data + offset + i * block_size;
                        const value_type* pj = data + offset + j * block_size;
                        for (std::size_t k = 0; k < block_size; ++k)
                        {
                            if (pi[k] != pj[k]) return pi[k] < pj[k];
                        }
                        return false;
                    });
                    std::vector<value_type> temp(axis_len * block_size);
                    for (std::size_t i = 0; i < axis_len; ++i)
                        std::copy(data + offset + i * block_size,
                                  data + offset + i * block_size + block_size,
                                  temp.begin() + i * block_size);
                    for (std::size_t i = 0; i < axis_len; ++i)
                        std::copy(temp.begin() + indices[i] * block_size,
                                  temp.begin() + indices[i] * block_size + block_size,
                                  data + offset + i * block_size);
                }
            }
            return arr;
        }

        /**********************************************
         * stable_sort
         **********************************************/
        template <class E>
        inline auto stable_sort(E&& e)
        {
            auto arr = xt::eval(std::forward<E>(e));
            std::stable_sort(arr.data(), arr.data() + arr.size());
            return arr;
        }

        template <class E>
        inline auto stable_sort(E&& e, std::size_t axis)
        {
            // same as sort but with stable sort along axis
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(std::forward<E>(e));
            auto shape = arr.shape();
            if (axis >= shape.size())
                throw std::runtime_error("stable_sort: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i) block_size *= shape[i];
            std::size_t outer_loop = arr.size() / (axis_len * block_size);
            value_type* data = arr.data();
            for (std::size_t o = 0; o < outer_loop; ++o)
            {
                std::size_t offset = o * axis_len * block_size;
                if (block_size == 1)
                {
                    std::stable_sort(data + offset, data + offset + axis_len);
                }
                else
                {
                    std::vector<std::size_t> indices(axis_len);
                    std::iota(indices.begin(), indices.end(), 0);
                    std::stable_sort(indices.begin(), indices.end(), [&](std::size_t i, std::size_t j) {
                        const value_type* pi = data + offset + i * block_size;
                        const value_type* pj = data + offset + j * block_size;
                        for (std::size_t k = 0; k < block_size; ++k)
                        {
                            if (pi[k] != pj[k]) return pi[k] < pj[k];
                        }
                        return false;
                    });
                    std::vector<value_type> temp(axis_len * block_size);
                    for (std::size_t i = 0; i < axis_len; ++i)
                        std::copy(data + offset + i * block_size,
                                  data + offset + i * block_size + block_size,
                                  temp.begin() + i * block_size);
                    for (std::size_t i = 0; i < axis_len; ++i)
                        std::copy(temp.begin() + indices[i] * block_size,
                                  temp.begin() + indices[i] * block_size + block_size,
                                  data + offset + i * block_size);
                }
            }
            return arr;
        }

        /**********************************************
         * argsort
         **********************************************/
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
            xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>> result({indices.size()});
            std::copy(indices.begin(), indices.end(), result.data());
            return result;
        }

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
                    std::sort(begin_it, end_it, [&](std::size_t i, std::size_t j) {
                        const value_type* pi = src + i;
                        const value_type* pj = src + j;
                        for (std::size_t k = 0; k < block_size; ++k)
                        {
                            if (pi[k] != pj[k]) return pi[k] < pj[k];
                        }
                        return false;
                    });
                }
            }
            auto result_arr = xarray_container<xt::uvector<std::size_t>, DEFAULT_LAYOUT, std::vector<std::size_t>>(shape);
            std::copy(result_idx.begin(), result_idx.end(), result_arr.data());
            return result_arr;
        }

        /**********************************************
         * partition
         **********************************************/
        template <class E>
        inline auto partition(E&& e, std::size_t kth)
        {
            auto arr = xt::eval(std::forward<E>(e));
            if (kth >= arr.size()) throw std::runtime_error("partition: kth out of bounds.");
            std::nth_element(arr.data(), arr.data() + kth, arr.data() + arr.size());
            return arr;
        }

        template <class E>
        inline auto partition(E&& e, std::size_t kth, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(std::forward<E>(e));
            auto shape = arr.shape();
            if (axis >= shape.size()) throw std::runtime_error("partition: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            if (kth >= axis_len) throw std::runtime_error("partition: kth out of axis bounds.");
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i) block_size *= shape[i];
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
                    // Cannot partition blocks lexicographically simply; fallback to simple nth_element on first element
                    std::nth_element(data + offset, data + offset + kth * block_size,
                                     data + offset + axis_len * block_size,
                                     [&](const value_type& a, const value_type& b) { return a < b; }); // compare only first element
                }
            }
            return arr;
        }

        /**********************************************
         * nth_element
         **********************************************/
        template <class E>
        inline auto nth_element(E&& e, std::size_t n)
        {
            auto arr = xt::eval(std::forward<E>(e));
            std::nth_element(arr.data(), arr.data() + n, arr.data() + arr.size());
            return arr;
        }

        /**********************************************
         * median
         **********************************************/
        template <class E>
        inline auto median(const E& e)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            std::size_t sz = arr.size();
            if (sz == 0) return value_type(0);
            std::nth_element(arr.data(), arr.data() + sz / 2, arr.data() + sz);
            if (sz % 2 == 0)
            {
                std::nth_element(arr.data(), arr.data() + sz / 2 - 1, arr.data() + sz);
                return (arr[sz / 2 - 1] + arr[sz / 2]) / value_type(2);
            }
            return arr[sz / 2];
        }

        template <class E>
        inline auto median(const E& e, std::size_t axis)
        {
            using value_type = typename std::decay_t<E>::value_type;
            auto arr = xt::eval(e);
            auto shape = arr.shape();
            if (axis >= shape.size()) throw std::runtime_error("median: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i) block_size *= shape[i];
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
                    // Median by first element of block only
                    std::vector<std::size_t> indices(axis_len);
                    std::iota(indices.begin(), indices.end(), 0);
                    auto comp = [&](std::size_t i, std::size_t j) {
                        return data[offset + i * block_size] < data[offset + j * block_size];
                    };
                    std::nth_element(indices.begin(), indices.begin() + axis_len / 2, indices.end(), comp);
                    medians[o] = data[offset + indices[axis_len / 2] * block_size];
                    if (axis_len % 2 == 0)
                    {
                        std::nth_element(indices.begin(), indices.begin() + axis_len / 2 - 1, indices.end(), comp);
                        medians[o] = (medians[o] + data[offset + indices[axis_len / 2 - 1] * block_size]) / value_type(2);
                    }
                }
            }
            std::vector<std::size_t> new_shape = shape;
            new_shape[axis] = 1;
            xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<std::size_t>> result(new_shape);
            std::copy(medians.begin(), medians.end(), result.data());
            return result;
        }

        /**********************************************
         * parallel_sort
         **********************************************/
        template <class E>
        inline auto parallel_sort(E&& e)
        {
            auto arr = xt::eval(std::forward<E>(e));
            detail::parallel_mergesort(arr.data(), arr.data() + arr.size(), detail::less<typename std::decay_t<E>::value_type>());
            return arr;
        }

        /**********************************************
         * is_sorted
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
            if (axis >= shape.size()) throw std::runtime_error("is_sorted: axis out of bounds.");
            std::size_t axis_len = shape[axis];
            std::size_t block_size = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i) block_size *= shape[i];
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
                    for (std::size_t i = 0; i + 1 < axis_len; ++i)
                    {
                        const auto* p1 = data + offset + i * block_size;
                        const auto* p2 = data + offset + (i + 1) * block_size;
                        if (std::lexicographical_compare(p2, p2 + block_size, p1, p1 + block_size))
                            return false;
                    }
                }
            }
            return true;
        }

    } // namespace sort
} // namespace xt

#endif // XTENSOR_XSORT_HPP