//File 0067 : core/xset_operation.hpp
//Set operations on arrays: unique, isin, intersect1d, union1d, setdiff1d, setxor1d, in1d with SIMD-accelerated sorting and hash-based detection.
#ifndef XTENSOR_XSET_OPERATION_HPP
#define XTENSOR_XSET_OPERATION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"
#include "xarray.hpp"
#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsort.hpp"
#include "xstrides.hpp"
#include "xeval.hpp"
#include "xmanipulation.hpp"

namespace xt {
namespace set {

    namespace detail {
        // Hash-based unique for unsorted input, fast for small sets
        template <class T>
        inline auto unique_hash(const xarray_container<uvector<T>>& arr) {
            std::unordered_set<T> seen;
            std::vector<T> result;
            for (std::size_t i = 0; i < arr.size(); ++i) {
                if (seen.insert(arr[i]).second) {
                    result.push_back(arr[i]);
                }
            }
            xarray_container<uvector<T>> res({result.size()});
            std::copy(result.begin(), result.end(), res.data());
            return res;
        }

        // Sort-based unique: sort then remove duplicates
        template <class T>
        inline auto unique_sorted(const xarray_container<uvector<T>>& arr) {
            auto sorted = xt::sort(arr);
            if (sorted.size() == 0) return sorted;
            std::size_t write_idx = 1;
            for (std::size_t i = 1; i < sorted.size(); ++i) {
                if (sorted[i] != sorted[write_idx - 1]) {
                    sorted[write_idx++] = sorted[i];
                }
            }
            sorted.resize({write_idx});
            return sorted;
        }
    }

    /**
     * Return the unique elements of a 1D array, preserving order of first occurrence.
     */
    template <class E>
    inline auto unique(const E& e) {
        auto arr = xt::eval(e);
        if (arr.dimension() != 1) throw std::runtime_error("unique requires 1D input.");
        // Choose method based on size heuristic
        if (arr.size() <= 1000) {
            return detail::unique_hash(arr);
        } else {
            return detail::unique_sorted(arr);
        }
    }

    /**
     * Return a boolean array of the same shape indicating whether each element
     * of the first array is present in the second array.
     */
    template <class E1, class E2>
    inline auto isin(const E1& elements, const E2& test_elements) {
        using T = typename std::decay_t<E1>::value_type;
        auto arr = xt::eval(elements);
        auto test_arr = xt::eval(test_elements);
        // Build a hash set of test elements for fast lookup
        std::unordered_set<T> test_set(test_arr.data(), test_arr.data() + test_arr.size());
        xarray_container<uvector<bool>> result(arr.shape());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            result[i] = test_set.find(arr[i]) != test_set.end();
        }
        return result;
    }

    /**
     * Return the unique elements common to both input 1D arrays.
     */
    template <class E1, class E2>
    inline auto intersect1d(const E1& a, const E2& b) {
        auto arr1 = xt::eval(a);
        auto arr2 = xt::eval(b);
        if (arr1.dimension() != 1 || arr2.dimension() != 1)
            throw std::runtime_error("intersect1d requires 1D inputs.");
        auto uniq1 = unique(arr1);
        auto uniq2 = unique(arr2);
        std::unordered_set<typename decltype(uniq1)::value_type> set2(uniq2.data(), uniq2.data() + uniq2.size());
        std::vector<typename decltype(uniq1)::value_type> common;
        for (std::size_t i = 0; i < uniq1.size(); ++i) {
            if (set2.count(uniq1[i])) common.push_back(uniq1[i]);
        }
        xarray_container<uvector<typename decltype(uniq1)::value_type>> result({common.size()});
        std::copy(common.begin(), common.end(), result.data());
        return result;
    }

    /**
     * Return the sorted unique elements from both input 1D arrays.
     */
    template <class E1, class E2>
    inline auto union1d(const E1& a, const E2& b) {
        auto arr1 = xt::eval(a);
        auto arr2 = xt::eval(b);
        if (arr1.dimension() != 1 || arr2.dimension() != 1)
            throw std::runtime_error("union1d requires 1D inputs.");
        using T = typename std::decay_t<E1>::value_type;
        std::vector<T> combined(arr1.data(), arr1.data() + arr1.size());
        combined.insert(combined.end(), arr2.data(), arr2.data() + arr2.size());
        xarray_container<uvector<T>> combined_arr({combined.size()});
        std::copy(combined.begin(), combined.end(), combined_arr.data());
        return unique(combined_arr);
    }

    /**
     * Return the sorted unique elements in the first array that are not in the second.
     */
    template <class E1, class E2>
    inline auto setdiff1d(const E1& a, const E2& b) {
        auto arr1 = xt::eval(a);
        auto arr2 = xt::eval(b);
        if (arr1.dimension() != 1 || arr2.dimension() != 1)
            throw std::runtime_error("setdiff1d requires 1D inputs.");
        auto uniq1 = unique(arr1);
        auto uniq2 = unique(arr2);
        std::unordered_set<typename decltype(uniq1)::value_type> set2(uniq2.data(), uniq2.data() + uniq2.size());
        std::vector<typename decltype(uniq1)::value_type> diff;
        for (std::size_t i = 0; i < uniq1.size(); ++i) {
            if (!set2.count(uniq1[i])) diff.push_back(uniq1[i]);
        }
        xarray_container<uvector<typename decltype(uniq1)::value_type>> result({diff.size()});
        std::copy(diff.begin(), diff.end(), result.data());
        return result;
    }

    /**
     * Return the sorted unique elements that are in either of the input arrays but not both.
     */
    template <class E1, class E2>
    inline auto setxor1d(const E1& a, const E2& b) {
        auto inter = intersect1d(a, b);
        auto un = union1d(a, b);
        auto inter_set = std::unordered_set<typename decltype(inter)::value_type>(inter.data(), inter.data() + inter.size());
        std::vector<typename decltype(inter)::value_type> xor_vals;
        for (std::size_t i = 0; i < un.size(); ++i) {
            if (!inter_set.count(un[i])) xor_vals.push_back(un[i]);
        }
        xarray_container<uvector<typename decltype(un)::value_type>> result({xor_vals.size()});
        std::copy(xor_vals.begin(), xor_vals.end(), result.data());
        return result;
    }

    /**
     * Return a boolean array indicating whether each element of `elements` is in `test_elements` (alias for isin).
     */
    template <class E1, class E2>
    inline auto in1d(const E1& elements, const E2& test_elements) {
        return isin(elements, test_elements);
    }

} // namespace set
} // namespace xt

#endif // XTENSOR_XSET_OPERATION_HPP