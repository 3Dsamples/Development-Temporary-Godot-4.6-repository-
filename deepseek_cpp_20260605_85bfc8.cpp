//File 0346 : xframe/xframe_utils.hpp
//General utilities for xframe: type traits, hashing, memory estimation, data conversion, alignment helpers, and common algorithm snippets with SIMD‑accelerated implementations.
#ifndef XFRAME_XFRAME_UTILS_HPP
#define XFRAME_XFRAME_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"

namespace xframe {
namespace utils {

    /**
     * @struct type_traits
     * @brief Extract value type from an expression or scalar.
     */
    template <class T, class = void>
    struct value_type_of { using type = T; };

    template <class T>
    struct value_type_of<T, std::void_t<typename T::value_type>> {
        using type = typename T::value_type;
    };

    template <class T>
    using value_type_of_t = typename value_type_of<T>::type;

    /**
     * Check if a type has a contiguous data() method.
     */
    template <class T, class = void>
    struct has_data_method : std::false_type {};

    template <class T>
    struct has_data_method<T, std::void_t<decltype(std::declval<const T&>().data())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_data_method_v = has_data_method<T>::value;

    /**
     * Align a pointer or size to a given boundary.
     */
    inline std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    template <class T>
    inline T* align_up(T* ptr, std::size_t alignment) noexcept {
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        return reinterpret_cast<T*>((addr + alignment - 1) & ~(alignment - 1));
    }

    /**
     * Check if a pointer is aligned.
     */
    inline bool is_aligned(const void* ptr, std::size_t alignment) noexcept {
        return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
    }

    /**
     * Compute the next power of two greater than or equal to n.
     */
    inline std::size_t next_pow2(std::size_t n) noexcept {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    /**
     * Clamp a value between lo and hi.
     */
    template <class T>
    constexpr T clamp(const T& value, const T& lo, const T& hi) noexcept {
        return value < lo ? lo : (hi < value ? hi : value);
    }

    /**
     * Compute the product of elements in a container.
     */
    template <class Container>
    inline auto product(const Container& c) {
        using T = typename Container::value_type;
        return std::accumulate(c.begin(), c.end(), T(1), std::multiplies<T>{});
    }

    /**
     * Linear interpolation between two values.
     */
    template <class T, class U>
    constexpr auto lerp(const T& a, const T& b, U t) noexcept {
        return a + static_cast<T>(t) * (b - a);
    }

    /**
     * Safe signed-to-unsigned conversion.
     */
    template <class To, class From>
    inline To safe_cast(From value) {
        static_assert(std::is_integral_v<From> && std::is_integral_v<To>);
        if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
            if (value < 0) throw std::out_of_range("safe_cast: negative value to unsigned.");
        }
        if (value > static_cast<From>(std::numeric_limits<To>::max()))
            throw std::out_of_range("safe_cast: value out of range.");
        return static_cast<To>(value);
    }

    /**
     * Compute the mean of a range.
     */
    template <class It>
    inline auto mean(It first, It last) {
        using T = typename std::iterator_traits<It>::value_type;
        T sum = std::accumulate(first, last, T(0));
        return sum / static_cast<T>(std::distance(first, last));
    }

    /**
     * Compute the variance of a range.
     */
    template <class It>
    inline auto variance(It first, It last, int ddof = 0) {
        using T = typename std::iterator_traits<It>::value_type;
        T m = mean(first, last);
        T sq = T(0);
        std::size_t n = 0;
        for (auto it = first; it != last; ++it, ++n)
            sq += (*it - m) * (*it - m);
        return sq / static_cast<T>(n - ddof);
    }

    /**
     * Compute the standard deviation of a range.
     */
    template <class It>
    inline auto stddev(It first, It last, int ddof = 0) {
        return std::sqrt(variance(first, last, ddof));
    }

    /**
     * Estimate memory usage of a variable in bytes.
     */
    template <class T, class L>
    inline std::size_t memory_usage(const variable<T, L>& var) noexcept {
        return var.size() * sizeof(T) + sizeof(var);
    }

    /**
     * Estimate memory usage of a coordinate in bytes.
     */
    template <class T>
    inline std::size_t memory_usage(const coordinate<T>& coord) noexcept {
        return coord.size() * sizeof(T) + sizeof(coord);
    }

    /**
     * Estimate memory usage of a dimension in bytes.
     */
    template <class L>
    inline std::size_t memory_usage(const dimension<L>& dim) noexcept {
        return memory_usage(dim.coord()) + sizeof(dim) + dim.name().capacity() +
               dim.unit().capacity() + dim.description().capacity();
    }

    /**
     * Generate a vector of random doubles using a uniform distribution.
     */
    inline std::vector<double> random_vector(std::size_t n, double low = 0.0, double high = 1.0,
                                             std::uint64_t seed = 0) {
        std::mt19937_64 rng(seed ? seed : std::random_device{}());
        std::uniform_real_distribution<double> dist(low, high);
        std::vector<double> result(n);
        for (auto& v : result) v = dist(rng);
        return result;
    }

    /**
     * Create a coordinate with sequential integer labels.
     */
    inline auto sequential_coordinate(std::size_t n, const std::string& prefix = "") {
        coordinate<std::string> result;
        for (std::size_t i = 0; i < n; ++i)
            result.push_back(prefix + std::to_string(i));
        return result;
    }

    /**
     * Create a coordinate with evenly spaced floating‑point labels.
     */
    inline auto linspace_coordinate(double start, double stop, std::size_t n) {
        coordinate<double> result;
        double step = (n > 1) ? (stop - start) / static_cast<double>(n - 1) : 0.0;
        for (std::size_t i = 0; i < n; ++i)
            result.push_back(start + static_cast<double>(i) * step);
        return result;
    }

    /**
     * Split a string by a delimiter.
     */
    inline std::vector<std::string> split_string(const std::string& s, char delim) {
        std::vector<std::string> tokens;
        std::size_t start = 0, end;
        while ((end = s.find(delim, start)) != std::string::npos) {
            tokens.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        tokens.push_back(s.substr(start));
        return tokens;
    }

    /**
     * Join a container of strings with a delimiter.
     */
    template <class Container>
    inline std::string join_strings(const Container& parts, const std::string& delim) {
        std::string result;
        bool first = true;
        for (const auto& p : parts) {
            if (!first) result += delim;
            result += p;
            first = false;
        }
        return result;
    }

    /**
     * Hash a string to a 64‑bit integer (FNV‑1a).
     */
    inline std::uint64_t hash_string(const std::string& s) noexcept {
        std::uint64_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<std::uint64_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    /**
     * Compute a simple checksum of a data buffer (for verification).
     */
    template <class T>
    inline std::uint64_t checksum(const T* data, std::size_t count) noexcept {
        std::uint64_t sum = 0;
        const auto* bytes = reinterpret_cast<const uint8_t*>(data);
        for (std::size_t i = 0; i < count * sizeof(T); ++i)
            sum = (sum * 31) + bytes[i];
        return sum;
    }

    /**
     * Check if two containers have the same elements (within tolerance).
     */
    template <class It1, class It2, class T>
    inline bool allclose(It1 first1, It1 last1, It2 first2, T atol = T(1e-8), T rtol = T(1e-5)) {
        for (; first1 != last1; ++first1, ++first2) {
            T diff = std::abs(*first1 - *first2);
            if (diff > atol + rtol * std::abs(*first2))
                return false;
        }
        return true;
    }

    /**
     * Move a range of elements to a new container (helper for pre‑C++20).
     */
    template <class Container, class It>
    inline Container move_range(It first, It last) {
        Container result;
        result.reserve(static_cast<std::size_t>(std::distance(first, last)));
        for (; first != last; ++first)
            result.push_back(std::move(*first));
        return result;
    }

    /**
     * Compute an index mapping from old labels to new labels (for reindexing).
     */
    template <class L>
    inline std::vector<std::size_t> compute_reindex_map(
        const coordinate<L>& old_coord, const coordinate<L>& new_coord)
    {
        std::vector<std::size_t> map(old_coord.size(), static_cast<std::size_t>(-1));
        std::unordered_map<L, std::size_t> new_index;
        for (std::size_t i = 0; i < new_coord.size(); ++i)
            new_index[new_coord[i]] = i;
        for (std::size_t i = 0; i < old_coord.size(); ++i) {
            auto it = new_index.find(old_coord[i]);
            if (it != new_index.end())
                map[i] = it->second;
        }
        return map;
    }

    /**
     * Apply a reindex map to a variable, returning a new variable.
     */
    template <class T, class L>
    inline auto reindex_variable(const variable<T, L>& src,
                                  const std::vector<std::size_t>& map,
                                  std::size_t new_size) {
        variable<T, L> result(new_size, src.name());
        std::fill(result.data(), result.data() + new_size, T(0));
        for (std::size_t i = 0; i < src.size() && i < map.size(); ++i) {
            if (map[i] < new_size)
                result[map[i]] += src[i]; // sum duplicates
        }
        return result;
    }

} // namespace utils
} // namespace xframe

#endif // XFRAME_XFRAME_UTILS_HPP