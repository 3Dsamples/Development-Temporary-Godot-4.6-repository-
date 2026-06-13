//File 0052 : core/xutils.hpp
//General utilities: make_sequence, has_data_interface, forwarding, C++17 polyfills, aligned allocation helpers, and type traits for xtensor.
#ifndef XTENSOR_XUTILS_HPP
#define XTENSOR_XUTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
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
         * Generate a sequence from 0 to N-1 at compile-time.
         */
        template <std::size_t... I>
        constexpr auto make_index_sequence_impl(std::index_sequence<I...>) noexcept
        {
            return std::index_sequence<I...>{};
        }

        template <std::size_t N>
        using make_index_sequence = std::make_index_sequence<N>;

        template <std::size_t... I>
        using index_sequence = std::index_sequence<I...>;

        /**
         * Make a vector of size_t from 0 to N-1.
         */
        template <std::size_t N>
        inline auto make_sequence() noexcept
        {
            std::array<std::size_t, N> result;
            std::iota(result.begin(), result.end(), std::size_t(0));
            return result;
        }

        inline auto make_sequence(std::size_t n)
        {
            std::vector<std::size_t> result(n);
            std::iota(result.begin(), result.end(), std::size_t(0));
            return result;
        }
    }

    /**
     * Trait to detect if a type has a contiguous data() member.
     */
    template <class T, class = void>
    struct has_data_interface : std::false_type {};

    template <class T>
    struct has_data_interface<T, std::void_t<decltype(std::declval<const T&>().data())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_data_interface_v = has_data_interface<T>::value;

    /**
     * Trait to detect if a type has a shape() member.
     */
    template <class T, class = void>
    struct has_shape_interface : std::false_type {};

    template <class T>
    struct has_shape_interface<T, std::void_t<decltype(std::declval<const T&>().shape())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_shape_interface_v = has_shape_interface<T>::value;

    /**
     * Trait to detect if a type has strides().
     */
    template <class T, class = void>
    struct has_strides_interface : std::false_type {};

    template <class T>
    struct has_strides_interface<T, std::void_t<decltype(std::declval<const T&>().strides())>>
        : std::true_type {};

    template <class T>
    inline constexpr bool has_strides_interface_v = has_strides_interface<T>::value;

    /**
     * Forward a sequence from one container type to another.
     */
    template <class To, class From>
    inline To forward_sequence(const From& from)
    {
        return To(from.begin(), from.end());
    }

    /**
     * Compute the product of a container's elements.
     */
    template <class Container>
    inline auto product(const Container& c)
    {
        using value_type = typename Container::value_type;
        return std::accumulate(c.begin(), c.end(), value_type(1), std::multiplies<value_type>());
    }

    /**
     * Safe signed/unsigned conversion.
     */
    template <class To, class From>
    inline To safe_cast(From value)
    {
        static_assert(std::is_integral_v<From> && std::is_integral_v<To>,
                      "safe_cast only for integral types.");
        if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>)
        {
            if (value < 0) throw std::out_of_range("safe_cast: negative value to unsigned.");
        }
        if (value > static_cast<From>(std::numeric_limits<To>::max()))
            throw std::out_of_range("safe_cast: value out of range.");
        return static_cast<To>(value);
    }

    /**
     * Apply a function to each element of a tuple.
     */
    template <class Tuple, class Func, std::size_t... I>
    inline void for_each_impl(Tuple&& t, Func&& f, std::index_sequence<I...>)
    {
        (f(std::get<I>(std::forward<Tuple>(t))), ...);
    }

    template <class Tuple, class Func>
    inline void for_each(Tuple&& t, Func&& f)
    {
        for_each_impl(std::forward<Tuple>(t), std::forward<Func>(f),
                      std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
    }

    /**
     * Apply a function to corresponding elements of two tuples.
     */
    template <class Tuple1, class Tuple2, class Func, std::size_t... I>
    inline void for_each_pair_impl(Tuple1&& t1, Tuple2&& t2, Func&& f, std::index_sequence<I...>)
    {
        (f(std::get<I>(std::forward<Tuple1>(t1)), std::get<I>(std::forward<Tuple2>(t2))), ...);
    }

    template <class Tuple1, class Tuple2, class Func>
    inline void for_each_pair(Tuple1&& t1, Tuple2&& t2, Func&& f)
    {
        static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>,
                      "Tuples must have same size.");
        for_each_pair_impl(std::forward<Tuple1>(t1), std::forward<Tuple2>(t2), std::forward<Func>(f),
                           std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple1>>>{});
    }

    /**
     * Check if all elements of a container satisfy a predicate.
     */
    template <class Container, class Predicate>
    inline bool all_of(const Container& c, Predicate pred)
    {
        return std::all_of(c.begin(), c.end(), pred);
    }

    /**
     * Check if any element of a container satisfies a predicate.
     */
    template <class Container, class Predicate>
    inline bool any_of(const Container& c, Predicate pred)
    {
        return std::any_of(c.begin(), c.end(), pred);
    }

    /**
     * Check if none of the elements satisfy a predicate.
     */
    template <class Container, class Predicate>
    inline bool none_of(const Container& c, Predicate pred)
    {
        return std::none_of(c.begin(), c.end(), pred);
    }

    /**
     * Compute the offset of a contiguous multi-dimensional index.
     */
    template <class Index, class Strides>
    inline auto compute_offset(const Index& idx, const Strides& strides)
    {
        using size_type = typename Index::value_type;
        size_type offset = 0;
        for (std::size_t i = 0; i < idx.size(); ++i)
            offset += idx[i] * strides[i];
        return offset;
    }

    /**
     * Align a pointer or size up to the given alignment.
     */
    template <class T>
    inline T* align_up(T* ptr, std::size_t alignment) noexcept
    {
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        auto aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<T*>(aligned);
    }

    inline std::size_t align_up(std::size_t size, std::size_t alignment) noexcept
    {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /**
     * Return the next power of two greater than or equal to n.
     */
    inline std::size_t next_pow2(std::size_t n) noexcept
    {
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
    constexpr T clamp(const T& value, const T& lo, const T& hi) noexcept
    {
        return value < lo ? lo : (hi < value ? hi : value);
    }

} // namespace xt

#endif // XTENSOR_XUTILS_HPP