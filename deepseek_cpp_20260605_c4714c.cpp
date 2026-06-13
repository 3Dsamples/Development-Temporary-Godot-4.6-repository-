//File 0326 : xframe/xaxis_meta.hpp
//Axis metadata utilities: extract dimension names, units, descriptions, validate dimension compatibility, and build axis descriptors from tuples.
#ifndef XFRAME_XAXIS_META_HPP
#define XFRAME_XAXIS_META_HPP

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_dimension.hpp"

namespace xframe {
namespace axis {
namespace meta {

    /**
     * Get the name of the i‑th dimension from a tuple of dimensions.
     */
    template <class Tuple, std::size_t I = 0>
    inline const label_type& dim_name(const Tuple& dims, std::size_t i)
    {
        if constexpr (I >= std::tuple_size_v<Tuple>) throw std::out_of_range("dim_name: index out of bounds.");
        else return (i == 0) ? std::get<I>(dims).name() : dim_name<Tuple, I+1>(dims, i-1);
    }

    /**
     * Get the unit of the i‑th dimension.
     */
    template <class Tuple, std::size_t I = 0>
    inline const label_type& dim_unit(const Tuple& dims, std::size_t i)
    {
        if constexpr (I >= std::tuple_size_v<Tuple>) throw std::out_of_range("dim_unit: index out of bounds.");
        else return (i == 0) ? std::get<I>(dims).unit() : dim_unit<Tuple, I+1>(dims, i-1);
    }

    /**
     * Get the description of the i‑th dimension.
     */
    template <class Tuple, std::size_t I = 0>
    inline const label_type& dim_description(const Tuple& dims, std::size_t i)
    {
        if constexpr (I >= std::tuple_size_v<Tuple>) throw std::out_of_range("dim_description: index out of bounds.");
        else return (i == 0) ? std::get<I>(dims).description() : dim_description<Tuple, I+1>(dims, i-1);
    }

    /**
     * Get the size of the i‑th dimension.
     */
    template <class Tuple, std::size_t I = 0>
    inline std::size_t dim_size(const Tuple& dims, std::size_t i)
    {
        if constexpr (I >= std::tuple_size_v<Tuple>) throw std::out_of_range("dim_size: index out of bounds.");
        else return (i == 0) ? std::get<I>(dims).size() : dim_size<Tuple, I+1>(dims, i-1);
    }

    /**
     * Collect the names of all dimensions into a vector.
     */
    template <class Tuple, std::size_t... I>
    inline std::vector<label_type> dim_names_impl(const Tuple& dims, std::index_sequence<I...>)
    {
        return { std::get<I>(dims).name()... };
    }

    template <class Tuple>
    inline std::vector<label_type> dim_names(const Tuple& dims)
    {
        return dim_names_impl(dims, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }

    /**
     * Find the index of a dimension by name. Returns size() if not found.
     */
    template <class Tuple, std::size_t I = 0>
    inline std::size_t find_dim_index(const Tuple& dims, const label_type& name)
    {
        if constexpr (I >= std::tuple_size_v<Tuple>) return std::tuple_size_v<Tuple>;
        else return (std::get<I>(dims).name() == name) ? I : find_dim_index<Tuple, I+1>(dims, name);
    }

    /**
     * Check that two dimension tuples are compatible (same names, sizes, units).
     */
    template <class Tuple1, class Tuple2, std::size_t... I>
    inline bool are_compatible_impl(const Tuple1& a, const Tuple2& b, std::index_sequence<I...>)
    {
        return ((std::get<I>(a).name() == std::get<I>(b).name() &&
                 std::get<I>(a).size() == std::get<I>(b).size() &&
                 std::get<I>(a).unit() == std::get<I>(b).unit()) && ...);
    }

    template <class Tuple1, class Tuple2>
    inline bool are_compatible(const Tuple1& a, const Tuple2& b)
    {
        if constexpr (std::tuple_size_v<Tuple1> != std::tuple_size_v<Tuple2>) return false;
        return are_compatible_impl(a, b, std::make_index_sequence<std::tuple_size_v<Tuple1>>{});
    }

} // namespace meta
} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_META_HPP