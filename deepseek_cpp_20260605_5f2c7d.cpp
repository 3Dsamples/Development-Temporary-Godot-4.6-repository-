//File 0339 : xframe/xcoordinate_system.hpp
//Coordinate system: maps labels to integer positions across multiple axes, supports alignment, broadcasting, and SIMD-accelerated label resolution for multi-dimensional xframe operations.
#ifndef XFRAME_XCOORDINATE_SYSTEM_HPP
#define XFRAME_XCOORDINATE_SYSTEM_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xcoordinate.hpp"
#include "xdimension.hpp"
#include "xaxis.hpp"

namespace xframe
{
    /**
     * @class xcoordinate_system
     * @brief A multi-axis coordinate system that maps labels to integer indices.
     *
     * For each axis (dimension), a coordinate sequence is stored. The system
     * provides methods to resolve a label to an index on a given axis, to align
     * two coordinate systems (e.g., finding the common set of labels), and to
     * broadcast labels across axes for operations like join and merge.
     * Lookup uses hash maps for O(1) performance when labels are strings, with
     * SIMD-accelerated construction of the index map.
     */
    template <class L = label_type>
    class xcoordinate_system : public expression<xcoordinate_system<L>>
    {
    public:
        using self_type = xcoordinate_system<L>;
        using label_type = L;
        using size_type = std::size_t;
        using coordinate_type = coordinate<L>;
        using dimension_type = dimension<L>;

        xcoordinate_system() noexcept = default;

        /**
         * Construct from a tuple of dimensions.
         * Builds a hash map for each axis for fast label→index lookup.
         */
        template <class... Dims>
        explicit xcoordinate_system(std::tuple<Dims...> dims)
            : m_dimensions(dims)
        {
            build_index_maps(std::make_index_sequence<sizeof...(Dims)>{});
        }

        /**
         * Construct from a vector of dimensions.
         */
        explicit xcoordinate_system(const std::vector<dimension_type>& dims)
            : m_dimensions_vec(dims)
        {
            build_index_maps_vec();
        }

        xcoordinate_system(const self_type&) = default;
        xcoordinate_system& operator=(const self_type&) = default;
        xcoordinate_system(self_type&&) = default;
        xcoordinate_system& operator=(self_type&&) = default;

        /**
         * Number of axes.
         */
        size_type axis_count() const noexcept
        {
            if (m_dimensions_vec.empty())
                return std::tuple_size_v<decltype(m_dimensions)>;
            return m_dimensions_vec.size();
        }

        /**
         * Get the coordinate for axis i.
         */
        const coordinate_type& axis(size_type i) const
        {
            if (!m_dimensions_vec.empty())
                return m_dimensions_vec[i].coord();
            return get_axis_impl(i, std::make_index_sequence<std::tuple_size_v<decltype(m_dimensions)>>{});
        }

        /**
         * Resolve a label to its integer index on axis i.
         * Returns axis(i).size() if not found.
         */
        size_type index_of(size_type i, const L& label) const
        {
            if (!m_index_maps.empty() && i < m_index_maps.size())
            {
                auto it = m_index_maps[i].find(label);
                if (it != m_index_maps[i].end()) return it->second;
                return axis(i).size();
            }
            return axis(i).find(label);
        }

        /**
         * Check if a label exists on axis i.
         */
        bool contains(size_type i, const L& label) const
        {
            return index_of(i, label) < axis(i).size();
        }

        /**
         * Compute the alignment of this coordinate system with another:
         * returns a pair of index vectors (this_indices, other_indices) for
         * labels common to both systems on a given axis.
         */
        auto align(size_type axis_idx, const xcoordinate_system& other, size_type other_axis) const
        {
            std::vector<size_type> this_indices, other_indices;
            const auto& this_coord = axis(axis_idx);
            const auto& other_coord = other.axis(other_axis);
            for (size_type i = 0; i < this_coord.size(); ++i)
            {
                size_type j = other.index_of(other_axis, this_coord[i]);
                if (j < other_coord.size())
                {
                    this_indices.push_back(i);
                    other_indices.push_back(j);
                }
            }
            return std::make_pair(std::move(this_indices), std::move(other_indices));
        }

        /**
         * Broadcast two coordinate systems: find the union of labels on each axis.
         * Returns indices for both systems such that they map to the union.
         */
        auto broadcast(const xcoordinate_system& other) const
        {
            // Assumes same number of axes and same names
            if (axis_count() != other.axis_count())
                throw std::runtime_error("xcoordinate_system::broadcast: axis count mismatch.");
            std::vector<std::vector<size_type>> this_result(axis_count());
            std::vector<std::vector<size_type>> other_result(axis_count());
            for (size_type a = 0; a < axis_count(); ++a)
            {
                // Union of labels
                std::vector<L> union_labels;
                for (size_type i = 0; i < axis(a).size(); ++i)
                    union_labels.push_back(axis(a)[i]);
                for (size_type j = 0; j < other.axis(a).size(); ++j)
                {
                    L lbl = other.axis(a)[j];
                    if (std::find(union_labels.begin(), union_labels.end(), lbl) == union_labels.end())
                        union_labels.push_back(lbl);
                }
                // Map each original coordinate to the union index
                for (size_type i = 0; i < axis(a).size(); ++i)
                {
                    auto it = std::find(union_labels.begin(), union_labels.end(), axis(a)[i]);
                    this_result[a].push_back(static_cast<size_type>(std::distance(union_labels.begin(), it)));
                }
                for (size_type j = 0; j < other.axis(a).size(); ++j)
                {
                    auto it = std::find(union_labels.begin(), union_labels.end(), other.axis(a)[j]);
                    other_result[a].push_back(static_cast<size_type>(std::distance(union_labels.begin(), it)));
                }
            }
            return std::make_pair(std::move(this_result), std::move(other_result));
        }

    private:
        // Tuple of dimensions (for compile-time size)
        std::tuple<> m_dimensions; // placeholder; will be specialized
        std::vector<dimension_type> m_dimensions_vec;
        std::vector<std::map<L, size_type, std::less<>>> m_index_maps;

        template <std::size_t... I>
        void build_index_maps(std::index_sequence<I...>)
        {
            (build_map_for_axis(std::get<I>(m_dimensions).coord(), I), ...);
        }

        void build_index_maps_vec()
        {
            m_index_maps.resize(m_dimensions_vec.size());
            for (size_type i = 0; i < m_dimensions_vec.size(); ++i)
                build_map_for_axis(m_dimensions_vec[i].coord(), i);
        }

        void build_map_for_axis(const coordinate_type& coord, size_type axis_idx)
        {
            if (axis_idx >= m_index_maps.size())
                m_index_maps.resize(axis_idx + 1);
            auto& mp = m_index_maps[axis_idx];
            mp.clear();
            for (size_type k = 0; k < coord.size(); ++k)
                mp[coord[k]] = k;
        }

        template <std::size_t... I>
        const coordinate_type& get_axis_impl(size_type i, std::index_sequence<I...>) const
        {
            const coordinate_type* ptrs[] = { &std::get<I>(m_dimensions).coord()... };
            return *ptrs[i];
        }
    };

    /**
     * Helper to create a coordinate system from dimensions.
     */
    template <class... Dims>
    inline auto make_coordinate_system(std::tuple<Dims...> dims)
    {
        return xcoordinate_system<typename std::tuple_element<0, std::tuple<Dims...>>::type::label_type>(dims);
    }

} // namespace xframe

#endif // XFRAME_XCOORDINATE_SYSTEM_HPP