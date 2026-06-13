//File 0304 : xframe/xframe_dimension.hpp
//Dimension class combining a name, a coordinate array, and optional metadata (unit, description) for labeled axes.
#ifndef XFRAME_DIMENSION_HPP
#define XFRAME_DIMENSION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_coordinate.hpp"

namespace xframe
{
    /**
     * @class dimension
     * @brief A named axis with a coordinate array and optional metadata.
     *
     * Dimensions provide labeled indexing for xframe arrays. Each dimension
     * consists of a name (label), a coordinate (sequence of labels/values),
     * and optional unit and description strings for documentation.
     */
    template <class L = label_type>
    class dimension : public expression<dimension<L>>
    {
    public:
        using self_type = dimension<L>;
        using label_type = L;
        using coordinate_type = coordinate<label_type>;
        using size_type = std::size_t;

        /**
         * Construct a dimension with a name and coordinate.
         * @param name Dimension name.
         * @param coord Coordinate values.
         * @param unit Optional unit string.
         * @param desc Optional description.
         */
        dimension(const label_type& name, const coordinate_type& coord,
                  const label_type& unit = label_type{},
                  const label_type& desc = label_type{})
            : m_name(name), m_coordinate(coord), m_unit(unit), m_description(desc)
        {
        }

        /**
         * Construct a dimension with a name and coordinate (move).
         */
        dimension(const label_type& name, coordinate_type&& coord,
                  const label_type& unit = label_type{},
                  const label_type& desc = label_type{})
            : m_name(name), m_coordinate(std::move(coord)), m_unit(unit), m_description(desc)
        {
        }

        /**
         * Construct a dimension with a name and size (generates integer coordinates 0..size-1).
         */
        dimension(const label_type& name, size_type size)
            : m_name(name), m_coordinate()
        {
            for (size_type i = 0; i < size; ++i)
                m_coordinate.push_back(label_type(std::to_string(i)));
        }

        dimension(const self_type&) = default;
        dimension& operator=(const self_type&) = default;
        dimension(self_type&&) = default;
        dimension& operator=(self_type&&) = default;

        /**
         * Dimension name.
         */
        const label_type& name() const noexcept { return m_name; }
        void set_name(const label_type& n) { m_name = n; }

        /**
         * Coordinate access.
         */
        const coordinate_type& coord() const noexcept { return m_coordinate; }
        coordinate_type& coord() noexcept { return m_coordinate; }
        size_type size() const noexcept { return m_coordinate.size(); }

        /**
         * Metadata.
         */
        const label_type& unit() const noexcept { return m_unit; }
        void set_unit(const label_type& u) { m_unit = u; }
        const label_type& description() const noexcept { return m_description; }
        void set_description(const label_type& d) { m_description = d; }

        /**
         * Element access: return the coordinate label at index i.
         */
        const label_type& operator[](size_type i) const { return m_coordinate[i]; }
        label_type& operator[](size_type i) { return m_coordinate[i]; }

        /**
         * Find the index of a label in the coordinate.
         * Returns size() if not found.
         */
        size_type index_of(const label_type& label) const
        {
            return m_coordinate.find(label);
        }

        /**
         * Check if the coordinate contains a label.
         */
        bool contains(const label_type& label) const
        {
            return m_coordinate.contains(label);
        }

        /**
         * Sort the coordinate in ascending order.
         */
        void sort() { m_coordinate.sort(); }

        /**
         * Expression interface.
         */
        self_type& derived() noexcept { return *this; }
        const self_type& derived() const noexcept { return *this; }

        /**
         * Comparison.
         */
        bool operator==(const self_type& rhs) const
        {
            return m_name == rhs.m_name && m_coordinate == rhs.m_coordinate;
        }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

    private:
        label_type m_name;
        coordinate_type m_coordinate;
        label_type m_unit;
        label_type m_description;
    };

    /**
     * Helper to create a dimension with a name and list of labels.
     */
    template <class L>
    inline auto make_dimension(const L& name, std::initializer_list<L> labels,
                               const L& unit = L{}, const L& desc = L{})
    {
        return dimension<L>(name, coordinate<L>(labels), unit, desc);
    }

    /**
     * Helper to create an integer-indexed dimension.
     */
    template <class L = label_type>
    inline auto make_int_dimension(const L& name, std::size_t size)
    {
        return dimension<L>(name, size);
    }

} // namespace xframe

#endif // XFRAME_DIMENSION_HPP