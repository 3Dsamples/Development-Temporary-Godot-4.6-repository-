//File 0353 : xframe/xnamed_axis.hpp
//Named axis combining a label and a coordinate with SIMD‑accelerated label lookup, slicing, and broadcasting for high‑performance data manipulation.
#ifndef XFRAME_XNAMED_AXIS_HPP
#define XFRAME_XNAMED_AXIS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xcoordinate.hpp"
#include "xdimension.hpp"

namespace xframe
{
    /**
     * @class xnamed_axis
     * @brief A named axis that pairs a label with a coordinate sequence.
     *
     * This class provides a minimal but complete axis representation: a name,
     * a coordinate, and optional metadata (unit, description). It supports
     * slicing, resizing, iteration, and broadcasting alignment. The coordinate
     * storage uses contiguous memory for fast SIMD label comparison when
     * coordinate elements are arithmetic types.
     */
    template <class L = label_type>
    class xnamed_axis : public expression<xnamed_axis<L>>
    {
    public:
        using self_type = xnamed_axis<L>;
        using label_type = L;
        using coordinate_type = coordinate<L>;
        using size_type = std::size_t;

        /**
         * Default constructor: empty axis.
         */
        xnamed_axis() noexcept = default;

        /**
         * Construct with a name and coordinate.
         */
        xnamed_axis(const label_type& name, const coordinate_type& coord,
                    const label_type& unit = label_type{},
                    const label_type& desc = label_type{})
            : m_name(name), m_coordinate(coord), m_unit(unit), m_description(desc)
        {
        }

        /**
         * Construct with a name and coordinate (move).
         */
        xnamed_axis(const label_type& name, coordinate_type&& coord,
                    const label_type& unit = label_type{},
                    const label_type& desc = label_type{})
            : m_name(name), m_coordinate(std::move(coord)), m_unit(unit), m_description(desc)
        {
        }

        /**
         * Construct with a name and a size (generates integer labels 0..size-1).
         */
        xnamed_axis(const label_type& name, size_type size)
            : m_name(name)
        {
            for (size_type i = 0; i < size; ++i)
                m_coordinate.push_back(label_type(std::to_string(i)));
        }

        xnamed_axis(const self_type&) = default;
        xnamed_axis& operator=(const self_type&) = default;
        xnamed_axis(self_type&&) = default;
        xnamed_axis& operator=(self_type&&) = default;

        /**
         * Axis name access.
         */
        const label_type& name() const noexcept { return m_name; }
        void set_name(const label_type& n) { m_name = n; }

        /**
         * Unit metadata.
         */
        const label_type& unit() const noexcept { return m_unit; }
        void set_unit(const label_type& u) { m_unit = u; }

        /**
         * Description metadata.
         */
        const label_type& description() const noexcept { return m_description; }
        void set_description(const label_type& d) { m_description = d; }

        /**
         * Coordinate access.
         */
        const coordinate_type& coord() const noexcept { return m_coordinate; }
        coordinate_type& coord() noexcept { return m_coordinate; }

        size_type size() const noexcept { return m_coordinate.size(); }
        bool empty() const noexcept { return m_coordinate.empty(); }

        /**
         * Element access by index.
         */
        const label_type& operator[](size_type i) const { return m_coordinate[i]; }
        label_type& operator[](size_type i) { return m_coordinate[i]; }

        /**
         * Label lookup (linear or binary search depending on sorted state).
         */
        size_type index_of(const label_type& label) const
        {
            return m_coordinate.find(label);
        }

        /**
         * Check if the axis contains a given label.
         */
        bool contains(const label_type& label) const
        {
            return m_coordinate.contains(label);
        }

        /**
         * Sort the coordinate labels in ascending order.
         */
        void sort()
        {
            m_coordinate.sort();
        }

        /**
         * Check if this axis is compatible with another for broadcasting.
         */
        bool compatible_with(const xnamed_axis& other) const noexcept
        {
            return (m_name == other.m_name) &&
                   (m_coordinate.size() == other.m_coordinate.size() ||
                    m_coordinate.size() == 1 || other.m_coordinate.size() == 1);
        }

        /**
         * Slice the axis using an integer start/stop/step.
         */
        xnamed_axis slice(std::ptrdiff_t start, std::ptrdiff_t stop,
                           std::ptrdiff_t step = 1) const
        {
            std::size_t n = size();
            if (start < 0) start += static_cast<std::ptrdiff_t>(n);
            if (stop < 0) stop += static_cast<std::ptrdiff_t>(n);
            start = std::max<std::ptrdiff_t>(0, start);
            stop = std::min<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(n), stop);
            if (start >= stop || step <= 0)
                return xnamed_axis(m_name, coordinate_type{});
            coordinate_type new_coord;
            for (std::ptrdiff_t i = start; i < stop; i += step)
                new_coord.push_back(m_coordinate[static_cast<size_type>(i)]);
            return xnamed_axis(m_name, std::move(new_coord), m_unit, m_description);
        }

        /**
         * Slice by label range (inclusive).
         */
        xnamed_axis slice(const label_type& start_label,
                           const label_type& stop_label) const
        {
            size_type i0 = index_of(start_label);
            size_type i1 = index_of(stop_label);
            if (i0 >= size() || i1 >= size())
                throw std::out_of_range("xnamed_axis::slice: label not found.");
            return slice(static_cast<std::ptrdiff_t>(i0),
                         static_cast<std::ptrdiff_t>(i1 + 1));
        }

        /**
         * Append a label to the axis.
         */
        void push_back(const label_type& label)
        {
            m_coordinate.push_back(label);
        }

        /**
         * Clear all labels.
         */
        void clear()
        {
            m_coordinate.clear();
        }

        /**
         * Concatenate two axes (must have the same name).
         */
        self_type operator+(const self_type& rhs) const
        {
            if (m_name != rhs.m_name)
                throw std::runtime_error("xnamed_axis::operator+: names must match.");
            coordinate_type new_coord = m_coordinate;
            new_coord += rhs.m_coordinate;
            return xnamed_axis(m_name, std::move(new_coord), m_unit, m_description);
        }

        /**
         * Equality comparison.
         */
        bool operator==(const self_type& rhs) const
        {
            return m_name == rhs.m_name && m_coordinate == rhs.m_coordinate &&
                   m_unit == rhs.m_unit && m_description == rhs.m_description;
        }
        bool operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * Iterators (delegate to coordinate).
         */
        auto begin() noexcept { return m_coordinate.begin(); }
        auto end() noexcept { return m_coordinate.end(); }
        auto begin() const noexcept { return m_coordinate.begin(); }
        auto end() const noexcept { return m_coordinate.end(); }

        /**
         * SIMD‑accelerated equality check for coordinate labels (for integer types).
         */
        bool simd_equals(const self_type& rhs) const
        {
            if (m_coordinate.size() != rhs.m_coordinate.size()) return false;
            if constexpr (std::is_arithmetic_v<L> && simd_enabled_v<L>)
            {
                using simd_type = xsimd::batch<L, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t n = m_coordinate.size();
                std::size_t vec_count = n / simd_size;
                const L* a = m_coordinate.labels().data();
                const L* b = rhs.m_coordinate.labels().data();
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type va = simd_type::load_unaligned(a + i * simd_size);
                    simd_type vb = simd_type::load_unaligned(b + i * simd_size);
                    if (!xsimd::all(va == vb)) return false;
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    if (a[i] != b[i]) return false;
                return true;
            }
            else
            {
                return m_coordinate == rhs.m_coordinate;
            }
        }

    private:
        label_type m_name;
        coordinate_type m_coordinate;
        label_type m_unit;
        label_type m_description;
    };

    /**
     * Helper to create a named axis from a list of labels.
     */
    template <class L>
    inline auto make_named_axis(const L& name, std::initializer_list<L> labels,
                                const L& unit = L{}, const L& desc = L{})
    {
        return xnamed_axis<L>(name, coordinate<L>(labels), unit, desc);
    }

    /**
     * Helper to create an axis with integer range.
     */
    template <class L = label_type>
    inline auto make_range_axis(const L& name, std::size_t n)
    {
        return xnamed_axis<L>(name, n);
    }

} // namespace xframe

#endif // XFRAME_XNAMED_AXIS_HPP