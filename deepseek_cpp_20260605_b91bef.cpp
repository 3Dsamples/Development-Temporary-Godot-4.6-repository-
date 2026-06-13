//File 0330 : xframe/xaxis.hpp
//Top-level include for the xframe axis module: aggregating all axis types, functions, and utilities with full C++17 support, SIMD, and real-time 2D/3D simulation capabilities.
#ifndef XFRAME_XAXIS_HPP
#define XFRAME_XAXIS_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>
#include <variant>
#include <algorithm>
#include <cmath>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_dimension.hpp"
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xaxis_meta.hpp"
#include "xaxis_scalar.hpp"
#include "xaxis_variant.hpp"
#include "xaxis_view.hpp"
#include "xaxis_index_slice.hpp"
#include "xaxis_label_slice.hpp"
#include "xaxis_math.hpp"

namespace xframe {
namespace axis {

    /**
     * @enum axis_kind
     * @brief Classification of axis types used in xframe.
     */
    enum class axis_kind : uint8_t {
        index,            // Integer‑indexed axis (0, 1, 2, …)
        label,            // String‑labeled axis
        continuous,       // Floating‑point continuous axis (time, frequency)
        categorical,      // Finite set of categories
        datetime,         // Date/time axis
        unknown
    };

    /**
     * Detect the kind of axis from its coordinate type.
     */
    template <class T>
    constexpr axis_kind deduce_axis_kind() noexcept {
        if constexpr (std::is_integral_v<T>) return axis_kind::index;
        else if constexpr (std::is_floating_point_v<T>) return axis_kind::continuous;
        else if constexpr (std::is_same_v<T, std::string>) return axis_kind::label;
        else return axis_kind::unknown;
    }

    /**
     * @class xaxis
     * @brief Represents a single axis (dimension) of a data frame.
     *
     * An axis combines a dimension (name + coordinate) with optional unit,
     * description, and metadata. It provides element access by index and
     * label, supports slicing via index or label ranges, and can broadcast
     * its coordinate values to match the shape of multi‑dimensional expressions.
     */
    template <class L = label_type>
    class xaxis : public expression<xaxis<L>>
    {
    public:
        using self_type = xaxis<L>;
        using label_type = L;
        using dimension_type = dimension<L>;
        using coordinate_type = coordinate<L>;
        using size_type = std::size_t;

        xaxis() noexcept = default;

        explicit xaxis(const dimension_type& dim) : m_dimension(dim) {}
        explicit xaxis(dimension_type&& dim) noexcept : m_dimension(std::move(dim)) {}

        xaxis(const label_type& name, const coordinate_type& coord,
              const label_type& unit = label_type{},
              const label_type& desc = label_type{})
            : m_dimension(name, coord, unit, desc) {}

        xaxis(const label_type& name, size_type size)
            : m_dimension(name, size) {}

        xaxis(const self_type&) = default;
        xaxis& operator=(const self_type&) = default;
        xaxis(self_type&&) = default;
        xaxis& operator=(self_type&&) = default;

        const label_type& name() const noexcept { return m_dimension.name(); }
        void set_name(const label_type& n) { m_dimension.set_name(n); }
        const label_type& unit() const noexcept { return m_dimension.unit(); }
        void set_unit(const label_type& u) { m_dimension.set_unit(u); }
        const label_type& description() const noexcept { return m_dimension.description(); }
        void set_description(const label_type& d) { m_dimension.set_description(d); }

        const coordinate_type& coord() const noexcept { return m_dimension.coord(); }
        coordinate_type& coord() noexcept { return m_dimension.coord(); }
        size_type size() const noexcept { return m_dimension.size(); }

        const dimension_type& dimension() const noexcept { return m_dimension; }
        dimension_type& dimension() noexcept { return m_dimension; }

        const label_type& operator[](size_type i) const { return m_dimension.coord()[i]; }
        label_type& operator[](size_type i) { return m_dimension.coord()[i]; }

        size_type index_of(const label_type& label) const { return m_dimension.index_of(label); }
        bool contains(const label_type& label) const { return m_dimension.contains(label); }

        /**
         * Create a slice of this axis (view) using integer start/stop/step.
         */
        auto slice(std::ptrdiff_t start, std::ptrdiff_t stop, std::ptrdiff_t step = 1) const
        {
            std::size_t n = this->size();
            if (start < 0) start += static_cast<std::ptrdiff_t>(n);
            if (stop < 0) stop += static_cast<std::ptrdiff_t>(n);
            if (start < 0) start = 0;
            if (stop > static_cast<std::ptrdiff_t>(n)) stop = static_cast<std::ptrdiff_t>(n);
            if (step <= 0) throw std::runtime_error("xaxis::slice: step must be positive.");
            coordinate_type new_coord;
            for (std::ptrdiff_t i = start; i < stop; i += step)
                new_coord.push_back(m_dimension.coord()[static_cast<size_type>(i)]);
            return xaxis<L>(m_dimension.name(), std::move(new_coord),
                            m_dimension.unit(), m_dimension.description());
        }

        /**
         * Create a slice using label range.
         */
        auto slice_labels(const label_type& start_label, const label_type& stop_label) const
        {
            size_type i0 = this->index_of(start_label);
            size_type i1 = this->index_of(stop_label);
            if (i0 >= this->size() || i1 >= this->size())
                throw std::out_of_range("xaxis::slice_labels: label not found.");
            return slice(static_cast<std::ptrdiff_t>(i0), static_cast<std::ptrdiff_t>(i1 + 1));
        }

        /**
         * Select a subset of coordinates by integer indices.
         */
        auto select_indices(const std::vector<size_type>& indices) const
        {
            coordinate_type new_coord;
            for (auto i : indices)
                new_coord.push_back(m_dimension.coord().at(i));
            return xaxis<L>(m_dimension.name(), std::move(new_coord),
                            m_dimension.unit(), m_dimension.description());
        }

        /**
         * Select a subset of coordinates by labels.
         */
        auto select_labels(const std::vector<label_type>& labels) const
        {
            coordinate_type new_coord;
            for (const auto& lbl : labels)
            {
                size_type i = this->index_of(lbl);
                if (i >= this->size()) throw std::out_of_range("xaxis::select_labels: label not found.");
                new_coord.push_back(lbl);
            }
            return xaxis<L>(m_dimension.name(), std::move(new_coord),
                            m_dimension.unit(), m_dimension.description());
        }

        /**
         * Concatenate two axes (must have same name).
         */
        self_type operator+(const self_type& rhs) const
        {
            if (this->name() != rhs.name())
                throw std::runtime_error("xaxis::operator+: axis names must match.");
            coordinate_type new_coord = this->coord();
            for (size_type i = 0; i < rhs.size(); ++i)
                new_coord.push_back(rhs[i]);
            return xaxis<L>(this->name(), std::move(new_coord), this->unit(), this->description());
        }

        /**
         * Equality comparison.
         */
        bool operator==(const self_type& rhs) const { return m_dimension == rhs.m_dimension; }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

    private:
        dimension_type m_dimension;
    };

    /**
     * Check that two axes are compatible for broadcasting (same name, or one is scalar).
     */
    template <class L>
    inline bool axes_compatible(const xaxis<L>& a, const xaxis<L>& b) noexcept
    {
        return a.name() == b.name() || a.size() == 1 || b.size() == 1;
    }

    /**
     * Align two axes: return the larger axis (union of labels).
     * If they have the same name and overlapping labels, the result contains the sorted union.
     */
    template <class L>
    inline xaxis<L> align_axes(const xaxis<L>& a, const xaxis<L>& b)
    {
        if (a.name() != b.name()) throw std::runtime_error("align_axes: names must match.");
        std::vector<L> union_labels;
        for (size_type i = 0; i < a.size(); ++i) union_labels.push_back(a[i]);
        for (size_type i = 0; i < b.size(); ++i)
            if (std::find(union_labels.begin(), union_labels.end(), b[i]) == union_labels.end())
                union_labels.push_back(b[i]);
        std::sort(union_labels.begin(), union_labels.end());
        coordinate<L> coord;
        for (auto& lbl : union_labels) coord.push_back(lbl);
        return xaxis<L>(a.name(), std::move(coord), a.unit(), a.description());
    }

    /**
     * Generate a range axis with integer coordinates 0..n-1.
     */
    template <class L = label_type>
    inline xaxis<L> range_axis(const L& name, std::size_t n)
    {
        return xaxis<L>(name, n);
    }

    /**
     * Create an axis from a vector of labels.
     */
    template <class L>
    inline xaxis<L> make_axis(const L& name, std::initializer_list<L> labels)
    {
        return xaxis<L>(name, coordinate<L>(labels));
    }

} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_HPP