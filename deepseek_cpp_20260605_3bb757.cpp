//File 0331 : xframe/xaxis_base.hpp
//Abstract base class for all axis types: defines the common interface for dimension access, coordinate iteration, and broadcasting with SIMD support.
#ifndef XFRAME_XAXIS_BASE_HPP
#define XFRAME_XAXIS_BASE_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xframe_config.hpp"
#include "xframe_forward.hpp"
#include "xframe_expression.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"

namespace xframe {
namespace axis {

    /**
     * @class xaxis_base
     * @brief CRTP base class for all axis types.
     *
     * Provides a uniform interface for accessing the dimension, coordinate
     * labels, size, and metadata (unit, description). Derived classes
     * implement specific coordinate types (integer, label, continuous).
     * The base also supports broadcasting via virtual interface for
     * runtime polymorphism when needed.
     */
    template <class D, class L = label_type>
    class xaxis_base : public expression<D>
    {
    public:
        using derived_type = D;
        using self_type = xaxis_base<D, L>;
        using label_type = L;
        using size_type = std::size_t;
        using coordinate_type = coordinate<L>;
        using dimension_type = dimension<L>;

        /**
         * Construct from a dimension descriptor.
         */
        explicit xaxis_base(const dimension_type& dim) : m_dimension(dim) {}
        explicit xaxis_base(dimension_type&& dim) noexcept : m_dimension(std::move(dim)) {}

        /**
         * Construct with name and size (generates integer labels 0..size-1).
         */
        xaxis_base(const label_type& name, size_type size)
            : m_dimension(name, size) {}

        /**
         * Construct with name and coordinate.
         */
        xaxis_base(const label_type& name, const coordinate_type& coord,
                   const label_type& unit = label_type{},
                   const label_type& desc = label_type{})
            : m_dimension(name, coord, unit, desc) {}

        xaxis_base(const self_type&) = default;
        xaxis_base& operator=(const self_type&) = default;
        xaxis_base(self_type&&) = default;
        xaxis_base& operator=(self_type&&) = default;

        // --- Dimension access ---
        const dimension_type& dimension() const noexcept { return m_dimension; }
        dimension_type& dimension() noexcept { return m_dimension; }

        const label_type& name() const noexcept { return m_dimension.name(); }
        void set_name(const label_type& n) { m_dimension.set_name(n); }
        const label_type& unit() const noexcept { return m_dimension.unit(); }
        void set_unit(const label_type& u) { m_dimension.set_unit(u); }
        const label_type& description() const noexcept { return m_dimension.description(); }
        void set_description(const label_type& d) { m_dimension.set_description(d); }

        // --- Coordinate access ---
        const coordinate_type& coord() const noexcept { return m_dimension.coord(); }
        coordinate_type& coord() noexcept { return m_dimension.coord(); }
        size_type size() const noexcept { return m_dimension.size(); }

        const label_type& operator[](size_type i) const { return m_dimension.coord()[i]; }
        label_type& operator[](size_type i) { return m_dimension.coord()[i]; }

        size_type index_of(const label_type& label) const { return m_dimension.index_of(label); }
        bool contains(const label_type& label) const { return m_dimension.contains(label); }

        // --- Expression interface ---
        derived_type& derived() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived() const noexcept { return *static_cast<const derived_type*>(this); }

        // --- Range access (to be overridden by derived) ---
        auto begin() const noexcept { return m_dimension.coord().begin(); }
        auto end() const noexcept { return m_dimension.coord().end(); }

        // --- Comparison ---
        bool operator==(const self_type& rhs) const { return m_dimension == rhs.m_dimension; }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

        // --- Broadcasting helper ---
        bool compatible_with(const self_type& other) const noexcept
        {
            return (name() == other.name()) && (size() == other.size() || size() == 1 || other.size() == 1);
        }

    protected:
        dimension_type m_dimension;
    };

} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_BASE_HPP