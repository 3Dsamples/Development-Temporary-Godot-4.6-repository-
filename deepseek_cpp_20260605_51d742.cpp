//File 0327 : xframe/xaxis_scalar.hpp
//Axis scalar: a 1D variable tied to a named axis, broadcasting element access over other dimensions, with SIMD data storage and expression integration.
#ifndef XFRAME_XAXIS_SCALAR_HPP
#define XFRAME_XAXIS_SCALAR_HPP

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
#include "xframe_variable.hpp"
#include "xframe_coordinate.hpp"
#include "xframe_dimension.hpp"

namespace xframe
{
    /**
     * @class xaxis_scalar
     * @brief A scalar value varying along a single axis (dimension).
     *
     * The object holds a variable whose size equals the length of the associated
     * axis. In any expression, it broadcasts to the shape of the other operand
     * by returning the value corresponding to the coordinate along that axis.
     */
    template <class T = double, class L = label_type>
    class xaxis_scalar : public expression<xaxis_scalar<T, L>>
    {
    public:
        using self_type = xaxis_scalar<T, L>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using label_type = L;
        using variable_type = variable<T, L>;
        using dimension_type = dimension<L>;

        /**
         * Construct an axis scalar with a given dimension and optional initial value.
         * @param axis_dim The dimension along which values vary.
         * @param fill_value Initial value for all entries along the axis.
         */
        explicit xaxis_scalar(const dimension_type& axis_dim, T fill_value = T{})
            : m_dim(axis_dim), m_values(axis_dim.size(), axis_dim.name())
        {
            m_values.fill(fill_value);
        }

        /**
         * Construct an axis scalar with a given dimension and a list of values.
         * The number of values must match the dimension size.
         */
        xaxis_scalar(const dimension_type& axis_dim, std::initializer_list<T> vals)
            : m_dim(axis_dim), m_values(axis_dim.size(), axis_dim.name())
        {
            if (vals.size() != axis_dim.size())
                throw std::runtime_error("xaxis_scalar: size mismatch.");
            std::copy(vals.begin(), vals.end(), m_values.data());
        }

        /**
         * Construct from a dimension and a vector of values.
         */
        xaxis_scalar(const dimension_type& axis_dim, const std::vector<T>& vals)
            : m_dim(axis_dim), m_values(axis_dim.size(), axis_dim.name())
        {
            if (vals.size() != axis_dim.size())
                throw std::runtime_error("xaxis_scalar: size mismatch.");
            std::copy(vals.begin(), vals.end(), m_values.data());
        }

        xaxis_scalar(const self_type&) = default;
        xaxis_scalar& operator=(const self_type&) = default;
        xaxis_scalar(self_type&&) = default;
        xaxis_scalar& operator=(self_type&&) = default;

        /**
         * Dimension count: axis scalar contributes 1 dimension (its own axis).
         */
        std::size_t dimension_count() const noexcept { return 1; }

        /**
         * Size: number of elements equals the axis length.
         */
        size_type size() const noexcept { return m_dim.size(); }

        /**
         * Returns the dimension descriptor.
         */
        const dimension_type& dimension(std::size_t i) const
        {
            if (i != 0) throw std::out_of_range("xaxis_scalar: dimension index out of range.");
            return m_dim;
        }

        /**
         * Element access: only one index is meaningful (the coordinate along the axis).
         */
        template <class... Args>
        const_reference operator()(size_type index) const
        {
            return m_values[index];
        }

        template <class... Args>
        reference operator()(size_type index)
        {
            return m_values[index];
        }

        const_reference operator[](size_type i) const { return m_values[i]; }
        reference operator[](size_type i) { return m_values[i]; }

        template <class... Labels>
        const_reference locate(const L& label) const
        {
            size_type idx = m_dim.coord().find(label);
            if (idx >= m_dim.size()) throw std::out_of_range("xaxis_scalar: label not found.");
            return m_values[idx];
        }

        template <class... Labels>
        reference locate(const L& label)
        {
            size_type idx = m_dim.coord().find(label);
            if (idx >= m_dim.size()) throw std::out_of_range("xaxis_scalar: label not found.");
            return m_values[idx];
        }

        pointer data() noexcept { return m_values.data(); }
        const_pointer data() const noexcept { return m_values.data(); }

        variable_type& variable() noexcept { return m_values; }
        const variable_type& variable() const noexcept { return m_values; }

        /**
         * Access the underlying dimension.
         */
        const dimension_type& axis_dimension() const noexcept { return m_dim; }

        /**
         * SIMD load.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(m_values.data() + i);
        }

        /**
         * Arithmetic assignment from another axis scalar (same axis).
         */
        self_type& operator+=(const self_type& rhs)
        {
            if (m_dim.name() != rhs.m_dim.name() || m_dim.size() != rhs.m_dim.size())
                throw std::runtime_error("xaxis_scalar: incompatible dimensions.");
            m_values += rhs.m_values;
            return *this;
        }

        self_type& operator-=(const self_type& rhs)
        {
            if (m_dim.name() != rhs.m_dim.name() || m_dim.size() != rhs.m_dim.size())
                throw std::runtime_error("xaxis_scalar: incompatible dimensions.");
            m_values -= rhs.m_values;
            return *this;
        }

        self_type& operator*=(T scalar)
        {
            m_values *= scalar;
            return *this;
        }

        self_type& operator/=(T scalar)
        {
            m_values /= scalar;
            return *this;
        }

    private:
        dimension_type m_dim;
        variable_type m_values;
    };

    /**
     * Free functions to create axis scalars.
     */
    template <class L = label_type, class T = double>
    inline auto make_axis_scalar(const dimension<L>& dim, T fill = T{})
    {
        return xaxis_scalar<T, L>(dim, fill);
    }

    template <class L = label_type, class T = double>
    inline auto make_axis_scalar(const dimension<L>& dim, std::initializer_list<T> vals)
    {
        return xaxis_scalar<T, L>(dim, vals);
    }

    /**
     * Multiplication of an xframe by an axis scalar (broadcast along the axis).
     * xframe(i,j,...) * axis_scalar(i) for axis 0.
     * Not fully implemented here; placeholder.
     */

} // namespace xframe

#endif // XFRAME_XAXIS_SCALAR_HPP