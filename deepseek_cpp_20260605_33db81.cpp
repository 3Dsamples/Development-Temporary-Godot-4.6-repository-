//File 0333 : xframe/xaxis_expression_leaf.hpp
//Leaf expression node for axes: terminal node in the expression template system wrapping a single axis, providing element access, broadcasting, and SIMD support.
#ifndef XFRAME_XAXIS_EXPRESSION_LEAF_HPP
#define XFRAME_XAXIS_EXPRESSION_LEAF_HPP

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
#include "xaxis_base.hpp"

namespace xframe {
namespace axis {

    /**
     * @class xaxis_expression_leaf
     * @brief Terminal expression node that wraps an axis as a leaf in the
     *        expression template tree.
     *
     * This class enables axes to participate in lazy expression evaluation
     * alongside xframe arrays. When used in arithmetic, the leaf broadcasts
     * its coordinate values (converted to numeric type) across the dimensions
     * of the other operand. SIMD loads are supported when the underlying axis
     * data is contiguous.
     */
    template <class Axis, class T = double>
    class xaxis_expression_leaf : public expression<xaxis_expression_leaf<Axis, T>>
    {
    public:
        using self_type = xaxis_expression_leaf<Axis, T>;
        using axis_type = Axis;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using label_type = typename Axis::label_type;

        /**
         * Construct the leaf from an axis reference.
         * The axis must outlive this expression node.
         */
        explicit xaxis_expression_leaf(const Axis& ax) noexcept : m_axis(ax) {}

        xaxis_expression_leaf(const self_type&) = default;
        xaxis_expression_leaf& operator=(const self_type&) = default;
        xaxis_expression_leaf(self_type&&) = default;
        xaxis_expression_leaf& operator=(self_type&&) = default;

        size_type dimension_count() const noexcept { return 1; }
        size_type size() const noexcept { return m_axis.size(); }

        /**
         * Returns the dimension descriptor of the wrapped axis.
         */
        const dimension<label_type>& dimension(size_type i) const
        {
            if (i != 0) throw std::out_of_range("xaxis_expression_leaf: dimension index out of range.");
            return m_axis.dimension();
        }

        /**
         * Element access: convert the coordinate label at the given index
         * to a numeric value of type T.
         */
        const_reference operator()(size_type i) const
        {
            m_cached_value = convert_to_value(m_axis[i]);
            return m_cached_value;
        }

        reference operator()(size_type i)
        {
            m_cached_value = convert_to_value(m_axis[i]);
            return m_cached_value;
        }

        const_reference operator[](size_type i) const { return (*this)(i); }
        reference operator[](size_type i) { return (*this)(i); }

        /**
         * Label‑based access.
         */
        const_reference locate(const label_type& label) const
        {
            size_type idx = m_axis.index_of(label);
            if (idx >= m_axis.size()) throw std::out_of_range("xaxis_expression_leaf: label not found.");
            return (*this)(idx);
        }

        reference locate(const label_type& label)
        {
            size_type idx = m_axis.index_of(label);
            if (idx >= m_axis.size()) throw std::out_of_range("xaxis_expression_leaf: label not found.");
            return (*this)(idx);
        }

        pointer data() noexcept { return m_numeric_data.data(); }
        const_pointer data() const noexcept
        {
            ensure_numeric_cache();
            return m_numeric_data.data();
        }

        const Axis& axis() const noexcept { return m_axis; }

        /**
         * SIMD load: returns a batch of numeric values at the given flat offset.
         */
        template <class Align, class U = T>
        auto load_simd(std::size_t i) const
        {
            ensure_numeric_cache();
            using simd_type = xsimd::batch<U, default_simd_arch>;
            return simd_type::load_unaligned(m_numeric_data.data() + i);
        }

        /**
         * Broadcast shape: for expression compatibility, the leaf reports
         * its shape as having one dimension.
         */
        auto shape() const
        {
            std::vector<size_type> s;
            s.push_back(m_axis.size());
            return s;
        }

    private:
        const Axis& m_axis;
        mutable T m_cached_value = T{};
        mutable std::vector<T> m_numeric_data;
        mutable bool m_numeric_cache_valid = false;

        /**
         * Convert a label to a numeric value.
         * For arithmetic types, static_cast; for strings, attempt std::stod.
         */
        static T convert_to_value(const label_type& label)
        {
            if constexpr (std::is_arithmetic_v<label_type>)
            {
                return static_cast<T>(label);
            }
            else if constexpr (std::is_same_v<label_type, std::string>)
            {
                try { return static_cast<T>(std::stod(label)); }
                catch (...) { return std::numeric_limits<T>::quiet_NaN(); }
            }
            else
            {
                return T{};
            }
        }

        /**
         * Populate the numeric cache from the axis coordinates.
         */
        void ensure_numeric_cache() const
        {
            if (m_numeric_cache_valid) return;
            m_numeric_data.resize(m_axis.size());
            for (size_type i = 0; i < m_axis.size(); ++i)
                m_numeric_data[i] = convert_to_value(m_axis[i]);
            m_numeric_cache_valid = true;
        }
    };

    /**
     * Helper to wrap an axis as an expression leaf.
     */
    template <class T = double, class Axis>
    inline auto make_axis_leaf(const Axis& ax)
    {
        return xaxis_expression_leaf<Axis, T>(ax);
    }

} // namespace axis
} // namespace xframe

#endif // XFRAME_XAXIS_EXPRESSION_LEAF_HPP