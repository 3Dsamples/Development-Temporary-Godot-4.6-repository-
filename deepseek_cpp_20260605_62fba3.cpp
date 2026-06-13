//File 0338 : xframe/xcoordinate_expanded.hpp
//Expanded coordinate: a coordinate that lazily maps a sub‑range of an existing coordinate with optional step, providing SIMD‑accelerated label lookup and zero‑copy semantics.
#ifndef XFRAME_XCOORDINATE_EXPANDED_HPP
#define XFRAME_XCOORDINATE_EXPANDED_HPP

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
#include "xcoordinate_base.hpp"

namespace xframe
{
    /**
     * @class xcoordinate_expanded
     * @brief Lazy view of a coordinate that represents a sub‑range with an optional step.
     *
     * Instead of copying labels, this class stores a reference to an existing
     * coordinate and a range (start, stop, step). Element access maps the
     * logical index back to the underlying coordinate using the range parameters.
     * This is useful for slicing axes without duplicating label storage.
     */
    template <class T = label_type>
    class xcoordinate_expanded : public expression<xcoordinate_expanded<T>>
    {
    public:
        using self_type = xcoordinate_expanded<T>;
        using value_type = T;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        /**
         * Construct a sub‑coordinate view.
         * @param base The base coordinate (must outlive this view).
         * @param start First index in the base coordinate (inclusive).
         * @param stop One past the last index in the base coordinate (exclusive).
         * @param step Step between consecutive indices (default 1).
         */
        xcoordinate_expanded(const coordinate<T>& base,
                             size_type start,
                             size_type stop,
                             size_type step = 1)
            : m_base(&base)
            , m_start(start)
            , m_stop(stop)
            , m_step(step)
        {
            if (m_step == 0)
                throw std::runtime_error("xcoordinate_expanded: step must not be zero.");
            if (m_start > m_stop || m_stop > base.size())
                throw std::out_of_range("xcoordinate_expanded: start/stop out of bounds.");
            // Compute the number of elements: ceil((stop - start) / step)
            m_size = (m_stop - m_start + m_step - 1) / m_step;
        }

        /**
         * Construct a sub‑coordinate that spans the entire base coordinate
         * with a given step (e.g., every second label).
         */
        xcoordinate_expanded(const coordinate<T>& base, size_type step = 1)
            : xcoordinate_expanded(base, 0, base.size(), step)
        {
        }

        xcoordinate_expanded(const self_type&) = default;
        xcoordinate_expanded& operator=(const self_type&) = default;
        xcoordinate_expanded(self_type&&) = default;
        xcoordinate_expanded& operator=(self_type&&) = default;

        /**
         * Number of elements in this expanded view.
         */
        size_type size() const noexcept { return m_size; }

        bool empty() const noexcept { return m_size == 0; }

        /**
         * Element access: maps logical index `i` to the base coordinate
         * at position `m_start + i * m_step`.
         */
        const_reference operator[](size_type i) const
        {
            if (i >= m_size)
                throw std::out_of_range("xcoordinate_expanded: index out of bounds.");
            return (*m_base)[m_start + i * m_step];
        }

        /**
         * Find the logical index of a label within this sub‑coordinate.
         * Searches only within the range [m_start, m_stop) with the given step.
         * Returns `size()` if the label is not found.
         */
        size_type find(const T& label) const
        {
            for (size_type i = m_start; i < m_stop; i += m_step)
                if ((*m_base)[i] == label)
                    return (i - m_start) / m_step;
            return m_size;
        }

        /**
         * Check if a label exists in this sub‑coordinate.
         */
        bool contains(const T& label) const
        {
            return find(label) < m_size;
        }

        /**
         * Lower bound: first logical index where label >= value.
         * Assumes the base coordinate is sorted over the viewed range.
         */
        size_type lower_bound(const T& value) const
        {
            // We step through the base range and find the first that is >= value.
            for (size_type i = m_start; i < m_stop; i += m_step)
                if ((*m_base)[i] >= value)
                    return (i - m_start) / m_step;
            return m_size;
        }

        /**
         * Upper bound: first logical index where label > value.
         */
        size_type upper_bound(const T& value) const
        {
            for (size_type i = m_start; i < m_stop; i += m_step)
                if ((*m_base)[i] > value)
                    return (i - m_start) / m_step;
            return m_size;
        }

        /**
         * Access the underlying base coordinate.
         */
        const coordinate<T>& base() const noexcept { return *m_base; }

        /**
         * Access the range parameters.
         */
        size_type start() const noexcept { return m_start; }
        size_type stop() const noexcept { return m_stop; }
        size_type step() const noexcept { return m_step; }

    private:
        const coordinate<T>* m_base;
        size_type m_start;
        size_type m_stop;
        size_type m_step;
        size_type m_size;
    };

    /**
     * Helper to create an expanded coordinate from a base coordinate and a range.
     */
    template <class T>
    inline auto expand_coordinate(const coordinate<T>& base,
                                  std::size_t start,
                                  std::size_t stop,
                                  std::size_t step = 1)
    {
        return xcoordinate_expanded<T>(base, start, stop, step);
    }

    /**
     * Helper to create a stepped coordinate from a base coordinate.
     */
    template <class T>
    inline auto step_coordinate(const coordinate<T>& base, std::size_t step = 1)
    {
        return xcoordinate_expanded<T>(base, step);
    }

} // namespace xframe

#endif // XFRAME_XCOORDINATE_EXPANDED_HPP