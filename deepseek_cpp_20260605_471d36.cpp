//File 0340 : xframe/xcoordinate_view.hpp
//Coordinate view: a lazy non‑owning slice of an existing coordinate, providing label access, search, and SIMD‑accelerated operations without copying labels.
#ifndef XFRAME_XCOORDINATE_VIEW_HPP
#define XFRAME_XCOORDINATE_VIEW_HPP

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
     * @class xcoordinate_view
     * @brief Non‑owning view of a sub‑range of a coordinate.
     *
     * The view stores a pointer to the base coordinate and a range
     * [start, stop). Element access directly returns base labels
     * without copying. Label lookup is restricted to the viewed
     * range. The base coordinate must outlive the view.
     */
    template <class T = label_type>
    class xcoordinate_view : public expression<xcoordinate_view<T>>
    {
    public:
        using self_type = xcoordinate_view<T>;
        using value_type = T;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        /**
         * Construct a view of an entire coordinate.
         */
        explicit xcoordinate_view(const coordinate<T>& base) noexcept
            : m_base(&base), m_start(0), m_stop(base.size())
        {
        }

        /**
         * Construct a view of a sub‑range [start, stop).
         */
        xcoordinate_view(const coordinate<T>& base, size_type start, size_type stop)
            : m_base(&base), m_start(start), m_stop(stop)
        {
            if (start > stop || stop > base.size())
                throw std::out_of_range("xcoordinate_view: invalid range.");
        }

        xcoordinate_view(const self_type&) = default;
        xcoordinate_view& operator=(const self_type&) = default;
        xcoordinate_view(self_type&&) = default;
        xcoordinate_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return m_stop - m_start; }
        bool empty() const noexcept { return m_start == m_stop; }

        const_reference operator[](size_type i) const
        {
            if (i >= size()) throw std::out_of_range("xcoordinate_view: index out of bounds.");
            return (*m_base)[m_start + i];
        }

        /**
         * Find the logical index of a label within the view.
         * Searches only the range [m_start, m_stop) of the base.
         * Returns size() if not found.
         */
        size_type find(const T& label) const
        {
            for (size_type i = m_start; i < m_stop; ++i)
                if ((*m_base)[i] == label)
                    return i - m_start;
            return size();
        }

        bool contains(const T& label) const
        {
            return find(label) < size();
        }

        /**
         * Lower bound within the view (assumes base sorted in range).
         */
        size_type lower_bound(const T& value) const
        {
            for (size_type i = m_start; i < m_stop; ++i)
                if ((*m_base)[i] >= value)
                    return i - m_start;
            return size();
        }

        size_type upper_bound(const T& value) const
        {
            for (size_type i = m_start; i < m_stop; ++i)
                if ((*m_base)[i] > value)
                    return i - m_start;
            return size();
        }

        const coordinate<T>& base() const noexcept { return *m_base; }
        size_type start() const noexcept { return m_start; }
        size_type stop() const noexcept { return m_stop; }

    private:
        const coordinate<T>* m_base;
        size_type m_start;
        size_type m_stop;
    };

    /**
     * Helper to create a coordinate view from a coordinate and a range.
     */
    template <class T>
    inline auto view_coordinate(const coordinate<T>& base,
                                std::size_t start, std::size_t stop)
    {
        return xcoordinate_view<T>(base, start, stop);
    }

    /**
     * Helper to create a full view of a coordinate.
     */
    template <class T>
    inline auto view_coordinate(const coordinate<T>& base)
    {
        return xcoordinate_view<T>(base);
    }

} // namespace xframe

#endif // XFRAME_XCOORDINATE_VIEW_HPP