//File 0303 : xframe/xframe_coordinate.hpp
//Coordinate type: stores a 1D array of labels for a dimension, with support for integer, string, and floating-point labels.
#ifndef XFRAME_COORDINATE_HPP
#define XFRAME_COORDINATE_HPP

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

namespace xframe
{
    /**
     * @class coordinate
     * @brief A sequence of labels (values) for a dimension.
     *
     * Coordinates can be of any type, but commonly are strings (labels) or
     * arithmetic types (times, locations). They provide named indexing for
     * dimensions and support sorting, searching, and binary search for
     * efficient label-based lookup.
     */
    template <class T = label_type>
    class coordinate : public expression<coordinate<T>>
    {
    public:
        using self_type = coordinate<T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        coordinate() noexcept = default;
        coordinate(std::initializer_list<T> labels) : m_labels(labels) {}
        explicit coordinate(const std::vector<T>& labels) : m_labels(labels) {}
        explicit coordinate(std::vector<T>&& labels) noexcept : m_labels(std::move(labels)) {}

        coordinate(const self_type&) = default;
        coordinate& operator=(const self_type&) = default;
        coordinate(self_type&&) = default;
        coordinate& operator=(self_type&&) = default;

        /**
         * Size of the coordinate array.
         */
        size_type size() const noexcept { return m_labels.size(); }
        bool empty() const noexcept { return m_labels.empty(); }

        /**
         * Element access.
         */
        reference operator[](size_type i) { return m_labels[i]; }
        const_reference operator[](size_type i) const { return m_labels[i]; }

        reference at(size_type i)
        {
            if (i >= m_labels.size()) throw std::out_of_range("coordinate::at");
            return m_labels[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= m_labels.size()) throw std::out_of_range("coordinate::at");
            return m_labels[i];
        }

        /**
         * Data access.
         */
        const std::vector<T>& labels() const noexcept { return m_labels; }
        std::vector<T>& labels() noexcept { return m_labels; }

        /**
         * Iterator support.
         */
        iterator begin() noexcept { return m_labels.begin(); }
        iterator end() noexcept { return m_labels.end(); }
        const_iterator begin() const noexcept { return m_labels.begin(); }
        const_iterator end() const noexcept { return m_labels.end(); }
        const_iterator cbegin() const noexcept { return m_labels.begin(); }
        const_iterator cend() const noexcept { return m_labels.end(); }

        /**
         * Find the index of a label.
         * Returns the index if found, otherwise size() (npos).
         */
        size_type find(const T& label) const
        {
            auto it = std::find(m_labels.begin(), m_labels.end(), label);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Check if a label is present.
         */
        bool contains(const T& label) const
        {
            return std::find(m_labels.begin(), m_labels.end(), label) != m_labels.end();
        }

        /**
         * Sort the coordinate (ascending).
         */
        void sort()
        {
            std::sort(m_labels.begin(), m_labels.end());
        }

        /**
         * Return the index of the first label not less than the given value.
         * Assumes the coordinate is sorted.
         */
        size_type lower_bound(const T& value) const
        {
            auto it = std::lower_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Return the index of the first label greater than the given value.
         * Assumes sorted.
         */
        size_type upper_bound(const T& value) const
        {
            auto it = std::upper_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Append a label.
         */
        void push_back(const T& label) { m_labels.push_back(label); }
        void push_back(T&& label) { m_labels.push_back(std::move(label)); }

        /**
         * Clear all labels.
         */
        void clear() { m_labels.clear(); }

        /**
         * Expression interface.
         */
        self_type& derived() noexcept { return *this; }
        const self_type& derived() const noexcept { return *this; }

        /**
         * Check equality.
         */
        bool operator==(const self_type& rhs) const { return m_labels == rhs.m_labels; }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

        /**
         * Concatenate two coordinates.
         */
        friend self_type operator+(const self_type& a, const self_type& b)
        {
            self_type result(a.m_labels);
            result.m_labels.insert(result.m_labels.end(), b.m_labels.begin(), b.m_labels.end());
            return result;
        }

    private:
        std::vector<T> m_labels;
    };

    /**
     * Type alias for common coordinate types.
     */
    using string_coordinate = coordinate<std::string>;
    using int_coordinate = coordinate<int>;
    using double_coordinate = coordinate<double>;

    /**
     * Helper to create a coordinate from an initializer list.
     */
    template <class T>
    inline auto make_coordinate(std::initializer_list<T> labels)
    {
        return coordinate<T>(labels);
    }

    /**
     * Helper to create a range coordinate (e.g., 0..n-1).
     */
    inline auto range_coordinate(std::size_t n)
    {
        coordinate<std::size_t> result;
        for (std::size_t i = 0; i < n; ++i) result.push_back(i);
        return result;
    }

} // namespace xframe

#endif // XFRAME_COORDINATE_HPP