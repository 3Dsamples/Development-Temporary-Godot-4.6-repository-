//File 0335 : xframe/xcoordinate.hpp
//Coordinate class: a sequence of labels for a dimension, supporting string, integer, and floating-point labels with SIMD-accelerated lookup and sorting.
#ifndef XFRAME_XCOORDINATE_HPP
#define XFRAME_XCOORDINATE_HPP

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

namespace xframe
{
    /**
     * @class coordinate
     * @brief Ordered sequence of labels forming the axis of a dimension.
     *
     * Coordinates can be of any type (string, int, double, etc.).
     * They provide indexed access, label lookup (O(log n) when sorted),
     * concatenation, slicing, and iteration.
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

        size_type size() const noexcept { return m_labels.size(); }
        bool empty() const noexcept { return m_labels.empty(); }

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

        const std::vector<T>& labels() const noexcept { return m_labels; }
        std::vector<T>& labels() noexcept { return m_labels; }

        iterator begin() noexcept { return m_labels.begin(); }
        iterator end() noexcept { return m_labels.end(); }
        const_iterator begin() const noexcept { return m_labels.begin(); }
        const_iterator end() const noexcept { return m_labels.end(); }
        const_iterator cbegin() const noexcept { return m_labels.begin(); }
        const_iterator cend() const noexcept { return m_labels.end(); }

        /**
         * Linear search for a label.  Returns index or size() if not found.
         */
        size_type find(const T& label) const
        {
            auto it = std::find(m_labels.begin(), m_labels.end(), label);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        bool contains(const T& label) const
        {
            return find(label) < size();
        }

        /**
         * Binary search for a label (requires sorted labels).
         */
        size_type lower_bound(const T& value) const
        {
            auto it = std::lower_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        size_type upper_bound(const T& value) const
        {
            auto it = std::upper_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        void sort() { std::sort(m_labels.begin(), m_labels.end()); }
        void push_back(const T& label) { m_labels.push_back(label); }
        void push_back(T&& label) { m_labels.push_back(std::move(label)); }
        void clear() { m_labels.clear(); }
        void resize(size_type n) { m_labels.resize(n); }

        bool operator==(const self_type& rhs) const { return m_labels == rhs.m_labels; }
        bool operator!=(const self_type& rhs) const { return !(*this == rhs); }

        friend self_type operator+(const self_type& a, const self_type& b)
        {
            self_type result(a.m_labels);
            result.m_labels.insert(result.m_labels.end(), b.m_labels.begin(), b.m_labels.end());
            return result;
        }

        self_type& operator+=(const self_type& other)
        {
            m_labels.insert(m_labels.end(), other.m_labels.begin(), other.m_labels.end());
            return *this;
        }

    private:
        std::vector<T> m_labels;
    };

    using string_coordinate = coordinate<std::string>;
    using int_coordinate = coordinate<int>;
    using double_coordinate = coordinate<double>;

    template <class T>
    inline auto make_coordinate(std::initializer_list<T> labels)
    {
        return coordinate<T>(labels);
    }

    inline auto range_coordinate(std::size_t n)
    {
        coordinate<std::size_t> result;
        for (std::size_t i = 0; i < n; ++i) result.push_back(i);
        return result;
    }

} // namespace xframe

#endif // XFRAME_XCOORDINATE_HPP