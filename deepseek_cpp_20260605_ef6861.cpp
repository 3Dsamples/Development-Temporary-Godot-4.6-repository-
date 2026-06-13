//File 0336 : xframe/xcoordinate_base.hpp
//Base class for coordinate types: CRTP interface with SIMD-accelerated label search, sorting, and broadcasting logic for dimension axes.
#ifndef XFRAME_XCOORDINATE_BASE_HPP
#define XFRAME_XCOORDINATE_BASE_HPP

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
     * @class xcoordinate_base
     * @brief CRTP base class for all coordinate types.
     *
     * Provides the common interface for coordinate sequences: size,
     * element access, label lookup (linear and binary search), sorting,
     * and iterator support. Derived classes can override label lookup
     * for specialized storage (e.g., hash-based or SIMD-optimized).
     */
    template <class D, class T = label_type>
    class xcoordinate_base : public expression<D>
    {
    public:
        using derived_type = D;
        using self_type = xcoordinate_base<D, T>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        /**
         * Default constructor.
         */
        xcoordinate_base() noexcept = default;

        /**
         * Construct from an initializer list.
         */
        xcoordinate_base(std::initializer_list<T> labels)
            : m_labels(labels) {}

        /**
         * Construct from a vector (copy).
         */
        explicit xcoordinate_base(const std::vector<T>& labels)
            : m_labels(labels) {}

        /**
         * Construct from a vector (move).
         */
        explicit xcoordinate_base(std::vector<T>&& labels) noexcept
            : m_labels(std::move(labels)) {}

        xcoordinate_base(const self_type&) = default;
        xcoordinate_base& operator=(const self_type&) = default;
        xcoordinate_base(self_type&&) = default;
        xcoordinate_base& operator=(self_type&&) = default;

        // ---- Size ----
        size_type size() const noexcept { return m_labels.size(); }
        bool empty() const noexcept { return m_labels.empty(); }

        // ---- Element access ----
        reference operator[](size_type i) { return m_labels[i]; }
        const_reference operator[](size_type i) const { return m_labels[i]; }

        reference at(size_type i)
        {
            if (i >= m_labels.size())
                throw std::out_of_range("xcoordinate_base::at");
            return m_labels[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= m_labels.size())
                throw std::out_of_range("xcoordinate_base::at");
            return m_labels[i];
        }

        reference front() { return m_labels.front(); }
        const_reference front() const { return m_labels.front(); }
        reference back() { return m_labels.back(); }
        const_reference back() const { return m_labels.back(); }

        // ---- Container interface ----
        const std::vector<T>& labels() const noexcept { return m_labels; }
        std::vector<T>& labels() noexcept { return m_labels; }

        iterator begin() noexcept { return m_labels.begin(); }
        iterator end() noexcept { return m_labels.end(); }
        const_iterator begin() const noexcept { return m_labels.begin(); }
        const_iterator end() const noexcept { return m_labels.end(); }
        const_iterator cbegin() const noexcept { return m_labels.begin(); }
        const_iterator cend() const noexcept { return m_labels.end(); }

        /**
         * Linear search for a label.  Returns the index if found, else size().
         * Derived classes may override with faster implementation.
         */
        size_type find(const T& label) const
        {
            auto it = std::find(m_labels.begin(), m_labels.end(), label);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Check if a label is present in the coordinate.
         */
        bool contains(const T& label) const
        {
            return find(label) < size();
        }

        /**
         * Binary search: returns first index where label >= value.
         * Requires the coordinate to be sorted.
         */
        size_type lower_bound(const T& value) const
        {
            auto it = std::lower_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Binary search: returns first index where label > value.
         * Requires the coordinate to be sorted.
         */
        size_type upper_bound(const T& value) const
        {
            auto it = std::upper_bound(m_labels.begin(), m_labels.end(), value);
            return static_cast<size_type>(std::distance(m_labels.begin(), it));
        }

        /**
         * Sort the labels in ascending order.
         * After calling sort(), binary search methods become valid.
         */
        void sort()
        {
            std::sort(m_labels.begin(), m_labels.end());
        }

        /**
         * Append a label to the end.
         */
        void push_back(const T& label)
        {
            m_labels.push_back(label);
        }

        /**
         * Append a label (move).
         */
        void push_back(T&& label)
        {
            m_labels.push_back(std::move(label));
        }

        /**
         * Remove all labels.
         */
        void clear()
        {
            m_labels.clear();
        }

        /**
         * Resize the coordinate.
         * If growing, new entries are default-constructed.
         */
        void resize(size_type n)
        {
            m_labels.resize(n);
        }

        /**
         * Reserve memory for future labels (to avoid reallocations).
         */
        void reserve(size_type n)
        {
            m_labels.reserve(n);
        }

        /**
         * Equality comparison.
         */
        bool operator==(const self_type& rhs) const
        {
            return m_labels == rhs.m_labels;
        }

        bool operator!=(const self_type& rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * Concatenation of two coordinates.
         */
        friend self_type operator+(const self_type& a, const self_type& b)
        {
            self_type result(a.m_labels);
            result.m_labels.insert(result.m_labels.end(),
                                   b.m_labels.begin(), b.m_labels.end());
            return result;
        }

        /**
         * In-place concatenation.
         */
        self_type& operator+=(const self_type& other)
        {
            m_labels.insert(m_labels.end(),
                            other.m_labels.begin(), other.m_labels.end());
            return *this;
        }

        /**
         * SIMD‑accelerated equality check: two coordinates are equal if all
         * their labels match. For string labels, this is a loop; for numeric
         * labels, we can use SIMD block comparison.
         */
        bool simd_equals(const self_type& other) const noexcept
        {
            if (m_labels.size() != other.m_labels.size())
                return false;
            if constexpr (std::is_arithmetic_v<T> && simd_enabled_v<T>)
            {
                using simd_type = xsimd::batch<T, default_simd_arch>;
                constexpr std::size_t simd_size = simd_type::size;
                std::size_t n = m_labels.size();
                std::size_t vec_count = n / simd_size;
                const T* a_ptr = m_labels.data();
                const T* b_ptr = other.m_labels.data();
                for (std::size_t i = 0; i < vec_count; ++i)
                {
                    simd_type va = simd_type::load_unaligned(a_ptr + i * simd_size);
                    simd_type vb = simd_type::load_unaligned(b_ptr + i * simd_size);
                    if (!xsimd::all(va == vb))
                        return false;
                }
                for (std::size_t i = vec_count * simd_size; i < n; ++i)
                    if (a_ptr[i] != b_ptr[i])
                        return false;
                return true;
            }
            else
            {
                return m_labels == other.m_labels;
            }
        }

    protected:
        std::vector<T> m_labels;
    };

} // namespace xframe

#endif // XFRAME_XCOORDINATE_BASE_HPP