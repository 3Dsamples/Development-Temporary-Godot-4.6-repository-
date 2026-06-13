/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file sequence_view.h
 * @brief Non‑owning view over a contiguous sequence of elements.
 *
 * This file provides `SequenceView<T>`, a lightweight alternative to `std::span`
 * (or `gsl::span`) that works in C++17 and adds convenience methods for
 * sub‑ranges, slicing, and arithmetic operations on numeric sequences.
 *
 * The view is immutable with respect to ownership but allows mutable access
 * to the elements if the underlying data is mutable. It is designed for
 * zero‑overhead abstraction in performance‑critical code (e.g., octree
 * traversal, batch processing of primitives).
 *
 * Features:
 * - Constant‑time `size()`, `data()`, `empty()`
 * - Random access iterators (contiguous iterator category)
 * - Slicing: `first(n)`, `last(n)`, `subspan(offset, count)`
 * - Element access: `operator[]`, `at()` (bounds‑checked in debug)
 * - Arithmetic operations on numeric views (element‑wise add, sub, scale)
 * - Conversion from `std::vector`, `std::array`, built‑in arrays
 */

#ifndef ORTHOTREE_DETAIL_SEQUENCE_VIEW_H_INCLUDED
#define ORTHOTREE_DETAIL_SEQUENCE_VIEW_H_INCLUDED

#include "../core/build_config.h"
#include "../core/types.h"
#include "common.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <algorithm>
#include <stdexcept>

namespace OrthoTree {
namespace detail {

/**
 * @brief A view over a contiguous sequence of objects of type T.
 *
 * @tparam T Element type (may be const‑qualified).
 */
template <typename T>
class SequenceView {
public:
    using element_type   = T;
    using value_type     = std::remove_cv_t<T>;
    using pointer        = T*;
    using const_pointer  = const T*;
    using reference      = T&;
    using const_reference = const T&;
    using iterator       = T*;
    using const_iterator = const T*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type      = std::size_t;
    using difference_type = std::ptrdiff_t;

    // ------------------------------------------------------------------------
    //  Constructors
    // ------------------------------------------------------------------------
    constexpr SequenceView() noexcept : m_data(nullptr), m_size(0) {}

    constexpr SequenceView(pointer data, size_type size) noexcept
        : m_data(data), m_size(size) {}

    template <size_type N>
    constexpr SequenceView(T (&array)[N]) noexcept : m_data(array), m_size(N) {}

    template <typename Container,
              typename = std::enable_if_t<
                  !std::is_same_v<std::decay_t<Container>, SequenceView> &&
                  std::is_convertible_v<decltype(std::declval<Container>().data()), pointer> &&
                  std::is_convertible_v<decltype(std::declval<Container>().size()), size_type>>>
    constexpr SequenceView(Container& cont) noexcept
        : m_data(cont.data()), m_size(cont.size()) {}

    template <typename Container,
              typename = std::enable_if_t<
                  std::is_convertible_v<decltype(std::declval<const Container>().data()), pointer> &&
                  std::is_convertible_v<decltype(std::declval<const Container>().size()), size_type>>>
    constexpr SequenceView(const Container& cont) noexcept
        : m_data(cont.data()), m_size(cont.size()) {}

    // ------------------------------------------------------------------------
    //  Observers
    // ------------------------------------------------------------------------
    constexpr pointer data() const noexcept { return m_data; }
    constexpr size_type size() const noexcept { return m_size; }
    constexpr bool empty() const noexcept { return m_size == 0; }

    // ------------------------------------------------------------------------
    //  Element access
    // ------------------------------------------------------------------------
    constexpr reference operator[](size_type idx) const noexcept {
        ORTHOTREE_ASSERT(idx < m_size);
        return m_data[idx];
    }

    constexpr reference at(size_type idx) const {
        if (idx >= m_size) throw std::out_of_range("SequenceView::at");
        return m_data[idx];
    }

    constexpr reference front() const noexcept {
        ORTHOTREE_ASSERT(!empty());
        return m_data[0];
    }

    constexpr reference back() const noexcept {
        ORTHOTREE_ASSERT(!empty());
        return m_data[m_size - 1];
    }

    // ------------------------------------------------------------------------
    //  Subviews
    // ------------------------------------------------------------------------
    constexpr SequenceView first(size_type count) const noexcept {
        ORTHOTREE_ASSERT(count <= m_size);
        return SequenceView(m_data, count);
    }

    constexpr SequenceView last(size_type count) const noexcept {
        ORTHOTREE_ASSERT(count <= m_size);
        return SequenceView(m_data + m_size - count, count);
    }

    constexpr SequenceView subspan(size_type offset, size_type count = static_cast<size_type>(-1)) const noexcept {
        ORTHOTREE_ASSERT(offset <= m_size);
        if (count == static_cast<size_type>(-1) || offset + count > m_size)
            count = m_size - offset;
        return SequenceView(m_data + offset, count);
    }

    // ------------------------------------------------------------------------
    //  Iterators
    // ------------------------------------------------------------------------
    constexpr iterator begin() const noexcept { return m_data; }
    constexpr iterator end() const noexcept { return m_data + m_size; }
    constexpr const_iterator cbegin() const noexcept { return m_data; }
    constexpr const_iterator cend() const noexcept { return m_data + m_size; }
    constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

    // ------------------------------------------------------------------------
    //  Comparison
    // ------------------------------------------------------------------------
    friend constexpr bool operator==(const SequenceView& a, const SequenceView& b) noexcept {
        if (a.size() != b.size()) return false;
        return std::equal(a.begin(), a.end(), b.begin());
    }
    friend constexpr bool operator!=(const SequenceView& a, const SequenceView& b) noexcept {
        return !(a == b);
    }

    // ------------------------------------------------------------------------
    //  Arithmetic operations (for numeric T)
    // ------------------------------------------------------------------------
    template <typename U = T,
              typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    SequenceView& add(const SequenceView& other) noexcept {
        ORTHOTREE_ASSERT(m_size == other.m_size);
        for (size_type i = 0; i < m_size; ++i) m_data[i] += other.m_data[i];
        return *this;
    }

    template <typename U = T,
              typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    SequenceView& sub(const SequenceView& other) noexcept {
        ORTHOTREE_ASSERT(m_size == other.m_size);
        for (size_type i = 0; i < m_size; ++i) m_data[i] -= other.m_data[i];
        return *this;
    }

    template <typename U = T,
              typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    SequenceView& scale(U scalar) noexcept {
        for (size_type i = 0; i < m_size; ++i) m_data[i] *= scalar;
        return *this;
    }

    template <typename U = T,
              typename = std::enable_if_t<std::is_arithmetic_v<U>>>
    SequenceView& normalize() noexcept {
        U sum_sq = U(0);
        for (size_type i = 0; i < m_size; ++i) sum_sq += m_data[i] * m_data[i];
        if (sum_sq > U(0)) {
            U inv_norm = U(1) / std::sqrt(sum_sq);
            scale(inv_norm);
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    //  Conversion to contiguous container (explicit)
    // ------------------------------------------------------------------------
    template <template <typename...> class Container = std::vector>
    Container<value_type> to_container() const {
        return Container<value_type>(begin(), end());
    }

private:
    pointer m_data;
    size_type m_size;
};

// ----------------------------------------------------------------------------
//  Deduction guides
// ----------------------------------------------------------------------------
template <typename T, size_t N>
SequenceView(T (&)[N]) -> SequenceView<T>;

template <typename Container>
SequenceView(Container&) -> SequenceView<typename Container::value_type>;

template <typename Container>
SequenceView(const Container&) -> SequenceView<const typename Container::value_type>;

// ----------------------------------------------------------------------------
//  Alias for const view
// ----------------------------------------------------------------------------
template <typename T>
using ConstSequenceView = SequenceView<const T>;

} // namespace detail
} // namespace OrthoTree

#endif // ORTHOTREE_DETAIL_SEQUENCE_VIEW_H_INCLUDED