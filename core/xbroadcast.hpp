// core/xbroadcast.hpp
#ifndef XTENSOR_XBROADCAST_HPP
#define XTENSOR_XBROADCAST_HPP

// ----------------------------------------------------------------------------
// xbroadcast.hpp – Broadcasting view and expression
// ----------------------------------------------------------------------------
// This header defines the xbroadcast class, which broadcasts an expression
// to a specified shape. It provides a non‑owning view that repeats the
// underlying data along dimensions of size 1 according to NumPy broadcasting
// semantics. Fully compatible with BigNumber value type and FFT acceleration.
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace detail
    {
        // --------------------------------------------------------------------
        // Check if a shape can be broadcast to a target shape
        // --------------------------------------------------------------------
        bool can_broadcast_to(const shape_type& from, const shape_type& to);

        // --------------------------------------------------------------------
        // Compute broadcast strides for the broadcasted view
        // --------------------------------------------------------------------
        strides_type compute_broadcast_strides(const shape_type& from_shape,
                                               const strides_type& from_strides,
                                               const shape_type& to_shape);

        // --------------------------------------------------------------------
        // Compute broadcasted size
        // --------------------------------------------------------------------
        size_type compute_broadcast_size(const shape_type& shape) noexcept;

    } // namespace detail

    // ========================================================================
    // xbroadcast – Broadcast an expression to a new shape
    // ========================================================================
    template <class CT>
    class xbroadcast : public xview_expression<xbroadcast<CT>>
    {
    public:
        using self_type = xbroadcast<CT>;
        using base_type = xview_expression<self_type>;
        using value_type = typename CT::value_type;
        using reference = typename CT::reference;
        using const_reference = typename CT::const_reference;
        using pointer = typename CT::pointer;
        using const_pointer = typename CT::const_pointer;
        using size_type = xt::size_type;
        using difference_type = xt::difference_type;
        using shape_type = xt::shape_type;
        using strides_type = xt::strides_type;

        using inner_types = xcontainer_inner_types<self_type>;
        using storage_type = typename inner_types::storage_type;
        using temporary_type = typename inner_types::temporary_type;

        static constexpr layout_type layout = CT::layout;

        // --------------------------------------------------------------------
        // Constructors
        // --------------------------------------------------------------------
        template <class CTA>
        xbroadcast(CTA&& e, const shape_type& target_shape);

        // --------------------------------------------------------------------
        // Copy / move
        // --------------------------------------------------------------------
        xbroadcast(const self_type&) = default;
        xbroadcast(self_type&&) = default;

        self_type& operator=(const self_type&) = delete;
        self_type& operator=(self_type&&) = delete;

        // --------------------------------------------------------------------
        // Shape and size
        // --------------------------------------------------------------------
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        size_type size() const noexcept;
        size_type dimension() const noexcept;
        bool empty() const noexcept;

        // --------------------------------------------------------------------
        // Access to underlying expression
        // --------------------------------------------------------------------
        const CT& expression() const noexcept;

        // --------------------------------------------------------------------
        // Element access (flat index)
        // --------------------------------------------------------------------
        const_reference operator[](size_type i) const;
        const_reference flat(size_type i) const;

        // --------------------------------------------------------------------
        // Element access (multi‑dimensional)
        // --------------------------------------------------------------------
        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class S>
        const_reference element(const S& indices) const;

        // --------------------------------------------------------------------
        // Assignment from expressions (if underlying is writable)
        // --------------------------------------------------------------------
        template <class E>
        self_type& operator=(const xexpression<E>& e);
        template <class E>
        self_type& assign(const xexpression<E>& e);
        template <class E>
        self_type& operator+=(const xexpression<E>& e);
        template <class E>
        self_type& operator-=(const xexpression<E>& e);
        template <class E>
        self_type& operator*=(const xexpression<E>& e);
        template <class E>
        self_type& operator/=(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Iterator support (read‑only)
        // --------------------------------------------------------------------
        class const_iterator
        {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type = self_type::value_type;
            using difference_type = std::ptrdiff_t;
            using pointer = const value_type*;
            using reference = value_type;

            const_iterator(const self_type* broadcast, size_type index);
            reference operator*() const;
            const_iterator& operator++();
            const_iterator operator++(int);
            const_iterator& operator--();
            const_iterator operator--(int);
            const_iterator& operator+=(difference_type n);
            const_iterator& operator-=(difference_type n);
            const_iterator operator+(difference_type n) const;
            const_iterator operator-(difference_type n) const;
            difference_type operator-(const const_iterator& other) const;
            bool operator==(const const_iterator& other) const;
            bool operator!=(const const_iterator& other) const;
            bool operator<(const const_iterator& other) const;
            bool operator>(const const_iterator& other) const;
            bool operator<=(const const_iterator& other) const;
            bool operator>=(const const_iterator& other) const;

        private:
            const self_type* m_broadcast;
            size_type m_index;
        };

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

    private:
        CT m_expression;
        shape_type m_shape;
        strides_type m_strides;
        size_type m_size;

        size_type map_flat_index(size_type flat_index) const;
        template <class S>
        size_type map_indices_to_flat(const S& indices) const;
        template <class E>
        void broadcast_assign(const E& expr);
        template <class E>
        void broadcast_assign_recursive(const E& expr, std::vector<size_type>& indices,
                                        size_type& flat_index, size_type dim);
        template <class E, class Op>
        self_type& assign_composite(const xexpression<E>& e, Op&& op);
        template <class E, class Op>
        void assign_composite_recursive(const E& expr, std::vector<size_type>& indices,
                                        size_type& flat_index, size_type dim, Op&& op);
    };

    // ========================================================================
    // Factory functions for broadcasting
    // ========================================================================
    template <class E>
    inline auto broadcast(const xexpression<E>& e, const shape_type& target_shape);

    template <class E>
    inline auto broadcast(E&& e, const shape_type& target_shape);

    template <class E1, class E2>
    inline auto broadcast_to(const xexpression<E1>& e, const xexpression<E2>& reference);

    template <class E1, class E2>
    inline auto broadcast_arrays(const xexpression<E1>& e1, const xexpression<E2>& e2);

    // ========================================================================
    // BigNumber‑specific broadcast optimizations
    // ========================================================================
    template <class E, XTL_REQUIRES(is_bignumber_expression<E>::value)>
    inline auto broadcast_bignumber(const xexpression<E>& e, const shape_type& target_shape);

    // ------------------------------------------------------------------------
    // Inner types specialization for xbroadcast
    // ------------------------------------------------------------------------
    template <class CT>
    struct xcontainer_inner_types<xbroadcast<CT>>
    {
        using temporary_type = xarray_container<typename CT::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (empty with TODO comments)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace detail
    {
        inline bool can_broadcast_to(const shape_type& from, const shape_type& to)
        { /* TODO: implement */ return true; }
        inline strides_type compute_broadcast_strides(const shape_type& from_shape,
                                                      const strides_type& from_strides,
                                                      const shape_type& to_shape)
        { /* TODO: implement */ return from_strides; }
        inline size_type compute_broadcast_size(const shape_type& shape) noexcept
        { size_type s = 1; for (auto d : shape) s *= d; return s; }
    }

    // xbroadcast member functions
    template <class CT> template <class CTA>
    xbroadcast<CT>::xbroadcast(CTA&& e, const shape_type& target_shape)
        : m_expression(std::forward<CTA>(e)), m_shape(target_shape),
          m_strides(detail::compute_broadcast_strides(m_expression.shape(), m_expression.strides(), m_shape)),
          m_size(detail::compute_broadcast_size(m_shape))
    { /* TODO: implement validation */ }

    template <class CT> inline auto xbroadcast<CT>::shape() const noexcept -> const shape_type& { return m_shape; }
    template <class CT> inline auto xbroadcast<CT>::strides() const noexcept -> const strides_type& { return m_strides; }
    template <class CT> inline auto xbroadcast<CT>::size() const noexcept -> size_type { return m_size; }
    template <class CT> inline auto xbroadcast<CT>::dimension() const noexcept -> size_type { return m_shape.size(); }
    template <class CT> inline bool xbroadcast<CT>::empty() const noexcept { return m_size == 0; }
    template <class CT> inline auto xbroadcast<CT>::expression() const noexcept -> const CT& { return m_expression; }

    template <class CT> inline auto xbroadcast<CT>::operator[](size_type i) const -> const_reference
    { return flat(i); }
    template <class CT> inline auto xbroadcast<CT>::flat(size_type i) const -> const_reference
    { size_type expr_flat = map_flat_index(i); return m_expression.flat(expr_flat); }

    template <class CT> template <class... Args>
    inline auto xbroadcast<CT>::operator()(Args... args) const -> const_reference
    { std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...}; return element(indices); }
    template <class CT> template <class S>
    inline auto xbroadcast<CT>::element(const S& indices) const -> const_reference
    { size_type expr_flat = map_indices_to_flat(indices); return m_expression.flat(expr_flat); }

    template <class CT> template <class E>
    inline auto xbroadcast<CT>::operator=(const xexpression<E>& e) -> self_type& { return assign(e); }
    template <class CT> template <class E>
    inline auto xbroadcast<CT>::assign(const xexpression<E>& e) -> self_type& { /* TODO: implement */ return *this; }
    template <class CT> template <class E>
    inline auto xbroadcast<CT>::operator+=(const xexpression<E>& e) -> self_type& { return assign_composite(e, detail::plus()); }
    template <class CT> template <class E>
    inline auto xbroadcast<CT>::operator-=(const xexpression<E>& e) -> self_type& { return assign_composite(e, detail::minus()); }
    template <class CT> template <class E>
    inline auto xbroadcast<CT>::operator*=(const xexpression<E>& e) -> self_type&
    { if constexpr (use_fft_multiplication<CT, E>::value) return assign_composite(e, detail::bignumber_fft_multiply()); else return assign_composite(e, detail::multiplies()); }
    template <class CT> template <class E>
    inline auto xbroadcast<CT>::operator/=(const xexpression<E>& e) -> self_type& { return assign_composite(e, detail::divides()); }

    // Iterator
    template <class CT> inline xbroadcast<CT>::const_iterator::const_iterator(const self_type* broadcast, size_type index) : m_broadcast(broadcast), m_index(index) {}
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator*() const -> reference { return (*m_broadcast)[m_index]; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator++() -> const_iterator& { ++m_index; return *this; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator++(int) -> const_iterator { const_iterator tmp = *this; ++m_index; return tmp; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator--() -> const_iterator& { --m_index; return *this; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator--(int) -> const_iterator { const_iterator tmp = *this; --m_index; return tmp; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator+=(difference_type n) -> const_iterator& { m_index += n; return *this; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator-=(difference_type n) -> const_iterator& { m_index -= n; return *this; }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator+(difference_type n) const -> const_iterator { return const_iterator(m_broadcast, m_index + n); }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator-(difference_type n) const -> const_iterator { return const_iterator(m_broadcast, m_index - n); }
    template <class CT> inline auto xbroadcast<CT>::const_iterator::operator-(const const_iterator& other) const -> difference_type { return static_cast<difference_type>(m_index - other.m_index); }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator==(const const_iterator& other) const { return m_broadcast == other.m_broadcast && m_index == other.m_index; }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator!=(const const_iterator& other) const { return !(*this == other); }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator<(const const_iterator& other) const { return m_index < other.m_index; }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator>(const const_iterator& other) const { return m_index > other.m_index; }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator<=(const const_iterator& other) const { return m_index <= other.m_index; }
    template <class CT> inline bool xbroadcast<CT>::const_iterator::operator>=(const const_iterator& other) const { return m_index >= other.m_index; }

    template <class CT> inline auto xbroadcast<CT>::begin() const -> const_iterator { return const_iterator(this, 0); }
    template <class CT> inline auto xbroadcast<CT>::end() const -> const_iterator { return const_iterator(this, m_size); }
    template <class CT> inline auto xbroadcast<CT>::cbegin() const -> const_iterator { return begin(); }
    template <class CT> inline auto xbroadcast<CT>::cend() const -> const_iterator { return end(); }

    // Private helpers
    template <class CT> inline auto xbroadcast<CT>::map_flat_index(size_type flat_index) const -> size_type
    { /* TODO: implement */ return flat_index; }
    template <class CT> template <class S>
    inline auto xbroadcast<CT>::map_indices_to_flat(const S& indices) const -> size_type
    { /* TODO: implement */ return 0; }
    template <class CT> template <class E>
    inline void xbroadcast<CT>::broadcast_assign(const E& expr) { /* TODO: implement */ }
    template <class CT> template <class E>
    inline void xbroadcast<CT>::broadcast_assign_recursive(const E& expr, std::vector<size_type>& indices,
                                                           size_type& flat_index, size_type dim)
    { /* TODO: implement */ }
    template <class CT> template <class E, class Op>
    inline auto xbroadcast<CT>::assign_composite(const xexpression<E>& e, Op&& op) -> self_type&
    { /* TODO: implement */ return *this; }
    template <class CT> template <class E, class Op>
    inline void xbroadcast<CT>::assign_composite_recursive(const E& expr, std::vector<size_type>& indices,
                                                           size_type& flat_index, size_type dim, Op&& op)
    { /* TODO: implement */ }

    // Factory functions
    template <class E> inline auto broadcast(const xexpression<E>& e, const shape_type& target_shape)
    { return xbroadcast<const E&>(e.derived_cast(), target_shape); }
    template <class E> inline auto broadcast(E&& e, const shape_type& target_shape)
    { using expr_type = std::decay_t<E>; return xbroadcast<expr_type>(std::forward<E>(e), target_shape); }
    template <class E1, class E2> inline auto broadcast_to(const xexpression<E1>& e, const xexpression<E2>& reference)
    { return broadcast(e.derived_cast(), reference.derived_cast().shape()); }
    template <class E1, class E2> inline auto broadcast_arrays(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { shape_type common_shape = detail::broadcast_shapes(e1.derived_cast().shape(), e2.derived_cast().shape()); return std::make_pair(broadcast(e1.derived_cast(), common_shape), broadcast(e2.derived_cast(), common_shape)); }
    template <class E, XTL_REQUIRES(is_bignumber_expression<E>::value)>
    inline auto broadcast_bignumber(const xexpression<E>& e, const shape_type& target_shape)
    { return broadcast(e.derived_cast(), target_shape); }

} // namespace xt

#endif // XTENSOR_XBROADCAST_HPPther's shape
    template <class E1, class E2>
    inline auto broadcast_to(const xexpression<E1>& e, const xexpression<E2>& reference)
    {
        return broadcast(e.derived_cast(), reference.derived_cast().shape());
    }

    // ------------------------------------------------------------------------
    // Helper to automatically broadcast two expressions to a common shape
    // ------------------------------------------------------------------------
    template <class E1, class E2>
    inline auto broadcast_arrays(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        shape_type common_shape = detail::broadcast_shapes(e1.derived_cast().shape(),
                                                           e2.derived_cast().shape());
        return std::make_pair(
            broadcast(e1.derived_cast(), common_shape),
            broadcast(e2.derived_cast(), common_shape)
        );
    }

    // ========================================================================
    // BigNumber‑specific broadcast optimizations
    // ========================================================================
    // When broadcasting BigNumber arrays, we can optionally pre‑compute
    // FFT plans for repeated operations. These functions are provided
    // for advanced use cases.

    template <class E,
              XTL_REQUIRES(is_bignumber_expression<E>::value)>
    inline auto broadcast_bignumber(const xexpression<E>& e, const shape_type& target_shape)
    {
        // Same as regular broadcast, but may be specialized later to cache
        // FFT plans for the broadcasted shape.
        return broadcast(e.derived_cast(), target_shape);
    }

    // ------------------------------------------------------------------------
    // Inner types specialization for xbroadcast
    // ------------------------------------------------------------------------
    template <class CT>
    struct xcontainer_inner_types<xbroadcast<CT>>
    {
        using temporary_type = xarray_container<typename CT::value_type>;
        using storage_type = typename temporary_type::storage_type;
    };

} // namespace xt

#endif // XTENSOR_XBROADCAST_HPP
