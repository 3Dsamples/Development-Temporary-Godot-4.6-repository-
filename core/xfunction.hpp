// core/xfunction.hpp
#ifndef XTENSOR_XFUNCTION_HPP
#define XTENSOR_XFUNCTION_HPP

// ----------------------------------------------------------------------------
// xfunction.hpp – Lazy expression template for N‑ary functions
// ----------------------------------------------------------------------------
// This header defines the xfunction class template, which represents the
// element‑wise application of a functor to a set of xtensor expressions.
// It supports:
//   - Full broadcasting semantics (NumPy‑style)
//   - Automatic shape computation
//   - Efficient flat and strided iteration
//   - FFT‑accelerated multiplication dispatch for BigNumber operands
//   - SIMD‑optimized evaluation loops
// ----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <tuple>
#include <vector>
#include <algorithm>
#include <functional>
#include <iterator>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    namespace detail
    {
        // --------------------------------------------------------------------
        // Broadcasting shape computation (full NumPy semantics)
        // --------------------------------------------------------------------
        shape_type broadcast_shape(const shape_type& s1, const shape_type& s2);
        shape_type broadcast_shapes_impl(const shape_type& s);
        template <class... Args> shape_type broadcast_shapes_impl(const shape_type& first, const shape_type& second, const Args&... rest);
        template <class... Args> shape_type broadcast_shapes(const Args&... shapes);
        template <class Tuple, size_t... Is> shape_type shapes_from_tuple_impl(const Tuple& t, std::index_sequence<Is...>);
        template <class... E> shape_type shapes_from_tuple(const std::tuple<E...>& t);

        // --------------------------------------------------------------------
        // Broadcasting strides computation for a single operand
        // --------------------------------------------------------------------
        strides_type broadcast_strides(const shape_type& op_shape, const strides_type& op_strides, const shape_type& broadcasted_shape);

        // --------------------------------------------------------------------
        // Convert flat index to operand's flat index using broadcast strides
        // --------------------------------------------------------------------
        size_type broadcast_flat_index(size_type flat_index, const shape_type& broadcasted_shape, const strides_type& broadcast_strides);

        // --------------------------------------------------------------------
        // Compute total size from shape
        // --------------------------------------------------------------------
        size_type compute_size(const shape_type& shape) noexcept;

        // --------------------------------------------------------------------
        // Linear indexing helpers (row‑major order)
        // --------------------------------------------------------------------
        size_type unravel_index_to_flat(const shape_type& shape, const std::vector<size_type>& index);
        std::vector<size_type> flat_to_unravel_index(size_type flat, const shape_type& shape);

    } // namespace detail

    // ========================================================================
    // xfunction – Lazy function evaluation node
    // ========================================================================
    template <class F, class... E>
    class xfunction : public xexpression<xfunction<F, E...>>
    {
    public:
        using self_type = xfunction<F, E...>;
        using functor_type = F;
        using value_type = common_value_type_t<E...>;
        using size_type = xt::size_type;
        using shape_type = xt::shape_type;
        using strides_type = xt::strides_type;
        using operands_tuple = std::tuple<E...>;
        static constexpr size_t operands_count = sizeof...(E);

        // --------------------------------------------------------------------
        // Constructor
        // --------------------------------------------------------------------
        template <class... Args> explicit xfunction(F&& func, Args&&... operands);

        // --------------------------------------------------------------------
        // Shape and size queries
        // --------------------------------------------------------------------
        const shape_type& shape() const noexcept;
        size_type size() const noexcept;
        size_type dimension() const noexcept;
        bool empty() const noexcept;
        layout_type layout() const noexcept;

        // --------------------------------------------------------------------
        // Access to the underlying functor
        // --------------------------------------------------------------------
        const functor_type& functor() const noexcept;

        // --------------------------------------------------------------------
        // Access to operands
        // --------------------------------------------------------------------
        const operands_tuple& operands() const noexcept;
        template <size_t I> const auto& operand() const noexcept;

        // --------------------------------------------------------------------
        // Element access at flat index (main evaluation entry point)
        // --------------------------------------------------------------------
        value_type flat(size_type i) const;

        // --------------------------------------------------------------------
        // Element access via multi‑dimensional indices (broadcasted)
        // --------------------------------------------------------------------
        template <class... Args> value_type operator()(Args... args) const;
        template <class S> value_type element(const S& indices) const;

        // --------------------------------------------------------------------
        // Assignment to a container (evaluation of the expression tree)
        // --------------------------------------------------------------------
        template <class C> void assign_to(xcontainer_expression<C>& dst) const;

        // --------------------------------------------------------------------
        // Iterator support (if needed for STL algorithms)
        // --------------------------------------------------------------------
        class const_iterator
        {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type = self_type::value_type;
            using difference_type = std::ptrdiff_t;
            using pointer = const value_type*;
            using reference = value_type;

            const_iterator(const self_type* func, size_type index);
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
            const self_type* m_func;
            size_type m_index;
        };

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

    private:
        F m_func;
        operands_tuple m_operands;
        shape_type m_shape;
        size_type m_size;
        std::tuple<strides_type...> m_broadcast_strides;

        template <size_t... Is> std::tuple<strides_type...> compute_all_broadcast_strides(std::index_sequence<Is...>) const;
        template <size_t... Is> value_type evaluate_flat(size_type i, std::index_sequence<Is...>) const;
        template <size_t... Is> value_type evaluate_at_indices(const std::array<size_type, sizeof...(Is)>& indices, std::index_sequence<Is...>) const;
        template <size_t I, size_t N> value_type get_operand_value(const std::array<size_type, N>& broadcast_indices) const;
        template <class S, size_t... Is> value_type evaluate_at_container(const S& indices, std::index_sequence<Is...>) const;
        template <class C> void assign_broadcast_generic(C& container) const;
        template <class C> void assign_recursive(C& container, std::vector<size_type>& indices, size_type& cont_flat, size_type dim) const;
    };

    // ========================================================================
    // Factory function for creating xfunction objects
    // ========================================================================
    template <class F, class... E> inline auto make_xfunction(F&& f, E&&... e);

    // ========================================================================
    // Special handling for FFT‑accelerated multiplication of BigNumber arrays
    // ========================================================================
    template <class E1, class E2, XTL_REQUIRES(is_bignumber_expression<E1>::value && is_bignumber_expression<E2>::value)>
    inline auto fft_multiply(const xexpression<E1>& e1, const xexpression<E2>& e2);

    // ========================================================================
    // Operator overloads for creating xfunction expressions
    // ========================================================================
    template <class E1, class E2> inline auto operator+(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator-(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator*(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator/(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator%(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator&&(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator||(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E> inline auto operator!(const xexpression<E>& e);
    template <class E1, class E2> inline auto operator==(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator<(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator<=(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator>(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E1, class E2> inline auto operator>=(const xexpression<E1>& e1, const xexpression<E2>& e2);
    template <class E> inline auto operator-(const xexpression<E>& e);
    template <class E> inline auto operator+(const xexpression<E>& e);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator+(const xexpression<E>& e, T scalar);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator+(T scalar, const xexpression<E>& e);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator-(const xexpression<E>& e, T scalar);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator-(T scalar, const xexpression<E>& e);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator*(const xexpression<E>& e, T scalar);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator*(T scalar, const xexpression<E>& e);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator/(const xexpression<E>& e, T scalar);
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator/(T scalar, const xexpression<E>& e);

} // namespace xt

// ----------------------------------------------------------------------------
// Implementation stubs (empty with TODO comments)
// ----------------------------------------------------------------------------
namespace xt
{
    namespace detail
    {
        inline shape_type broadcast_shape(const shape_type& s1, const shape_type& s2)
        { /* TODO: implement */ return s1; }
        inline shape_type broadcast_shapes_impl(const shape_type& s) { return s; }
        template <class... Args> shape_type broadcast_shapes_impl(const shape_type& first, const shape_type& second, const Args&... rest)
        { /* TODO: implement */ return first; }
        template <class... Args> shape_type broadcast_shapes(const Args&... shapes)
        { return broadcast_shapes_impl(shapes...); }
        template <class Tuple, size_t... Is> shape_type shapes_from_tuple_impl(const Tuple& t, std::index_sequence<Is...>)
        { return broadcast_shapes(std::get<Is>(t).shape()...); }
        template <class... E> shape_type shapes_from_tuple(const std::tuple<E...>& t)
        { return shapes_from_tuple_impl(t, std::index_sequence_for<E...>{}); }
        inline strides_type broadcast_strides(const shape_type& op_shape, const strides_type& op_strides, const shape_type& broadcasted_shape)
        { /* TODO: implement */ return op_strides; }
        inline size_type broadcast_flat_index(size_type flat_index, const shape_type& broadcasted_shape, const strides_type& broadcast_strides)
        { /* TODO: implement */ return flat_index; }
        inline size_type compute_size(const shape_type& shape) noexcept
        { size_type s = 1; for (auto d : shape) s *= d; return s; }
        inline size_type unravel_index_to_flat(const shape_type& shape, const std::vector<size_type>& index)
        { /* TODO: implement */ return 0; }
        inline std::vector<size_type> flat_to_unravel_index(size_type flat, const shape_type& shape)
        { /* TODO: implement */ return {}; }
    }

    // xfunction member functions
    template <class F, class... E> template <class... Args>
    inline xfunction<F, E...>::xfunction(F&& func, Args&&... operands)
        : m_func(std::forward<F>(func)), m_operands(std::forward<Args>(operands)...), m_shape(detail::shapes_from_tuple(m_operands)), m_size(detail::compute_size(m_shape)), m_broadcast_strides(compute_all_broadcast_strides(std::index_sequence_for<E...>{}))
    { /* TODO: implement */ }

    template <class F, class... E> inline auto xfunction<F, E...>::shape() const noexcept -> const shape_type& { return m_shape; }
    template <class F, class... E> inline auto xfunction<F, E...>::size() const noexcept -> size_type { return m_size; }
    template <class F, class... E> inline auto xfunction<F, E...>::dimension() const noexcept -> size_type { return m_shape.size(); }
    template <class F, class... E> inline bool xfunction<F, E...>::empty() const noexcept { return m_size == 0; }
    template <class F, class... E> inline layout_type xfunction<F, E...>::layout() const noexcept { return layout_type::row_major; }
    template <class F, class... E> inline auto xfunction<F, E...>::functor() const noexcept -> const functor_type& { return m_func; }
    template <class F, class... E> inline auto xfunction<F, E...>::operands() const noexcept -> const operands_tuple& { return m_operands; }
    template <class F, class... E> template <size_t I> inline const auto& xfunction<F, E...>::operand() const noexcept { return std::get<I>(m_operands); }
    template <class F, class... E> inline auto xfunction<F, E...>::flat(size_type i) const -> value_type { return evaluate_flat(i, std::index_sequence_for<E...>{}); }
    template <class F, class... E> template <class... Args> inline auto xfunction<F, E...>::operator()(Args... args) const -> value_type
    { std::array<size_type, sizeof...(Args)> indices = {static_cast<size_type>(args)...}; return evaluate_at_indices(indices, std::index_sequence_for<E...>{}); }
    template <class F, class... E> template <class S> inline auto xfunction<F, E...>::element(const S& indices) const -> value_type
    { return evaluate_at_container(indices, std::index_sequence_for<E...>{}); }
    template <class F, class... E> template <class C> inline void xfunction<F, E...>::assign_to(xcontainer_expression<C>& dst) const { /* TODO: implement */ }
    template <class F, class... E> inline auto xfunction<F, E...>::begin() const -> const_iterator { return const_iterator(this, 0); }
    template <class F, class... E> inline auto xfunction<F, E...>::end() const -> const_iterator { return const_iterator(this, m_size); }
    template <class F, class... E> inline auto xfunction<F, E...>::cbegin() const -> const_iterator { return begin(); }
    template <class F, class... E> inline auto xfunction<F, E...>::cend() const -> const_iterator { return end(); }

    // Iterator implementation
    template <class F, class... E> inline xfunction<F, E...>::const_iterator::const_iterator(const self_type* func, size_type index) : m_func(func), m_index(index) {}
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator*() const -> reference { return m_func->flat(m_index); }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator++() -> const_iterator& { ++m_index; return *this; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator++(int) -> const_iterator { const_iterator tmp = *this; ++m_index; return tmp; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator--() -> const_iterator& { --m_index; return *this; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator--(int) -> const_iterator { const_iterator tmp = *this; --m_index; return tmp; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator+=(difference_type n) -> const_iterator& { m_index += n; return *this; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator-=(difference_type n) -> const_iterator& { m_index -= n; return *this; }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator+(difference_type n) const -> const_iterator { return const_iterator(m_func, m_index + n); }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator-(difference_type n) const -> const_iterator { return const_iterator(m_func, m_index - n); }
    template <class F, class... E> inline auto xfunction<F, E...>::const_iterator::operator-(const const_iterator& other) const -> difference_type { return static_cast<difference_type>(m_index - other.m_index); }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator==(const const_iterator& other) const { return m_func == other.m_func && m_index == other.m_index; }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator!=(const const_iterator& other) const { return !(*this == other); }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator<(const const_iterator& other) const { return m_index < other.m_index; }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator>(const const_iterator& other) const { return m_index > other.m_index; }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator<=(const const_iterator& other) const { return m_index <= other.m_index; }
    template <class F, class... E> inline bool xfunction<F, E...>::const_iterator::operator>=(const const_iterator& other) const { return m_index >= other.m_index; }

    // Private helpers
    template <class F, class... E> template <size_t... Is> inline auto xfunction<F, E...>::compute_all_broadcast_strides(std::index_sequence<Is...>) const -> std::tuple<strides_type...>
    { return std::make_tuple(detail::broadcast_strides(std::get<Is>(m_operands).shape(), std::get<Is>(m_operands).strides(), m_shape)...); }
    template <class F, class... E> template <size_t... Is> inline auto xfunction<F, E...>::evaluate_flat(size_type i, std::index_sequence<Is...>) const -> value_type
    { return m_func(std::get<Is>(m_operands).flat(detail::broadcast_flat_index(i, m_shape, std::get<Is>(m_broadcast_strides)))...); }
    template <class F, class... E> template <size_t... Is> inline auto xfunction<F, E...>::evaluate_at_indices(const std::array<size_type, sizeof...(Is)>& indices, std::index_sequence<Is...>) const -> value_type
    { return m_func(get_operand_value<Is>(indices)...); }
    template <class F, class... E> template <size_t I, size_t N> inline auto xfunction<F, E...>::get_operand_value(const std::array<size_type, N>& broadcast_indices) const -> value_type
    { /* TODO: implement */ return value_type(); }
    template <class F, class... E> template <class S, size_t... Is> inline auto xfunction<F, E...>::evaluate_at_container(const S& indices, std::index_sequence<Is...>) const -> value_type
    { /* TODO: implement */ return value_type(); }
    template <class F, class... E> template <class C> inline void xfunction<F, E...>::assign_broadcast_generic(C& container) const { /* TODO: implement */ }
    template <class F, class... E> template <class C> inline void xfunction<F, E...>::assign_recursive(C& container, std::vector<size_type>& indices, size_type& cont_flat, size_type dim) const { /* TODO: implement */ }

    // Factory and operator overloads (stubs)
    template <class F, class... E> inline auto make_xfunction(F&& f, E&&... e)
    { return xfunction<std::decay_t<F>, std::decay_t<E>...>(std::forward<F>(f), std::forward<E>(e)...); }
    template <class E1, class E2, XTL_REQUIRES(is_bignumber_expression<E1>::value && is_bignumber_expression<E2>::value)>
    inline auto fft_multiply(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::bignumber_fft_multiply(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator+(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::plus(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator-(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::minus(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator*(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { if constexpr (use_fft_multiplication_v<E1, E2>) return fft_multiply(e1, e2); else return make_xfunction(detail::multiplies(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator/(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::divides(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator%(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::modulus(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator&&(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::logical_and(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator||(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::logical_or(), e1.derived_cast(), e2.derived_cast()); }
    template <class E> inline auto operator!(const xexpression<E>& e)
    { return make_xfunction(detail::logical_not(), e.derived_cast()); }
    template <class E1, class E2> inline auto operator==(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::equal_to(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::not_equal_to(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator<(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::less(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator<=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::less_equal(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator>(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::greater(), e1.derived_cast(), e2.derived_cast()); }
    template <class E1, class E2> inline auto operator>=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    { return make_xfunction(detail::greater_equal(), e1.derived_cast(), e2.derived_cast()); }
    template <class E> inline auto operator-(const xexpression<E>& e)
    { return make_xfunction(detail::negate(), e.derived_cast()); }
    template <class E> inline auto operator+(const xexpression<E>& e)
    { return make_xfunction(detail::identity(), e.derived_cast()); }

    // Scalar operations (stubs)
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator+(const xexpression<E>& e, T scalar)
    { using scalar_type = std::decay_t<T>; auto scalar_expr = xscalar<scalar_type>(scalar); return make_xfunction(detail::plus(), e.derived_cast(), std::move(scalar_expr)); }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator+(T scalar, const xexpression<E>& e) { return e + scalar; }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator-(const xexpression<E>& e, T scalar) { return e + (-scalar); }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator-(T scalar, const xexpression<E>& e) { return scalar + (-e); }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator*(const xexpression<E>& e, T scalar) { return e * scalar; }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator*(T scalar, const xexpression<E>& e) { return e * scalar; }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator/(const xexpression<E>& e, T scalar) { return e * (T(1) / scalar); }
    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)> inline auto operator/(T scalar, const xexpression<E>& e) { /* TODO: implement */ return e; }

} // namespace xt

#endif // XTENSOR_XFUNCTION_HPP)>
    inline auto operator-(const xexpression<E>& e, T scalar)
    {
        using scalar_type = std::decay_t<T>;
        auto scalar_expr = xscalar<scalar_type>(scalar);
        return make_xfunction(detail::minus(), e.derived_cast(), std::move(scalar_expr));
    }

    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)>
    inline auto operator-(T scalar, const xexpression<E>& e)
    {
        using scalar_type = std::decay_t<T>;
        auto scalar_expr = xscalar<scalar_type>(scalar);
        return make_xfunction(detail::minus(), std::move(scalar_expr), e.derived_cast());
    }

    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)>
    inline auto operator*(const xexpression<E>& e, T scalar)
    {
        using scalar_type = std::decay_t<T>;
        auto scalar_expr = xscalar<scalar_type>(scalar);
        if constexpr (is_bignumber_expression_v<E> && std::is_same_v<scalar_type, bignumber::BigNumber>)
        {
            return make_xfunction(detail::bignumber_fft_multiply(),
                                  e.derived_cast(), std::move(scalar_expr));
        }
        else
        {
            return make_xfunction(detail::multiplies(),
                                  e.derived_cast(), std::move(scalar_expr));
        }
    }

    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)>
    inline auto operator*(T scalar, const xexpression<E>& e)
    {
        return e * scalar;
    }

    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)>
    inline auto operator/(const xexpression<E>& e, T scalar)
    {
        using scalar_type = std::decay_t<T>;
        auto scalar_expr = xscalar<scalar_type>(scalar);
        return make_xfunction(detail::divides(), e.derived_cast(), std::move(scalar_expr));
    }

    template <class E, class T, XTL_REQUIRES(!is_xexpression<T>::value)>
    inline auto operator/(T scalar, const xexpression<E>& e)
    {
        using scalar_type = std::decay_t<T>;
        auto scalar_expr = xscalar<scalar_type>(scalar);
        return make_xfunction(detail::divides(), std::move(scalar_expr), e.derived_cast());
    }

} // namespace xt

#endif // XTENSOR_XFUNCTION_HPP
