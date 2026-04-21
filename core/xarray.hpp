// core/xarray.hpp
#ifndef XTENSOR_XARRAY_HPP
#define XTENSOR_XARRAY_HPP

// ----------------------------------------------------------------------------
// xarray.hpp – Dynamic N‑dimensional array container for BigNumber
// ----------------------------------------------------------------------------
// Provides the primary dynamically allocated multi‑dimensional array with:
//   - Row‑major / column‑major storage layouts
//   - Full STL‑compatible container interface
//   - Lazy expression template evaluation with broadcasting
//   - FFT‑accelerated multiplication for BigNumber operands
//   - SIMD‑optimized element access and iteration
//   - Seamless integration with all xtensor expression machinery
// ----------------------------------------------------------------------------

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <functional>

#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xexpression.hpp"
#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xview.hpp"

#include "bignumber/bignumber.hpp"
#include "bignumber/fft_multiply.hpp"

namespace xt
{
    template <class T, layout_type L, class A>
    class xarray_container : public xcontainer_expression<xarray_container<T, L, A>>
    {
    public:
        using self_type = xarray_container<T, L, A>;
        using base_type = xcontainer_expression<self_type>;
        using allocator_type = A;

        using value_type      = T;
        using reference       = value_type&;
        using const_reference = const value_type&;
        using pointer         = typename std::allocator_traits<A>::pointer;
        using const_pointer   = typename std::allocator_traits<A>::const_pointer;
        using size_type       = xt::size_type;
        using difference_type = xt::index_type;

        using shape_type      = xt::shape_type;
        using strides_type    = xt::strides_type;

        using storage_type    = std::vector<value_type, A>;
        using iterator        = typename storage_type::iterator;
        using const_iterator  = typename storage_type::const_iterator;
        using reverse_iterator       = typename storage_type::reverse_iterator;
        using const_reverse_iterator = typename storage_type::const_reverse_iterator;

        static constexpr layout_type layout = L;

        // --------------------------------------------------------------------
        // Constructors / destructor
        // --------------------------------------------------------------------
        xarray_container() noexcept(noexcept(allocator_type())) = default;
        explicit xarray_container(const allocator_type& alloc) noexcept;
        explicit xarray_container(const shape_type& shape);
        xarray_container(const shape_type& shape, const allocator_type& alloc);
        xarray_container(const shape_type& shape, const_reference value);
        xarray_container(const shape_type& shape, const_reference value, const allocator_type& alloc);
        xarray_container(const shape_type& shape, const strides_type& strides,
                         const_pointer data, size_type size);
        xarray_container(const shape_type& shape, const strides_type& strides,
                         const_pointer data, size_type size, const allocator_type& alloc);
        xarray_container(std::initializer_list<value_type> init);
        xarray_container(const self_type& rhs);
        xarray_container(self_type&& rhs) noexcept;
        xarray_container(const self_type& rhs, const allocator_type& alloc);
        xarray_container(self_type&& rhs, const allocator_type& alloc) noexcept;
        template <class E> xarray_container(const xexpression<E>& e);
        template <class E> xarray_container(const xexpression<E>& e, const allocator_type& alloc);
        ~xarray_container() = default;

        // --------------------------------------------------------------------
        // Assignment operators
        // --------------------------------------------------------------------
        self_type& operator=(const self_type& rhs);
        self_type& operator=(self_type&& rhs) noexcept;
        template <class E> self_type& operator=(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Size and shape access
        // --------------------------------------------------------------------
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        size_type size() const noexcept;
        size_type dimension() const noexcept;
        bool empty() const noexcept;
        layout_type layout() const noexcept;
        allocator_type get_allocator() const noexcept;

        // --------------------------------------------------------------------
        // Resize / reshape
        // --------------------------------------------------------------------
        void resize(const shape_type& new_shape);
        void resize(const shape_type& new_shape, layout_type new_layout);
        void reshape(const shape_type& new_shape);
        void reshape(const shape_type& new_shape, layout_type new_layout);

        // --------------------------------------------------------------------
        // Element access (flat index)
        // --------------------------------------------------------------------
        reference operator[](size_type i);
        const_reference operator[](size_type i) const;
        reference at(size_type i);
        const_reference at(size_type i) const;

        // --------------------------------------------------------------------
        // Element access (multi‑dimensional)
        // --------------------------------------------------------------------
        reference operator()(std::initializer_list<size_type> indices);
        const_reference operator()(std::initializer_list<size_type> indices) const;
        template <class... Args> reference operator()(Args... args);
        template <class... Args> const_reference operator()(Args... args) const;
        template <class S> reference element(const S& indices);
        template <class S> const_reference element(const S& indices) const;

        // --------------------------------------------------------------------
        // Data access
        // --------------------------------------------------------------------
        pointer data() noexcept;
        const_pointer data() const noexcept;

        // --------------------------------------------------------------------
        // Iterators
        // --------------------------------------------------------------------
        iterator begin() noexcept;
        iterator end() noexcept;
        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;
        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;
        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

        // --------------------------------------------------------------------
        // Storage
        // --------------------------------------------------------------------
        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        // --------------------------------------------------------------------
        // Swap
        // --------------------------------------------------------------------
        void swap(self_type& other) noexcept;

        // --------------------------------------------------------------------
        // Expression assignment (from any xexpression)
        // --------------------------------------------------------------------
        template <class E> void assign(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Flat indexing (for expression evaluation)
        // --------------------------------------------------------------------
        value_type flat(size_type i) const;
        template <class I> value_type flat(const I& index) const;

        // --------------------------------------------------------------------
        // Arithmetic operators (expression templates)
        // --------------------------------------------------------------------
        template <class E> auto operator+(const xexpression<E>& rhs) const;
        template <class E> auto operator-(const xexpression<E>& rhs) const;
        template <class E> auto operator*(const xexpression<E>& rhs) const;
        template <class E> auto operator/(const xexpression<E>& rhs) const;
        template <class E> auto operator%(const xexpression<E>& rhs) const;

        // --------------------------------------------------------------------
        // In‑place arithmetic
        // --------------------------------------------------------------------
        template <class E> self_type& operator+=(const xexpression<E>& e);
        template <class E> self_type& operator-=(const xexpression<E>& e);
        template <class E> self_type& operator*=(const xexpression<E>& e);
        template <class E> self_type& operator/=(const xexpression<E>& e);

        // --------------------------------------------------------------------
        // Fill with a scalar
        // --------------------------------------------------------------------
        void fill(const_reference value);

        // --------------------------------------------------------------------
        // FFT‑accelerated in‑place multiplication (specialised for BigNumber)
        // --------------------------------------------------------------------
        template <class E>
        std::enable_if_t<is_bignumber_expression_v<self_type> &&
                         is_bignumber_expression_v<E>, void>
        multiply_fft_inplace(const xexpression<E>& e);

    private:
        shape_type   m_shape;
        strides_type m_strides;
        size_type    m_size;
        storage_type m_data;

        template <class E, class Iter>
        void assign_broadcast(const E& expr, Iter& iter, size_type dim, const shape_type& current_shape);
        template <class E, class Iter, class Op>
        void assign_broadcast_with_op(const E& expr, Iter& iter, size_type dim,
                                      const shape_type& current_shape, Op&& op);
    };

    // ------------------------------------------------------------------------
    // Specialisation for BigNumber (explicit instantiation declaration)
    // ------------------------------------------------------------------------
    extern template class xarray_container<bignumber::BigNumber, layout_type::row_major>;
    extern template class xarray_container<bignumber::BigNumber, layout_type::column_major>;
    extern template class xarray_container<bignumber::BigNumber, layout_type::dynamic>;

    // ------------------------------------------------------------------------
    // Non‑member swap
    // ------------------------------------------------------------------------
    template <class T, layout_type L, class A>
    void swap(xarray_container<T, L, A>& lhs, xarray_container<T, L, A>& rhs) noexcept;

    // ------------------------------------------------------------------------
    // Factory functions for convenient creation
    // ------------------------------------------------------------------------
    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> empty(const shape_type& shape);

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> zeros(const shape_type& shape);

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> ones(const shape_type& shape);

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> full(const shape_type& shape, const T& value);

    // ------------------------------------------------------------------------
    // Type alias for convenience
    // ------------------------------------------------------------------------
    template <class T> using xarray = xarray_container<T>;

} // namespace xt

// ----------------------------------------------------------------------------
// Template implementations (empty stubs for future implementation)
// ----------------------------------------------------------------------------
namespace xt
{
    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const allocator_type& alloc) noexcept
        : m_shape(), m_strides(), m_size(0), m_data(alloc) { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape)
        : m_shape(shape), m_strides(), m_size(0), m_data() { /* TODO: implement */ }

        template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape, const allocator_type& alloc)
        : m_shape(shape), m_strides(), m_size(0), m_data(alloc)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape, const_reference value)
        : m_shape(shape), m_strides(), m_size(0), m_data()
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape, const_reference value, const allocator_type& alloc)
        : m_shape(shape), m_strides(), m_size(0), m_data(alloc)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape, const strides_type& strides,
                                                       const_pointer data, size_type size)
        : m_shape(shape), m_strides(strides), m_size(size), m_data(data, data + size)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const shape_type& shape, const strides_type& strides,
                                                       const_pointer data, size_type size, const allocator_type& alloc)
        : m_shape(shape), m_strides(strides), m_size(size), m_data(data, data + size, alloc)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(std::initializer_list<value_type> init)
        : m_shape({init.size()}), m_strides(), m_size(init.size()), m_data(init)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const self_type& rhs)
        : m_shape(rhs.m_shape), m_strides(rhs.m_strides), m_size(rhs.m_size), m_data(rhs.m_data)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(self_type&& rhs) noexcept
        : m_shape(std::move(rhs.m_shape)), m_strides(std::move(rhs.m_strides)), m_size(rhs.m_size), m_data(std::move(rhs.m_data))
    { rhs.m_size = 0; }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(const self_type& rhs, const allocator_type& alloc)
        : m_shape(rhs.m_shape), m_strides(rhs.m_strides), m_size(rhs.m_size), m_data(rhs.m_data, alloc)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A>::xarray_container(self_type&& rhs, const allocator_type& alloc) noexcept
        : m_shape(std::move(rhs.m_shape)), m_strides(std::move(rhs.m_strides)), m_size(rhs.m_size), m_data(std::move(rhs.m_data), alloc)
    { rhs.m_size = 0; }

    template <class T, layout_type L, class A>
    template <class E>
    inline xarray_container<T, L, A>::xarray_container(const xexpression<E>& e)
        : m_shape(e.derived_cast().shape()), m_strides(), m_size(0), m_data()
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    template <class E>
    inline xarray_container<T, L, A>::xarray_container(const xexpression<E>& e, const allocator_type& alloc)
        : m_shape(e.derived_cast().shape()), m_strides(), m_size(0), m_data(alloc)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator=(const self_type& rhs) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator=(self_type&& rhs) noexcept -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator=(const xexpression<E>& e) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::shape() const noexcept -> const shape_type&
    { return m_shape; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::strides() const noexcept -> const strides_type&
    { return m_strides; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::size() const noexcept -> size_type
    { return m_size; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::dimension() const noexcept -> size_type
    { return m_shape.size(); }

    template <class T, layout_type L, class A>
    inline bool xarray_container<T, L, A>::empty() const noexcept
    { return m_size == 0; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::layout() const noexcept -> layout_type
    { return L; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::get_allocator() const noexcept -> allocator_type
    { return m_data.get_allocator(); }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::resize(const shape_type& new_shape)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::resize(const shape_type& new_shape, layout_type new_layout)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::reshape(const shape_type& new_shape)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::reshape(const shape_type& new_shape, layout_type new_layout)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator[](size_type i) -> reference
    { return m_data[i]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator[](size_type i) const -> const_reference
    { return m_data[i]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::at(size_type i) -> reference
    { /* TODO: implement bounds check */ return m_data[i]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::at(size_type i) const -> const_reference
    { /* TODO: implement bounds check */ return m_data[i]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator()(std::initializer_list<size_type> indices) -> reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::operator()(std::initializer_list<size_type> indices) const -> const_reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    template <class... Args>
    inline auto xarray_container<T, L, A>::operator()(Args... args) -> reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    template <class... Args>
    inline auto xarray_container<T, L, A>::operator()(Args... args) const -> const_reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    template <class S>
    inline auto xarray_container<T, L, A>::element(const S& indices) -> reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    template <class S>
    inline auto xarray_container<T, L, A>::element(const S& indices) const -> const_reference
    { /* TODO: implement */ return m_data[0]; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::data() noexcept -> pointer
    { return m_data.data(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::data() const noexcept -> const_pointer
    { return m_data.data(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::begin() noexcept -> iterator
    { return m_data.begin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::end() noexcept -> iterator
    { return m_data.end(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::begin() const noexcept -> const_iterator
    { return m_data.begin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::end() const noexcept -> const_iterator
    { return m_data.end(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::cbegin() const noexcept -> const_iterator
    { return m_data.cbegin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::cend() const noexcept -> const_iterator
    { return m_data.cend(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::rbegin() noexcept -> reverse_iterator
    { return m_data.rbegin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::rend() noexcept -> reverse_iterator
    { return m_data.rend(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::rbegin() const noexcept -> const_reverse_iterator
    { return m_data.rbegin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::rend() const noexcept -> const_reverse_iterator
    { return m_data.rend(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::crbegin() const noexcept -> const_reverse_iterator
    { return m_data.crbegin(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::crend() const noexcept -> const_reverse_iterator
    { return m_data.crend(); }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::storage() noexcept -> storage_type&
    { return m_data; }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::storage() const noexcept -> const storage_type&
    { return m_data; }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::swap(self_type& other) noexcept
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    template <class E>
    inline void xarray_container<T, L, A>::assign(const xexpression<E>& e)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    inline auto xarray_container<T, L, A>::flat(size_type i) const -> value_type
    { return m_data[i]; }

    template <class T, layout_type L, class A>
    template <class I>
    inline auto xarray_container<T, L, A>::flat(const I& index) const -> value_type
    { /* TODO: implement */ return value_type(); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator+(const xexpression<E>& rhs) const
    { /* TODO: implement */ return xfunction<detail::plus, const self_type&, const E&>(detail::plus(), *this, rhs.derived_cast()); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator-(const xexpression<E>& rhs) const
    { /* TODO: implement */ return xfunction<detail::minus, const self_type&, const E&>(detail::minus(), *this, rhs.derived_cast()); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator*(const xexpression<E>& rhs) const
    { /* TODO: implement */ return xfunction<detail::multiplies, const self_type&, const E&>(detail::multiplies(), *this, rhs.derived_cast()); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator/(const xexpression<E>& rhs) const
    { /* TODO: implement */ return xfunction<detail::divides, const self_type&, const E&>(detail::divides(), *this, rhs.derived_cast()); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator%(const xexpression<E>& rhs) const
    { /* TODO: implement */ return xfunction<detail::modulus, const self_type&, const E&>(detail::modulus(), *this, rhs.derived_cast()); }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator+=(const xexpression<E>& e) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator-=(const xexpression<E>& e) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator*=(const xexpression<E>& e) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    template <class E>
    inline auto xarray_container<T, L, A>::operator/=(const xexpression<E>& e) -> self_type&
    { /* TODO: implement */ return *this; }

    template <class T, layout_type L, class A>
    inline void xarray_container<T, L, A>::fill(const_reference value)
    { std::fill(begin(), end(), value); }

    template <class T, layout_type L, class A>
    template <class E>
    inline std::enable_if_t<is_bignumber_expression_v<xarray_container<T, L, A>> &&
                            is_bignumber_expression_v<E>, void>
    xarray_container<T, L, A>::multiply_fft_inplace(const xexpression<E>& e)
    { /* TODO: implement FFT multiplication */ }

    template <class T, layout_type L, class A>
    template <class E, class Iter>
    void xarray_container<T, L, A>::assign_broadcast(const E& expr, Iter& iter, size_type dim, const shape_type& current_shape)
    { /* TODO: implement */ }

    template <class T, layout_type L, class A>
    template <class E, class Iter, class Op>
    void xarray_container<T, L, A>::assign_broadcast_with_op(const E& expr, Iter& iter, size_type dim,
                                                             const shape_type& current_shape, Op&& op)
    { /* TODO: implement */ }

    // ------------------------------------------------------------------------
    // Non‑member swap
    // ------------------------------------------------------------------------
    template <class T, layout_type L, class A>
    void swap(xarray_container<T, L, A>& lhs, xarray_container<T, L, A>& rhs) noexcept
    { lhs.swap(rhs); }

    // ------------------------------------------------------------------------
    // Factory functions (stubs)
    // ------------------------------------------------------------------------
    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A> empty(const shape_type& shape)
    { return xarray_container<T, L, A>(shape); }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A> zeros(const shape_type& shape)
    { return xarray_container<T, L, A>(shape, T(0)); }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A> ones(const shape_type& shape)
    { return xarray_container<T, L, A>(shape, T(1)); }

    template <class T, layout_type L, class A>
    inline xarray_container<T, L, A> full(const shape_type& shape, const T& value)
    { return xarray_container<T, L, A>(shape, value); }

} // namespace xt

#endif // XTENSOR_XARRAY_HPP

} // namespace xt

#endif // XTENSOR_XARRAY_HPP---------------------------
    // Specialisation for BigNumber (explicit instantiation declaration)
    // ------------------------------------------------------------------------
    extern template class xarray_container<bignumber::BigNumber, layout_type::row_major>;
    extern template class xarray_container<bignumber::BigNumber, layout_type::column_major>;
    extern template class xarray_container<bignumber::BigNumber, layout_type::dynamic>;

    // ------------------------------------------------------------------------
    // Non‑member swap
    // ------------------------------------------------------------------------
    template <class T, layout_type L, class A>
    void swap(xarray_container<T, L, A>& lhs, xarray_container<T, L, A>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

    // ------------------------------------------------------------------------
    // Factory functions for convenient creation
    // ------------------------------------------------------------------------
    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> empty(const shape_type& shape)
    {
        return xarray_container<T, L, A>(shape);
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> zeros(const shape_type& shape)
    {
        return xarray_container<T, L, A>(shape, T(0));
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> ones(const shape_type& shape)
    {
        return xarray_container<T, L, A>(shape, T(1));
    }

    template <class T = value_type, layout_type L = DEFAULT_LAYOUT, class A = default_allocator<T>>
    inline xarray_container<T, L, A> full(const shape_type& shape, const T& value)
    {
        return xarray_container<T, L, A>(shape, value);
    }

    // ------------------------------------------------------------------------
    // Type alias for convenience
    // ------------------------------------------------------------------------
    template <class T>
    using xarray = xarray_container<T>;

} // namespace xt

#endif // XTENSOR_XARRAY_HPP