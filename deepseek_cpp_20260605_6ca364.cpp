//File 0061 : views/xstrided_view_base.hpp
//Common base class for all strided views providing shape/stride storage, offset handling, element access, and layout deduction.
#ifndef XTENSOR_XSTRIDED_VIEW_BASE_HPP
#define XTENSOR_XSTRIDED_VIEW_BASE_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xlayout.hpp"
#include "../core/xshape.hpp"
#include "../core/xaccessible.hpp"

namespace xt
{
    /*********************************************
     * xstrided_view_base – common base for views
     *********************************************/
    template <class D, class CT, class S, layout_type L, class FST>
    class xstrided_view_base : public xexpression<D>,
                               public xview_semantic<D>,
                               public xaccessible<D>
    {
    public:
        using derived_type = D;
        using expression_type = std::decay_t<CT>;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using inner_types = xcontainer_inner_types<D>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using storage_type = typename inner_types::storage_type;

        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;
        using iterator = xiterator<D>;
        using const_iterator = xconst_iterator<D>;

        /**
         * Construct from expression, shape, strides, offset, and layout.
         */
        template <class E>
        xstrided_view_base(E&& e, const shape_type& shape, const strides_type& strides,
                           size_type offset = 0, layout_type layout = L) noexcept
            : m_e(std::forward<E>(e))
            , m_shape(shape)
            , m_strides(strides)
            , m_offset(offset)
            , m_layout(layout)
        {
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
            if (m_layout == layout_type::dynamic)
                m_layout = deduce_layout(m_shape, m_strides);
        }

        /**
         * Construct from expression and shape with implicit strides.
         */
        template <class E>
        xstrided_view_base(E&& e, const shape_type& shape, layout_type layout = L) noexcept
            : m_e(std::forward<E>(e))
            , m_shape(shape)
            , m_offset(0)
            , m_layout(layout)
        {
            m_strides = compute_strides(m_shape, m_layout);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        xstrided_view_base(const xstrided_view_base&) = default;
        xstrided_view_base& operator=(const xstrided_view_base&) = default;
        xstrided_view_base(xstrided_view_base&&) = default;
        xstrided_view_base& operator=(xstrided_view_base&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; compute_strides_from_shape(); }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        reference operator[](size_type i) { return derived_cast()(i); }
        const_reference operator[](size_type i) const { return derived_cast()(i); }

        /**
         * Multi-dimensional element access from index pair iterators.
         */
        template <class It>
        const_reference element(It first, It last) const
        {
            auto idx = shape_type(first, last);
            return m_e.data()[m_offset + compute_offset(idx)];
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const derived_type*>(this)->element(first, last));
        }

        pointer data() noexcept { return m_e.data() + m_offset; }
        const_pointer data() const noexcept { return m_e.data() + m_offset; }

        // Iterators
        iterator begin() noexcept { return iterator(&derived_cast(), 0); }
        iterator end() noexcept { return iterator(&derived_cast(), size()); }
        const_iterator begin() const noexcept { return const_iterator(&derived_cast(), 0); }
        const_iterator end() const noexcept { return const_iterator(&derived_cast(), size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper
        stepper stepper_begin() noexcept { return stepper(&derived_cast(), m_offset); }
        stepper stepper_end() noexcept { return stepper(&derived_cast(), m_offset + size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(&derived_cast(), m_offset); }
        const_stepper stepper_end() const noexcept { return const_stepper(&derived_cast(), m_offset + size()); }

        // SIMD load
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            bool contiguous = true;
            std::size_t expected = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_shape.size()) - 1; d >= 0; --d)
            {
                if (m_strides[static_cast<std::size_t>(d)] != expected)
                {
                    contiguous = false;
                    break;
                }
                expected *= m_shape[static_cast<std::size_t>(d)];
            }
            if (contiguous)
            {
                return simd_type::load_unaligned(m_e.data() + m_offset + i);
            }
            alignas(64) std::array<T, simd_type::size> buffer;
            for (std::size_t k = 0; k < simd_type::size; ++k)
                buffer[k] = derived_cast()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

        expression_type& expression() noexcept { return m_e; }
        const expression_type& expression() const noexcept { return m_e; }
        size_type offset() const noexcept { return m_offset; }
        layout_type layout() const noexcept { return m_layout; }

        template <class E>
        void assign_temporary(E&& tmp)
        {
            for (size_type i = 0; i < size(); ++i)
            {
                auto idx = unravel_index(i, m_shape, m_layout);
                derived_cast()(i) = tmp.element(idx.begin(), idx.end());
            }
        }

    protected:
        CT m_e;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_offset;
        layout_type m_layout;

        derived_type& derived_cast() noexcept { return *static_cast<derived_type*>(this); }
        const derived_type& derived_cast() const noexcept { return *static_cast<const derived_type*>(this); }

        size_type compute_offset(const shape_type& idx) const noexcept
        {
            size_type off = 0;
            for (std::size_t d = 0; d < idx.size(); ++d)
                off += idx[d] * m_strides[d];
            return off;
        }

        void compute_strides_from_shape()
        {
            m_strides = xt::compute_strides(m_shape, m_layout);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }
    };

    // Inner types for xstrided_view_base – only used as base, but needed for traits
    template <class D, class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xstrided_view_base<D, CT, S, L, FST>>
    {
        using value_type = typename std::decay_t<CT>::value_type;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using storage_type = typename std::decay_t<CT>::storage_type;
        static constexpr layout_type layout = L;
    };

} // namespace xt

#endif // XTENSOR_XSTRIDED_VIEW_BASE_HPP