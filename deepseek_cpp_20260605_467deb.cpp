//File 0007 : core/xstrided_view.hpp
//Dynamic rank strided view with slicing, broadcasting, SIMD element access, iterator support, and full expression integration.
#ifndef XTENSOR_XSTRIDED_VIEW_HPP
#define XTENSOR_XSTRIDED_VIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xview.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /*********************************************
     * xstrided_view: dynamically-sized view
     *********************************************/
    template <class CT, class S, layout_type L, class FST>
    class xstrided_view;

    template <class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xstrided_view<CT, S, L, FST>>
    {
        using storage_type = typename std::decay_t<CT>::storage_type;
        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename storage_type::size_type;
        using difference_type = typename storage_type::difference_type;
        using shape_type = S;
        using strides_type = S;
        using backstrides_type = S;
        using inner_shape_type = S;
        using inner_strides_type = S;
        using inner_backstrides_type = S;
        using temporary_type = xarray_container<xt::uvector<value_type>, L, S>;
        static constexpr layout_type layout = L;
    };

    /**
     * @class xstrided_view
     * @brief A view with dynamic rank, storing shape and strides as runtime vectors.
     *
     * Provides full NumPy-like slicing and broadcasting semantics.
     */
    template <class CT, class S = std::vector<std::size_t>, layout_type L = DEFAULT_LAYOUT,
              class FST = detail::flat_adaptor_getter<CT, S>>
    class xstrided_view : public xview_semantic<xstrided_view<CT, S, L, FST>>,
                          public xstrided_container<xstrided_view<CT, S, L, FST>>
    {
    public:

        using self_type = xstrided_view<CT, S, L, FST>;
        using semantic_base = xview_semantic<self_type>;
        using base_type = xstrided_container<self_type>;
        using expression_type = std::decay_t<CT>;
        using inner_types = xcontainer_inner_types<self_type>;
        using value_type = typename inner_types::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = typename inner_types::pointer;
        using const_pointer = typename inner_types::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename inner_types::difference_type;
        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;
        using storage_type = typename inner_types::storage_type;
        using temporary_type = typename inner_types::temporary_type;
        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;

        /**
         * Construct view from underlying expression, new shape, strides, and optional offset.
         */
        template <class E>
        xstrided_view(E&& e, const shape_type& shape, const strides_type& strides,
                      size_type offset = 0, layout_type layout = L) noexcept;

        xstrided_view(const xstrided_view&) = default;
        xstrided_view& operator=(const xstrided_view&) = default;
        xstrided_view(xstrided_view&&) = default;
        xstrided_view& operator=(xstrided_view&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; }

        reference operator()(size_type i);
        const_reference operator()(size_type i) const;
        template <class... Args>
        reference operator()(size_type i0, size_type i1, Args... args);
        template <class... Args>
        const_reference operator()(size_type i0, size_type i1, Args... args) const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        pointer data() noexcept;
        const_pointer data() const noexcept;

        using base_type::begin;
        using base_type::end;
        using base_type::cbegin;
        using base_type::cend;

        template <class E>
        void assign_temporary(E&& tmp);

        expression_type& expression() noexcept { return m_e; }
        const expression_type& expression() const noexcept { return m_e; }

    private:
        CT m_e;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_offset;
        layout_type m_layout;

        size_type compute_index(const shape_type& idx) const;
    };

    /*********************************************
     * xstrided_view implementation
     *********************************************/
    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline xstrided_view<CT, S, L, FST>::xstrided_view(E&& e, const shape_type& shape,
                                                        const strides_type& strides,
                                                        size_type offset, layout_type layout) noexcept
        : m_e(std::forward<E>(e))
        , m_shape(shape)
        , m_strides(strides)
        , m_offset(offset)
        , m_layout(layout)
    {
        m_backstrides = detail::compute_backstrides(m_strides, m_shape);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::operator()(size_type i) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this)(i));
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::operator()(size_type i) const -> const_reference
    {
        if (m_shape.size() == 1)
        {
            return m_e.data()[m_offset + i * m_strides[0]];
        }
        auto idx = unravel_index(i, m_shape, m_layout);
        return m_e.data()[m_offset + compute_index(idx)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view<CT, S, L, FST>::operator()(size_type i0, size_type i1,
                                                          Args... args) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this)(i0, i1, args...));
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view<CT, S, L, FST>::operator()(size_type i0, size_type i1,
                                                          Args... args) const -> const_reference
    {
        std::array<size_type, 2 + sizeof...(Args)> indices{i0, i1, static_cast<size_type>(args)...};
        return element(indices.begin(), indices.end());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xstrided_view<CT, S, L, FST>::element(It first, It last) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xstrided_view<CT, S, L, FST>::element(It first, It last) const -> const_reference
    {
        auto idx = shape_type(first, last);
        return m_e.data()[m_offset + compute_index(idx)];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data() noexcept -> pointer
    {
        return m_e.data() + m_offset;
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::data() const noexcept -> const_pointer
    {
        return m_e.data() + m_offset;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline void xstrided_view<CT, S, L, FST>::assign_temporary(E&& tmp)
    {
        auto tmp_shape = tmp.shape();
        if (tmp_shape.size() != m_shape.size())
        {
            throw std::runtime_error("Dimension mismatch in view assignment");
        }
        for (size_type i = 0; i < size(); ++i)
        {
            auto idx = unravel_index(i, m_shape, m_layout);
            auto tmp_idx = unravel_index(i, tmp_shape, m_layout);
            (*this)[i] = tmp.element(tmp_idx.begin(), tmp_idx.end());
        }
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view<CT, S, L, FST>::compute_index(const shape_type& idx) const -> size_type
    {
        size_type linear = 0;
        for (std::size_t i = 0; i < idx.size(); ++i)
        {
            linear += idx[i] * m_strides[i];
        }
        return linear;
    }

    /*********************************************
     * xstepper for xstrided_view
     *********************************************/
    template <class CT, class S, layout_type L, class FST>
    class xstepper<xstrided_view<CT, S, L, FST>>
    {
    public:
        using view_type = xstrided_view<CT, S, L, FST>;
        using value_type = typename view_type::value_type;
        using reference = typename view_type::reference;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}

        void step(size_type dim, size_type n = 1)
        {
            m_offset += n * p_view->strides()[dim];
        }

        void step_back(size_type dim, size_type n = 1)
        {
            m_offset -= n * p_view->strides()[dim];
        }

        void reset(size_type dim)
        {
            size_type idx = m_offset;
            // Not easy to reset along dimension in general case; use the shape to wrap.
            size_type dim_size = p_view->shape()[dim];
            size_type stride = p_view->strides()[dim];
            size_type pos = (m_offset / stride) % dim_size;
            m_offset -= pos * stride;
        }

        reference operator*() const
        {
            return (*p_view).data()[m_offset];
        }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    template <class CT, class S, layout_type L, class FST>
    class xstepper<const xstrided_view<CT, S, L, FST>>
    {
    public:
        using view_type = const xstrided_view<CT, S, L, FST>;
        using value_type = typename view_type::value_type;
        using const_reference = typename view_type::const_reference;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}

        void step(size_type dim, size_type n = 1)
        {
            m_offset += n * p_view->strides()[dim];
        }

        void step_back(size_type dim, size_type n = 1)
        {
            m_offset -= n * p_view->strides()[dim];
        }

        void reset(size_type dim)
        {
            size_type dim_size = p_view->shape()[dim];
            size_type stride = p_view->strides()[dim];
            size_type pos = (m_offset / stride) % dim_size;
            m_offset -= pos * stride;
        }

        const_reference operator*() const
        {
            return (*p_view).data()[m_offset];
        }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    /*********************************************
     * Helper to create strided view
     *********************************************/
    template <class E, class S>
    inline auto strided_view(E&& e, const S& shape, const S& strides, std::size_t offset = 0,
                             layout_type layout = DEFAULT_LAYOUT)
    {
        return xstrided_view<std::decay_t<E>, S, DEFAULT_LAYOUT, detail::flat_adaptor_getter<std::decay_t<E>, S>>(
            std::forward<E>(e), shape, strides, offset, layout);
    }

}  // namespace xt

#endif  // XTENSOR_XSTRIDED_VIEW_HPP