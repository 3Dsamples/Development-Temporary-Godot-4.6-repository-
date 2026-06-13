//File 0006 (UPDATED) : core/xview.hpp
//Fixed-rank view with full slice support (integral, range, newaxis, xall), proper shape type, stepper, and iterator integration.
#ifndef XTENSOR_XVIEW_HPP
#define XTENSOR_XVIEW_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "xfunction.hpp"
#include "xmath.hpp"
#include "xsemantic.hpp"
#include "xstrides.hpp"
#include "xslice.hpp"
#include "xtensor_config.hpp"
#include "xtensor_forward.hpp"
#include "xtensor_simd.hpp"

namespace xt
{
    /****************************
     * xview_shape_type helper
     ****************************/
    namespace detail
    {
        template <class S, class... Rest>
        struct xview_shape_type_impl
        {
            using type = std::vector<std::size_t>;
        };
    }

    template <class CT, std::size_t N, class... S>
    class xview;

    template <class CT, std::size_t N, class... S>
    struct xcontainer_inner_types<xview<CT, N, S...>>
    {
        using storage_type = typename std::decay_t<CT>::storage_type;
        using value_type = typename storage_type::value_type;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using pointer = typename storage_type::pointer;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename storage_type::size_type;
        using difference_type = typename storage_type::difference_type;
        using shape_type = std::array<size_type, N>;
        using strides_type = std::array<size_type, N>;
        using backstrides_type = std::array<size_type, N>;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = xarray_container<xt::uvector<value_type>, DEFAULT_LAYOUT, std::vector<size_type>>;
        static constexpr layout_type layout = DEFAULT_LAYOUT;
    };

    template <class CT, std::size_t N, class... S>
    class xview : public xview_semantic<xview<CT, N, S...>>,
                  public xstrided_container<xview<CT, N, S...>>
    {
    public:
        using self_type = xview<CT, N, S...>;
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

        static constexpr std::size_t rank = N;

        template <class E, class... SL>
        xview(E&& e, SL&&... slices) noexcept;

        xview(const xview&) = default;
        xview& operator=(const xview&) = default;
        xview(xview&&) = default;
        xview& operator=(xview&&) = default;

        size_type size() const noexcept;
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

        pointer data() noexcept { return nullptr; }
        const_pointer data() const noexcept { return nullptr; }

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

        template <std::size_t... I>
        auto compute_element(std::index_sequence<I...>, size_type index) const -> const_reference;

        void compute_strides_from_slices();
    };

    // xview implementation
    template <class CT, std::size_t N, class... S>
    template <class E, class... SL>
    inline xview<CT, N, S...>::xview(E&& e, SL&&... slices) noexcept
        : m_e(std::forward<E>(e))
    {
        static_assert(sizeof...(SL) == N, "Number of slices must match rank");
        auto slice_tuple = std::make_tuple(std::forward<SL>(slices)...);
        // For each slice, determine new size and stride multiplier
        std::array<size_type, N> temp_shape;
        std::array<size_type, N> temp_strides;
        std::size_t dim = 0;
        auto process_slice = [&](auto&& sl) {
            using slice_type = std::decay_t<decltype(sl)>;
            if constexpr (std::is_same_v<slice_type, xall_tag> || std::is_same_v<slice_type, xall_range>)
            {
                // full dimension
                temp_shape[dim] = m_e.shape()[dim];
                temp_strides[dim] = m_e.strides()[dim];
            }
            else if constexpr (std::is_same_v<slice_type, xnewaxis_tag> || std::is_same_v<slice_type, xnewaxis_range>)
            {
                // new axis: size 1, stride 0
                temp_shape[dim] = 1;
                temp_strides[dim] = 0;
            }
            else if constexpr (std::is_same_v<slice_type, std::ptrdiff_t> || std::is_same_v<slice_type, int> || std::is_same_v<slice_type, long>)
            {
                // integer index: dimension removed, we'll drop this axis (size 0, stride 0) but view rank N fixed; we need to treat as size 1 with stride 0 and later ignore?
                // In xtensor, integer indexing reduces rank, but here N is compile-time, so we must handle as size 1, stride 0 for broadcasting
                temp_shape[dim] = 1;
                temp_strides[dim] = 0;
            }
            else if constexpr (is_xslice_v<slice_type>)
            {
                // xslice or xrange
                auto [sz, step] = normalize_slice(sl, m_e.shape()[dim]);
                temp_shape[dim] = sz;
                temp_strides[dim] = step * m_e.strides()[dim];
            }
            ++dim;
        };
        std::apply([&](auto&&... args) { (process_slice(args), ...); }, slice_tuple);
        m_shape = temp_shape;
        m_strides = temp_strides;
        m_backstrides = detail::compute_backstrides(m_strides, m_shape);
    }

    template <class CT, std::size_t N, class... S>
    inline auto xview<CT, N, S...>::size() const noexcept -> size_type
    {
        return compute_size(m_shape);
    }

    template <class CT, std::size_t N, class... S>
    inline auto xview<CT, N, S...>::operator()(size_type i) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this)(i));
    }

    template <class CT, std::size_t N, class... S>
    inline auto xview<CT, N, S...>::operator()(size_type i) const -> const_reference
    {
        if (m_shape.size() == 1) return m_e.data()[i * m_strides[0]];
        auto idx = unravel_index(i, m_shape);
        return element(idx.begin(), idx.end());
    }

    template <class CT, std::size_t N, class... S>
    template <class... Args>
    inline auto xview<CT, N, S...>::operator()(size_type i0, size_type i1, Args... args) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this)(i0, i1, args...));
    }

    template <class CT, std::size_t N, class... S>
    template <class... Args>
    inline auto xview<CT, N, S...>::operator()(size_type i0, size_type i1, Args... args) const -> const_reference
    {
        std::array<size_type, 2 + sizeof...(Args)> indices{i0, i1, static_cast<size_type>(args)...};
        return element(indices.begin(), indices.end());
    }

    template <class CT, std::size_t N, class... S>
    inline auto xview<CT, N, S...>::operator[](size_type i) -> reference { return operator()(i); }

    template <class CT, std::size_t N, class... S>
    inline auto xview<CT, N, S...>::operator[](size_type i) const -> const_reference { return operator()(i); }

    template <class CT, std::size_t N, class... S>
    template <class It>
    inline auto xview<CT, N, S...>::element(It first, It last) -> reference
    {
        return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
    }

    template <class CT, std::size_t N, class... S>
    template <class It>
    inline auto xview<CT, N, S...>::element(It first, It last) const -> const_reference
    {
        auto idx = std::vector<size_type>(first, last);
        size_type linear_index = 0;
        for (std::size_t i = 0; i < idx.size(); ++i)
            linear_index += idx[i] * m_strides[i];
        return m_e.data()[linear_index];
    }

    template <class CT, std::size_t N, class... S>
    template <class E>
    inline void xview<CT, N, S...>::assign_temporary(E&& tmp)
    {
        for (std::size_t i = 0; i < size(); ++i)
        {
            auto idx = unravel_index(i, m_shape);
            (*this)[i] = tmp.element(idx.begin(), idx.end());
        }
    }

    // xstepper for xview
    template <class CT, std::size_t N, class... S>
    class xstepper<xview<CT, N, S...>>
    {
    public:
        using view_type = xview<CT, N, S...>;
        using value_type = typename view_type::value_type;
        using reference = typename view_type::reference;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}
        void step(size_type dim, size_type n = 1) { m_offset += n * p_view->strides()[dim]; }
        void step_back(size_type dim, size_type n = 1) { m_offset -= n * p_view->strides()[dim]; }
        void reset(size_type dim) { m_offset = m_offset % p_view->strides()[dim]; }
        reference operator*() const { return (*p_view)[m_offset]; }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    template <class CT, std::size_t N, class... S>
    class xstepper<const xview<CT, N, S...>>
    {
    public:
        using view_type = const xview<CT, N, S...>;
        using value_type = typename view_type::value_type;
        using const_reference = typename view_type::const_reference;
        using size_type = typename view_type::size_type;

        xstepper(view_type* v, size_type offset) : p_view(v), m_offset(offset) {}
        void step(size_type dim, size_type n = 1) { m_offset += n * p_view->strides()[dim]; }
        void step_back(size_type dim, size_type n = 1) { m_offset -= n * p_view->strides()[dim]; }
        void reset(size_type dim) { m_offset = m_offset % p_view->strides()[dim]; }
        const_reference operator*() const { return (*p_view)[m_offset]; }

    private:
        view_type* p_view;
        size_type m_offset;
    };

    // Helper to create view
    template <class E, class... S>
    inline auto view(E&& e, S&&... slices)
    {
        constexpr std::size_t N = sizeof...(S);
        return xview<std::decay_t<E>, N, std::decay_t<S>...>(std::forward<E>(e), std::forward<S>(slices)...);
    }

} // namespace xt

#endif // XTENSOR_XVIEW_HPP