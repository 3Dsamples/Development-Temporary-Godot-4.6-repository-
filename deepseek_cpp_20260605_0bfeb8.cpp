//File 0058 : views/xdynamic_view.hpp
//Dynamic-rank view supporting runtime-variable slices, broadcasting, SIMD-accelerated element access, and full expression/iterator integration.
#ifndef XTENSOR_XDYNAMIC_VIEW_HPP
#define XTENSOR_XDYNAMIC_VIEW_HPP

#include <algorithm>
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

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xtensor_simd.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xmath.hpp"
#include "../core/xsemantic.hpp"
#include "../core/xstrides.hpp"
#include "../core/xslice.hpp"
#include "../core/xaccessible.hpp"
#include "../core/xiterable.hpp"
#include "../core/xlayout.hpp"

namespace xt
{
    /*********************************************
     * xdynamic_view: dynamic slicing and broadcasting
     *********************************************/
    template <class CT, class S, layout_type L, class FST>
    class xdynamic_view;

    template <class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xdynamic_view<CT, S, L, FST>>
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
        using temporary_type = xarray_container<uvector<value_type>, L, S>;
        static constexpr layout_type layout = L;
    };

    template <class CT, class S = std::vector<std::size_t>, layout_type L = DEFAULT_LAYOUT,
              class FST = detail::flat_adaptor_getter<CT, S>>
    class xdynamic_view : public xexpression<xdynamic_view<CT, S, L, FST>>,
                          public xview_semantic<xdynamic_view<CT, S, L, FST>>,
                          public xaccessible<xdynamic_view<CT, S, L, FST>>
    {
    public:
        using self_type = xdynamic_view<CT, S, L, FST>;
        using base_type = xexpression<self_type>;
        using semantic_base = xview_semantic<self_type>;
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
        using expression_type = std::decay_t<CT>;

        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<self_type>;
        using const_iterator = xconst_iterator<self_type>;

        /**
         * Build a dynamic view by slicing the expression with any number of slices at runtime.
         * Slices can be: xall_tag, xnewaxis_tag, xrange, xslice, integral indices.
         */
        template <class E, class... Slices>
        xdynamic_view(E&& e, Slices&&... slices)
            : m_e(std::forward<E>(e))
        {
            auto slice_tuple = std::make_tuple(std::forward<Slices>(slices)...);
            auto old_shape = m_e.shape();
            auto old_strides = compute_strides(old_shape, L);
            // Compute new shape and strides using the slice tuple
            std::tie(m_shape, m_strides) = compute_sliced_view(slice_tuple, old_shape, old_strides);
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
            m_layout = L;
        }

        /**
         * Construct a dynamic view by providing explicit shape, strides, and offset.
         */
        template <class E>
        xdynamic_view(E&& e, const shape_type& shape, const strides_type& strides,
                     size_type offset = 0, layout_type layout = L) noexcept
            : m_e(std::forward<E>(e)), m_shape(shape), m_strides(strides), m_offset(offset), m_layout(layout)
        {
            m_backstrides = detail::compute_backstrides(m_strides, m_shape);
        }

        xdynamic_view(const self_type&) = default;
        xdynamic_view& operator=(const self_type&) = default;
        xdynamic_view(self_type&&) = default;
        xdynamic_view& operator=(self_type&&) = default;

        size_type size() const noexcept { return compute_size(m_shape); }
        const shape_type& shape() const noexcept { return m_shape; }
        const strides_type& strides() const noexcept { return m_strides; }
        const backstrides_type& backstrides() const noexcept { return m_backstrides; }

        void set_shape(const shape_type& s) { m_shape = s; }
        void set_strides(const strides_type& st) { m_strides = st; m_backstrides = detail::compute_backstrides(st, m_shape); }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            std::array<size_type, sizeof...(Args)> idx{static_cast<size_type>(args)...};
            return element(idx.begin(), idx.end());
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this)(args...));
        }

        reference operator[](size_type i) { return operator()(i); }
        const_reference operator[](size_type i) const { return operator()(i); }

        template <class It>
        const_reference element(It first, It last) const
        {
            std::vector<size_type> idx(first, last);
            size_type offset = m_offset;
            for (std::size_t d = 0; d < idx.size(); ++d)
                offset += idx[d] * m_strides[d];
            return m_e.data()[offset];
        }

        template <class It>
        reference element(It first, It last)
        {
            return const_cast<reference>(static_cast<const self_type&>(*this).element(first, last));
        }

        pointer data() noexcept { return m_e.data() + m_offset; }
        const_pointer data() const noexcept { return m_e.data() + m_offset; }

        // Iterators
        iterator begin() noexcept { return iterator(this, 0); }
        iterator end() noexcept { return iterator(this, size()); }
        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }
        const_iterator cbegin() const noexcept { return begin(); }
        const_iterator cend() const noexcept { return end(); }

        // Stepper
        stepper stepper_begin() noexcept { return stepper(this, m_offset); }
        stepper stepper_end() noexcept { return stepper(this, m_offset + size()); }
        const_stepper stepper_begin() const noexcept { return const_stepper(this, m_offset); }
        const_stepper stepper_end() const noexcept { return const_stepper(this, m_offset + size()); }

        // SIMD load
        template <class Align, class T = value_type>
        auto load_simd(std::size_t i) const
        {
            using simd_type = xsimd::batch<T, default_simd_arch>;
            // For contiguous views, we can load directly; otherwise fallback to scalar.
            bool contiguous = true;
            std::size_t expected_stride = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(m_shape.size()) - 1; d >= 0; --d)
            {
                if (m_strides[static_cast<std::size_t>(d)] != expected_stride)
                {
                    contiguous = false;
                    break;
                }
                expected_stride *= m_shape[static_cast<std::size_t>(d)];
            }
            if (contiguous)
            {
                return simd_type::load_unaligned(m_e.data() + m_offset + i);
            }
            // Fallback: scalar gather
            alignas(64) std::array<T, simd_type::size> buffer;
            for (std::size_t k = 0; k < simd_type::size; ++k)
                buffer[k] = operator()(i + k);
            return simd_type::load_aligned(buffer.data());
        }

        template <class E>
        void assign_temporary(E&& tmp)
        {
            for (size_type i = 0; i < size(); ++i)
            {
                auto idx = unravel_index(i, m_shape);
                (*this)[i] = tmp.element(idx.begin(), idx.end());
            }
        }

        const expression_type& expression() const noexcept { return m_e; }
        size_type offset() const noexcept { return m_offset; }
        layout_type layout() const noexcept { return m_layout; }

    private:
        CT m_e;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        size_type m_offset = 0;
        layout_type m_layout = L;
    };

    template <class E, class... Slices>
    inline auto dynamic_view(E&& e, Slices&&... slices)
    {
        return xdynamic_view<std::decay_t<E>, std::vector<std::size_t>>(
            std::forward<E>(e), std::forward<Slices>(slices)...);
    }

} // namespace xt

#endif // XTENSOR_XDYNAMIC_VIEW_HPP